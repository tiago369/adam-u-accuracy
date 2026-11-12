"""Build the nominal model (used by IK, FK and feed-forward) and the error-injected plant.

Fixes applied to the stock `adam_u.xml` for both models:
  * the base freejoint is removed, so the lifting column is welded at 1.05 m (the stock model drops ~18 mm at start);
  * a TCP site `tcp_l` / `tcp_r` is added 0.12 m below each wristRoll frame (palm centre);
  * arm motor torque limits are set to PND's peak torques (stock: 65 N.m on shoulderYaw instead of 17.5 N.m);
  * meshes are stripped (every body carries an explicit <inertial>, so dynamics are unchanged) and contacts are
    disabled: the test poses are collision-free, and contacts would only slow the simulation down.
The SDK's virtual elastic band lives in `simulate_python`, not in the MJCF, so it is simply not used here.

Errors are injected ONLY into the plant. Structural errors (friction, backlash, series compliance, rotor inertia)
are XML edits; parametric errors (joint zero offsets, link geometry, mass/CoM, payload) are edits on the compiled
model arrays, drawn from a per-unit random stream so that a "unit" is one virtual robot.
"""
import os
import copy
import dataclasses
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from .config import STOCK_XML, ARM, SIDES, DT, PEAK_TORQUE, REFLECTED_INERTIA, TCP_OFFSET

# Passive joints added after the motor: their intermediate inertia must keep the spring mode well below Nyquist.
# With 0.001 kg.m^2 the 3000 N.m/rad mode sits at omega*dt = 1.7 and rattles inside the backlash gap for ever
# (a numerical limit cycle that shows up as fake repeatability); 0.01 kg.m^2 gives omega*dt = 0.55.
J_MID = 0.01          # kg.m^2, gear-output-side inertia of the passive joints
DAMP_K = 5.0          # N.m.s/rad, damping of the series spring (zeta ~ 0.5)
SOLREF_GAP = 0.004    # s, time constant of the backlash end stops (MuJoCo recommends >= 2 dt)

ARM_BODIES = ['shoulderPitch', 'shoulderRoll', 'shoulderYaw', 'elbow', 'wristYaw', 'wristPitch', 'wristRoll']


@dataclasses.dataclass(frozen=True)
class Scenario:
    """Error magnitudes injected into the plant. Zero / None means the mechanism is absent."""
    zero_offset_deg: float = 0.0     # sigma of each arm/waist joint zero offset (encoder vs link), deg
    geometry_mm: float = 0.0         # sigma of each arm body origin offset, per axis, mm
    mass_frac: float = 0.0           # every body mass scaled by U(1 - f, 1 + f) (feed-forward stays nominal)
    com_mm: float = 0.0              # every body CoM shifted by U(-c, c) per axis, mm
    friction: float = 0.0            # Coulomb friction (frictionloss) on the 7 arm joints, N.m
    backlash_deg: float = 0.0        # total gear-play gap on the 7 arm joints, deg (after the encoder)
    stiffness: float | None = None   # series compliance after the encoder on the 4 proximal arm joints, N.m/rad
    gains: str = 'sdk'               # PD gain set, see config.GAINS
    payload_kg: float = 0.0          # unmodelled point mass 0.10 m below each wristRoll origin
    armature: bool = True            # reflected rotor inertia (dynamics only)
    noise: bool = True               # encoder noise + quantisation, torque noise, latency (cycle-to-cycle variation)

    def label(self):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return ', '.join(f'{k}={v}' for k, v in d.items() if v != getattr(NOMINAL_PLANT, k)) or 'baseline'


NOMINAL_PLANT = Scenario()


def _base_tree():
    tree = ET.parse(STOCK_XML)
    root = tree.getroot()
    # strip meshes, mesh geoms and the contact section
    asset = root.find('asset')
    for mesh in asset.findall('mesh'):
        asset.remove(mesh)
    for parent in root.iter():
        for g in list(parent.findall('geom')):
            if 'mesh' in g.attrib:
                parent.remove(g)
    contact = root.find('contact')
    if contact is not None:
        root.remove(contact)
    # weld the base
    col = root.find(".//body[@name='lifting_Columns']")
    col.remove(col.find("joint[@name='floating_base']"))
    # TCP sites and a torso reference site
    for s in SIDES:
        wr = root.find(f".//body[@name='wristRoll{s}']")
        jnt = wr.find('joint')
        site = ET.Element('site', name=f'tcp_{s[0].lower()}', pos=' '.join(map(str, TCP_OFFSET)), size='0.01')
        wr.insert(list(wr).index(jnt) + 1, site)
    # arm torque limits from the datasheet
    act = root.find('actuator')
    for s in SIDES:
        for a, tau in zip(ARM, PEAK_TORQUE):
            m = act.find(f"motor[@joint='{a}_{s}']")
            m.set('ctrlrange', f'{-tau:g} {tau:g}')
            j = root.find(f".//joint[@name='{a}_{s}']")
            j.set('actuatorfrcrange', f'{-tau:g} {tau:g}')
    opt = ET.SubElement(root, 'option', timestep=str(DT), integrator='implicitfast')
    ET.SubElement(opt, 'flag', contact='disable')
    return tree


_BASE = None


def _xml_for(sc: Scenario | None):
    """XML string for the nominal model (sc=None) or for a plant with the structural errors of `sc`."""
    global _BASE
    if _BASE is None:
        _BASE = _base_tree()
    tree = copy.deepcopy(_BASE)
    root = tree.getroot()
    if sc is not None:
        for s in SIDES:
            for k, (a, J) in enumerate(zip(ARM, REFLECTED_INERTIA)):
                body = root.find(f".//body[@name='{a}{s}']")
                j = body.find(f"joint[@name='{a}_{s}']")
                if sc.armature:
                    j.set('armature', f'{J:g}')
                if sc.friction > 0:
                    j.set('frictionloss', f'{sc.friction:g}')
                axis = j.get('axis')
                pos = list(body).index(j) + 1
                if sc.stiffness is not None and k < 4:
                    body.insert(pos, ET.Element('joint', name=f'{a}_{s}_k', axis=axis, stiffness=f'{sc.stiffness:g}',
                                                damping=f'{DAMP_K:g}', armature=f'{J_MID:g}'))
                    pos += 1
                if sc.backlash_deg > 0:
                    h = np.radians(sc.backlash_deg) / 2
                    body.insert(pos, ET.Element('joint', name=f'{a}_{s}_bl', axis=axis, limited='true',
                                                range=f'{-h:.7g} {h:.7g}', solreflimit=f'{SOLREF_GAP:g} 1',
                                                solimplimit='0.95 0.99 0.0001', damping='0.05', armature=f'{J_MID:g}'))
    return ET.tostring(root, encoding='unicode')


def nominal_model():
    return mujoco.MjModel.from_xml_string(_xml_for(None))


def plant_model(sc: Scenario, unit_rng: np.random.Generator | None = None):
    """Compiled plant for scenario `sc`; parametric errors are drawn from `unit_rng` (one virtual robot)."""
    m = mujoco.MjModel.from_xml_string(_xml_for(sc))
    if sc.friction > 0:
        m.opt.noslip_iterations = 3      # without it, frictionloss lets the arm creep away under gravity
    rng = unit_rng if unit_rng is not None else np.random.default_rng(0)
    # draw every random quantity in a fixed order, whether used or not, so that units stay comparable across cells
    z = rng.standard_normal(3 + 14)
    g = rng.standard_normal((2, 7, 3))
    mf = rng.uniform(-1, 1, m.nbody)
    cf = rng.uniform(-1, 1, (m.nbody, 3))
    if sc.zero_offset_deg > 0:
        names = ['waistRoll', 'waistPitch', 'waistYaw'] + [f'{a}_{s}' for s in SIDES for a in ARM]
        for n, zi in zip(names, z):
            m.qpos0[m.joint(n).qposadr[0]] += np.radians(sc.zero_offset_deg) * zi
    if sc.geometry_mm > 0:
        for i, s in enumerate(SIDES):
            for k, b in enumerate(ARM_BODIES):
                m.body_pos[m.body(b + s).id] += sc.geometry_mm * 1e-3 * g[i, k]
    if sc.mass_frac > 0:
        m.body_mass[1:] *= 1 + sc.mass_frac * mf[1:]
    if sc.com_mm > 0:
        m.body_ipos[1:] += sc.com_mm * 1e-3 * cf[1:]
    if sc.payload_kg > 0:
        for s in SIDES:
            b = m.body('wristRoll' + s).id
            m0, pp = m.body_mass[b], np.array([0, 0, -0.10])
            m.body_ipos[b] = (m0 * m.body_ipos[b] + sc.payload_kg * pp) / (m0 + sc.payload_kg)
            m.body_mass[b] = m0 + sc.payload_kg
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


def write_viewer_scene(path, sc: Scenario | None = None):
    """Write a standalone XML of the (meshless) model, e.g. for `python -m mujoco.viewer --mjcf=...`."""
    with open(path, 'w') as f:
        f.write(_xml_for(sc))
    return path
