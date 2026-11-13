"""E4 - the remaining blocks of the proposal's design.

  a. Palm-in orientation: the E1 cells repeated with the tool palm-in, so that the other wrist joints carry gravity.
  b. vAP at the extra poses: P6 (arm almost hanging, low gravity load, backlash not preloaded) and P7 (arm extended
     forward, largest lever arm), approach from 50 mm along +-x, +-y, +-z (directions out of reach are skipped).
  c. Dwell sensitivity, stabilisation and creep (ISO 9283 sec. 7.5-7.7): the ISO cycle with an 8 s dwell and the TCP
     logged every 1 ms. Per arrival: AP measured in windows ending at 2, 4 and 8 s; stabilisation time (last exit of
     the 50 ms-smoothed TCP from a 0.2 mm band around the final position, i.e. the mean over 7.5-8 s); overshoot past
     the final position along the move direction, after arrival; creep rate between the 3.5-4 s and 7.5-8 s windows
     (on the baseline this is the noise wander of the window means, the reference for the other plants).
  d. ISO 230-2 per joint, left arm: each joint alone, 5 targets over +-0.15 rad around its P1 angle, linear cycle
     (up through the targets, then down), 5 cycles, 0.05 rad overtravel at both ends, 5 s dwell
     (the SDK gains give a ~0.4 s joint time constant; a 2 s dwell leaves ~1 mrad of settling lag that reads as reversal). Deviation = attained - target,
     read at the encoder and at the true link (after zero offset, backlash and series spring). Per joint: reversal
     B, repeatability R, systematic deviation E and accuracy A (ISO 230-2:2014 definitions, angles in mrad).
  e. Static compliance at P1 and P7: +-1 N at the left TCP along x, y, z, central difference -> 3x3 compliance
     matrix (mm/N), for SDK vs pnd_mujoco gains, with and without the series spring. Friction is off (it would make
     the response hysteretic rather than elastic).

usage: python -m experiments.e4_extras [--quick] [--procs N]
"""
import time
import numpy as np

from adam_u_accuracy.config import ARM, SEQUENCE, SETTLE, DT
from adam_u_accuracy.model import Scenario
from adam_u_accuracy.sim import Rig
from adam_u_accuracy.poses import load_poses
from adam_u_accuracy.campaign import run_jobs, scenario_dict, UNIT_SEED0, CYCLE_SEED0
from experiments.common import parse, save
from experiments.e1_ofat import CELLS
from experiments.e3_realistic import ALL_ON, VAP_PLANTS, pre_poses, job_vap

PLANTS = {'baseline': Scenario(), 'friction': Scenario(friction=0.1), 'backlash': Scenario(backlash_deg=0.2),
          'compliance': Scenario(stiffness=3000.0), 'all_on': ALL_ON}
BAND = 0.2e-3        # stabilisation band; must exceed the noise wander of the TCP (~RP = 0.09 mm)
SMOOTH = 50          # moving-average length for the stabilisation test, control periods (50 ms)


# ------------------------------------------------------------------ c. dwell / stabilisation / creep
def job_dwell(job):
    t0 = time.perf_counter()
    rig = Rig(Scenario(**job['scenario']), UNIT_SEED0 + job['unit'])
    rng = np.random.default_rng(CYCLE_SEED0 + job['unit'])
    P = rig.poses
    rig.reset(P['P1']); rig.segment(P['P1'], P['P1'], rng, T=0.0, dwell=SETTLE, record=False)
    pc, _ = rig.commanded()
    n_move = int(round(2.0 / DT))
    win = lambda tr, a, b: tr[n_move + int(a / DT):n_move + int(b / DT)].mean(0)
    rows, prev = [], 'P1'
    for c in range(1 + job['n']):
        for i, name in enumerate(SEQUENCE):
            m = rig.segment(P[prev], P[name], rng, dwell=8.0, record=False, trace=True)
            tr = m['trace']
            x0 = tr[0]
            prev = name
            if c == 0:
                continue
            final = win(tr, 7.5, 8.0)                                # (2, 3)
            ap = {f'{t:g}s': np.linalg.norm(win(tr, t - 0.5, t) - pc[i], axis=-1) * 1e3 for t in (2, 4, 8)}
            dist = np.linalg.norm(tr[n_move:] - final, axis=-1)      # (n_dwell, 2)
            ker = np.ones(SMOOTH) / SMOOTH
            dist = np.stack([np.convolve(dist[:, a], ker, mode='valid') for a in range(2)], 1)
            out = np.where(dist > BAND, np.arange(len(dist))[:, None], -1).max(0)
            t_stab = (out + 1 + SMOOTH // 2) * DT                    # time after arrival of the last exit from the band
            u = final - x0
            u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-12)
            over = np.maximum(0, np.einsum('tak,ak->ta', tr[n_move:] - final, u).max(0)) * 1e3
            creep = np.linalg.norm(final - win(tr, 3.5, 4.0), axis=-1) / 4.0 * 1e6      # um/s
            rows.append(dict(pose=name, AP=ap, t_stab=t_stab, overshoot_mm=over, creep_um_s=creep))
    agg = lambda f: float(np.mean([f(r) for r in rows]))
    res = dict(AP_2s=agg(lambda r: r['AP']['2s'].mean()), AP_4s=agg(lambda r: r['AP']['4s'].mean()),
               AP_8s=agg(lambda r: r['AP']['8s'].mean()),
               dAP_4_8_max=float(max(np.abs(r['AP']['8s'] - r['AP']['4s']).max() for r in rows)),
               t_stab_mean=agg(lambda r: r['t_stab'].mean()), t_stab_max=float(max(r['t_stab'].max() for r in rows)),
               overshoot_mean=agg(lambda r: r['overshoot_mm'].mean()),
               overshoot_max=float(max(r['overshoot_mm'].max() for r in rows)),
               creep_mean=agg(lambda r: r['creep_um_s'].mean()), creep_max=float(max(r['creep_um_s'].max() for r in rows)))
    return dict(cell=f"dwell/{job['plant']}", unit=job['unit'], plant=job['plant'], res=res, rows=rows,
                wall_s=time.perf_counter() - t0,
                summary=f"t_stab {res['t_stab_mean']:.2f}s creep {res['creep_mean']:.1f}um/s dAP {res['dAP_4_8_max']:.3f}mm")


# ------------------------------------------------------------------ d. ISO 230-2 per joint
def iso230_stats(dev_up, dev_dn):
    """dev_up, dev_dn: (runs, targets) deviations. ISO 230-2:2014 bidirectional statistics."""
    mu, md = dev_up.mean(0), dev_dn.mean(0)
    su, sd = dev_up.std(0, ddof=1), dev_dn.std(0, ddof=1)
    Bi = mu - md
    Ri = np.maximum.reduce([2 * su + 2 * sd + np.abs(Bi), 4 * su, 4 * sd])
    A = max((mu + 2 * su).max(), (md + 2 * sd).max()) - min((mu - 2 * su).min(), (md - 2 * sd).min())
    E = max(mu.max(), md.max()) - min(mu.min(), md.min())
    return dict(B=float(np.abs(Bi).max()), B_mean=float(Bi.mean()), R=float(Ri.max()), R_up=float(4 * su.max()),
                R_dn=float(4 * sd.max()), E=float(E), A=float(A), mean_bias=float((mu + md).mean() / 2))


def job_230(job):
    t0 = time.perf_counter()
    rig = Rig(Scenario(**job['scenario']), UNIT_SEED0 + job['unit'])
    rng = np.random.default_rng(CYCLE_SEED0 + job['unit'] + 7 * job['joint'])
    j = 5 + job['joint']                                            # index in J19 (left arm)
    base = rig.poses['P1'].copy()
    targets = base[j] + np.linspace(-0.15, 0.15, 5)
    over = 0.05

    def at(v):
        q = base.copy(); q[j] = v; return q
    rig.reset(at(targets[0] - over))
    rig.segment(at(targets[0] - over), at(targets[0] - over), rng, T=0.0, dwell=SETTLE, record=False)
    cur = targets[0] - over
    up = {'enc': [], 'link': []}; dn = {'enc': [], 'link': []}
    for c in range(job['n']):
        ru, rd = {'enc': [], 'link': []}, {'enc': [], 'link': []}
        for seq, store in [(list(targets), ru), (list(targets[::-1]), rd)]:
            for v in seq:
                m = rig.segment(at(cur), at(v), rng, T=0.5, dwell=5.0, window=(4.5, 5.0))
                cur = v
                store['enc'].append(m['q_enc'][j] - v)
                store['link'].append(m['q_link'][j] - v)
            end = targets[-1] + over if store is ru else targets[0] - over
            rig.segment(at(cur), at(end), rng, T=0.5, dwell=0.5, record=False)
            cur = end
        for k in ('enc', 'link'):
            up[k].append(ru[k]); dn[k].append(rd[k][::-1])
    res = {k: iso230_stats(np.array(up[k]) * 1e3, np.array(dn[k]) * 1e3) for k in ('enc', 'link')}
    name = ARM[job['joint']]
    return dict(cell=f"230/{job['plant']}/{name}", unit=job['unit'], plant=job['plant'], joint=name, res=res,
                wall_s=time.perf_counter() - t0,
                summary=f"B link {res['link']['B']:.2f} enc {res['enc']['B']:.2f} mrad, R link {res['link']['R']:.2f} mrad")


# ------------------------------------------------------------------ e. static compliance
def job_stiff(job):
    t0 = time.perf_counter()
    rig = Rig(Scenario(**job['scenario']), UNIT_SEED0 + job['unit'], poses='extra' if job['pose'] == 'P7' else 'palm_down')
    rng = np.random.default_rng(CYCLE_SEED0 + job['unit'])
    q = rig.poses[job['pose']]
    rig.reset(q); rig.segment(q, q, rng, T=0.0, dwell=SETTLE, record=False)
    C = np.zeros((3, 3))
    F = job['force']
    for k in range(3):
        x = []
        for sgn in (1, -1):
            f = np.zeros(3); f[k] = sgn * F
            rig.wrench = (0, f)
            x.append(rig.segment(q, q, rng, T=0.0, dwell=3.0, window=(2.5, 3.0))['p'][0])
        C[:, k] = (x[0] - x[1]) / (2 * F) * 1e3                     # mm/N
    rig.wrench = None
    ev = np.linalg.eigvalsh((C + C.T) / 2)
    return dict(cell=f"stiff/{job['plant']}/{job['pose']}", unit=job['unit'], plant=job['plant'], pose=job['pose'],
                C=C, eig=ev, wall_s=time.perf_counter() - t0, summary=f"compliance eig {np.round(ev, 2)} mm/N")


def main():
    args = parse(__doc__)
    q = args.quick
    out = {}

    # a. palm-in
    n = 5 if q else 30
    jobs = [dict(exp='e4_palm_in', cell=c, scenario=scenario_dict(sc), unit=u, n_cycles=n, poses='palm_in')
            for c, (sc, nu) in CELLS.items() for u in range(1 if q else nu)]
    out['palm_in'] = dict(n_cycles=n, runs=run_jobs(jobs, args.procs))

    # b. vAP at P6, P7
    pre = {k: v.tolist() for k, v in pre_poses('extra', ('P6', 'P7')).items()}
    extra = {k: v.tolist() for k, v in load_poses('extra').items()}
    n_rep = 3 if q else 10
    jobs = [dict(plant=k, scenario=scenario_dict(sc), unit=0, pose=pn, n=n_rep, pre=pre, poses=extra)
            for k, sc in VAP_PLANTS.items() for pn in ['P6', 'P7']]
    out['vap_extra'] = dict(n_rep=n_rep, runs=run_jobs(jobs, args.procs, fn=job_vap))

    # c. dwell / stabilisation / creep
    jobs = [dict(plant=k, scenario=scenario_dict(sc), unit=0, n=1 if q else 3) for k, sc in PLANTS.items()]
    out['dwell'] = dict(band_mm=BAND * 1e3, runs=run_jobs(jobs, args.procs, fn=job_dwell))

    # d. ISO 230-2
    jobs = [dict(plant=k, scenario=scenario_dict(sc), unit=0, joint=jn, n=2 if q else 5)
            for k, sc in PLANTS.items() for jn in range(7)]
    out['iso230'] = dict(targets_rad=0.15, n_cycles=2 if q else 5, runs=run_jobs(jobs, args.procs, fn=job_230))

    # e. static compliance
    stiff = {'sdk': Scenario(), 'sdk+spring': Scenario(stiffness=3000.0), 'mujoco_ex': Scenario(gains='mujoco_ex'),
             'mujoco_ex+spring': Scenario(gains='mujoco_ex', stiffness=3000.0)}
    jobs = [dict(plant=k, scenario=scenario_dict(sc), unit=0, pose=pn, force=1.0)
            for k, sc in stiff.items() for pn in ['P1', 'P7']]
    out['stiffness'] = dict(force_N=1.0, runs=run_jobs(jobs, args.procs, fn=job_stiff))
    save('e4_extras', out, q)


if __name__ == '__main__':
    main()
