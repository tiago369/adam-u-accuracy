"""ISO 9283 test poses P1-P5 for each arm, solved once by IK on the nominal model and frozen to JSON.

Cube (chosen in the design study): side 175 mm, centre (0.3875, +-0.2625, -0.0125) m in the torso frame, limited by
the +-0.96 rad wrist range rather than by reach. P1 = centre; P2-P5 lie on the diagonal plane through the near-bottom
and far-top edges, 10 % of the diagonal in from the corners. Two tool orientations, fingers forward: palm down
(results/poses.json) and palm in (results/poses_palm_in.json), so that each wrist joint is gravity-loaded in one.
Extra poses (results/poses_extra.json, joint-space, both orientations' cube excluded):
  P6 - arm almost hanging, low gravity load: backlash is not preloaded there;
  P7 - arm extended forward at shoulder width: largest lever arm, for compliance.
IK: damped least squares with a null-space pull to a fixed rest posture (fixed redundancy resolution). The right arm
is the exact mirror q_R = S q_L. Run `python -m adam_u_accuracy.poses` to regenerate `results/poses.json`.
"""
import os
import json
import numpy as np
import mujoco

from .config import ARM, RESULTS, S_MIRROR
from .model import nominal_model

CUBE_CENTRE = np.array([0.3875, 0.2625, -0.0125])
CUBE_SIDE = 0.175
R_PALM_DOWN = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)   # columns: hand x, y, z in torso frame
R_PALM_IN = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)      # palm faces the body midline
ORIENTATIONS = {'palm_down': R_PALM_DOWN, 'palm_in': R_PALM_IN}
EXTRA_Q = {'P6': [-0.15, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0], 'P7': [-1.4, 0.4, 0.0, -0.15, 0.0, 0.0, 0.0]}
Q_REST = np.array([-0.6, 0.25, 0.0, -1.3, 0.0, 0.0, 0.0])
POSES_JSON = os.path.join(RESULTS, 'poses.json')
POSE_FILES = {'palm_down': POSES_JSON, 'palm_in': os.path.join(RESULTS, 'poses_palm_in.json'),
              'extra': os.path.join(RESULTS, 'poses_extra.json')}


def iso_points(C=CUBE_CENTRE, L=CUBE_SIDE):
    h = L / 2
    corners = [C + np.array(c) for c in [(-h, -h, -h), (-h, h, -h), (h, h, h), (h, -h, h)]]
    return [C] + [C + 0.8 * (c - C) for c in corners]


class LeftArmIK:
    def __init__(self, m=None):
        self.m = m if m is not None else nominal_model()
        self.d = mujoco.MjData(self.m)
        names = [f'{a}_Left' for a in ARM]
        self.qa = np.array([self.m.joint(n).qposadr[0] for n in names])
        self.va = np.array([self.m.joint(n).dofadr[0] for n in names])
        jid = [self.m.joint(n).id for n in names]
        self.lo, self.hi = self.m.jnt_range[jid, 0].copy(), self.m.jnt_range[jid, 1].copy()
        self.site = self.m.site('tcp_l').id
        mujoco.mj_kinematics(self.m, self.d)
        self.p_torso = self.d.xpos[self.m.body('torso').id].copy()
        self.jp, self.jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))

    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site] - self.p_torso, self.d.site_xmat[self.site].reshape(3, 3).copy()

    def _step(self, q, p_t, R_t, null_gain=0.5, rot_w=0.3):
        p, R = self.fk(q)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
        J = np.vstack([self.jp[:, self.va], rot_w * self.jr[:, self.va]])
        qt, qc, rv = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(qt, R_t.flatten()); mujoco.mju_mat2Quat(qc, R.flatten()); mujoco.mju_subQuat(rv, qt, qc)
        e = np.concatenate([p_t - p, rot_w * (R @ rv)])
        lam = 1e-4 + 0.05 * min(1.0, np.linalg.norm(e))
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
        W = 1.0 / (self.hi - self.lo)
        dq += (np.eye(7) - Jp @ J) @ (null_gain * (Q_REST - q) * W / W.max())
        s = np.abs(dq).max()
        if s > 0.2:
            dq *= 0.2 / s
        return np.clip(q + dq, self.lo, self.hi), np.linalg.norm(e[:3]), np.linalg.norm(rv)

    def solve(self, p_t, R_t, iters=6000):
        """Iterate until both the task error and the null-space motion converge (unique, reproducible solution)."""
        q = np.clip(Q_REST.copy(), self.lo, self.hi)
        for _ in range(iters):
            q2, ep, er = self._step(q, p_t, R_t)
            if np.abs(q2 - q).max() < 1e-12:
                break
            q = q2
        return q2, ep, er


def solve_poses(orientation='palm_down'):
    ik = LeftArmIK()
    R_t = ORIENTATIONS[orientation]
    out = dict(frame='torso (waist at 0)', cube_centre=CUBE_CENTRE.tolist(), cube_side=CUBE_SIDE,
               orientation=orientation, R_tool=R_t.tolist(), poses={})
    for i, p in enumerate(iso_points()):
        q, ep, er = ik.solve(p, R_t)
        frac = np.minimum(q - ik.lo, ik.hi - q) / (ik.hi - ik.lo)
        out['poses'][f'P{i + 1}'] = dict(p_torso=p.tolist(), q_left=q.tolist(), q_right=(S_MIRROR * q).tolist(),
                                         ik_err_mm=ep * 1e3, ik_err_deg=float(np.degrees(er)),
                                         min_limit_margin=float(frac.min()))
    return out


def extra_poses():
    ik = LeftArmIK()
    out = dict(frame='torso (waist at 0)', poses={})
    for k, q in EXTRA_Q.items():
        q = np.array(q)
        p, _ = ik.fk(q)
        frac = np.minimum(q - ik.lo, ik.hi - q) / (ik.hi - ik.lo)
        out['poses'][k] = dict(p_torso=p.tolist(), q_left=q.tolist(), q_right=(S_MIRROR * q).tolist(),
                               min_limit_margin=float(frac.min()))
    return out


def load_poses(which='palm_down'):
    """dict pose name -> 19-vector of commanded joint angles (waist and neck at 0). which: palm_down | palm_in | extra"""
    with open(POSE_FILES[which]) as f:
        P = json.load(f)['poses']
    out = {}
    for k, v in P.items():
        q = np.zeros(19)
        q[5:12], q[12:19] = v['q_left'], v['q_right']
        out[k] = q
    return out


if __name__ == '__main__':
    os.makedirs(RESULTS, exist_ok=True)
    for orientation in ORIENTATIONS:
        out = solve_poses(orientation)
        for k, v in out['poses'].items():
            print(f"{orientation} {k}: p={np.round(v['p_torso'], 4)} q={np.round(v['q_left'], 3)} IK err "
                  f"{v['ik_err_mm']:.1e} mm {v['ik_err_deg']:.1e} deg, min limit margin {v['min_limit_margin']:.2f}")
        with open(POSE_FILES[orientation], 'w') as f:
            json.dump(out, f, indent=1)
    out = extra_poses()
    for k, v in out['poses'].items():
        print(f"extra {k}: p={np.round(v['p_torso'], 3)} min limit margin {v['min_limit_margin']:.2f}")
    with open(POSE_FILES['extra'], 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', ', '.join(POSE_FILES.values()))
