"""E3 - the realistic robot: accuracy across a fleet, direction dependence, and visit order.

All mechanisms on at their declared levels (zero offsets 0.3 deg, link geometry 1 mm, mass 5 % / CoM 5 mm, friction
0.1 N.m, backlash 0.2 deg, series compliance 3000 N.m/rad), SDK gains.

  a. Fleet Monte Carlo: U virtual robots x 3 cycles of the ISO order. Accuracy varies between robots (each has its
     own geometry and offsets), so it needs many units and few cycles. Reported per instrument, and split into
     servo error (encoder FK - command, fixable by gains/feed-forward) and kinematic + post-encoder error
     (truth - encoder FK, fixable only by calibration).
  b. Multi-directional accuracy vAP (ISO 9283 sec. 7.2.4) at P1, P2, P4: approach from 50 mm along +-x, +-y, +-z,
     n repetitions per direction, for the baseline, friction only, backlash only and all-on plants. This is where
     friction and backlash appear, because the ISO cycle approaches each pose from one direction only.
  c. Visit order: the same 5 poses, 30 cycles in ISO order vs in a fresh random order per cycle (so the approach
     direction changes between visits), on the all-on plant.

usage: python -m experiments.e3_realistic [--quick] [--procs N]
"""
import time
import numpy as np

from adam_u_accuracy.config import S_MIRROR, N_WARMUP, SETTLE, T_MOVE
from adam_u_accuracy.model import Scenario
from adam_u_accuracy.sim import Rig, Nominal
from adam_u_accuracy.poses import LeftArmIK, R_PALM_DOWN, load_poses, POSES_JSON
from adam_u_accuracy.campaign import run_jobs, scenario_dict, UNIT_SEED0, CYCLE_SEED0
from adam_u_accuracy.metrics import pose_metrics, rp_l, vap
from experiments.common import parse, save

ALL_ON = Scenario(zero_offset_deg=0.3, geometry_mm=1.0, mass_frac=0.05, com_mm=5.0, friction=0.1,
                  backlash_deg=0.2, stiffness=3000.0)
VAP_PLANTS = {'baseline': Scenario(), 'friction': Scenario(friction=0.1), 'backlash': Scenario(backlash_deg=0.2),
              'all_on': ALL_ON}
DIRS = ['+x', '-x', '+y', '-y', '+z', '-z']
APPROACH = 0.05


def pre_poses():
    """19-vector pre-poses: pose P approached along direction k starts at P - 50 mm * e_k (left arm; right mirrored)."""
    import json
    P = json.load(open(POSES_JSON))['poses']
    ik = LeftArmIK()
    out = {}
    for name in ['P1', 'P2', 'P4']:
        p = np.array(P[name]['p_torso'])
        for dname in DIRS:
            e = np.zeros(3); e['xyz'.index(dname[1])] = 1.0 if dname[0] == '+' else -1.0
            q, ep, er = ik.solve(p - APPROACH * e, R_PALM_DOWN)
            assert ep < 1e-5 and er < 1e-3, (name, dname, ep, er)
            q19 = np.zeros(19); q19[5:12], q19[12:19] = q, S_MIRROR * q
            out[f'{name}{dname}'] = q19
    return out


def job_vap(job):
    t0 = time.perf_counter()
    sc = Scenario(**job['scenario'])
    rig = Rig(sc, UNIT_SEED0 + job['unit'])
    rng = np.random.default_rng(CYCLE_SEED0 + job['unit'])
    P, pre, name = rig.poses, job['pre'], job['pose']
    rig.reset(P[name])
    rig.segment(P[name], P[name], rng, T=0.0, dwell=SETTLE, record=False)
    cur = P[name]
    meas = {k: [] for k in DIRS}
    for rep in range(job['n'] + 1):              # first round is warm-up
        for dname in DIRS:
            q_pre = np.array(pre[f'{name}{dname}'])
            rig.segment(cur, q_pre, rng, T=1.0, dwell=1.0, record=False)
            m = rig.segment(q_pre, P[name], rng, T=1.0)
            cur = P[name]
            if rep > 0:
                meas[dname].append(m['p'])
    pc, _ = rig.nom.fk(P[name])
    res = {}
    for a, arm in enumerate(['Left', 'Right']):
        groups = [np.array(meas[k])[:, a] for k in DIRS]
        res[arm] = dict(vAP=vap(groups),
                        AP_dir={k: float(np.linalg.norm(g.mean(0) - pc[a]) * 1e3) for k, g in zip(DIRS, groups)},
                        RP_dir={k: float(rp_l(g) * 1e3) for k, g in zip(DIRS, groups)},
                        bary_dir_mm={k: (g.mean(0) - pc[a]) * 1e3 for k, g in zip(DIRS, groups)})
    return dict(cell=f"{job['plant']}/{name}", unit=job['unit'], plant=job['plant'], pose=name, res=res,
                wall_s=time.perf_counter() - t0,
                summary=f"vAP L {res['Left']['vAP']:.3f} mm R {res['Right']['vAP']:.3f} mm")


def job_order(job):
    t0 = time.perf_counter()
    rig = Rig(Scenario(**job['scenario']), UNIT_SEED0 + job['unit'])
    seq0 = ['P5', 'P4', 'P3', 'P2', 'P1']
    if job['order'] == 'iso':
        run = rig.run_cycles(job['n'], cycle_seed=CYCLE_SEED0 + job['unit'])
        p = run['p']                                    # (C, 5, 2, 3), already in seq0 order
    else:
        rng = np.random.default_rng(CYCLE_SEED0 + job['unit'])
        order_rng = np.random.default_rng(99)
        P = rig.poses
        rig.reset(P['P1']); rig.segment(P['P1'], P['P1'], rng, T=0.0, dwell=SETTLE, record=False)
        prev, store = 'P1', {k: [] for k in seq0}
        for c in range(N_WARMUP + job['n']):
            seq = list(order_rng.permutation(seq0))
            if seq[0] == prev:                          # never "move" to the pose we are already at
                seq = seq[1:] + seq[:1]
            for name in seq:
                m = rig.segment(P[prev], P[name], rng, record=c >= N_WARMUP)
                if m is not None:
                    store[name].append(m['p'])
                prev = name
        p = np.stack([np.array(store[k]) for k in seq0], 1)
    pc, Rc = rig.commanded()
    R_dummy = np.broadcast_to(Rc, p.shape[:1] + Rc.shape)
    m = pose_metrics(p, R_dummy, pc, Rc)
    return dict(cell=f"order/{job['order']}", unit=job['unit'], order=job['order'], AP_p=m['AP_p'], RP_l=m['RP_l'],
                AP_mean=m['AP_mean'], RP_mean=m['RP_mean'], wall_s=time.perf_counter() - t0,
                summary=f"AP {m['AP_mean']:.3f} mm RP {m['RP_mean']:.3f} mm")


def main():
    args = parse(__doc__)
    q = args.quick
    out = {'scenario_all_on': scenario_dict(ALL_ON)}

    # a. fleet Monte Carlo
    n_units = 4 if q else 30
    jobs = [dict(exp='e3_quick' if q else 'e3', cell='fleet', scenario=scenario_dict(ALL_ON), unit=u, n_cycles=3,
                 instruments=['ground_truth', 'encoder_fk', 'mocap']) for u in range(n_units)]
    fleet = run_jobs(jobs, args.procs)
    out['fleet'] = dict(n_units=n_units, n_cycles=3, runs=fleet)

    # b. vAP
    pre = {k: v.tolist() for k, v in pre_poses().items()}
    n_rep = 3 if q else 10
    jobs = [dict(plant=k, scenario=scenario_dict(sc), unit=0, pose=pn, n=n_rep, pre=pre)
            for k, sc in VAP_PLANTS.items() for pn in ['P1', 'P2', 'P4']]
    out['vap'] = dict(n_rep=n_rep, approach_m=APPROACH, runs=run_jobs(jobs, args.procs, fn=job_vap))

    # c. visit order
    n_cyc = 5 if q else 30
    units = 1 if q else 3
    jobs = [dict(order=o, scenario=scenario_dict(ALL_ON), unit=u, n=n_cyc) for o in ['iso', 'random'] for u in range(units)]
    out['order'] = dict(n_cycles=n_cyc, runs=run_jobs(jobs, args.procs, fn=job_order))
    save('e3_realistic', out, q)


if __name__ == '__main__':
    main()
