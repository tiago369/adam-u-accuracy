"""E1 - one factor at a time: how much AP and RP does each physical error source produce on its own, and which
instrument sees it?

Each cell adds one mechanism to the baseline plant at its declared "on" level. Random mechanisms (zero offsets,
geometry, mass/CoM) are drawn per virtual robot, so they get several units; deterministic ones get one unit.
All cells run the ISO 9283 cycle P5 -> P1, 3 warm-up cycles + 30 measured cycles, both arms.

usage: python -m experiments.e1_ofat [--quick] [--procs N]
"""
from adam_u_accuracy.model import Scenario
from adam_u_accuracy.campaign import run_jobs, scenario_dict
from experiments.common import parse, save

CELLS = {   # name: (scenario, units)
    'baseline':     (Scenario(), 1),
    'zero_offset':  (Scenario(zero_offset_deg=0.3), 3),
    'geometry':     (Scenario(geometry_mm=1.0), 3),
    'mass_com':     (Scenario(mass_frac=0.05, com_mm=5.0), 3),
    'friction':     (Scenario(friction=0.1), 1),
    'backlash':     (Scenario(backlash_deg=0.2), 1),
    'compliance':   (Scenario(stiffness=3000.0), 1),
    'gains_mjex':   (Scenario(gains='mujoco_ex'), 1),
    'payload':      (Scenario(payload_kg=0.5), 1),
}


def main():
    args = parse(__doc__)
    n = 5 if args.quick else 30
    jobs = [dict(exp='e1', cell=c, scenario=scenario_dict(sc), unit=u, n_cycles=n, ci=True)
            for c, (sc, nu) in CELLS.items() for u in range(1 if args.quick else nu)]
    res = run_jobs(jobs, args.procs)
    save('e1_ofat', dict(n_cycles=n, cells={c: scenario_dict(sc) for c, (sc, _) in CELLS.items()}, runs=res), args.quick)


if __name__ == '__main__':
    main()
