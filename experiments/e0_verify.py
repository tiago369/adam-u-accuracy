"""E0 - verification: does the test rig measure anything, and how small can it measure?

  1. Ideal plant (stock dynamics after the model fixes: no armature, ideal sensing, no noise), run twice with the
     same seed: bitwise identical, RP = 0, and AP below 0.3 mm. A perfect simulator returns empty numbers.
  2. Baseline plant (rotor inertia, encoder quantisation/noise, torque noise, 1 ms latency): the floor of RP and AP
     that the controller itself produces, before any physical error source is added.
  3. Time-step convergence: the full plant (all mechanisms on) at 1 ms vs 0.5 ms physics step, same controller.
  4. Instrument floor: the virtual mocap's AP/RP on the baseline against ground truth.

usage: python -m experiments.e0_verify [--quick]
"""
import numpy as np

from adam_u_accuracy.model import Scenario
from adam_u_accuracy.sim import Rig
from adam_u_accuracy.instruments import INSTRUMENT_FUNCS
from adam_u_accuracy.metrics import pose_metrics
from experiments.common import parse, save


def main():
    args = parse(__doc__)
    n = 5 if args.quick else 30
    out = {}

    ideal = Scenario(armature=False, noise=False)
    r1, r2 = Rig(ideal, 0), Rig(ideal, 0)
    a, b = r1.run_cycles(3, cycle_seed=0), r2.run_cycles(3, cycle_seed=0)
    pc, Rc = r1.commanded()
    m = pose_metrics(a['p'], a['R'], pc, Rc)
    out['ideal'] = dict(bitwise_identical=bool(all(np.array_equal(a[k], b[k]) for k in a)),
                        cycle_spread_max_m=float(np.abs(a['p'] - a['p'][0]).max()),
                        AP_mean_mm=m['AP_mean'], AP_max_mm=m['AP_max'], RP_max_mm=m['RP_max'])
    print('ideal plant:', out['ideal'])

    rig = Rig(Scenario(), 0)
    run = rig.run_cycles(n, cycle_seed=1)
    out['baseline'] = {}
    rng = np.random.default_rng(3)
    for name, f in INSTRUMENT_FUNCS.items():
        p, R = f(run, rig.nom, rng)
        mm = pose_metrics(p, R, pc, Rc, ci=name == 'ground_truth', rng=rng)
        out['baseline'][name] = mm
        print(f"baseline, {name:13s}: AP mean {mm['AP_mean']:.4f} mm, RP mean {mm['RP_mean']:.4f} mm, "
              f"RP_a/b/c mean {np.round(mm['RP_abc'].mean((0, 1)), 4)} deg")
    gt = run['p']
    for name in ['mocap', 'laser_tracker']:
        p, _ = INSTRUMENT_FUNCS[name](run, rig.nom, rng)
        out.setdefault('instrument_floor_mm', {})[name] = float(np.sqrt(((p - gt) ** 2).sum(-1).mean()) * 1e3)
    print('instrument RMS error vs truth (mm):', out['instrument_floor_mm'])

    full = Scenario(zero_offset_deg=0.3, mass_frac=0.05, com_mm=5.0, friction=0.1, backlash_deg=0.2, stiffness=3000.0)
    runs = [Rig(full, 0, substeps=s).run_cycles(3 if args.quick else 10, cycle_seed=2) for s in (1, 2)]
    ms = [pose_metrics(r['p'], r['R'], pc, Rc) for r in runs]
    out['timestep'] = dict(AP_mean_mm={'1ms': ms[0]['AP_mean'], '0.5ms': ms[1]['AP_mean']},
                           RP_mean_mm={'1ms': ms[0]['RP_mean'], '0.5ms': ms[1]['RP_mean']},
                           barycentre_diff_max_mm=float(np.abs(runs[0]['p'].mean(0) - runs[1]['p'].mean(0)).max() * 1e3))
    print('time step check:', out['timestep'])
    out['n_cycles'] = n
    save('e0_verify', out, args.quick)


if __name__ == '__main__':
    main()
