"""E2 - 2^(6-1) fractional factorial (resolution VI, generator F = ABCDE): which mechanisms drive AP and RP, and do
they interact?

Factors, low / high:
    A  joint zero offsets        0 / sigma 0.3 deg
    B  inertial model error      0 / mass x U(0.95, 1.05), CoM +-5 mm (feed-forward stays nominal)
    C  Coulomb friction          0 / 0.1 N.m on the arm joints
    D  backlash                  0 / 0.2 deg total gap on the arm joints
    E  series compliance         rigid / 3000 N.m/rad on the 4 proximal arm joints
    F  gain set                  SDK / pnd_mujoco example (10x kp on shoulder pitch and roll)
32 cells x U units (common random numbers: unit u is the same virtual robot in every cell) x 30 cycles.
Resolution VI: main effects and all two-factor interactions are clear of each other (aliased only with 4- and
5-factor interactions). Responses per unit: AP (ground truth), AP seen by encoder FK, RP (ground truth).
Effects are computed on cell means; % contribution = SS_effect / SS_total.

usage: python -m experiments.e2_factorial [--quick] [--procs N]
"""
import itertools
import numpy as np

from adam_u_accuracy.model import Scenario
from adam_u_accuracy.campaign import run_jobs, scenario_dict
from experiments.common import parse, save

FACTORS = 'ABCDEF'
NAMES = dict(A='zero offsets', B='inertial error', C='friction', D='backlash', E='compliance', F='gains (mujoco_ex)')


def design():
    rows = []
    for a, b, c, d, e in itertools.product([-1, 1], repeat=5):
        rows.append((a, b, c, d, e, a * b * c * d * e))
    return np.array(rows)


def scenario(row):
    a, b, c, d, e, f = (x > 0 for x in row)
    return Scenario(zero_offset_deg=0.3 if a else 0.0, mass_frac=0.05 if b else 0.0, com_mm=5.0 if b else 0.0,
                    friction=0.1 if c else 0.0, backlash_deg=0.2 if d else 0.0, stiffness=3000.0 if e else None,
                    gains='mujoco_ex' if f else 'sdk')


def cell_name(row):
    return ''.join(FACTORS[i].lower() if x > 0 else '-' for i, x in enumerate(row))


def effects(X, y):
    """Main effects and two-factor interactions of response y (one value per row of X)."""
    cols = {f: X[:, i] for i, f in enumerate(FACTORS)}
    for i, j in itertools.combinations(range(6), 2):
        cols[FACTORS[i] + FACTORS[j]] = X[:, i] * X[:, j]
    ss_tot = ((y - y.mean()) ** 2).sum()
    out = {}
    for k, c in cols.items():
        eff = y[c > 0].mean() - y[c < 0].mean()
        out[k] = dict(effect=float(eff), pct=float(100 * len(y) * eff ** 2 / 4 / ss_tot))
    return out


def main():
    args = parse(__doc__)
    units, n = (1, 5) if args.quick else (3, 30)
    X = design()
    jobs = [dict(exp='e2', cell=cell_name(r), scenario=scenario_dict(scenario(r)), unit=u, n_cycles=n,
                 instruments=['ground_truth', 'encoder_fk', 'mocap'])
            for r in X for u in range(units)]
    res = run_jobs(jobs, args.procs)
    by_cell = {}
    for r in res:
        by_cell.setdefault(r['cell'], []).append(r)
    resp = {}
    for key, inst, metric in [('AP_truth', 'ground_truth', 'AP_mean'), ('AP_encoder', 'encoder_fk', 'AP_mean'),
                              ('RP_truth', 'ground_truth', 'RP_mean'), ('AP_mocap', 'mocap', 'AP_mean')]:
        y = np.array([np.mean([u['metrics'][inst][metric] for u in by_cell[cell_name(r)]]) for r in X])
        resp[key] = dict(cell_means=y, effects=effects(X, y))
    for key, v in resp.items():
        top = sorted(v['effects'].items(), key=lambda kv: -kv[1]['pct'])[:6]
        print(f'{key}: ' + ', '.join(f"{k} {e['effect']:+.3f} mm ({e['pct']:.0f}%)" for k, e in top))
    save('e2_factorial', dict(factors=NAMES, design=X, cells=[cell_name(r) for r in X], units=units, n_cycles=n,
                              responses=resp, runs=res), args.quick)


if __name__ == '__main__':
    main()
