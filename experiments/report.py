"""Build figures and results/SUMMARY.md from the experiment JSON files.

usage: python -m experiments.report [--quick]     (--quick reads the *_quick.json files)
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adam_u_accuracy.config import RESULTS

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1-3 (validated reference palette)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'legend.frameon': False})
FIG = os.path.join(RESULTS, 'figures')     # --quick writes to figures_quick/


def load(name, quick):
    p = os.path.join(RESULTS, f"{name}{'_quick' if quick else ''}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def by_cell(runs):
    out = {}
    for r in runs:
        out.setdefault(r['cell'], []).append(r)
    return out


def fig_ofat(e1, lines):
    order = ['baseline', 'compliance', 'backlash', 'mass_com', 'friction', 'geometry', 'zero_offset', 'gains_mjex', 'payload']
    labels = dict(baseline='baseline (noise only)', compliance='series compliance 3000 N·m/rad',
                  backlash='backlash 0.2°', mass_com='mass ±5 %, CoM ±5 mm', friction='friction 0.1 N·m',
                  geometry='link geometry 1 mm', zero_offset='zero offsets σ 0.3°', gains_mjex='gains: pnd_mujoco example',
                  payload='payload 0.5 kg, unmodelled')
    cells = by_cell(e1['runs'])
    order = [c for c in order if c in cells]
    get = lambda c, inst, k: [u['metrics'][inst][k] for u in cells[c]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(order))[::-1]
    for inst, col, dy, name in [('ground_truth', C1, 0.17, 'ground truth'), ('encoder_fk', C2, -0.17, 'encoder FK')]:
        v = [np.mean(get(c, inst, 'AP_mean')) for c in order]
        axs[0].barh(y + dy, v, height=0.32, color=col, label=name)
        v = [np.mean(get(c, inst, 'RP_mean')) for c in order]
        axs[1].barh(y + dy, v, height=0.32, color=col, label=name)
    axs[0].set_xscale('log'); axs[1].set_xlim(left=0)
    axs[0].set_yticks(y, [labels[c] for c in order])
    axs[0].set_xlabel('AP, mean of 5 poses × 2 arms (mm, log)')
    axs[1].set_xlabel('RP, mean of 5 poses × 2 arms (mm)')
    axs[0].set_title('Accuracy: each mechanism alone', loc='left', color=INK)
    axs[1].set_title('Repeatability: each mechanism alone', loc='left', color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.55, 0.0), ncol=2)
    fig.savefig(os.path.join(FIG, 'e1_ofat.png'), dpi=160, bbox_inches='tight'); plt.close(fig)

    lines += ['## E1: one mechanism at a time', '',
              f"ISO 9283 cycle P5→P1, both arms, {e1['n_cycles']} measured cycles per unit. "
              'Random mechanisms: mean over units. AP/RP in mm, mean (max) over the 10 pose×arm combinations.', '',
              '| Mechanism | AP truth | AP encoder FK | AP mocap | RP truth | RP mocap | RP 95 % CI (worst pose) |',
              '|---|---|---|---|---|---|---|']
    for c in order:
        gt = cells[c][0]['metrics']['ground_truth']
        ci = np.array(gt['RP_l_ci']).reshape(-1, 2) if 'RP_l_ci' in gt else None
        worst = int(np.argmax(np.array(gt['RP_l']).ravel()))
        f = lambda inst, k: f"{np.mean(get(c, inst, k + '_mean')):.3f} ({np.mean(get(c, inst, k + '_max')):.3f})"
        lines.append(f"| {labels[c]} | {f('ground_truth', 'AP')} | {f('encoder_fk', 'AP')} | {f('mocap', 'AP')} | "
                     f"{f('ground_truth', 'RP')} | {f('mocap', 'RP')} | "
                     f"{'' if ci is None else f'{ci[worst, 0]:.3f}–{ci[worst, 1]:.3f}'} |")
    lines += ['', f'![E1]({os.path.basename(FIG)}/e1_ofat.png)', '']


def fig_factorial(e2, lines):
    fac = e2['factors']
    keys = [('AP_truth', 'AP, ground truth'), ('AP_encoder', 'AP seen by encoder FK'), ('RP_truth', 'RP, ground truth')]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw=dict(wspace=0.55))
    for ax, (k, title) in zip(axs, keys):
        eff = e2['responses'][k]['effects']
        top = sorted(eff.items(), key=lambda kv: -kv[1]['pct'])[:8][::-1]
        names = [n if len(n) == 2 else f"{n} {fac[n]}" for n, _ in top]
        ax.barh(range(len(top)), [e['pct'] for _, e in top], color=C1, height=0.6)
        for i, (_, e) in enumerate(top):
            ax.text(e['pct'] + 1, i, f"{e['effect']:+.2f} mm", va='center', color=INK2, fontsize=8)
        ax.set_yticks(range(len(top)), names)
        ax.set_xlim(0, 115); ax.set_xlabel('% of variance between cells')
        ax.set_title(title, loc='left', color=INK)
    fig.savefig(os.path.join(FIG, 'e2_factorial.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
    lines += ['## E2: 2^(6-1) fractional factorial', '',
              f"32 cells × {e2['units']} units × {e2['n_cycles']} cycles. Factors: " +
              ', '.join(f'{k} = {v}' for k, v in fac.items()) + '. Effect = mean(high) − mean(low) of the cell means.', '',
              '| Response | Top effects (effect in mm, % of variance) |', '|---|---|']
    for k, title in keys + [('AP_mocap', 'AP, virtual mocap')]:
        eff = e2['responses'][k]['effects']
        top = sorted(eff.items(), key=lambda kv: -kv[1]['pct'])[:5]
        lines.append(f"| {title} | " + ', '.join(f"{n} {e['effect']:+.3f} ({e['pct']:.0f} %)" for n, e in top) + ' |')
    y = np.array(e2['responses']['RP_truth']['cell_means'])
    lines += ['', f'RP across the 32 cells: min {y.min():.3f} mm, median {np.median(y):.3f} mm, max {y.max():.3f} mm.', '',
              f'![E2]({os.path.basename(FIG)}/e2_factorial.png)', '']


def fleet_split(e3, quick):
    """AP per unit: truth, servo (encoder FK), kinematic + post-encoder (truth - encoder FK), from the raw states."""
    from adam_u_accuracy.sim import Nominal
    from adam_u_accuracy.poses import load_poses
    from adam_u_accuracy.config import SEQUENCE
    nom, P = Nominal(), load_poses()
    pc = np.array([nom.fk(P[n])[0] for n in SEQUENCE])
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, 'raw', 'e3_quick' if quick else 'e3', 'fleet_u*.npz'))):
        z = np.load(f)
        enc = np.array([[nom.fk(q)[0] for q in qc] for qc in z['q_enc']])
        t = z['p'].mean(0) - pc; s = enc.mean(0) - pc
        nrm = lambda e: float(np.linalg.norm(e, axis=-1).mean() * 1e3)
        out.append((nrm(t), nrm(s), nrm(t - s)))
    return np.array(out)


def fig_e3(e3, quick, lines):
    sp = fleet_split(e3, quick)
    vap = e3['vap']['runs']
    plants = ['baseline', 'backlash', 'friction', 'all_on']
    poses = ['P1', 'P2', 'P4']
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(wspace=0.35, width_ratios=[1.1, 1]))
    rng = np.random.default_rng(0)
    for i, (lab, col) in enumerate([('total (ground truth)', C1), ('servo (encoder FK)', C2),
                                    ('kinematic + post-encoder', C3)]):
        if len(sp):
            yy = 2 - i + rng.uniform(-0.15, 0.15, len(sp))
            axs[0].scatter(sp[:, i], yy, s=22, color=col, edgecolor=SURF, linewidth=0.8, label=lab, zorder=3)
            axs[0].plot([np.median(sp[:, i])] * 2, [2 - i - 0.3, 2 - i + 0.3], color=INK, lw=2, zorder=4)
    axs[0].set_yticks([2, 1, 0], ['total\n(truth)', 'servo\n(encoder FK)', 'kinematic +\npost-encoder'])
    axs[0].set_xlabel('AP per robot, mean of 10 pose×arm (mm)'); axs[0].set_xlim(left=0)
    axs[0].set_title(f'Fleet of {len(sp)} virtual robots (bar = median)', loc='left', color=INK)
    w = 0.26
    for j, (pn, col) in enumerate(zip(poses, [C1, C2, C3])):
        v = [np.mean([r['res']['Left']['vAP'] for r in vap if r['plant'] == pl and r['pose'] == pn]) for pl in plants]
        axs[1].bar(np.arange(len(plants)) + (j - 1) * w, v, width=w - 0.03, color=col, label=pn)
    axs[1].set_yscale('log'); axs[1].set_xticks(range(len(plants)), ['baseline', 'backlash\n0.2°', 'friction\n0.1 N·m', 'all on'])
    axs[1].set_ylabel('vAP, left arm (mm, log)'); axs[1].legend(title='pose', loc='upper left')
    axs[1].set_title('Approach from ±x, ±y, ±z (50 mm)', loc='left', color=INK)
    fig.savefig(os.path.join(FIG, 'e3_realistic.png'), dpi=160, bbox_inches='tight'); plt.close(fig)

    gtA = np.array([r['metrics']['ground_truth']['AP_mean'] for r in e3['fleet']['runs']])
    moA = np.array([r['metrics']['mocap']['AP_mean'] for r in e3['fleet']['runs']])
    gtM = np.array([r['metrics']['ground_truth']['AP_max'] for r in e3['fleet']['runs']])
    lines += ['## E3: the realistic robot', '',
              'All mechanisms on (zero offsets 0.3°, geometry 1 mm, mass 5 %/CoM 5 mm, friction 0.1 N·m, '
              'backlash 0.2°, compliance 3000 N·m/rad), SDK gains.', '',
              f"**Fleet ({len(gtA)} robots × {e3['fleet']['n_cycles']} cycles).** AP (mean over poses): median "
              f"{np.median(gtA):.2f} mm, 90th percentile {np.percentile(gtA, 90):.2f} mm, worst pose on worst robot "
              f"{gtM.max():.2f} mm. Virtual mocap agrees with truth to {np.abs(moA - gtA).max() * 1e3:.0f} µm."]
    if len(sp):
        lines += [f"Split per robot (median): servo {np.median(sp[:, 1]):.2f} mm, kinematic + post-encoder "
                  f"{np.median(sp[:, 2]):.2f} mm, total {np.median(sp[:, 0]):.2f} mm."]
    lines += ['', '**Multi-directional accuracy vAP (left arm, mm).**', '',
              '| Plant | ' + ' | '.join(poses) + ' |', '|---|' + '---|' * len(poses)]
    for pl in plants:
        lines.append(f'| {pl} | ' + ' | '.join(
            f"{np.mean([r['res']['Left']['vAP'] for r in vap if r['plant'] == pl and r['pose'] == pn]):.3f}"
            for pn in poses) + ' |')
    od = by_cell(e3['order']['runs'])
    lines += ['', '**Visit order (all-on plant).**', '', '| Order | AP mean (mm) | RP mean (mm) | RP max (mm) |', '|---|---|---|---|']
    for k in ['order/iso', 'order/random']:
        rs = od.get(k, [])
        if rs:
            lines.append(f"| {k.split('/')[1]} | {np.mean([r['AP_mean'] for r in rs]):.3f} | "
                         f"{np.mean([r['RP_mean'] for r in rs]):.3f} | {np.max([np.max(r['RP_l']) for r in rs]):.3f} |")
    lines += ['', f'![E3]({os.path.basename(FIG)}/e3_realistic.png)', '']


def main():
    global FIG
    quick = '--quick' in sys.argv
    if quick:
        FIG += '_quick'
    os.makedirs(FIG, exist_ok=True)
    lines = ['# Results', '', f"Generated by `python -m experiments.report{' --quick' if quick else ''}` "
             'from the JSON files in `results/`. All lengths in mm; AP and RP follow ISO 9283 (see README).', '']
    e0 = load('e0_verify', quick)
    if e0:
        b = e0['baseline']
        lines += ['## E0: verification', '',
                  f"- Ideal plant (stock dynamics, ideal sensing): bitwise identical across runs = "
                  f"{e0['ideal']['bitwise_identical']}, RP = {e0['ideal']['RP_max_mm']:.1e} mm, "
                  f"AP = {e0['ideal']['AP_mean_mm']:.4f} mm. A perfect simulator returns empty numbers.",
                  f"- Baseline plant (rotor inertia, encoder quantisation and noise, torque noise, 1 ms latency): "
                  f"AP {b['ground_truth']['AP_mean']:.3f} mm, RP {b['ground_truth']['RP_mean']:.3f} mm. "
                  'This is the floor the controller produces before any physical error.',
                  f"- Instrument RMS error against truth: mocap {e0['instrument_floor_mm']['mocap']:.3f} mm, laser "
                  f"tracker {e0['instrument_floor_mm']['laser_tracker']:.3f} mm. On the baseline the mocap reads RP "
                  f"{b['mocap']['RP_mean']:.3f} mm (truth {b['ground_truth']['RP_mean']:.3f}): the uncertainty "
                  'ratio is below 4:1 there, so a sub-0.1 mm RP needs a laser tracker.',
                  f"- Time step: full plant AP {e0['timestep']['AP_mean_mm']['1ms']:.3f} mm at 1 ms vs "
                  f"{e0['timestep']['AP_mean_mm']['0.5ms']:.3f} mm at 0.5 ms; barycentres differ by at most "
                  f"{e0['timestep']['barycentre_diff_max_mm'] * 1e3:.1f} µm.", '']
    e1 = load('e1_ofat', quick)
    if e1:
        fig_ofat(e1, lines)
    e2 = load('e2_factorial', quick)
    if e2:
        fig_factorial(e2, lines)
    e3 = load('e3_realistic', quick)
    if e3:
        fig_e3(e3, quick, lines)
    path = os.path.join(RESULTS, f"SUMMARY{'_quick' if quick else ''}.md")
    open(path, 'w').write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
