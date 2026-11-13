"""Run batches of (scenario, unit) jobs in parallel, store raw states and per-instrument ISO 9283 metrics.

Random streams: unit u draws its parametric errors from seed UNIT_SEED0 + u and its cycle noise from
CYCLE_SEED0 + u, identically in every cell (common random numbers), so differences between cells are not noise.
Raw states go to results/raw/<experiment>/ (git-ignored); metrics are returned for the experiment's JSON.
"""
import os
import time
import json
import pickle
import hashlib
import dataclasses
import multiprocessing as mp
import numpy as np

from .config import RESULTS, N_WARMUP, SEQUENCE
from .model import Scenario
from .sim import Rig, Nominal
from .instruments import INSTRUMENT_FUNCS
from .metrics import pose_metrics, slim

UNIT_SEED0, CYCLE_SEED0 = 1000, 50000
_NOM = None


def _nominal():
    global _NOM
    if _NOM is None:
        _NOM = Nominal()
    return _NOM


def run_job(job):
    """job: dict(exp, cell, scenario: dict, unit, n_cycles[, n_warmup, instruments, ci])."""
    sc = Scenario(**job['scenario'])
    nom = _nominal()
    t0 = time.perf_counter()
    rig = Rig(sc, UNIT_SEED0 + job['unit'], nominal=nom, poses=job.get('poses', 'palm_down'))
    raw_dir = os.path.join(RESULTS, 'raw', job['exp'])
    os.makedirs(raw_dir, exist_ok=True)
    npz = os.path.join(raw_dir, f"{job['cell']}_u{job['unit']}.npz")
    run = _resume_raw(npz)
    # job keys: exp, cell, scenario, unit, n_cycles; optional n_warmup, instruments, ci, slim, poses
    if run is None:
        run = rig.run_cycles(job['n_cycles'], cycle_seed=CYCLE_SEED0 + job['unit'],
                             n_warmup=job.get('n_warmup', N_WARMUP))
        wall = time.perf_counter() - t0
        np.savez_compressed(npz, **run)
    else:   # raw states recovered from an interrupted campaign: metrics recomputed, timing unknown
        wall, rig.nsteps = float('nan'), 1
    pc, Rc = rig.commanded()
    rng = np.random.default_rng(7 + job['unit'])
    metrics = {}
    for name in job.get('instruments', list(INSTRUMENT_FUNCS)):
        p, R = INSTRUMENT_FUNCS[name](run, nom, rng)
        mm = pose_metrics(p, R, pc, Rc, ci=job.get('ci', False) and name == 'ground_truth', rng=rng)
        metrics[name] = slim(mm) if job.get('slim') else mm
    return dict(exp=job['exp'], cell=job['cell'], unit=job['unit'], scenario=job['scenario'], label=sc.label(),
                n_cycles=job['n_cycles'], wall_s=wall, us_per_step=wall / rig.nsteps * 1e6, metrics=metrics)


def _resume_raw(npz):
    """With ADAMU_RESUME_SINCE=<epoch s>, reuse a raw file written after that time (interrupted campaign)."""
    since = os.environ.get('ADAMU_RESUME_SINCE')
    if not since or not os.path.exists(npz) or os.path.getmtime(npz) < float(since):
        return None
    try:
        with np.load(npz) as f:
            return {k: f[k] for k in f.files}
    except Exception:
        return None


def _cache_path(job):
    key = hashlib.sha1(json.dumps(job, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return os.path.join(RESULTS, 'raw', '_cache', str(job.get('exp', 'misc')), f"{job.get('cell', 'x')}_u{job.get('unit', 0)}_{key}.pkl")


def _cached(args):
    """Run fn(job) unless a cached result exists; cache results so an interrupted campaign can resume."""
    fn, job = args
    path = _cache_path(job)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    r = fn(job)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(r, f)
    os.replace(path + '.tmp', path)
    return r


def scenario_dict(sc: Scenario):
    return dataclasses.asdict(sc)


def run_jobs(jobs, processes=None, log=print, fn=None):
    """Run `fn` (default run_job) over jobs in a spawn pool; results sorted by (cell, unit)."""
    fn = fn or run_job
    processes = processes or int(os.environ.get('ADAMU_PROCS', max(1, min(4, os.cpu_count() - 1))))
    out, t0 = [], time.perf_counter()
    args = [(fn, j) for j in jobs]
    if processes == 1:
        it = map(_cached, args)
    else:
        pool = mp.get_context('spawn').Pool(processes)
        it = pool.imap_unordered(_cached, args)
    for k, r in enumerate(it, 1):
        head = f"[{k}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {r['cell']:>12s} u{r['unit']}: "
        if 'metrics' in r:
            g = r['metrics']['ground_truth']
            log(head + f"AP {g['AP_mean']:7.3f} mm  RP {g['RP_mean']:6.3f} mm  ({r['wall_s']:.0f}s)")
        else:
            log(head + r.get('summary', '') + f" ({r['wall_s']:.0f}s)")
        out.append(r)
    if processes != 1:
        pool.close(); pool.join()
    return sorted(out, key=lambda r: (r['cell'], r['unit']))


