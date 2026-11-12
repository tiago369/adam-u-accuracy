"""CLI and output helpers shared by the experiment scripts."""
import os
import json
import argparse
import platform
import time
import mujoco
import numpy as np

from adam_u_accuracy.config import RESULTS
from adam_u_accuracy.metrics import to_jsonable


def parse(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='few units/cycles: checks the pipeline in minutes')
    ap.add_argument('--procs', type=int, default=None, help='parallel worker processes (default: min(4, ncpu-1))')
    return ap.parse_args()


def save(name, payload, quick=False):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}{'_quick' if quick else ''}.json")
    payload = dict(meta=dict(mujoco=mujoco.__version__, numpy=np.__version__, python=platform.python_version(),
                             date=time.strftime('%Y-%m-%d %H:%M'), quick=quick), **payload)
    with open(path, 'w') as f:
        json.dump(to_jsonable(payload), f, indent=1)
    print('wrote', path)
    return path
