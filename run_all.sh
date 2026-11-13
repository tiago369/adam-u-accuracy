#!/usr/bin/env bash
# Full step-5 campaign (~52 CPU-hours, about 6 h with ADAMU_PROCS=10). Pass --quick for a ~6 min pipeline check.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONUNBUFFERED=1
$PY -m adam_u_accuracy.poses
$PY -m experiments.e0_verify "$@"
$PY -m experiments.e1_ofat "$@"
$PY -m experiments.e2_factorial "$@"
$PY -m experiments.e3_realistic "$@"
$PY -m experiments.e4_extras "$@"
$PY -m experiments.report "$@"
