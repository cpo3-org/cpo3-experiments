#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/paths.sh"
cd "$CPO3_REPO_ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
uv run --no-sync python scripts/run_experiment.py "$@" --local \
    --data-root "$CPO3_DATA_ROOT" --results-root "$CPO3_RESULTS_ROOT"
