#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../scripts/paths_local.sh"
cd "$CPO3_REPO_ROOT"
export OPENBLAS_NUM_THREADS=1
uv run --no-sync python -u experiments/cplex_dc_opf_residuals/run.py
