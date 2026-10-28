#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/savio/paths.sh
export OPENBLAS_NUM_THREADS=1
export LD_PRELOAD="$HOME/ibm/cplex222/cplex/bin/x86-64_linux/libcplex2220.so${LD_PRELOAD:+:$LD_PRELOAD}"
uv run --no-sync python -u experiments/cplex_dc_opf_residuals/run.py
