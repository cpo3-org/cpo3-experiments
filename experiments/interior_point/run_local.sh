#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../scripts/paths_local.sh"
cd "$CPO3_REPO_ROOT"
uv run --no-sync python scripts/run_experiment.py interior_point --local "$@"
