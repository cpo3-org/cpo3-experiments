#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/paths_local.sh"
bash "$CPO3_REPO_ROOT/scripts/prepare_data.sh"
