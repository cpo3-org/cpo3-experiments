#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/paths.sh"
bash "$CPO3_REPO_ROOT/scripts/prepare_data.sh"
