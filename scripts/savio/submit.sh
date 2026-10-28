#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/paths.sh"
cd "$CPO3_REPO_ROOT"

case "${1:-}" in
    presolve_off|presolve_default|cpo3_vs_clarabel) experiment=$1 ;;
    *)
        echo "Usage: bash scripts/savio/submit.sh {presolve_off|presolve_default|cpo3_vs_clarabel} [configs ...]" >&2
        exit 1
        ;;
esac
shift

config_dir="experiments/$experiment/configs"
configs=("$@")
if (( ${#configs[@]} == 0 )); then
    for path in "$config_dir"/*.yaml; do
        config=${path##*/}
        configs+=("${config%.yaml}")
    done
fi

# Validate every selection before submitting any jobs.
for config in "${configs[@]}"; do
    if [[ "$config" == */* || ! -f "$config_dir/$config.yaml" ]]; then
        echo "Unknown configuration for $experiment: $config" >&2
        exit 1
    fi
done

logs="$CPO3_REPO_ROOT/experiments/$experiment/logs"
mkdir -p "$logs"
for config in "${configs[@]}"; do
    sbatch --job-name="${experiment//_/-}-$config" \
        --chdir="$CPO3_REPO_ROOT" --output="$logs/slurm-%j.out" \
        "$CPO3_REPO_ROOT/scripts/savio/run.sbatch" "$experiment" "$config"
done
