#!/usr/bin/env bash
#SBATCH --account=fc_radar
#SBATCH --partition=savio3_bigmem
#SBATCH --nodelist=n0[006-009,030-041].savio3
#SBATCH --qos=savio_normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0

set -euo pipefail

# From the repository root: bash experiments/cpo3_vs_clarabel/run_savio.sh [config ...]
# No arguments submits every config. --worker is used by the submitted jobs.
source "$(dirname "$0")/../../scripts/paths_savio.sh"
EXPERIMENT_ROOT="$CPO3_REPO_ROOT/experiments/cpo3_vs_clarabel"

worker=false
if [[ "${1:-}" == --worker ]]; then
    if [[ $# -ne 2 ]]; then
        echo "--worker requires exactly one config" >&2
        exit 2
    fi
    worker=true
    shift
fi

configs=("$@")
if [[ ${#configs[@]} -eq 0 ]]; then
    configs=(clarabel cpo3 clarabel_infeasible cpo3_infeasible)
fi
for config in "${configs[@]}"; do
    case "$config" in
        clarabel|cpo3|clarabel_infeasible|cpo3_infeasible) ;;
        *) echo "Unknown config: $config" >&2; exit 2 ;;
    esac
done

if [[ "$worker" == false ]]; then
    mkdir -p "$EXPERIMENT_ROOT/logs"
    for config in "${configs[@]}"; do
        sbatch --job-name="cpo3-vs-clarabel-$config" \
            --chdir="$CPO3_REPO_ROOT" \
            --output="$EXPERIMENT_ROOT/logs/slurm-%j.out" \
            "$EXPERIMENT_ROOT/run_savio.sh" --worker "$config"
    done
    exit
fi

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

cd "$CPO3_REPO_ROOT"
uv run --no-sync python scripts/run_experiment.py cpo3_vs_clarabel "${configs[0]}"
