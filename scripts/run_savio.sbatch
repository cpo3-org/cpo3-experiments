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
# Submission sets --chdir to the repository; Slurm copies this script elsewhere.
source "$PWD/scripts/paths_savio.sh"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
# Current CPLEX configurations are named cplex in both benchmark experiments.
if [[ "${2:-}" == cplex ]]; then
    export LD_PRELOAD="$HOME/ibm/cplex222/cplex/bin/x86-64_linux/libcplex2220.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
cd "$CPO3_REPO_ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
uv run --no-sync python scripts/run_experiment.py "$@" \
    --data-root "$CPO3_DATA_ROOT" --results-root "$CPO3_RESULTS_ROOT"
