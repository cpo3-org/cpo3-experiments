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

# From the repository root: bash experiments/central_path/run.sh [solver ...]
# No arguments submits every solver. Each solver job runs three full corpus passes
# consecutively; solvers run in parallel. --worker is used by the submitted jobs.

source "${CPO3_REPO_ROOT:-$PWD}/scripts/paths.sh"
EXPERIMENT_ROOT="$CPO3_REPO_ROOT/experiments/central_path"
RESULTS_ROOT="$CPO3_RESULTS_ROOT/central_path/results"

worker=false
if [[ "${1:-}" == --worker ]]; then
    if [[ $# -ne 2 ]]; then
        echo "--worker requires exactly one solver" >&2
        exit 2
    fi
    worker=true
    shift
fi

solvers=("$@")
if [[ ${#solvers[@]} -eq 0 ]]; then
    solvers=(clarabel cplex gurobi highs mosek piqp cpo3)
fi
for solver in "${solvers[@]}"; do
    case "$solver" in
        clarabel|cplex|gurobi|highs|mosek|piqp|cpo3) ;;
        *) echo "Unknown solver: $solver" >&2; exit 2 ;;
    esac
done

if [[ "$worker" == false ]]; then
    mkdir -p "$EXPERIMENT_ROOT/logs"
    for solver in "${solvers[@]}"; do
        sbatch --job-name="central-path-$solver" \
            --chdir="$CPO3_REPO_ROOT" \
            --output="$EXPERIMENT_ROOT/logs/slurm-%j.out" \
            "$EXPERIMENT_ROOT/run.sh" --worker "$solver"
    done
    exit
fi

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
# Point to commercial CPLEX instead of the Community Edition.
export LD_PRELOAD="$HOME/ibm/cplex222/cplex/bin/x86-64_linux/libcplex2220.so${LD_PRELOAD:+:$LD_PRELOAD}"

cd "$CPO3_REPO_ROOT"
solver="${solvers[0]}"
run_dir="$RESULTS_ROOT/$solver"

echo "Clearing previous results: $run_dir"
rm -rf -- "$run_dir"

for repeat in 1 2 3; do
    echo "Running $solver: corpus repetition $repeat/3"
    uv run --no-sync bench run "$EXPERIMENT_ROOT/configs/$solver.yaml" \
        --data-root "$CPO3_DATA_ROOT" \
        --run-dir "$run_dir/repeat-$repeat" \
        --no-stream-output
done
