# cpo3-experiments

Solver benchmarks and CPLEX diagnostics. Dependencies and solver revisions are pinned in `pyproject.toml` and `uv.lock`.

## Reproduce the manuscript tables and figures


From the repository root, install the pinned dependencies and regenerate tables and plots from the committed Parquet files. This does not require datasets, solver runs, or Savio access:

```bash
uv sync --locked
uv run --no-sync python experiments/presolve_off/analyze_results.py
uv run --no-sync python experiments/presolve_default/analyze_results.py
uv run --no-sync python experiments/cpo3_vs_clarabel/analyze_results.py
```

## Experiments

| Directory | Purpose |
| --- | --- |
| `experiments/presolve_off/` | Compare seven solvers with presolve and crossover disabled. |
| `experiments/presolve_default/` | Compare seven solvers with their default presolve and crossover settings; see configs for solver-specific settings. |
| `experiments/cpo3_vs_clarabel/` | Compare cpo3 and Clarabel (Faer), including infeasibility tolerances from `1e-7` to `1e-10`. |
| `experiments/cplex_dc_opf_residuals/` | Reproduce DC-OPF LP failures from `presolve_off` and compare independent residuals with CPLEX diagnostics. |

## Prepare results from new runs

After new benchmark runs, first source `scripts/savio/paths.sh` or `scripts/local/paths.sh` and prepare their results:

```bash
uv run --no-sync python experiments/presolve_off/prepare_results.py
uv run --no-sync python experiments/presolve_default/prepare_results.py
uv run --no-sync python experiments/cpo3_vs_clarabel/prepare_results.py
```

Preparation reads raw JSONL and writes `results.parquet`; use `--raw-results-dir` to select an explicit input directory. Analysis reads only Parquet data, applies the verification threshold, and generates tables and plots. For repeated experiments, it also writes `median_results.parquet`. Both scripts accept `--results-dir` to select the prepared-data and output directory.

Outputs are saved under each experiment's `results/`. Solutions are independently verified at `1e-6` with a 300-second cutoff. The `presolve_off` and `presolve_default` profiles use median runtimes over three repetitions, treating failures as infinity; at least two repetitions must verify. The cpo3–Clarabel comparison runs each configuration once.

LP/QP classifications are stored in `src/cpo3_experiments/problem_types.csv`, keyed by dataset and problem. Reports require every problem to appear in this catalog. To regenerate it after changing the datasets, prepare the complete data collection, source the matching paths script, then run:

```bash
uv run --no-sync python scripts/classify_problems.py --data-root "$CPO3_DATA_ROOT"
```

## Run locally

From the repository root (Linux or macOS, subject to solver availability):

```bash
uv sync --locked
source scripts/local/paths.sh
bash scripts/local/prepare_data.sh  # Only needed when datasets are missing.
bash scripts/local/run.sh presolve_off
bash scripts/local/run.sh presolve_default
bash scripts/local/run.sh cpo3_vs_clarabel
```

Local runs execute configurations sequentially, with one problem worker and a 4096 MB per-worker memory limit. Override these with `--workers 4 --memory-mb 8192`. Solver settings, timeouts, and repetitions match Savio. Omitting config names runs every configuration, including commercial solvers. To run a subset, append names such as `cpo3 clarabel`; the existing reports require the complete set of configurations.

Data and raw results default to `.local/data` and `.local/experiments`. Set `CPO3_DATA_ROOT` and `CPO3_RESULTS_ROOT` before sourcing the paths file to use other locations. Both local and Savio reruns replace the selected configurations' raw results.

Commercial solvers require their normal local installation/licenses; local scripts do not preload Savio's CPLEX library. The pinned Git dependencies require Git access and appropriate native build tools. macOS execution has not been validated here. Timings and timeout-sensitive solve counts depend on the machine and resource limits.

## Run on Savio

From the repository root on Savio, check the paths in `scripts/savio/paths.sh`, then set up the environment:

```bash
uv sync --locked
source scripts/savio/paths.sh
```

If datasets are missing, run `sbatch scripts/savio/prepare_data.sbatch` and wait for it to finish. Submit the desired experiment:

```bash
bash scripts/savio/submit.sh presolve_off
bash scripts/savio/submit.sh presolve_default
bash scripts/savio/submit.sh cpo3_vs_clarabel
```

Pass solver/config names to select jobs, for example `bash scripts/savio/submit.sh cpo3_vs_clarabel cpo3`. Reruns replace the selected jobs' raw results. Logs go to `experiments/<experiment>/logs/`; raw results go to `$CPO3_RESULTS_ROOT/<experiment>/results/`.

Benchmark jobs share the Slurm resource settings in `scripts/savio/run.sbatch`. Available configurations are discovered from each experiment's `configs/*.yaml` files.

## Reproduce CPLEX residual diagnostics

After `presolve_off` CPLEX results are available, use the entry point matching their location:

```bash
bash experiments/cplex_dc_opf_residuals/run_savio.sh
# Or locally:
bash experiments/cplex_dc_opf_residuals/run_local.sh
```

This runs directly in the current session, selects failed DC-OPF LPs from repetition 1, and writes `comparison.csv` and `details.jsonl` to the diagnostic experiment's `results/` directory.
