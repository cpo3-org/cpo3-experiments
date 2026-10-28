# cpo3-experiments

Solver benchmarks and CPLEX diagnostics. Dependencies and solver revisions are pinned in `pyproject.toml` and `uv.lock`.

## Experiments

| Directory | Purpose |
| --- | --- |
| `experiments/central_path/` | Compare seven solvers with presolve and crossover disabled. |
| `experiments/interior_point/` | Compare seven solvers with default preprocessing; see configs for solver-specific settings. |
| `experiments/cpo3_vs_clarabel/` | Compare cpo3 and Clarabel (Faer), including infeasibility tolerances from `1e-7` to `1e-10`. |
| `experiments/cplex_dc_opf_residuals/` | Reproduce central-path DC-OPF LP failures and compare independent residuals with CPLEX diagnostics. |

## Run on Savio

From the repository root on Savio, check the paths in `scripts/paths_savio.sh`, then set up the environment:

```bash
uv sync --locked
source scripts/paths_savio.sh
```

If datasets are missing, run `sbatch scripts/prepare_data_savio.sbatch` and wait for it to finish. Submit the desired experiment:

```bash
bash scripts/submit_savio.sh central_path
bash scripts/submit_savio.sh interior_point
bash scripts/submit_savio.sh cpo3_vs_clarabel
```

Pass solver/config names to select jobs, for example `bash scripts/submit_savio.sh cpo3_vs_clarabel cpo3`. Reruns replace the selected jobs' raw results. Logs go to `experiments/<experiment>/logs/`; raw results go to `$CPO3_RESULTS_ROOT/<experiment>/results/`.

Benchmark jobs share the Slurm resource settings in `scripts/run_savio.sbatch`. Available configurations are discovered from each experiment's `configs/*.yaml` files.

## Run locally

From the repository root (Linux or macOS, subject to solver availability):

```bash
uv sync --locked
source scripts/paths_local.sh
bash scripts/prepare_data_local.sh  # Only needed when datasets are missing.
bash scripts/run_local.sh central_path
bash scripts/run_local.sh interior_point
bash scripts/run_local.sh cpo3_vs_clarabel
```

Local runs execute configurations sequentially, with one problem worker and a 4096 MB per-worker memory limit. Override these with `--workers 4 --memory-mb 8192`. Solver settings, timeouts, and repetitions match Savio. Omitting config names runs every configuration, including commercial solvers. To run a subset, append names such as `cpo3 clarabel`; the existing reports require the complete set of configurations.

Data and raw results default to `.local/data` and `.local/experiments`. Set `CPO3_DATA_ROOT` and `CPO3_RESULTS_ROOT` before sourcing the paths file to use other locations. Both local and Savio reruns replace the selected configurations' raw results.

Commercial solvers require their normal local installation/licenses; local scripts do not preload Savio's CPLEX library. The pinned Git dependencies require Git access and appropriate native build tools. macOS execution has not been validated here. Timings and timeout-sensitive solve counts depend on the machine and resource limits.

## Compile reports

Source the matching paths file (`scripts/paths_savio.sh` for Savio or `scripts/paths_local.sh` for local results), then run after the benchmarks finish:

```bash
uv run --no-sync python experiments/central_path/report.py
uv run --no-sync python experiments/interior_point/report.py
uv run --no-sync python experiments/cpo3_vs_clarabel/report.py
```

Reports save Parquet files, plots, and applicable CSV summaries under each experiment's `results/`. Solutions are independently verified at `1e-6` with a 300-second cutoff. Central-path and interior-point profiles use median runtimes over three repetitions, treating failures as infinity; at least two repetitions must verify. The cpo3–Clarabel comparison runs each configuration once.

## Reproduce CPLEX residual diagnostics

After central-path CPLEX results are available, use the entry point matching their location:

```bash
bash experiments/cplex_dc_opf_residuals/run_savio.sh
# Or locally:
bash experiments/cplex_dc_opf_residuals/run_local.sh
```

This runs directly in the current session, selects failed DC-OPF LPs from repetition 1, and writes `comparison.csv` and `details.jsonl` to the diagnostic experiment's `results/` directory.
