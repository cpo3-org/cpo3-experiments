# cpo3-experiments

Solver benchmarks and CPLEX diagnostics. Dependencies and solver revisions are pinned in `pyproject.toml` and `uv.lock`.

## Experiments

| Directory | Purpose |
| --- | --- |
| `experiments/central_path/` | Compare seven solvers with presolve and crossover disabled. |
| `experiments/interior_point/` | Compare seven solvers with default preprocessing; see configs for solver-specific settings. |
| `experiments/cpo3_vs_clarabel/` | Compare cpo3 and Clarabel (Faer), including infeasibility tolerances from `1e-7` to `1e-10`. |
| `experiments/cplex_dc_opf_residuals/` | Reproduce central-path DC-OPF LP failures and compare independent residuals with CPLEX diagnostics. |

## Run benchmarks

From the repository root on Savio, check the paths in `scripts/paths.sh`, then set up the environment:

```bash
uv sync --locked
source scripts/paths.sh
```

If datasets are missing, run `sbatch scripts/prepare_data.sbatch` and wait for it to finish. Submit the desired experiment:

```bash
bash experiments/central_path/run.sh
bash experiments/interior_point/run.sh
bash experiments/cpo3_vs_clarabel/run.sh
```

Pass solver/config names to select jobs, for example `bash experiments/cpo3_vs_clarabel/run.sh cpo3`. Reruns replace the selected jobs' raw results. Logs go to `experiments/<experiment>/logs/`; raw results go to `$CPO3_RESULTS_ROOT/<experiment>/results/`.

After jobs finish, compile the corresponding report:

```bash
uv run --no-sync python experiments/central_path/report.py
uv run --no-sync python experiments/interior_point/report.py
uv run --no-sync python experiments/cpo3_vs_clarabel/report.py
```

Reports save Parquet files, plots, and applicable CSV summaries under each experiment's `results/`. Solutions are independently verified at `1e-6` with a 300-second cutoff. Central-path and interior-point profiles use median runtimes over three repetitions, treating failures as infinity; at least two repetitions must verify. The cpo3–Clarabel comparison runs each configuration once.

## Reproduce CPLEX residual diagnostics

After central-path CPLEX results are available:

```bash
bash experiments/cplex_dc_opf_residuals/run.sh
```

This runs directly in the current session, selects failed DC-OPF LPs from repetition 1, and writes `comparison.csv` and `details.jsonl` to the diagnostic experiment's `results/` directory.
