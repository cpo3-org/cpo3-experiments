# cpo3-experiments

Benchmark code and recorded results accompanying *Strongly Monotone Central Path Equations for Quadratic Cone Programs*. See the manuscript for the experimental methodology and interpretation of results.

| Experiment | Manuscript content |
| --- | --- |
| [`presolve_off`](experiments/presolve_off/) | Solver performance on feasible problems; solution residuals and LP/QP performance profiles. |
| [`cpo3_vs_clarabel`](experiments/cpo3_vs_clarabel/) | Comparison with Clarabel: solve counts, linear-algebra work, and infeasibility certificates. |
| [`presolve_default`](experiments/presolve_default/) | Appendix: performance with default presolve and crossover settings. |
| [`cplex_dc_opf_residuals`](experiments/cplex_dc_opf_residuals/) | CPLEX residual diagnostics discussed in the LP results. |
| [`warm_start_perturb`](experiments/warm_start_perturb/) | Warm starts on perturbed Netlib LP and Maros–Mészáros QP problems. |
| [`warm_start_mpc`](experiments/warm_start_mpc/) | Warm starts in closed-loop model predictive control. |

## Setup

Run all commands from the repository root. Install `uv`, Git, and the Rust/native build tools needed by the solver dependencies. Python 3.11 or later is required.

```bash
uv sync --locked
```

Dependency versions and solver revisions are pinned in `pyproject.toml` and `uv.lock`.

## Regenerate reports from the supplied results

These commands analyze the supplied `results.parquet` files without rerunning solvers. No benchmark datasets, solver licenses, or Savio access are needed for this step.

```bash
uv run --no-sync python experiments/presolve_off/analyze_results.py
uv run --no-sync python experiments/presolve_default/analyze_results.py
uv run --no-sync python experiments/cpo3_vs_clarabel/analyze_results.py
```

Tables are printed to the terminal and figures are saved to each experiment's `results/plots/`. The Clarabel comparison also saves CSV summaries in `results/analysis/`. These scripts use recorded residuals; they do not recompute solutions or residuals from problem data.

## Warm-start experiments


To run the perturbation experiments, run `perturb.py` before `analyse.py`

```bash
uv run --no-sync python experiments/warm_start_perturb/perturb.py
uv run --no-sync python experiments/warm_start_perturb/analyse.py
```

`perturb.py` saves every cold and warm solve to `data/warm_runs.csv`. `analyse.py` prints Table 6 and saves it as `results/warm_start_table.tex`. The full run takes a long time; for a quick smoke test, set `LIMIT` at the top of `perturb.py` to the number of problems to run from each suite. The perturbation sizes are set by `DELTAS`.

To reproduce the MPC figure:

```bash
uv run --no-sync python experiments/warm_start_mpc/mpc.py
```

This prints iteration counts and saves `figure_warm_mpc.pdf` and `figure_warm_mpc.png` in `experiments/warm_start_mpc/`.

## Rerun the benchmarks

Full runs require the relevant commercial solver installations and licenses. For example, to rerun `presolve_off` locally and analyze the new results:

```bash
source scripts/local/paths.sh
bash scripts/local/prepare_data.sh  # Skip if datasets are already available.
bash scripts/local/run.sh presolve_off
uv run --no-sync python experiments/presolve_off/prepare_results.py
uv run --no-sync python experiments/presolve_off/analyze_results.py
```

Substitute `presolve_default` or `cpo3_vs_clarabel` for the other experiments. To run only selected configurations, append their names, for example `bash scripts/local/run.sh presolve_off cpo3 clarabel`. The analysis scripts expect complete experiment results.

Local datasets are stored in `.local/data/` and raw results in `.local/experiments/`. Rerunning a configuration replaces its previous raw results. Preparing results replaces the experiment’s `results/results.parquet`.

Local runs use one worker with a 4096 MB memory limit by default. Override these settings with `--workers` and `--memory-mb`. The full experiments require 20 GB of memory (`--memory-mb 20480`); with the 4096 MB default, problems that need more memory return errors. Runtimes depend on the machine and resource limits.

For Savio, check and source [`scripts/savio/paths.sh`](scripts/savio/paths.sh), prepare missing datasets with `sbatch scripts/savio/prepare_data.sbatch`, and wait for completion before submitting with `bash scripts/savio/submit.sh presolve_off`. After the jobs finish, use the same preparation and analysis commands above.

The supplied CPLEX diagnostic results are in [`experiments/cplex_dc_opf_residuals/results/`](experiments/cplex_dc_opf_residuals/results/). To regenerate them after a `presolve_off` CPLEX run, execute `bash experiments/cplex_dc_opf_residuals/run_local.sh` (or `run_savio.sh` on Savio).
