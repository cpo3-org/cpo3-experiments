"""Shared execution for Savio workers and local experiment entry points."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CONFIGS = {
    "central_path": ("clarabel", "cplex", "gurobi", "highs", "mosek", "piqp", "cpo3"),
    "interior_point": ("clarabel", "cplex", "gurobi", "highs", "mosek", "piqp", "cpo3"),
    "cpo3_vs_clarabel": ("clarabel", "cpo3", "clarabel_infeasible", "cpo3_infeasible"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=CONFIGS)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1, help="Local problem workers (default: 1)"
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=4096,
        help="Local per-worker memory limit (default: 4096 MB)",
    )
    args = parser.parse_args()
    configs = args.configs or CONFIGS[args.experiment]
    if any(config not in CONFIGS[args.experiment] for config in configs):
        parser.error(
            f"Choose configurations from: {', '.join(CONFIGS[args.experiment])}"
        )
    if args.workers < 1 or args.memory_mb < 1:
        parser.error("--workers and --memory-mb must be positive")

    repo = Path(__file__).resolve().parent.parent
    data_root = Path(os.environ["CPO3_DATA_ROOT"]).resolve()
    results_root = (
        Path(os.environ["CPO3_RESULTS_ROOT"]).resolve() / args.experiment / "results"
    )
    repeats = 1 if args.experiment == "cpo3_vs_clarabel" else 3
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[variable] = "1"

    # A temporary copy changes only harness resources, never solver settings.
    with tempfile.TemporaryDirectory(prefix="cpo3-configs-") as temporary:
        for config in configs:
            config_path = (
                repo / "experiments" / args.experiment / "configs" / f"{config}.yaml"
            )
            if args.local:
                settings = yaml.safe_load(config_path.read_text())
                settings["run"]["parallelism"] = args.workers
                settings["run"]["worker_memory_limit_mb"] = args.memory_mb
                config_path = Path(temporary) / f"{config}.yaml"
                config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
            run_dir = results_root / config
            print(f"Clearing previous results: {run_dir}", flush=True)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            for repeat in range(1, repeats + 1):
                destination = run_dir / f"repeat-{repeat}" if repeats > 1 else run_dir
                print(f"Running {config}: repetition {repeat}/{repeats}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "solver_benchmarks.cli",
                        "run",
                        str(config_path),
                        "--data-root",
                        str(data_root),
                        "--run-dir",
                        str(destination),
                        "--no-stream-output",
                    ],
                    cwd=repo,
                    check=True,
                )


if __name__ == "__main__":
    main()
