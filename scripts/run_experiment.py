"""Run benchmark configurations locally or inside a Savio job."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPEATS = {"presolve_off": 3, "presolve_default": 3, "cpo3_vs_clarabel": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=REPEATS)
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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    config_dir = repo / "experiments" / args.experiment / "configs"
    available = sorted(path.stem for path in config_dir.glob("*.yaml"))
    configs = args.configs or available
    if not configs or any(config not in available for config in configs):
        parser.error(f"Choose configurations from: {', '.join(available)}")
    if args.workers < 1 or args.memory_mb < 1:
        parser.error("--workers and --memory-mb must be positive")

    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve() / args.experiment / "results"
    repeats = REPEATS[args.experiment]

    # A temporary copy changes only harness resources, never solver settings.
    with tempfile.TemporaryDirectory(prefix="cpo3-configs-") as temporary:
        for config in configs:
            config_path = config_dir / f"{config}.yaml"
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
