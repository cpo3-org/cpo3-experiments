"""Prepare benchmark JSONL results as a Parquet file."""

import argparse
import json
import os
from pathlib import Path

import polars as pl
from solver_benchmarks.analysis.derive import worst_relative_residual

from cpo3_experiments.toolkit import add_problem_types

SOLVERS = {"cpo3", "cplex", "gurobi", "piqp", "clarabel", "highs", "mosek"}
REPEATS = {1, 2, 3}


def prepare_results(results_dir: Path) -> pl.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*/repeat-*/results.jsonl")):
        solver = path.parent.parent.name
        repeat = int(path.parent.name.removeprefix("repeat-"))
        if solver not in SOLVERS or repeat not in REPEATS:
            raise ValueError(f"Unexpected solver or repetition: {path}")
        with path.open() as file:
            for line in file:
                result = json.loads(line)
                if (
                    result["solver"] != solver
                    or result["solver_id"] != f"{solver}_eps=1e-8"
                ):
                    raise ValueError(f"Unexpected solver identity in {path}")
                rows.append(
                    {
                        "solver": result["solver"],
                        "repeat": repeat,
                        "dataset": result["dataset"],
                        "problem": result["problem"],
                        "tolerance": float(result["solver_id"].split("eps=")[1]),
                        "run_time_seconds": result["run_time_seconds"],
                        "status": result["status"],
                        "worst_relative_residual": worst_relative_residual(result),
                    }
                )
    if not rows:
        raise ValueError(
            f"No repeated results found in {results_dir}; expected <solver>/repeat-<1|2|3>/results.jsonl."
        )
    return add_problem_types(
        pl.DataFrame(rows, schema_overrides={"worst_relative_residual": pl.Float64})
    )


def main() -> None:
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    results_root = os.environ.get("CPO3_RESULTS_ROOT")
    parser.add_argument(
        "--raw-results-dir",
        type=Path,
        default=Path(results_root) / experiment.name / "results"
        if results_root
        else None,
        help="Raw input directory (default: $CPO3_RESULTS_ROOT/<experiment>/results)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=experiment / "results",
        help="Output directory for results.parquet",
    )
    args = parser.parse_args()
    if args.raw_results_dir is None:
        parser.error("Source a paths script or pass --raw-results-dir.")
    results = prepare_results(args.raw_results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "results.parquet"
    results.write_parquet(output)
    print(
        f"Prepared {results.height} results from {args.raw_results_dir} into {output}"
    )


if __name__ == "__main__":
    main()
