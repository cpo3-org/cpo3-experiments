"""Prepare benchmark JSONL results as a Parquet file."""

import argparse
import json
import os
from math import isfinite
from pathlib import Path

import polars as pl
from solver_benchmarks.analysis.derive import worst_relative_residual

from cpo3_experiments.toolkit import add_problem_types


def certificate_residual(result: dict, normalized: bool = False) -> float | None:
    kkt = result.get("kkt") or {}
    if kkt.get("certificate") != "primal_infeasible" or kkt.get("unsupported_cones"):
        return None
    aty_key = "Aty_inf" if normalized else "Aty_rel"
    bty, aty, cone = (kkt.get(key) for key in ("bty", aty_key, "dual_cone_res"))
    if bty is None or aty is None or cone is None:
        return None
    if not all(isfinite(value) for value in (bty, aty, cone)):
        return None
    # Scaling y by -1/(bᵀy) scales both absolute residuals by the same factor.
    return max(aty, cone) / (-bty if normalized else 1) if bty < 0 else None


def prepare_results(results_dir: Path) -> pl.DataFrame:
    rows = []
    paths = sorted(results_dir.glob("*/results.jsonl"))
    if not paths:
        paths = sorted(results_dir.glob("*/*/results.jsonl"))
    for path in paths:
        for line in path.read_text().splitlines():
            result = json.loads(line)
            infeasible = result["dataset"].endswith(
                "_infeasible"
            ) or path.parent.name.endswith("_infeasible")
            tolerance, _, infeasibility_tolerance = result["solver_id"].partition(
                "_infeas="
            )
            info = result.get("info") or {}
            rows.append(
                {
                    "solver": result["solver"],
                    "dataset": result["dataset"].removesuffix("_infeasible")
                    + ("_infeasible" if infeasible else ""),
                    "problem": result["problem"],
                    "tolerance": float(tolerance.split("eps=")[1]),
                    "infeasibility_tolerance": (
                        float(infeasibility_tolerance) if infeasible else None
                    ),
                    "run_time_seconds": result["run_time_seconds"],
                    "iterations": result["iterations"],
                    "factorizations": info.get("factorizations"),
                    "kkt_solves": info.get("primary_solves", info.get("kkt_solves")),
                    "refinement_passes": info.get(
                        "refinement_solves", info.get("refinement_passes")
                    ),
                    "worst_relative_residual": worst_relative_residual(result),
                    "certificate_residual": certificate_residual(result),
                    "normalized_certificate_residual": certificate_residual(
                        result, normalized=True
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No benchmark results found in {results_dir}.")
    return add_problem_types(
        pl.DataFrame(
            rows,
            schema_overrides={
                "infeasibility_tolerance": pl.Float64,
                "run_time_seconds": pl.Float64,
                "iterations": pl.Int64,
                "factorizations": pl.Int64,
                "kkt_solves": pl.Int64,
                "refinement_passes": pl.Int64,
                "worst_relative_residual": pl.Float64,
                "certificate_residual": pl.Float64,
                "normalized_certificate_residual": pl.Float64,
            },
        )
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
