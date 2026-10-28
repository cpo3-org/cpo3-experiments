"""Analyze prepared Parquet results and generate tables and plots."""

import argparse
from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from cpo3_experiments.toolkit import (
    PLOT_STYLE,
    TIME_LIMIT_SECONDS,
    aggregate_repeats,
    classify_results,
    plot_dolan_more,
    plot_residuals,
    save_plot,
    summarize_problem_type,
    summarize_results,
    summarize_runtimes,
)

LABELS = {
    "cpo3": "cpo3",
    "cplex": "CPLEX",
    "gurobi": "Gurobi",
    "piqp": "PIQP",
    "clarabel": "Clarabel",
    "highs": "HiGHS",
    "mosek": "MOSEK",
}

# Distinguish solvers in grayscale as well as in color.
LINESTYLES = {
    "cpo3": "-",
    "cplex": "--",
    "gurobi": "-.",
    "piqp": ":",
    "clarabel": (0, (5, 1, 1, 1, 1, 1)),
    "highs": (0, (3, 1, 1, 1)),
    "mosek": (0, (8, 2)),
}

REPEATS = {1, 2, 3}


def create_plots(
    results: pl.DataFrame, repetitions: pl.DataFrame, output_dir: Path, tol: float
) -> None:
    plt.rcParams.update(PLOT_STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for group in ("all", "lp", "qp"):
        subset = (
            results
            if group == "all"
            else results.filter(pl.col("problem_type") == group)
        )
        groups[group] = subset

    for name in ("dolan_more", "residuals_violin"):
        panels, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (group, subset) in zip(axes, groups.items()):
            if name == "residuals_violin":
                subset = (
                    repetitions
                    if group == "all"
                    else repetitions.filter(pl.col("problem_type") == group)
                )
            if subset.is_empty():
                ax.set_title(f"{group.upper()} (0 problems)")
                continue
            if name == "dolan_more":
                plot_dolan_more(ax, subset, LABELS, LINESTYLES)
            else:
                plot_residuals(ax, subset, tol, LABELS)
            n = subset.select("dataset", "problem").unique().height
            if name == "dolan_more":
                label = {"all": "All problems", "lp": "LP", "qp": "QP"}[group]
                ax.set_title(f"{label} ({n} problems)")
                if group == "all":
                    continue
            if name == "residuals_violin":
                ax.set_title(f"{group.upper()} ({n} problems)")
                if group != "all":
                    continue
            # Keep 9 pt lettering at the manuscript's full width; reduce height
            # rather than scaling down the entire figure and its labels.
            height = 2.8 if name == "dolan_more" else 2.5
            fig, individual = plt.subplots(figsize=(119 / 25.4, height))
            if name == "dolan_more":
                plot_dolan_more(individual, subset, LABELS, LINESTYLES)
            else:
                plot_residuals(individual, subset, tol, LABELS)
            save_plot(fig, output_dir, f"{name}_{group}.pdf")
        save_plot(panels, output_dir, f"{name}_panels.pdf")
    print(f"Plots saved to {output_dir}")


def main() -> None:
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=experiment / "results",
        help="Directory containing prepared Parquet data, tables, and plots",
    )
    parser.add_argument(
        "--verify-tol",
        "--tol",
        dest="verify_tol",
        type=float,
        default=1e-6,
        help="KKT verification tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=10.0,
        help="Shift in seconds for the geometric mean (default: 10)",
    )
    args = parser.parse_args()
    if not isfinite(args.verify_tol) or args.verify_tol <= 0:
        parser.error("--verify-tol must be finite and positive")
    if not isfinite(args.shift) or args.shift <= 0:
        parser.error("--shift must be finite and positive")

    parquet = args.results_dir / "results.parquet"
    if not parquet.is_file():
        parser.error(f"No prepared results at {parquet}; run prepare_results.py first.")
    print(f"Reading prepared results from {parquet}")
    prepared = pl.read_parquet(parquet)
    repetitions = classify_results(prepared, args.verify_tol)
    results = aggregate_repeats(repetitions, list(LABELS), REPEATS)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results.write_parquet(args.results_dir / "median_results.parquet")
    status_counts = summarize_results(repetitions)
    runtimes = summarize_runtimes(results, args.shift)
    print("Solver stopping tolerance: 1e-8 for every solver")
    print(f"Verification threshold: {args.verify_tol:g}")
    print(f"Verified solves must finish within {TIME_LIMIT_SECONDS:g}s.")
    print(
        "Profiles use median of 3 runtimes; failed/unverified runs contribute infinity."
    )
    print(
        "Finite medians require at least 2 verified runs. Residual plots include all verified repetitions."
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=240, float_precision=4):
        print("Status counts across individual repetitions:")
        print(status_counts)
        print("Problems by number of verified repetitions:")
        print(
            results.group_by("solver", "verified_repeats")
            .len()
            .sort("solver", "verified_repeats")
        )
        print(
            f"Shifted geometric means: shift={args.shift:g}s; failures penalized at {TIME_LIMIT_SECONDS:g}s."
        )
        print(runtimes)
        for problem_type in ("qp", "lp"):
            print(f"{problem_type.upper()} results:")
            print(
                summarize_problem_type(results, args.shift, problem_type, list(LABELS))
            )
    create_plots(results, repetitions, args.results_dir / "plots", args.verify_tol)


if __name__ == "__main__":
    main()
