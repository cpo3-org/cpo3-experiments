import argparse
import json
import os
from bisect import bisect_right
from math import isfinite, log10
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from solver_benchmarks.analysis.derive import worst_relative_residual
from solver_benchmarks.core import status

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
TIME_LIMIT_SECONDS = 300.0

CATEGORIES: tuple[str, ...] = (
    "solve_count",
    "inaccurate",
    "unverified",
    "time_limit",
    "iter_limit",
    "infeasible_or_unbounded",
    "error",
)


# LPs missing types in the results; confirmed from the benchmark collections.
MISSING_PROBLEM_TYPES = {
    "Dual2_5000": "lp",
    "L1_sixm1000obs": "lp",
    "Primal2_1000": "lp",
    "fhnw-binschedule1": "lp",
    "set-cover-model": "lp",
    "neos-5104907-jarama": "lp",
    "s82": "lp",  # Source MPS has no quadratic sections.
}


def problem_type(result: dict[str, Any]) -> str | None:
    nnz_p = result["metadata"].get("nnz_p")
    return None if nnz_p is None else "qp" if nnz_p else "lp"


def prepare_results(results_dir: Path) -> pl.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*/repeat-*/results.jsonl")):
        solver = path.parent.parent.name
        repeat = int(path.parent.name.removeprefix("repeat-"))
        if solver not in LABELS or repeat not in REPEATS:
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
                        "problem_type": problem_type(result),
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
    return pl.DataFrame(
        rows, schema_overrides={"worst_relative_residual": pl.Float64}
    ).with_columns(
        pl.col("problem_type")
        .drop_nulls()
        .first()
        .over("dataset", "problem")
        .fill_null(
            pl.col("problem").replace_strict(MISSING_PROBLEM_TYPES, default=None)
        )
    )


def classify_result(result: dict[str, Any], tol: float) -> str | None:
    runtime = result["run_time_seconds"]
    if runtime is not None and runtime > TIME_LIMIT_SECONDS:
        return "time_limit"

    residual = result["worst_relative_residual"]
    if residual is not None and isfinite(residual) and 0 <= residual <= tol:
        return "solve_count"

    reported = result["status"]
    if reported in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
        return "unverified" if residual is None else "inaccurate"
    if reported in status.ANY_INFEASIBLE:
        return "infeasible_or_unbounded"
    return {
        status.TIME_LIMIT: "time_limit",
        status.MAX_ITER_REACHED: "iter_limit",
        status.SOLVER_ERROR: "error",
        status.WORKER_ERROR: "error",
    }.get(reported)


def classify_results(results: pl.DataFrame, tol: float) -> pl.DataFrame:
    return results.with_columns(
        pl.Series(
            "category",
            [classify_result(result, tol) for result in results.iter_rows(named=True)],
            dtype=pl.String,
        ),
    ).with_columns(
        pl.when(
            (pl.col("category") == "solve_count")
            & pl.col("run_time_seconds").is_finite()
            & (pl.col("run_time_seconds") > 0)
        )
        .then(pl.col("run_time_seconds"))
        .otherwise(float("inf"))
        .alias("time")
    )


def validate_results(results: pl.DataFrame) -> None:
    if not (results["tolerance"] == 1e-8).all():
        raise ValueError("Expected stopping tolerance 1e-8.")
    if set(results["solver"]) != set(LABELS):
        raise ValueError("Expected results for all configured solvers.")
    keys = ["dataset", "problem", "solver"]
    if results.select(*keys, "repeat").is_duplicated().any():
        raise ValueError("Duplicate solver/problem/repetition results.")
    for row in results.group_by(keys).agg(pl.col("repeat")).iter_rows(named=True):
        if set(row["repeat"]) != REPEATS:
            raise ValueError(f"Expected repetitions 1, 2, 3 for {row}.")
    coverage = results.group_by("dataset", "problem").agg(pl.col("solver").n_unique())
    if not (coverage["solver"] == len(LABELS)).all():
        raise ValueError("Some problems are missing solver results.")
    if results["category"].null_count():
        raise ValueError("Unclassified solver statuses.")
    if results["problem_type"].null_count():
        raise ValueError("Unclassified LP/QP types.")


def aggregate_repeats(results: pl.DataFrame) -> pl.DataFrame:
    """Median of three verified runtimes; each failed run contributes infinity.

    Two verified runs suffice for a finite median. One verified run does not.
    With one failure, the median is the slower of the two verified runs.
    """
    validate_results(results)
    return (
        results.group_by("solver", "dataset", "problem")
        .agg(
            pl.col("problem_type").first(),
            pl.col("tolerance").first(),
            pl.len().alias("repeats"),
            pl.col("time").is_finite().sum().alias("verified_repeats"),
            pl.col("run_time_seconds").median(),
            pl.col("time").median(),
        )
        .sort("solver", "dataset", "problem")
    )


def summarize_results(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("solver")
        .agg(
            (pl.col("category") == category).sum().alias(category)
            for category in CATEGORIES
        )
        .with_columns(pl.sum_horizontal(*CATEGORIES).alias("total"))
        .sort("solver")
    )


def save_plot(fig: Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)


def summarize_collections(results: pl.DataFrame, shift: float) -> pl.DataFrame:
    """Solve counts and shifted geometric means on the plotted problems."""
    runtime = pl.when(pl.col("time").is_finite()).then(pl.col("time")).otherwise(TIME_LIMIT_SECONDS)
    return (
        results.group_by("solver", "dataset")
        .agg(
            pl.len().alias("dataset_size"),
            pl.col("time").is_finite().sum().alias("solved"),
            ((runtime + shift).log().mean().exp() - shift).alias(
                "shifted_geomean_seconds"
            ),
        )
        .sort("dataset", "solver")
    )


def summarize_runtimes(results: pl.DataFrame, shift: float) -> pl.DataFrame:
    summary = summarize_collections(results, shift)
    means = summary.pivot(
        on="dataset",
        index="solver",
        values="shifted_geomean_seconds",
    ).sort("solver")
    sizes = (
        summary.select("dataset", "dataset_size")
        .unique()
        .with_columns(pl.lit("dataset_size").alias("solver"))
        .pivot(on="dataset", index="solver", values="dataset_size")
        .select(means.columns)
    )
    return pl.concat([means, sizes], how="vertical_relaxed")


def summarize_problem_type(
    results: pl.DataFrame, shift: float, problem_type: str
) -> pl.DataFrame:
    datasets = {
        "qp": ["maros_meszaros", "qplib", "mpc_qpbenchmark", "dc_opf"],
        "lp": ["dc_opf", "netlib", "kennington", "mittelmann", "miplib"],
    }[problem_type]
    summary = summarize_collections(
        results.filter(pl.col("problem_type") == problem_type), shift
    )
    table = summary.rename({"shifted_geomean_seconds": "seconds"}).pivot(
        on="dataset", index="solver", values=["solved", "seconds"]
    )
    solvers = pl.DataFrame({"solver": list(LABELS)})
    return solvers.join(table, on="solver", maintain_order="left").select(
        "solver",
        *(
            f"{metric}_{dataset}"
            for dataset in datasets
            for metric in ("solved", "seconds")
        ),
    )


def plot_dolan_more(ax: Axes, results: pl.DataFrame, solvers: list[str]) -> None:
    times = results.pivot(on="solver", index=["dataset", "problem"], values="time")
    n = times.height
    assert n > 0
    fastest = times.select(pl.min_horizontal(*solvers)).to_series().to_list()
    ratios = {
        solver: sorted(
            t / fastest[i] for i, t in enumerate(times[solver].to_list()) if isfinite(t)
        )
        for solver in solvers
    }
    right = max(
        1.01, max((r for values in ratios.values() for r in values), default=1.0)
    )
    for solver, values in ratios.items():
        x = sorted({1.0, right, *values})
        # Unsolved problems remain in n, including those no solver solved.
        ax.step(
            x,
            [bisect_right(values, ratio) / n for ratio in x],
            where="post",
            label=LABELS[solver],
            linestyle=LINESTYLES[solver],
        )
    ax.set(
        xlabel="Median time / fastest median time",
        ylabel="Fraction of problems",
        xlim=(1, right),
        ylim=(0, 1),
    )
    ax.set_xscale("log")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        [ax.lines[solvers.index(s)] for s in LABELS],
        list(LABELS.values()),
        loc="lower right",
        ncols=2,
        fontsize=8,
        handlelength=2,
        columnspacing=1,
        labelspacing=0.2,
    )


def plot_residuals(ax: Axes, results: pl.DataFrame, tol: float) -> None:
    solvers = list(LABELS)
    verified = results.filter(pl.col("category") == "solve_count")
    values = [
        verified.filter(pl.col("solver") == solver)["worst_relative_residual"]
        .clip(lower_bound=1e-16)
        .log10()
        .to_numpy()
        for solver in solvers
    ]
    # A solver may have no verified runs; keep its tick without a violin.
    for position, (solver, samples) in enumerate(zip(solvers, values), start=1):
        if not len(samples):
            continue
        color = plt.get_cmap("tab10")(sorted(solvers).index(solver))
        violin = ax.violinplot(
            [samples],
            positions=[position],
            showmedians=True,
            facecolor=color,
            linecolor=color,
        )
        for body in cast(list[PolyCollection], violin["bodies"]):
            body.set_alpha(0.65)
    ax.set_xticks(range(1, len(solvers) + 1), list(LABELS.values()))
    ax.axhline(log10(tol), color="gray", linestyle=":")
    ax.yaxis.set_major_formatter(lambda value, _: f"$10^{{{value:g}}}$")
    ax.set_ylabel("KKT residual")


def create_plots(
    results: pl.DataFrame, repetitions: pl.DataFrame, output_dir: Path, tol: float
) -> None:
    # MPC recommends Helvetica/Arial at 8–12 pt; allow local sans-serif fallbacks.
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
            "pdf.use14corefonts": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    solvers = sorted(LABELS)
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
                plot_dolan_more(ax, subset, solvers)
            else:
                plot_residuals(ax, subset, tol)
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
                plot_dolan_more(individual, subset, solvers)
            else:
                plot_residuals(individual, subset, tol)
            save_plot(fig, output_dir, f"{name}_{group}.pdf")
        save_plot(panels, output_dir, f"{name}_panels.pdf")
    print(f"Plots saved to {output_dir}")


def main() -> None:
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    results_root = os.environ.get("CPO3_RESULTS_ROOT")
    parser.add_argument(
        "--raw-results-dir",
        type=Path,
        default=Path(results_root) / experiment.name / "results"
        if results_root
        else None,
        help="Raw input directory (default: $CPO3_RESULTS_ROOT/central_path/results)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=experiment / "results",
        help="Output directory for prepared data, tables, and plots",
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
    if args.raw_results_dir is None:
        parser.error(
            "Source scripts/paths.sh to set CPO3_RESULTS_ROOT, or pass --raw-results-dir."
        )
    if not isfinite(args.verify_tol) or args.verify_tol <= 0:
        parser.error("--verify-tol must be finite and positive")
    if not isfinite(args.shift) or args.shift <= 0:
        parser.error("--shift must be finite and positive")

    prepared = prepare_results(args.raw_results_dir)
    repetitions = classify_results(prepared, args.verify_tol)
    results = aggregate_repeats(repetitions)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    repetitions.write_parquet(args.results_dir / "results.parquet")
    results.write_parquet(args.results_dir / "median_results.parquet")
    print(f"Prepared {prepared.height} results from {args.raw_results_dir}")
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
            print(summarize_problem_type(results, args.shift, problem_type))
    create_plots(results, repetitions, args.results_dir / "plots", args.verify_tol)


if __name__ == "__main__":
    main()
