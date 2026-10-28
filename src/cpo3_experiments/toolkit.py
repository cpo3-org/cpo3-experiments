"""Shared calculations and plotting helpers for experiment reports."""

from bisect import bisect_right
from math import isfinite, log10
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import RcParams
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from solver_benchmarks.core import status

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


# MPC recommends Helvetica/Arial at 8–12 pt; allow local sans-serif fallbacks.
PLOT_STYLE = RcParams(
    {
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
        "pdf.use14corefonts": False,
    }
)


def add_problem_types(results: pl.DataFrame) -> pl.DataFrame:
    """Attach the saved classification for every dataset/problem pair."""
    catalog = pl.read_csv(
        Path(__file__).with_name("problem_types.csv"),
        schema={"dataset": pl.String, "problem": pl.String, "problem_type": pl.String},
    )
    results = results.join(
        catalog,
        on=["dataset", "problem"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    missing = (
        results.filter(pl.col("problem_type").is_null())
        .select("dataset", "problem")
        .unique()
    )
    if not missing.is_empty():
        raise ValueError(f"Problems missing from problem_types.csv: {missing.rows()}")
    return results


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


def validate_results(
    results: pl.DataFrame, solvers: list[str], repeats: set[int]
) -> None:
    if not (results["tolerance"] == 1e-8).all():
        raise ValueError("Expected stopping tolerance 1e-8.")
    if set(results["solver"]) != set(solvers):
        raise ValueError("Expected results for all configured solvers.")
    keys = ["dataset", "problem", "solver"]
    if results.select(*keys, "repeat").is_duplicated().any():
        raise ValueError("Duplicate solver/problem/repetition results.")
    for row in results.group_by(keys).agg(pl.col("repeat")).iter_rows(named=True):
        if set(row["repeat"]) != repeats:
            raise ValueError(f"Expected repetitions {sorted(repeats)} for {row}.")
    coverage = results.group_by("dataset", "problem").agg(pl.col("solver").n_unique())
    if not (coverage["solver"] == len(solvers)).all():
        raise ValueError("Some problems are missing solver results.")
    if results["category"].null_count():
        raise ValueError("Unclassified solver statuses.")
    if results["problem_type"].null_count():
        raise ValueError("Unclassified LP/QP types.")


def aggregate_repeats(
    results: pl.DataFrame, solvers: list[str], repeats: set[int]
) -> pl.DataFrame:
    """Median of three verified runtimes; each failed run contributes infinity.

    Two verified runs suffice for a finite median. One verified run does not.
    With one failure, the median is the slower of the two verified runs.
    """
    validate_results(results, solvers, repeats)
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
    runtime = (
        pl.when(pl.col("time").is_finite())
        .then(pl.col("time"))
        .otherwise(TIME_LIMIT_SECONDS)
    )
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
    results: pl.DataFrame, shift: float, problem_type: str, solver_order: list[str]
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
    solvers = pl.DataFrame({"solver": solver_order})
    return solvers.join(table, on="solver", maintain_order="left").select(
        "solver",
        *(
            f"{metric}_{dataset}"
            for dataset in datasets
            for metric in ("solved", "seconds")
        ),
    )


def plot_dolan_more(
    ax: Axes, results: pl.DataFrame, labels: dict[str, str], linestyles: dict[str, Any]
) -> None:
    solvers = sorted(labels)
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
            label=labels[solver],
            linestyle=linestyles[solver],
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
        [ax.lines[solvers.index(s)] for s in labels],
        list(labels.values()),
        loc="lower right",
        ncols=2,
        fontsize=8,
        handlelength=2,
        columnspacing=1,
        labelspacing=0.2,
    )


def plot_residuals(
    ax: Axes, results: pl.DataFrame, tol: float, labels: dict[str, str]
) -> None:
    solvers = list(labels)
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
    ax.set_xticks(range(1, len(solvers) + 1), list(labels.values()))
    ax.axhline(log10(tol), color="gray", linestyle=":")
    ax.yaxis.set_major_formatter(lambda value, _: f"$10^{{{value:g}}}$")
    ax.set_ylabel("KKT residual")
