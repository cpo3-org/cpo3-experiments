import argparse
import json
import os
from math import isfinite, log10
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import PercentFormatter
from solver_benchmarks.analysis.derive import worst_relative_residual

# Missing types confirmed against the central-path and interior-point experiments.
MISSING_PROBLEM_TYPES = {
    "Dual2_5000": "lp",
    "L1_sixm1000obs": "lp",
    "L1_sixm250obs": "lp",
    "L2CTA3D": "lp",
    "Primal2_1000": "lp",
    "a2864": "lp",
    "bdry2": "lp",
    "dlr1": "lp",
    "dlr2": "lp",
    "ex10": "lp",
    "fhnw-binschedule1": "lp",
    "highschool1-aigio": "lp",
    "irish-electricity": "lp",
    "neos-2075418-temuka": "lp",
    "neos-3025225": "lp",
    "neos-5104907-jarama": "lp",
    "neos-5114902-kasavu": "lp",
    "ns1644855": "lp",
    "physiciansched3-3": "lp",
    "rail02": "lp",
    "rmine15": "lp",
    "s250r10": "lp",
    "s82": "lp",
    "s100": "lp",
    "set-cover-model": "lp",
    "square41": "lp",
    "square47": "lp",
    "thk_48": "lp",
    "thk_63": "lp",
    "tpl-tub-ws1617": "lp",
    "8547": "qp",
    "9008": "qp",
}

SOLVERS = ("clarabel", "cpo3")
INFEASIBILITY_TOLERANCES = (1e-7, 1e-8, 1e-9, 1e-10)
TIME_LIMIT_SECONDS = 300.0


def verified_solve(residual: str, verify_tol: float) -> pl.Expr:
    return (
        (pl.col(residual) <= verify_tol)
        & (pl.col("run_time_seconds") <= TIME_LIMIT_SECONDS)
    ).fill_null(False)


def summarize(
    results: pl.DataFrame,
    residual: str = "worst_relative_residual",
    verify_tol: float = 1e-6,
) -> pl.DataFrame:
    assert set(results.select("solver", "tolerance").unique().iter_rows()) == {
        ("clarabel", 1e-8),
        ("cpo3", 1e-8),
    }
    assert not results.select("dataset", "problem", "solver").is_duplicated().any()
    coverage = results.group_by("dataset", "problem").agg(pl.col("solver").n_unique())
    assert (coverage["solver"] == 2).all()
    assert results["problem_type"].drop_nulls().is_in(["lp", "qp"]).all()

    results = results.with_columns(
        verified_solve(residual, verify_tol).alias("solved")
    )
    paired = results.pivot(
        on="solver",
        index=["dataset", "problem", "problem_type"],
        values=["solved", "run_time_seconds", "iterations"],
    )
    rows = []
    for label, subset in (
        ("All", paired),
        ("LP", paired.filter(pl.col("problem_type") == "lp")),
        ("QP", paired.filter(pl.col("problem_type") == "qp")),
    ):
        clarabel = pl.col("solved_clarabel")
        cpo3 = pl.col("solved_cpo3")
        shared = subset.filter(clarabel & cpo3)
        row = {
            "problems": label,
            "clarabel_only": subset.filter(clarabel & ~cpo3).height,
            "cpo3_only": subset.filter(cpo3 & ~clarabel).height,
            "shared": shared.height,
        }
        for solver in SOLVERS:
            row[f"{solver}_median_seconds"] = shared[
                f"run_time_seconds_{solver}"
            ].median()
            row[f"{solver}_median_iterations"] = shared[f"iterations_{solver}"].median()
        rows.append(row)
    return pl.DataFrame(rows)


def summarize_infeasible(
    results: pl.DataFrame, verify_tol: float = 1e-6
) -> pl.DataFrame:
    combinations = {
        (solver, tolerance)
        for solver in SOLVERS
        for tolerance in INFEASIBILITY_TOLERANCES
    }
    assert (
        set(results.select("solver", "infeasibility_tolerance").unique().iter_rows())
        == combinations
    )
    assert (
        not results.select("dataset", "problem", "solver", "infeasibility_tolerance")
        .is_duplicated()
        .any()
    )
    coverage = results.group_by("dataset", "problem").len()
    assert (coverage["len"] == len(combinations)).all()
    selected = results.filter(pl.col("infeasibility_tolerance") == 1e-8)
    return summarize(
        selected, "normalized_certificate_residual", verify_tol
    ).with_columns(
        pl.lit(1e-8).alias(f"{solver}_infeasibility_tolerance") for solver in SOLVERS
    )


def coverage_table(summary: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "Problems": row["problems"],
                "Solver": solver,
                "Solved": row["shared"] + row[f"{solver}_only"],
                "Unique": row[f"{solver}_only"],
                "Shared": row["shared"],
                "Median iterations": row[f"{solver}_median_iterations"],
            }
            for row in summary.iter_rows(named=True)
            for solver in ("cpo3", "clarabel")
        ]
    )


def shared_verified_results(results: pl.DataFrame, verify_tol: float) -> pl.DataFrame:
    return results.with_columns(
        verified_solve("worst_relative_residual", verify_tol).alias("solved")
    ).filter(pl.col("solved").all().over("dataset", "problem"))


def linear_work_table(results: pl.DataFrame, verify_tol: float) -> pl.DataFrame:
    shared = shared_verified_results(results, verify_tol)
    shared = shared.with_columns(
        (pl.col("kkt_solves") + pl.col("refinement_passes")).alias("applications")
    )
    rows = []
    for label, subset in (
        ("All", shared),
        ("LP", shared.filter(pl.col("problem_type") == "lp")),
        ("QP", shared.filter(pl.col("problem_type") == "qp")),
    ):
        for solver in ("cpo3", "clarabel"):
            solves = subset.filter(pl.col("solver") == solver)
            rows.append(
                {
                    "Problems": label,
                    "Solver": solver,
                    "Shared": solves.height,
                    "Factorizations": solves["factorizations"].median(),
                    "KKT solves": solves["kkt_solves"].median(),
                    "Refinement steps": solves["refinement_passes"].median(),
                    "Applications": solves["applications"].median(),
                }
            )
    return pl.DataFrame(rows)


def certificate_counts_table(results: pl.DataFrame, verify_tol: float) -> pl.DataFrame:
    return (
        results.group_by("infeasibility_tolerance", "solver")
        .agg(
            verified_solve("normalized_certificate_residual", verify_tol)
            .sum()
            .alias("verified")
        )
        .pivot(on="solver", index="infeasibility_tolerance", values="verified")
        .sort("infeasibility_tolerance", descending=True)
        .select(
            pl.col("infeasibility_tolerance").alias("Infeasibility tolerance"),
            "cpo3",
            "clarabel",
        )
    )


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
            nnz_p = result["metadata"].get("nnz_p")
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
                    "problem_type": None if nnz_p is None else "qp" if nnz_p else "lp",
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
    return pl.DataFrame(
        rows,
        schema_overrides={
            "problem_type": pl.String,
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
    ).with_columns(
        pl.col("problem_type")
        .drop_nulls()
        .first()
        .over("dataset", "problem")
        .fill_null(
            pl.col("problem").replace_strict(MISSING_PROBLEM_TYPES, default=None)
        )
    )


def create_plots(results: pl.DataFrame, output: Path, verify_tol: float) -> None:
    results = results.filter(pl.col("dataset").str.ends_with("_infeasible"))
    solvers = ["cpo3", "clarabel"]
    tolerances = sorted(results["infeasibility_tolerance"].unique(), reverse=True)
    colors = [plt.get_cmap("tab10")(i) for i in range(len(tolerances))]
    # MPC recommends Helvetica/Arial at 8–12 pt; allow local sans-serif fallbacks.
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
            "pdf.use14corefonts": False,
        }
    )
    fig, axes = plt.subplots(
        1, len(solvers), figsize=(119 / 25.4, 2.5), sharey=True, layout="constrained"
    )
    for ax, solver in zip(axes, solvers):
        ax.set_title("cpo3" if solver == "cpo3" else "Clarabel", fontsize=9)
        values = [
            results.filter(
                (pl.col("solver") == solver)
                & (pl.col("infeasibility_tolerance") == tolerance)
                & (pl.col("normalized_certificate_residual") <= verify_tol)
            )["normalized_certificate_residual"]
            .drop_nulls()
            .clip(lower_bound=1e-16)
            .log10()
            .to_numpy()
            for tolerance in tolerances
        ]
        populated = [i for i, sample in enumerate(values) if len(sample)]
        if populated:
            violin = ax.violinplot(
                [values[i] for i in populated],
                positions=[i + 1 for i in populated],
                showmedians=True,
                facecolor=[colors[i] for i in populated],
                linecolor=[colors[i] for i in populated],
            )
            for body in violin["bodies"]:
                body.set_alpha(0.65)
        ax.set_xticks(range(1, len(tolerances) + 1), [f"{t:g}" for t in tolerances])
        ax.axhline(log10(verify_tol), color="gray", linestyle=":")
        ax.yaxis.set_major_formatter(lambda value, _: f"$10^{{{value:g}}}$")
    fig.supylabel("Certificate residual", fontsize=9)
    fig.supxlabel("Infeasibility stopping tolerance", fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "certificates_violin.pdf")
    plt.close(fig)
    print(f"Saved {output / 'certificates_violin.pdf'}")


def create_refinement_plot(
    results: pl.DataFrame, output: Path, verify_tol: float
) -> None:
    shared = shared_verified_results(
        results.filter(~pl.col("dataset").str.ends_with("_infeasible")), verify_tol
    )
    if shared.is_empty():
        print("No shared verified problems for the refinement ECDF.")
        return
    if shared["refinement_passes"].null_count():
        print("Skipping refinement ECDF: some shared verified runs lack refinement counters.")
        return
    assert (shared["refinement_passes"] >= 0).all()
    count = shared.select("dataset", "problem").unique().height
    maximum = max(1, cast(int, shared["refinement_passes"].max()))
    upper = 10 ** max(1, int(np.ceil(np.log10(maximum))))
    # MPC artwork: 119 mm width, consistent 8–12 pt sans-serif lettering,
    # embedded fonts, lines >= 0.3 pt, and titles/captions in the manuscript.
    with plt.rc_context(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 3,
            "pdf.use14corefonts": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(119 / 25.4, 2.0), layout="constrained")
        for solver, label, linestyle in (
            ("cpo3", "cpo3", "-"),
            ("clarabel", "Clarabel", "--"),
        ):
            values = shared.filter(pl.col("solver") == solver)[
                "refinement_passes"
            ].to_numpy()
            assert len(values) == count
            unique, frequencies = np.unique(values, return_counts=True)
            cumulative = frequencies.cumsum() / len(values)
            line, = ax.step(
                np.r_[0, unique, upper],
                np.r_[0, cumulative, 1],
                where="post",
                linewidth=1.7,
                linestyle=linestyle,
                label=f"{label} (median {np.median(values):g})",
            )
            if unique[0] == 0:
                ax.plot(0, cumulative[0], "o", color=line.get_color(), markersize=4)
        ax.set_xscale("function", functions=(np.log1p, np.expm1))
        ticks = [0, *[10**power for power in range(int(np.log10(upper)) + 1)]]
        ax.set_xticks(ticks, [f"{tick:,}" for tick in ticks])
        ax.set_xlim(0, upper)
        ax.set_ylim(0, 1.025)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Refinement solves per problem, log1p(x) scale")
        ax.set_ylabel("Cumulative share\nof problems")
        ax.grid(axis="y", color="0.75", alpha=1, linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=True)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "refinement_solves_ecdf.pdf"
        fig.savefig(path)
        print(f"Saved {path} ({count} shared verified problems)")
        plt.close(fig)


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
        help="Raw input directory (default: $CPO3_RESULTS_ROOT/cpo3_vs_clarabel/results)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=experiment / "results",
        help="Output directory for prepared data, tables, and plots",
    )
    parser.add_argument("--verify-tol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.raw_results_dir is None:
        parser.error(
            "Source scripts/paths_local.sh or scripts/paths_savio.sh, or pass --raw-results-dir."
        )
    if not isfinite(args.verify_tol) or args.verify_tol <= 0:
        parser.error("--verify-tol must be finite and positive")
    results = prepare_results(args.raw_results_dir)
    expected = {
        "dc_opf",
        "kennington",
        "maros_meszaros",
        "miplib",
        "mittelmann",
        "mpc_qpbenchmark",
        "netlib",
        "qplib",
        "dc_opf_infeasible",
        "netlib_infeasible",
    }
    assert set(results["dataset"]) == expected, "Unexpected dataset coverage."
    unclassified = (
        results.filter(pl.col("problem_type").is_null())
        .select("dataset", "problem")
        .unique()
        .sort("dataset", "problem")
    )
    assert unclassified.is_empty(), (
        f"Unclassified LP/QP types (missing nnz_p metadata): {unclassified.rows()}"
    )
    output = args.results_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    infeasible = pl.col("dataset").str.ends_with("_infeasible")
    feasible_summary = summarize(
        results.filter(~infeasible), verify_tol=args.verify_tol
    )
    infeasible_summary = summarize_infeasible(
        results.filter(infeasible), args.verify_tol
    )
    results.write_parquet(args.results_dir / "results.parquet")
    print(f"Prepared {results.height} results from {args.raw_results_dir}")
    feasible_summary.write_csv(output / "comparison.csv")
    infeasible_summary.write_csv(output / "comparison_infeasible.csv")
    print(f"Independent verification threshold: {args.verify_tol:g}.")
    print(f"Verified solves and certificates must finish within {TIME_LIMIT_SECONDS:g}s.")
    print("Iteration and linear-algebra medians use shared verified feasible problems.")
    print(
        "Applications = KKT solves + refinement steps, computed before taking medians."
    )
    print("Certificates use absolute residuals after bᵀy = -1 normalization.")
    for label, table in (
        (
            "Solve counts, unique solves, and iterations",
            coverage_table(feasible_summary),
        ),
        (
            "Median cumulative linear-algebra counts",
            linear_work_table(results.filter(~infeasible), args.verify_tol),
        ),
        (
            "Verified certificate counts",
            certificate_counts_table(results.filter(infeasible), args.verify_tol),
        ),
    ):
        print(f"\n{label}")
        with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=180):
            print(table)

    create_plots(results, args.results_dir / "plots", args.verify_tol)
    create_refinement_plot(results, args.results_dir / "plots", args.verify_tol)


if __name__ == "__main__":
    main()
