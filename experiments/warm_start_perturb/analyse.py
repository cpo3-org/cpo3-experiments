"""Print the full comparison and write its CSV and LaTeX summaries.

Run from this directory with: uv run python analyse.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parent / "data" / "warm_runs.csv"
SUMMARY = DATA.with_name("warm_summary.csv")
RESULTS = DATA.parent.parent / "results"
SUMMARY_COLUMNS = (
    "family", "problem", "delta", "baseline_status", "cold_status",
    "warm_status", "warm_used", "warm_accepted", "status_agreement",
    "cold_iters", "warm_iters", "iters_saved", "cold_time_s", "warm_time_s",
    "speedup_x", "cold_kkt_solves", "warm_kkt_solves", "cold_s_per_iter",
    "warm_s_per_iter", "s_per_iter_ratio", "warm_fixed_overhead_s",
    "cold_refinement_passes", "warm_refinement_passes",
    "cold_refine_per_kkt", "warm_refine_per_kkt", "refine_per_kkt_ratio",
    "warm_start_score",
)
DELTA_LABELS = {
    0.0: r"$0$",
    1e-4: r"$10^{-4}$",
    1e-3: r"$10^{-3}$",
    1e-2: r"$10^{-2}$",
    1e-1: r"$10^{-1}$",
}


def trial_summary(runs: pd.DataFrame) -> tuple[str, float, float, str]:
    """Show median work and time even when a run reports infeasibility."""
    statuses = ", ".join(sorted(runs["status"].unique()))
    accepted = int(runs["warm_start_used"].eq(True).sum())
    acceptance = f"{accepted}/{len(runs)}"
    return (
        statuses,
        float(runs["iters"].median()),
        float(runs["solve_time_s"].median()),
        acceptance,
    )


def median_of(runs: pd.DataFrame, column: str) -> float:
    """Median of a column that may be absent from an older CSV."""
    if column not in runs.columns:
        return float("nan")
    return float(pd.to_numeric(runs[column], errors="coerce").median())


def ratio_of(numerator: float, denominator: float) -> float:
    """Guarded division, since either side may be missing or zero."""
    if pd.isna(denominator) or not denominator:
        return float("nan")
    return numerator / denominator


def per_iteration(seconds: float, iterations: float) -> float:
    """Seconds per iteration, the unit price of the work being bought."""
    return seconds / iterations if iterations and iterations > 0 else float("nan")


def fixed_overhead(
    cold_time: float, cold_iters: float, warm_time: float, warm_iters: float
) -> float:
    """Warm time left over after charging its iterations at the cold rate.

    Screening a warm start costs real time before the first iteration, and
    the iteration counter cannot see it. Charging the warm iterations at the
    cold unit price and taking the remainder attributes that gap. It is an
    estimate, not a measurement: it also absorbs any per-iteration cost the
    warm path carries, so read it beside `s_per_iter_ratio` rather than
    alone.
    """
    rate = per_iteration(cold_time, cold_iters)
    if pd.isna(rate) or pd.isna(warm_iters):
        return float("nan")
    return warm_time - rate * warm_iters


def all_accepted(runs: pd.DataFrame) -> bool:
    """True when every warm trial in a case took the supplied point.

    A rejected warm start runs as a cold start under a warm label, so its
    iterations and time carry no information about warm starting. Anything
    that is not an outright acceptance counts against the case, including a
    skipped or errored trial, whose flag is missing rather than false.
    """
    return bool(len(runs) > 0 and runs["warm_start_used"].eq(True).all())


def latex_number(value: float, decimals: int = 1) -> str:
    """Keep integer medians tidy, and show missing measurements as dashes."""
    if pd.isna(value):
        return "--"
    if decimals == 1 and float(value).is_integer():
        return str(int(value))
    return f"{value:.{decimals}f}"


def write_latex_tables(frame: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    """Make one four-row table per family and a document that displays both."""
    keys = ["family", "problem", "delta", "trial"]
    cold = comparisons.loc[comparisons["mode"] == "cold", keys + ["status"]]
    warm = comparisons.loc[
        comparisons["mode"] == "warm", keys + ["status", "warm_start_used"]
    ]
    paired = cold.merge(warm, on=keys, suffixes=("_cold", "_warm"))
    # A skipped row has no attempted warm solve, so it is not a percentage trial.
    paired = paired.loc[paired["status_warm"] != "skipped"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    for family, title in (("qp", r"Maros--M\'esz\'aros QPs"), ("lp", "Netlib LPs")):
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{Warm-start results for {title}. The accepted column uses every paired trial with a warm attempt, and agreement is the rate over accepted trials. Iterations and speedup are conditioned on both acceptance and status agreement, since a rejected warm start runs as a cold start and a disagreeing pair did not solve the same problem; they are medians of problem medians, over whichever status the pair agreed on rather than solved cases alone.}}",
            rf"\label{{tab:warm-{family}}}",
            r"\begin{tabular}{ccccc}",
            r"\hline",
            r"$\delta$ & Accepted (\%) & Median iterations (warm / cold) & Median speedup ($\times$) & Status agreement (\%) \\",
            r"\hline",
        ]
        for delta in DELTA_LABELS:
            cases = frame.loc[(frame["family"] == family) & (frame["delta"] == delta)]
            attempts = paired.loc[(paired["family"] == family) & (paired["delta"] == delta)]
            # Acceptance is the rate itself, so it alone spans every attempt.
            accepted = 100 * attempts["warm_start_used"].eq(True).mean()
            # The rest describe warm starting, so they need an accepted point
            # and two runs that agreed on the answer, whatever that answer was.
            taken = cases.loc[cases["warm_accepted"] & cases["status_agreement"]]
            agreed = attempts.loc[attempts["warm_start_used"].eq(True)]
            agreement = 100 * agreed["status_cold"].eq(agreed["status_warm"]).mean()
            counted = taken.dropna(subset=["warm_iters", "cold_iters"])
            iterations = (
                f"{latex_number(counted['warm_iters'].median())} / "
                f"{latex_number(counted['cold_iters'].median())}"
            )
            lines.append(
                f"{DELTA_LABELS[delta]} & {latex_number(accepted)}\\% & "
                f"{iterations} & {latex_number(taken['speedup_x'].median(), 2)} & "
                f"{latex_number(agreement)}\\% " + r"\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (RESULTS / f"{family}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compile from the results directory so these relative inputs resolve.
    document = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\input{qp.tex}",
        r"\input{lp.tex}",
        r"\end{document}",
    ]
    (RESULTS / "results.tex").write_text("\n".join(document) + "\n", encoding="utf-8")


def report_closeness_cost(counted: pd.DataFrame) -> None:
    """Test whether iterations get dearer as the starting point gets closer.

    A warm start skips the cheap approach and spends all its iterations near
    the optimum, where residuals are small and iterative refinement works
    hardest. If that is what makes a warm iteration dear, the cost ratio
    should fall monotonically as `delta` grows and the start moves away,
    while the factorization count per iteration stays flat. Printing both
    together makes the claim falsifiable rather than plausible.
    """
    live = counted.loc[counted["s_per_iter_ratio"].notna()]
    if live.empty:
        return
    print()
    print("  cost per iteration against distance moved:")
    print(
        f"    {'delta':>8}{'n':>5}{'s/iter w/c':>12}{'kkt/iter w':>12}"
        f"{'kkt/iter c':>12}{'refine/kkt w/c':>16}"
    )
    for delta, group in live.groupby("delta"):
        refine = group["refine_per_kkt_ratio"].median()
        print(
            f"    {delta:>8}{len(group):>5}"
            f"{group['s_per_iter_ratio'].median():>12.3f}"
            f"{ratio_of(group['warm_kkt_solves'].median(), group['warm_iters'].median()):>12.3f}"
            f"{ratio_of(group['cold_kkt_solves'].median(), group['cold_iters'].median()):>12.3f}"
            f"{refine:>16.3f}"
        )
    print(
        "    falling cost with flat factorizations supports refinement;"
        " a rising factorization column would refute it"
    )


def report_time_decomposition(counted: pd.DataFrame) -> None:
    """Explain any gap between the iteration saving and the speedup.

    Iterations have no fixed component, wall time does, so a median speedup
    taken over a population of small problems can look far worse than the
    iteration ratio without anything being wrong. Print both averages and
    split by problem size so the composition is visible.
    """
    live = counted.loc[(counted["delta"] > 0) & counted["speedup_x"].notna()]
    print()
    print(f"Time decomposition over {len(live)} counted cases with delta > 0:")
    if live.empty:
        print("  nothing to decompose")
        return

    iters_ratio = live["warm_iters"].sum() / live["cold_iters"].sum()
    time_ratio = live["warm_time_s"].sum() / live["cold_time_s"].sum()
    print(f"  iteration ratio, totals      {iters_ratio:.3f}")
    print(f"  time ratio, totals           {time_ratio:.3f}")
    print(f"  speedup from totals          {1 / time_ratio:.3f}")
    print(f"  speedup, median of per-case  {live['speedup_x'].median():.3f}")
    print(f"  median seconds per iteration ratio {live['s_per_iter_ratio'].median():.3f}")
    print(f"  median fixed overhead        {live['warm_fixed_overhead_s'].median():.6f} s")
    slower = ((live["warm_time_s"] > live["cold_time_s"]) & (
        live["warm_iters"] < live["cold_iters"])).sum()
    print(f"  cases slower in time but fewer iterations: {slower} / {len(live)}")

    if live["cold_kkt_solves"].notna().any():
        cold_kkt = live["cold_kkt_solves"].sum()
        warm_kkt = live["warm_kkt_solves"].sum()
        if cold_kkt > 0:
            print(f"  factorization ratio, totals  {warm_kkt / cold_kkt:.3f}")
    report_closeness_cost(counted)

    # A fixed cost is harmless on a slow problem and fatal on a fast one, so
    # the quartiles matter more than any single average.
    if len(live) >= 8:
        bands = pd.qcut(live["cold_time_s"], 4,
                        labels=["fastest", "q2", "q3", "slowest"])
        print("  by cold solve time:")
        for band, group in live.groupby(bands, observed=True):
            print(
                f"    {str(band):<8} n={len(group):<4} "
                f"median cold {group['cold_time_s'].median():.5f}s  "
                f"median speedup {group['speedup_x'].median():.3f}  "
                f"total time ratio {group['warm_time_s'].sum() / group['cold_time_s'].sum():.3f}"
            )


def main() -> None:
    data = pd.read_csv(DATA)
    # Drop problems excluded from the experiment, even if an older CSV has them.
    from perturb import EXCLUDED
    family_of = {"netlib": "lp", "maros-meszaros": "qp"}
    for suite, names in EXCLUDED.items():
        data = data.loc[~((data["family"] == family_of[suite]) & data["problem"].isin(names))]
    baseline = data.loc[data["mode"] == "baseline"].set_index(["family", "problem"])
    comparisons = data.loc[data["mode"] != "baseline"]
    rows = []

    for (family, problem, delta), runs in comparisons.groupby(
        ["family", "problem", "delta"], sort=True
    ):
        cold = runs.loc[runs["mode"] == "cold"]
        warm = runs.loc[runs["mode"] == "warm"]
        cold_status, cold_iters, cold_time, _ = trial_summary(cold)
        warm_status, warm_iters, warm_time, warm_used = trial_summary(warm)
        # Agreement alone, not agreement on "solved". Two runs that both
        # prove infeasibility are still doing the same work on the same
        # problem, so their iterations and times are comparable.
        comparable = cold_status == warm_status

        # A positive saved-iteration count or a speedup above one favors warm.
        rows.append({
            "family": family,
            "problem": problem,
            "delta": delta,
            "baseline_status": baseline.loc[(family, problem), "status"],
            "cold_status": cold_status,
            "warm_status": warm_status,
            "warm_used": warm_used,
            "warm_accepted": all_accepted(warm),
            "status_agreement": comparable,
            "cold_iters": cold_iters,
            "warm_iters": warm_iters,
            "iters_saved": cold_iters - warm_iters if comparable else float("nan"),
            "cold_time_s": cold_time,
            "warm_time_s": warm_time,
            "speedup_x": (
                cold_time / warm_time
                if comparable and warm_time > 0
                else float("nan")
            ),
            # Iterations are not the unit of work the clock charges for. A
            # warm point that enters badly conditioned makes the solver
            # refactorize inside an iteration, so track factorizations and
            # the unit price beside the count.
            "cold_kkt_solves": median_of(cold, "kkt_solves"),
            "warm_kkt_solves": median_of(warm, "kkt_solves"),
            "cold_s_per_iter": per_iteration(cold_time, cold_iters),
            "warm_s_per_iter": per_iteration(warm_time, warm_iters),
            "s_per_iter_ratio": (
                per_iteration(warm_time, warm_iters)
                / per_iteration(cold_time, cold_iters)
                if comparable
                else float("nan")
            ),
            "warm_fixed_overhead_s": (
                fixed_overhead(cold_time, cold_iters, warm_time, warm_iters)
                if comparable
                else float("nan")
            ),
            # Correction back-solves per factorization. A start near the
            # optimum meets small residuals, so refinement works harder and
            # each solve costs more even at an unchanged factorization count.
            "cold_refinement_passes": median_of(cold, "refinement_passes"),
            "warm_refinement_passes": median_of(warm, "refinement_passes"),
            "cold_refine_per_kkt": ratio_of(
                median_of(cold, "refinement_passes"), median_of(cold, "kkt_solves")
            ),
            "warm_refine_per_kkt": ratio_of(
                median_of(warm, "refinement_passes"), median_of(warm, "kkt_solves")
            ),
            "refine_per_kkt_ratio": ratio_of(
                ratio_of(
                    median_of(warm, "refinement_passes"), median_of(warm, "kkt_solves")
                ),
                ratio_of(
                    median_of(cold, "refinement_passes"), median_of(cold, "kkt_solves")
                ),
            ),
            "warm_start_score": median_of(warm, "warm_start_score"),
        })

    # Keep the same column names and order as the original 16-problem CSV.
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    # Save the same full table that is printed below for spreadsheet inspection.
    frame.to_csv(SUMMARY, index=False)
    write_latex_tables(frame, comparisons)
    print(frame.to_string(
        index=False, max_rows=None, max_cols=None,
        float_format=lambda x: f"{x:.6g}",
    ))
    print()
    # Counting rejected cases here would just count cold starts beating
    # themselves, so these tallies follow the tables and use accepted cases.
    taken = frame.loc[frame["warm_accepted"] & frame["status_agreement"]]
    print(f"Cases with an accepted warm start: {frame['warm_accepted'].sum()} / {len(frame)}")
    print(f"Of those, cases where both runs agreed: {len(taken)}")
    print(
        f"Counted cases with fewer warm iterations: "
        f"{(taken['iters_saved'] > 0).sum()} / {len(taken)}"
    )
    print(
        f"Counted cases with lower warm solve time: "
        f"{(taken['speedup_x'] > 1).sum()} / {len(taken)}"
    )
    report_time_decomposition(taken)
    print(f"Saved full table to {SUMMARY}")
    print(f"Saved LaTeX tables to {RESULTS}")


if __name__ == "__main__":
    main()
