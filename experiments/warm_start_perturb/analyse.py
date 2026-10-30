"""Make the warm-start table (Table 6 in the paper) from the runs of perturb.py.

A pair is a problem at one delta solved cold and warm. Only pairs where both
runs report "solved" are counted. Speedup is cold over warm solve time.

Run with: uv run python experiments/warm_start_perturb/analyse.py
"""

import math
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RUNS = HERE / "data" / "warm_runs.csv"
TABLE = HERE / "results" / "warm_start_table.tex"


def main() -> None:
    runs = pd.read_csv(RUNS)
    for family, problems in runs.groupby("family")["problem"]:
        print(f"{family.upper()}: {problems.nunique()} problems")

    cold = runs[runs["mode"] == "cold"]
    warm = runs[runs["mode"] == "warm"]
    pairs = cold.merge(warm, on=["family", "problem", "delta"], suffixes=("_cold", "_warm"))
    solved = pairs[(pairs["status_cold"] == "solved") & (pairs["status_warm"] == "solved")]
    solved = solved.assign(speedup=solved["solve_time_s_cold"] / solved["solve_time_s_warm"])

    table = solved.groupby(["family", "delta"]).agg(
        pairs=("problem", "size"),
        cold_iters=("iters_cold", "median"),
        warm_iters=("iters_warm", "median"),
        speedup=("speedup", "median"),
    )
    print(table.round(2).to_string())

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Problems & $\varepsilon_{\mathrm{pert}}$ & Pairs & \multicolumn{2}{c}{Median iterations} & Median \\",
        r" & & & Cold & Warm & speedup \\",
        r"\hline",
    ]
    for (family, delta), row in table.iterrows():
        lines.append(
            f"{family.upper()} & $10^{{{round(math.log10(delta))}}}$ & {int(row.pairs)} & "
            f"{row.cold_iters:g} & {row.warm_iters:g} & {row.speedup:.2f}$\\times$ \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]

    TABLE.parent.mkdir(exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"Saved {TABLE}")


if __name__ == "__main__":
    main()
