"""Run with: bash experiments/cplex_dc_opf_residuals/run.sh

Reproduce failed DC-OPF LPs from central_path repetition 1.
Write both sets of absolute metrics to results/comparison.csv and full
CPLEX quality metrics, our residuals, and the original records to details.jsonl.
"""

import csv
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import cplex
import yaml
from solver_benchmarks.analysis.derive import worst_relative_residual
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.datasets.dc_opf import load_matpower_case
from solver_benchmarks.solvers.cplex_adapter import CPLEXSolverAdapter
from solver_benchmarks.transforms.dc_opf import dc_opf_lp

METRICS = {
    "primal": ("primal_res", "primal_slack_bound_error_max"),
    "stationarity": ("dual_res", "dual_error_max"),
    "gap": ("duality_gap", "duality_gap"),
    "complementarity": ("comp_slack", "row_complementarity_max"),
}


def solve_with_quality(problem, settings, output):
    """Capture CPLEX diagnostics before the adapter closes its model."""
    quality = {}
    original_solve = cplex.Cplex.solve

    def capture_quality(model):
        original_solve(model)
        quality.update(
            (name, value)
            for name, value in vars(model.solution.get_quality_metrics()).items()
            if not name.startswith("_")
        )

    with patch.object(cplex.Cplex, "solve", capture_quality):
        result = CPLEXSolverAdapter(settings).solve(problem, output)
    return result, quality


def main():
    experiment = Path(__file__).resolve().parent
    raw_results = (
        Path(os.environ["CPO3_RESULTS_ROOT"])
        / "central_path/results/cplex/repeat-1/results.jsonl"
    )
    config = experiment.parent / "central_path/configs/cplex.yaml"
    settings = yaml.safe_load(config.read_text())["solvers"][0]["settings"]
    output = experiment / "results"
    output.mkdir(exist_ok=True)
    columns = ["problem", "status"] + [
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("ours", "cplex", "agree")
    ]
    with (
        raw_results.open() as source,
        (output / "comparison.csv").open("w", newline="") as csv_file,
        (output / "details.jsonl").open("w") as details,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for line in source:
            previous = json.loads(line)
            if (
                previous["dataset"] != "dc_opf"
                or previous["metadata"].get("nnz_p") != 0
            ):
                continue
            residual = worst_relative_residual(previous)
            if residual is None or residual <= 1e-6:
                continue

            name = previous["problem"]
            print(f"Loading and solving {name}", flush=True)
            case = load_matpower_case(Path(previous["metadata"]["source"]))
            qp, metadata = dc_opf_lp(case)
            problem = ProblemData("dc_opf", name, QP, qp, metadata)
            result, quality = solve_with_quality(problem, settings, output)
            record = {
                "previous": previous,
                "settings": settings,
                "ours": result.kkt,
                "cplex": quality,
            }
            details.write(json.dumps(record) + "\n")
            details.flush()
            if result.kkt is None or not quality:
                raise RuntimeError(f"{name}: missing diagnostics; see details.jsonl")

            row = {"problem": name, "status": result.status}
            for metric, (our_key, cplex_key) in METRICS.items():
                ours = abs(result.kkt[our_key])
                theirs = abs(quality[cplex_key])
                row[f"{metric}_ours"] = ours
                row[f"{metric}_cplex"] = theirs
                row[f"{metric}_agree"] = math.isclose(
                    ours, theirs, rel_tol=1e-5, abs_tol=1e-8
                )
            writer.writerow(row)
            csv_file.flush()
    print(f"Saved results to {output}")
    print("Agreement uses relative tolerance 1e-5 or absolute tolerance 1e-8.")


if __name__ == "__main__":
    main()
