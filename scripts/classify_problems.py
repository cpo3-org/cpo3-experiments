"""Build the LP/QP catalog from the prepared benchmark datasets."""

import argparse
import csv
from pathlib import Path

import yaml
from solver_benchmarks.datasets.mps import MPSLPDataset
from solver_benchmarks.datasets.registry import get_dataset


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    # These configs cover all feasible and infeasible datasets in the reports.
    configs = (
        repo / "experiments/presolve_off/configs/cpo3.yaml",
        repo / "experiments/cpo3_vs_clarabel/configs/cpo3_infeasible.yaml",
    )
    rows = []
    for config in configs:
        for entry in yaml.safe_load(config.read_text())["datasets"]:
            entry = {"name": entry} if isinstance(entry, str) else entry
            name = entry["name"]
            dataset_id = entry.get("id", name)
            dataset = get_dataset(name)(
                data_root=args.data_root, **entry.get("dataset_options", {})
            )
            problems = dataset.list_problems()
            if not problems:
                raise ValueError(f"No problems found for {dataset_id}")
            print(f"Classifying {dataset_id}: {len(problems)} problems", flush=True)
            for problem in problems:
                if isinstance(dataset, MPSLPDataset):
                    # This adapter constructs P=0 unless add_quadratic is set.
                    quadratic = bool(dataset.options.get("add_quadratic", False))
                else:
                    loaded = dataset.load_problem(problem.name)
                    quadratic = bool(loaded.qp["P"].nnz)
                    del loaded
                rows.append(
                    {
                        "dataset": dataset_id,
                        "problem": problem.name,
                        "problem_type": "qp" if quadratic else "lp",
                    }
                )

    output = repo / "src/cpo3_experiments/problem_types.csv"
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "problem", "problem_type"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["problem"])))
    print(f"Saved {len(rows)} classifications to {output}")


if __name__ == "__main__":
    main()
