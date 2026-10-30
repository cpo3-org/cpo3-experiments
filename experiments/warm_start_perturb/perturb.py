"""Solve perturbed Netlib LPs and Maros-Meszaros QPs from cold and warm starts.

For each problem we solve the original once, then perturb every entry of b and
c by delta times its own magnitude (random signs, fixed per problem) and solve
the perturbed problem twice: cold, and warm from the original solution.

Run with: uv run python experiments/warm_start_perturb/perturb.py
"""

import csv
import gzip
import tempfile
import zlib
from pathlib import Path

import numpy as np
import cpo3


HERE = Path(__file__).resolve().parent
PROBLEMS = HERE / "problems"
OUTPUT = HERE / "data" / "warm_runs.csv"

SUITES = {"lp": "netlib", "qp": "maros-meszaros"}
LIMIT = None  # Problems per suite, in name order; None runs them all.
DELTAS = (1e-4, 1e-3, 1e-2, 1e-1)
SEED = 20260918
EXCLUDED = {"QAP8", "QAP12", "QAP15", "STOCFOR3", "TRUSS"}
SETTINGS = cpo3.Settings(
    presolve=False, max_iters=100, time_limit=300.0, force_warm_start=True
)
COLUMNS = ("family", "problem", "delta", "mode", "status", "iters", "solve_time_s")


def read_problem(path: Path) -> cpo3.Problem:
    """cpo3 reads plain MPS only, so unzip to a temporary file first."""
    with tempfile.NamedTemporaryFile(suffix=".mps") as plain:
        plain.write(gzip.decompress(path.read_bytes()))
        plain.flush()
        return cpo3.read_mps(plain.name, first_named_set=True)


def solve(problem: cpo3.Problem, warm_start=None) -> cpo3.Solution:
    return cpo3.solve(problem, settings=SETTINGS, warm_start=warm_start)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()

        for family, suite in SUITES.items():
            for path in sorted((PROBLEMS / suite).glob("*.mps.gz"))[:LIMIT]:
                name = path.name.removesuffix(".mps.gz")
                if name in EXCLUDED:
                    continue
                problem = read_problem(path)

                # The solver cannot warm start a problem with no inequalities.
                if all(isinstance(cone, cpo3.ZeroCone) for cone in problem.cones):
                    print(f"skip {name}: equality constraints only")
                    continue
                original = solve(problem)
                if original.status.value != "solved":
                    print(f"skip {name}: original problem {original.status.value}")
                    continue
                warm_start = (original.x, original.y, original.s)

                # Seed by name so each problem always gets the same signs.
                rng = np.random.default_rng(SEED + zlib.crc32(name.encode()))
                b_signs = rng.choice((-1.0, 1.0), size=problem.b.size)
                c_signs = rng.choice((-1.0, 1.0), size=problem.c.size)

                for delta in DELTAS:
                    perturbed = cpo3.Problem(
                        P=problem.P,
                        c=problem.c + delta * np.abs(problem.c) * c_signs,
                        A=problem.A,
                        b=problem.b + delta * np.abs(problem.b) * b_signs,
                        cones=problem.cones,
                    )
                    for mode, start in (("cold", None), ("warm", warm_start)):
                        solution = solve(perturbed, start)
                        writer.writerow({
                            "family": family,
                            "problem": name,
                            "delta": delta,
                            "mode": mode,
                            "status": solution.status.value,
                            "iters": solution.info.iters,
                            "solve_time_s": solution.info.solve_time,
                        })
                file.flush()  # Keep finished problems if a long run stops.
                print(f"done {suite}/{name}", flush=True)


if __name__ == "__main__":
    main()
