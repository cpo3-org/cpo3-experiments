"""Compare cold and warm starts on the shipped Netlib LP and Maros QP suites.

Run from this directory with: uv run python perturb.py
"""

from __future__ import annotations

import csv
import gzip
import shutil
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import cpo3


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "experiments" / "warm_start_perturb" / "problems"
OUTPUT = HERE / "data" / "warm_runs.csv"

# How many problems to draw from each suite. `None` uses every eligible
# problem, `0` skips the suite entirely, and a positive count draws that
# many at random. The draw is seeded by `SELECTION_SEED`, so a given count
# always yields the same problems, and a larger count is not guaranteed to
# contain the smaller one.
NETLIB_SAMPLE: int | None = None
MAROS_SAMPLE: int | None = None

DELTAS = (1e-4, 1e-3, 1e-2, 1e-1)
SEED = 20260918
SELECTION_SEED = 20260921
# Problems left out of the experiment entirely, by suite.
EXCLUDED = {
    "netlib": {"QAP8", "QAP12", "QAP15", "STOCFOR3", "TRUSS"},
}
REPEATS = 1
SETTINGS = cpo3.Settings(presolve=False, max_iters=100, time_limit=300.0, force_warm_start=True)

COLUMNS = (
    "family", "problem", "delta", "mode", "trial", "seed",
    "b_relative_change", "c_relative_change", "b_moved_fraction",
    "c_moved_fraction", "status", "iters", "kkt_solves", "refinement_passes",
    "refinement_settled", "solve_time_s", "wall_time_s", "warm_start_used",
    "warm_start_score", "error",
)


def problem_seed(name: str) -> int:
    """Seed a problem by name, so its directions do not depend on sampling."""
    return SEED + zlib.crc32(name.encode("utf-8"))


def load_problem(source: Path, workdir: Path) -> cpo3.Problem:
    """Expand one gzipped MPS file and read it; the archive stays untouched."""
    plain = workdir / f"{source.name.removesuffix('.mps.gz')}.mps"
    try:
        with gzip.open(source, "rb") as compressed, plain.open("wb") as expanded:
            shutil.copyfileobj(compressed, expanded)
        return cpo3.read_mps(plain, first_named_set=True)
    finally:
        plain.unlink(missing_ok=True)


def is_pure_equality(problem: cpo3.Problem) -> bool:
    """True when no row carries a barrier, so `nu == 0`.

    The solver refuses to warm start such a problem, because without an
    inequality pair there is no complementarity to read a path level from.
    Running one would only ever record a cold start under a warm label. A
    problem with no rows at all counts here too, for the same reason.
    """
    return all(isinstance(cone, cpo3.ZeroCone) for cone in problem.cones)


def select_problems(
    paths: list[Path], sample: int | None, workdir: Path, suite: str
) -> list[Path]:
    """Draw eligible problems from one suite, excluding pure-equality ones.

    Eligibility needs the cones, so each candidate is read once here and
    again when it runs. Reads are negligible beside the solves they gate.
    """
    if sample == 0:
        return []
    if sample is not None and sample < 0:
        raise ValueError(f"{suite} sample must be None or nonnegative, got {sample}")

    order = list(paths)
    if sample is not None:
        # Shuffle so a count draws across the suite rather than its first few.
        rng = np.random.default_rng(SELECTION_SEED)
        order = [order[index] for index in rng.permutation(len(order))]

    chosen: list[Path] = []
    for path in order:
        if sample is not None and len(chosen) >= sample:
            break
        name = path.name.removesuffix(".mps.gz")
        if name in EXCLUDED.get(suite, set()):
            print(f"  skip {suite}/{name}: excluded")
            continue
        try:
            problem = load_problem(path, workdir)
        except Exception as exc:
            print(f"  skip {suite}/{name}: unreadable ({type(exc).__name__})")
            continue
        if is_pure_equality(problem):
            print(f"  skip {suite}/{name}: all constraints are equalities")
            continue
        chosen.append(path)

    if sample is not None and len(chosen) < sample:
        print(f"  {suite}: only {len(chosen)} eligible problems, wanted {sample}")
    # Sort so the run order does not depend on the shuffle.
    return sorted(chosen)


def signs_for(size: int, rng: np.random.Generator) -> np.ndarray:
    """One fixed set of directions, reused at every delta."""
    return rng.choice((-1.0, 1.0), size=size)


def perturb(vector: np.ndarray, signs: np.ndarray, delta: float) -> np.ndarray:
    """Move every entry by `delta` of its own magnitude.

    Each coordinate changes by exactly `delta` relative to itself, which
    makes the perturbed problem invariant to row scaling: rescaling a row of
    `[A | b]` rescales its perturbation with it, leaving the same problem.
    Drawing one global step and spreading it over the rows does not have
    that property, and concentrates the change on whichever rows the model
    happened to be written with large coefficients.

    The 2-norm change is still exactly `delta`, since every entry
    contributes `delta` times its own square. Entries that are exactly zero
    stay zero, so a structural row stays structural, but `delta` then
    overstates how much of a sparse vector moved. `moved_fraction` records
    how much of it was eligible to move at all.
    """
    return vector + delta * np.abs(vector) * signs


def moved_fraction(vector: np.ndarray) -> float:
    """Share of entries this rule can move, since zeros stay zero."""
    return float(np.mean(vector != 0.0)) if vector.size else 0.0


def relative_change(original: np.ndarray, changed: np.ndarray) -> float:
    """Record the achieved change so the scale rule can be checked in the CSV."""
    length = np.linalg.norm(original)
    return float(np.linalg.norm(changed - original) / length) if length else 0.0


def solve_one(
    problem: cpo3.Problem,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[cpo3.Solution | None, dict[str, object]]:
    """Keep a failed solve as a data row instead of losing the experiment."""
    started = time.perf_counter()
    try:
        solution = cpo3.solve(problem, settings=SETTINGS, warm_start=warm_start)
    except Exception as exc:
        return None, {
            **_blank_measurements(),
            "status": "error",
            "wall_time_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    info = solution.info
    return solution, {
        "status": solution.status.value,
        "iters": info.iters,
        # Factorizations, not iterations. A warm point that enters badly
        # conditioned makes the solver shift and retry inside an iteration,
        # which the iteration counter never sees but the clock does.
        "kkt_solves": info.primary_solves,
        # Correction back-solves inside those factorizations. A start near
        # the optimum meets small residuals, so refinement works harder per
        # solve; this is what makes a warm iteration dearer than a cold one
        # even when it needs no extra factorizations.
        "refinement_passes": info.refinement_solves,
        "refinement_settled": info.refinement_settled,
        "solve_time_s": info.solve_time,
        "wall_time_s": time.perf_counter() - started,
        "warm_start_used": info.warm_start_used,
        # The acceptance-screen score of the supplied point; None when no
        # warm start was offered. The path level and proximity recorded by
        # the earlier solver are no longer reported.
        "warm_start_score": info.warm_start_score,
        "error": "",
    }


def _blank_measurements() -> dict[str, object]:
    """Every measured field absent, so a failed row still has the columns."""
    return {
        "iters": None, "kkt_solves": None, "refinement_passes": None,
        "refinement_settled": None, "solve_time_s": None,
        "wall_time_s": None, "warm_start_used": None,
        "warm_start_score": None,
    }


def skipped(reason: str) -> dict[str, object]:
    """Keep the same CSV fields when a comparison cannot be run."""
    return {**_blank_measurements(), "status": "skipped", "error": reason}


def main() -> None:
    for suite in ("netlib", "maros-meszaros"):
        if not (SOURCE / suite).is_dir():
            raise FileNotFoundError(f"Expected benchmark suite {SOURCE / suite}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Write one completed problem at a time so a long sweep leaves useful data.
    with tempfile.TemporaryDirectory() as temporary, OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        workdir = Path(temporary)

        # Sorting first gives the shuffle, and the run order, a stable basis.
        print("Selecting problems...")
        netlib = select_problems(
            sorted((SOURCE / "netlib").glob("*.mps.gz")),
            NETLIB_SAMPLE, workdir, "netlib",
        )
        maros = select_problems(
            sorted((SOURCE / "maros-meszaros").glob("*.mps.gz")),
            MAROS_SAMPLE, workdir, "maros-meszaros",
        )
        problems = [("lp", "netlib", path) for path in netlib] + [
            ("qp", "maros-meszaros", path) for path in maros
        ]
        print(f"Running {len(netlib)} Netlib and {len(maros)} Maros problems")
        if not problems:
            print("Nothing to run; both sample counts are zero.")

        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        row_count = 0
        for index, (family, suite, source) in enumerate(problems):
            name = source.name.removesuffix(".mps.gz")
            seed = problem_seed(name)
            rows: list[dict[str, object]] = []
            try:
                original = load_problem(source, workdir)
            except Exception as exc:
                original = None
                result = {**skipped(f"{type(exc).__name__}: {exc}"), "status": "error"}

            if original is not None:
                # Each row moves by a share of its own entry, so a rescaled
                # row sees a rescaled move and the problem is unchanged.
                b, c = original.b.copy(), original.c.copy()
                a, p, cones = original.A, original.P, original.cones
                rng = np.random.default_rng(seed)
                b_signs = signs_for(b.size, rng)
                c_signs = signs_for(c.size, rng)
                b_moved, c_moved = moved_fraction(b), moved_fraction(c)
                baseline, result = solve_one(original)
            else:
                baseline = None
                b_moved = c_moved = float("nan")
            baseline_status = result["status"]
            common = {
                "family": family, "problem": name, "seed": seed,
                "b_moved_fraction": b_moved, "c_moved_fraction": c_moved,
            }
            rows.append({
                **common, "delta": 0.0, "mode": "baseline", "trial": 0,
                "b_relative_change": 0.0, "c_relative_change": 0.0,
                **result,
            })

            # The baseline is the sole source of warm values at every delta.
            warm_values = None
            if baseline is not None and baseline.status.value == "solved":
                values = (baseline.x, baseline.y, baseline.s)
                if all(np.all(np.isfinite(value)) for value in values):
                    warm_values = values

            for delta in DELTAS:
                if original is None:
                    problem = None
                    case_error = "problem could not be read"
                    b_change = c_change = float("nan")
                else:
                    try:
                        changed_b = perturb(b, b_signs, delta)
                        changed_c = perturb(c, c_signs, delta)
                        problem = cpo3.Problem(
                            P=p, c=changed_c, A=a, b=changed_b, cones=cones
                        )
                        b_change = relative_change(b, changed_b)
                        c_change = relative_change(c, changed_c)
                        case_error = ""
                    except Exception as exc:
                        problem = None
                        case_error = f"{type(exc).__name__}: {exc}"
                        b_change = c_change = float("nan")
                case = {
                    **common, "delta": delta,
                    "b_relative_change": b_change,
                    "c_relative_change": c_change,
                }
                for trial in range(1, REPEATS + 1):
                    # Alternate order so one mode does not always get the first run.
                    modes = ("cold", "warm") if trial % 2 else ("warm", "cold")
                    for mode in modes:
                        if problem is None:
                            result = skipped(case_error)
                        elif mode == "warm" and warm_values is None:
                            result = skipped(
                                "baseline was not solved with finite vectors"
                            )
                        else:
                            _, result = solve_one(
                                problem, warm_values if mode == "warm" else None
                            )
                        rows.append({**case, "mode": mode, "trial": trial, **result})
            writer.writerows(rows)
            output.flush()
            row_count += len(rows)
            print(
                f"[{index + 1}/{len(problems)}] {suite}/{name}: "
                f"baseline {baseline_status}",
                flush=True,
            )
    print(f"Saved {row_count} runs to {OUTPUT}")


if __name__ == "__main__":
    main()
