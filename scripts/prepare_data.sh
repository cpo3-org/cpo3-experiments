#!/usr/bin/env bash
set -euo pipefail
cd "$CPO3_REPO_ROOT"
SOLVER_BENCHMARKS_REVISION=$(uv run --no-sync python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["tool"]["uv"]["sources"]["solver-benchmarks"]["rev"])')

bundle_dir=$(mktemp -d)
trap 'rm -rf "$bundle_dir"' EXIT

curl --fail --location \
    "https://github.com/bodono/solver_benchmarks/archive/$SOLVER_BENCHMARKS_REVISION.tar.gz" |
    tar -xz -C "$bundle_dir" --strip-components=1 \
        "solver_benchmarks-$SOLVER_BENCHMARKS_REVISION/problem_classes/netlib_data" \
        "solver_benchmarks-$SOLVER_BENCHMARKS_REVISION/problem_classes/maros_meszaros_data" \
        "solver_benchmarks-$SOLVER_BENCHMARKS_REVISION/problem_classes/kennington"

mkdir -p "$CPO3_DATA_ROOT"
for folder in netlib_data maros_meszaros_data kennington; do
    printf 'Copying bundled dataset: %s\n' "$folder"
    mkdir -p "$CPO3_DATA_ROOT/$folder"
    cp -a "$bundle_dir/problem_classes/$folder/." \
        "$CPO3_DATA_ROOT/$folder/"
done

datasets=(
    qplib
    miplib
    dc_opf
    mittelmann
    mpc_qpbenchmark
)
for dataset in "${datasets[@]}"; do
    printf 'Preparing dataset: %s\n' "$dataset"
    uv run --no-sync bench data prepare "$dataset" --all \
        --option "data_root=$CPO3_DATA_ROOT"
done

shopt -s nullglob # Skip the loop if no .mps files match.
for file in "$CPO3_DATA_ROOT"/mittelmann/*.mps; do
    printf 'Compressing: %s\n' "$file"
    gzip -1 -- "$file"
done
