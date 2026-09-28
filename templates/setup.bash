#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)"

DIRS=(
    "$PROJECT_ROOT/data/train"
    "$PROJECT_ROOT/data/eval"
    "$PROJECT_ROOT/data/test"

    "$PROJECT_ROOT/src"
    "$PROJECT_ROOT/checkpoints"
    "$PROJECT_ROOT/outputs"
    "$PROJECT_ROOT/notebooks"
)

echo "Creating PyTorch project structure..."

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    echo "Created: $dir"
done

echo
echo "Done."

conda env create -f environment.yml