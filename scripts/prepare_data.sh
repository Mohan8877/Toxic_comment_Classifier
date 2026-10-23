#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python}

echo "Running data validation and creating splits..."
${PYTHON} -m src.data_validation --input data/raw/train.csv --out data/processed --seed 42
echo "Done. Processed files in data/processed/"
