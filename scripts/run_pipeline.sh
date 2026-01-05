#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  echo "Missing .env (copy .env.template and fill in read-only URI)."
  exit 1
fi

set -a
source .env
set +a

export PYTHONPATH=.

echo "Running phase 2 dataset build (reads MongoDB, writes local data/derived)..."
python3 -m src.data.phase2_dataset

echo "Running phase 3 baselines..."
python3 -m src.data.phase3_baselines

echo "Running phase 3 modeling..."
python3 -m src.data.phase3_modeling

echo "Running phase 3 analysis..."
python3 -m src.data.phase3_analysis

echo "Exporting CSV snapshots..."
python3 scripts/export_csv.py

echo "Done. You can now run the dashboard with: python3 src/dashboard/app.py"
