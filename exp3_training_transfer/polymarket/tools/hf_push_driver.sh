#!/usr/bin/env bash
# Standalone HF push driver: regenerates the auto dataset-card README and uploads
# the ALREADY-VALIDATED export + status files, using the production wrapper's own
# update_hf_readme / upload_to_hf functions (hf_functions.sh, extracted verbatim).
# NOTE: this uploads the AUTO-generated README; re-upload the custom card after.
# Usage: hf_push_driver.sh /abs/path/to/polymarket_full_market_dataset_<RUNID>.json
set -euo pipefail

ROOT="/n/fs/similarity/kalshi/agentic_forecasting"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.runtime/tinker_py311_venv/bin/python}"
HF_PYTHON_BIN="${HF_PYTHON_BIN:-$ROOT/.runtime/qwen_vllm/venv/bin/python}"
HF_REPO_ID="${HF_REPO_ID:-od2961/polymarket-full-market-dataset}"

DATASET_PATH="${1:?usage: hf_push_driver.sh DATASET_PATH}"
[[ -f "$DATASET_PATH" ]] || { echo "missing dataset: $DATASET_PATH"; exit 1; }
[[ -f "${DATASET_PATH%.json}.summary.json" ]] || { echo "missing summary"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hf_functions.sh"

cd "$ROOT"
echo "=== regenerating auto README ==="
update_hf_readme "$DATASET_PATH"
echo "=== uploading to $HF_REPO_ID ==="
upload_to_hf "$DATASET_PATH"
echo "=== updating latest pointer ==="
printf 'reports/%s\n' "$(basename "$DATASET_PATH")" > "$ROOT/reports/latest_polymarket_export_path.txt"
echo "done"
