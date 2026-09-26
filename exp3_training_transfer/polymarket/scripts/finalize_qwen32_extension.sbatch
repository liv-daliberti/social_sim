#!/bin/bash
#SBATCH --job-name=exp4_qwen32_finalize
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_finalize_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_finalize_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=all
#SBATCH --account=mltheory

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${EVALUATION_SPECS:?EVALUATION_SPECS is required}"
: "${REGISTRATION_LEDGER:?REGISTRATION_LEDGER is required}"
"$PY" "$POLY/scripts/render_qwen32_extension.py" \
  --evaluations "$EVALUATION_SPECS" \
  --registration-ledger "$REGISTRATION_LEDGER" \
  --output-prefix "$POLY/reports/exp4_qwen32_scale_latest"

echo "Qwen3-32B scale summary: $POLY/reports/exp4_qwen32_scale_latest.summary.json"
