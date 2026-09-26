#!/bin/bash
#SBATCH --job-name=exp4_scale_finalize
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/scale_finalize_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/scale_finalize_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=all

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${EVALUATION_SPECS:?EVALUATION_SPECS is required}"
"$PY" "$POLY/scripts/render_scale_extension.py" \
  --evaluations "$EVALUATION_SPECS" \
  --output-prefix "$POLY/reports/exp4_scale_latest"

echo "scale summary: $POLY/reports/exp4_scale_latest.summary.json"
