#!/bin/bash
#SBATCH --job-name=exp4_q32_five_seed_schedule
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/q32_schedule_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/q32_schedule_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
: "${FIVE_SEED_LEDGER:?FIVE_SEED_LEDGER is required}"
mkdir -p "$REPO/exp3_training_transfer/five_seed_extension/logs"
"$PY" "$REPO/exp3_training_transfer/five_seed_extension/launch.py" \
  --schedule-qwen32-eval "$FIVE_SEED_LEDGER"
