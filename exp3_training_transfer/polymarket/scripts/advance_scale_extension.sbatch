#!/bin/bash
#SBATCH --job-name=exp4_scale_advance
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/scale_advance_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/scale_advance_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=all

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"

: "${CANARY_JOB_ID:?CANARY_JOB_ID is required}"
: "${SCALE_PARTITION:=all}"
: "${SCALE_EXCLUDE:=node206}"
: "${SCALE_VLLM_RATIO:=0.78}"
: "${SCALE_TRAIN_WALLTIME:=30:00:00}"
: "${SCALE_EVAL_WALLTIME:=08:00:00}"
: "${SCALE_FINALIZE_WALLTIME:=01:00:00}"

python "$POLY/scripts/launch_scale_extension.py" \
  --stage full \
  --canary-job-id "$CANARY_JOB_ID" \
  --partition "$SCALE_PARTITION" \
  --exclude "$SCALE_EXCLUDE" \
  --vllm-ratio "$SCALE_VLLM_RATIO" \
  --walltime "$SCALE_TRAIN_WALLTIME" \
  --eval-walltime "$SCALE_EVAL_WALLTIME" \
  --finalize-walltime "$SCALE_FINALIZE_WALLTIME"
