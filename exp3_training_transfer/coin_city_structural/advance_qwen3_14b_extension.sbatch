#!/bin/bash
#SBATCH --job-name=c3_q14_advance
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/q14_advance_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/q14_advance_%j.err
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"

: "${CANARY_JOB_ID:?CANARY_JOB_ID is required}"
: "${CANARY_LEDGER:?CANARY_LEDGER is required}"
: "${SCALE_ACCOUNT:=mltheory}"
: "${SCALE_PARTITION:=mltheory}"
: "${SCALE_EXCLUDE:=node206}"
: "${SCALE_VLLM_RATIO:=0.78}"
: "${SCALE_TRAIN_WALLTIME:=30:00:00}"
: "${SCALE_BASE_WALLTIME:=06:00:00}"
: "${SCALE_FINALIZE_WALLTIME:=01:00:00}"

python "$FAMILY/launch_qwen3_14b_extension.py" \
  --stage full \
  --canary-job-id "$CANARY_JOB_ID" \
  --canary-ledger "$CANARY_LEDGER" \
  --account "$SCALE_ACCOUNT" \
  --partition "$SCALE_PARTITION" \
  --exclude "$SCALE_EXCLUDE" \
  --vllm-ratio "$SCALE_VLLM_RATIO" \
  --walltime "$SCALE_TRAIN_WALLTIME" \
  --base-walltime "$SCALE_BASE_WALLTIME" \
  --finalize-walltime "$SCALE_FINALIZE_WALLTIME"
