#!/bin/bash
#SBATCH --job-name=exp4_qwen32_advance
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_advance_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_advance_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=all
#SBATCH --account=mltheory

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"

: "${REGISTRATION_LEDGER:?REGISTRATION_LEDGER is required}"
: "${CANARY_LEDGER:?CANARY_LEDGER is required}"
: "${CANARY_JOB_ID:?CANARY_JOB_ID is required}"
: "${QWEN32_ACCOUNT:=mltheory}"
: "${QWEN32_PARTITION:=all}"
: "${QWEN32_SUBMIT_ACCOUNT:=allcs}"
: "${QWEN32_SUBMIT_PARTITION:=cs}"
: "${QWEN32_SUBMIT_QOS:=medium}"
: "${QWEN32_EXCLUDE:=node206}"
: "${QWEN32_TRAIN_WALLTIME:=30:00:00}"
: "${QWEN32_EVAL_WALLTIME:=12:00:00}"

python "$POLY/scripts/launch_qwen32_extension.py" \
  --stage full \
  --registration-ledger "$REGISTRATION_LEDGER" \
  --canary-ledger "$CANARY_LEDGER" \
  --canary-job-id "$CANARY_JOB_ID" \
  --account "$QWEN32_ACCOUNT" \
  --partition "$QWEN32_PARTITION" \
  --submit-account "$QWEN32_SUBMIT_ACCOUNT" \
  --submit-partition "$QWEN32_SUBMIT_PARTITION" \
  --submit-qos "$QWEN32_SUBMIT_QOS" \
  --exclude "$QWEN32_EXCLUDE" \
  --walltime "$QWEN32_TRAIN_WALLTIME" \
  --eval-walltime "$QWEN32_EVAL_WALLTIME"
