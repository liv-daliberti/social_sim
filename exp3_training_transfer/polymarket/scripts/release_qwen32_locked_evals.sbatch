#!/bin/bash
#SBATCH --job-name=exp4_qwen32_release
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_release_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_release_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=all
#SBATCH --account=mltheory

set -euo pipefail

: "${HELD_JOB_IDS:?HELD_JOB_IDS is required}"
IFS=: read -r -a jobs <<< "$HELD_JOB_IDS"
if [ "${#jobs[@]}" -ne 4 ]; then
  echo "expected exactly three parent evaluators and one parent finalizer" >&2
  exit 2
fi

for job in "${jobs[@]}"; do
  if ! [[ "$job" =~ ^[0-9]+$ ]]; then
    echo "invalid held job ID: $job" >&2
    exit 2
  fi
  state=$(scontrol show job "$job" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p' | head -n 1)
  reason=$(scontrol show job "$job" | sed -n 's/.*Reason=\([^ ]*\).*/\1/p' | head -n 1)
  if [ "$state" != PENDING ] || [ "$reason" != JobHeldUser ]; then
    echo "refusing release: job $job is state=$state reason=$reason" >&2
    exit 2
  fi
done

scontrol release "${jobs[@]}"
echo "released parent locked jobs: $HELD_JOB_IDS"
