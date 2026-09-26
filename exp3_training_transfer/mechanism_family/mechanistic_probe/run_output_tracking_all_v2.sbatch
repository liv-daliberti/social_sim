#!/bin/bash
#SBATCH --job-name=c3_output_v2
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/mechanistic_probe/logs/output_v2_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/mechanistic_probe/logs/output_v2_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PROBE="$REPO/exp3_training_transfer/mechanism_family/mechanistic_probe"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${MODEL_KEY:?MODEL_KEY is required}"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

mkdir -p "$PROBE/logs" "$PROBE/runs"
"$PY" "$PROBE/run_output_tracking_v2.py" \
  --model "$MODEL_KEY" \
  --seeds 42 43 44 \
  --bootstrap 10000 \
  --output "$PROBE/runs/output_tracking_${MODEL_KEY}_all_seeds_v2.json" \
  --predictions "$PROBE/runs/output_tracking_${MODEL_KEY}_all_seeds_predictions_v2.jsonl"

echo "Output tracking v2 complete: $MODEL_KEY"
