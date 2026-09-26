#!/bin/bash
#SBATCH --job-name=c3_scale_output
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/output_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/output_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PROBE="$REPO/exp3_training_transfer/mechanism_family/mechanistic_probe"
PY="$REPO/.runtime/oat_conda/bin/python"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${SCORE_MANIFEST:?SCORE_MANIFEST is required}"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

"$PY" "$PROBE/run_output_tracking_v2.py" \
  --aggregate "$SCORE_MANIFEST" \
  --model "$MODEL_KEY" \
  --seeds 42 43 44 \
  --bootstrap 10000 \
  --output "$PROBE/runs/output_tracking_${MODEL_KEY}_scale_v2_1.json" \
  --predictions "$PROBE/runs/output_tracking_${MODEL_KEY}_scale_v2_1_predictions.jsonl"
