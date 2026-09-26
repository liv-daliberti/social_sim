#!/bin/bash
#SBATCH --job-name=c3_latent_probe
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/mechanistic_probe/logs/probe_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/mechanistic_probe/logs/probe_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PROBE="$REPO/exp3_training_transfer/mechanism_family/mechanistic_probe"
PY="$REPO/.runtime/oat_conda/bin/python"
TASK_DIR="$PROBE/data"

: "${MODEL:?MODEL is required}"
: "${MODEL_COMMIT:?MODEL_COMMIT is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${ENDPOINT_LABEL:?ENDPOINT_LABEL is required}"
: "${RUN_DIR:?RUN_DIR is required}"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export TOKENIZERS_PARALLELISM=false

nvidia-smi
mkdir -p "$RUN_DIR" "$PROBE/logs"
exec 9>"$RUN_DIR/.probe.lock"
if ! flock -n 9; then
  echo "Another probe owns $RUN_DIR/.probe.lock; exiting without changes."
  exit 0
fi

adapter_args=()
if [[ -n "${ADAPTER:-}" ]]; then
  adapter_args=(--adapter "$ADAPTER")
fi

"$PY" "$PROBE/extract_hidden_states.py" \
  --task-dir "$TASK_DIR" \
  --run-dir "$RUN_DIR" \
  --model "$MODEL" \
  --model-commit "$MODEL_COMMIT" \
  --model-label "$MODEL_LABEL" \
  --endpoint-label "$ENDPOINT_LABEL" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-input-tokens 3072 \
  "${adapter_args[@]}"

"$PY" "$PROBE/analyze_probe.py" \
  --task-dir "$TASK_DIR" \
  --run-dir "$RUN_DIR" \
  --permutations 10000 \
  --status probe_complete

echo "$MODEL_LABEL/$ENDPOINT_LABEL probe complete: $RUN_DIR"
