#!/bin/bash
#SBATCH --job-name=c3_scale_probe_extract
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/extract_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/extract_%j.err

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PROBE="$REPO/exp3_training_transfer/mechanism_family/mechanistic_probe"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${MODEL:?MODEL is required}"
: "${MODEL_COMMIT:?MODEL_COMMIT is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${ENDPOINT_LABEL:?ENDPOINT_LABEL is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${DEVICE_MAP:?DEVICE_MAP is required}"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_DIR" "$REPO/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs"
exec 9>"$RUN_DIR/.scale_v2_1_extract.lock"
flock -n 9 || { echo "another extraction owns $RUN_DIR"; exit 0; }

adapter_args=()
if [ -n "${ADAPTER:-}" ]; then
  adapter_args=(--adapter "$ADAPTER")
fi

"$PY" "$PROBE/extract_hidden_states.py" \
  --task-dir "$PROBE/data" \
  --run-dir "$RUN_DIR" \
  --model "$MODEL" \
  --model-commit "$MODEL_COMMIT" \
  --model-label "$MODEL_LABEL" \
  --endpoint-label "$ENDPOINT_LABEL" \
  --batch-size 1 \
  --max-input-tokens 3072 \
  --device-map "$DEVICE_MAP" \
  --gpu-memory-gib 44 \
  "${adapter_args[@]}"

echo "large-model extraction complete: $RUN_DIR"
