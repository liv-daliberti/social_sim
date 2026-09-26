#!/bin/bash
#SBATCH --job-name=c3_probe_s45_46
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/probe_extract_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/probe_extract_%j.err

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
PROBE="$FAMILY/mechanistic_probe"
EXT="$REPO/exp3_training_transfer/five_seed_extension"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${REPORT_PATTERN:?REPORT_PATTERN is required}"
: "${MODEL:?MODEL is required}"
: "${MODEL_COMMIT:?MODEL_COMMIT is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${ENDPOINT_LABEL:?ENDPOINT_LABEL is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${DEVICE_MAP:?DEVICE_MAP is required}"

source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_DIR" "$EXT/logs"
mapfile -t ROOTS < <(find "$FAMILY/reports" -maxdepth 1 -type d -name "$REPORT_PATTERN" -print)
[ "${#ROOTS[@]}" -eq 1 ] || { echo "expected one report for $REPORT_PATTERN, found ${#ROOTS[@]}" >&2; exit 3; }
ADAPTER=$(find "${ROOTS[0]}" -type d -path '*/saved_models/step_*' | sort -V | tail -1)
[ -n "$ADAPTER" ] && [ -f "$ADAPTER/adapter_model.safetensors" ] || exit 4

exec 9>"$RUN_DIR/.five_seed_extract.lock"
flock -n 9 || { echo "another extraction owns $RUN_DIR"; exit 0; }
"$PY" "$PROBE/extract_hidden_states.py" \
  --task-dir "$PROBE/data" --run-dir "$RUN_DIR" --model "$MODEL" \
  --model-commit "$MODEL_COMMIT" --model-label "$MODEL_LABEL" \
  --endpoint-label "$ENDPOINT_LABEL" --adapter "$ADAPTER" \
  --batch-size "${BATCH_SIZE:-1}" --max-input-tokens 3072 \
  --device-map "$DEVICE_MAP" --gpu-memory-gib 44
