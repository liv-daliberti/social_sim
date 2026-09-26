#!/bin/bash
#SBATCH --job-name=cc_qwen_rel_extract
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_rel_extract_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_rel_extract_%j.err
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00

set -euo pipefail
: "${RUN_NAME:?RUN_NAME is required}"
: "${MODEL_ID:?MODEL_ID is required}"
: "${MODEL_COMMIT:?MODEL_COMMIT is required}"
SEAL_SOURCE=${SEAL_SOURCE:-0}
SOURCE_PROBE_NAME=${SOURCE_PROBE_NAME:-}

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
DATA="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe"
RUN_DIR="$DATA/$RUN_NAME"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$MECH/logs" "$RUN_DIR"
exec 9>"$RUN_DIR/.extract_select.lock"
flock -n 9 || { echo "$RUN_NAME extraction already owned"; exit 0; }

if [[ "$SEAL_SOURCE" == 1 && ! -f "$DATA/$SOURCE_PROBE_NAME/causal_source_receipt.json" ]]; then
  [[ -f "$DATA/$SOURCE_PROBE_NAME/probe_results.json" ]] || {
    echo "source probe is not complete: $SOURCE_PROBE_NAME" >&2
    exit 1
  }
  "$PY" "$MECH/seal_symbol_probe_for_causal_replication.py" \
    --run-dir "$DATA/$SOURCE_PROBE_NAME" \
    --model-id "$MODEL_ID" --model-commit "$MODEL_COMMIT"
fi
if [[ ! -f "$RUN_DIR/relational_protocol_manifest.json" ]]; then
  "$PY" "$MECH/freeze_symbol_relational_protocol.py" --run-dir "$RUN_DIR"
fi
for name in symbol_relational_common.py extract_symbol_label_states.py \
  analyze_symbol_relational_probe.py patch_symbol_label_activations.py \
  freeze_symbol_relational_protocol.py; do
  expected=$(jq -r --arg name "$name" '.code_sha256[$name]' \
    "$RUN_DIR/relational_protocol_manifest.json")
  observed=$(sha256sum "$MECH/$name" | cut -d' ' -f1)
  [[ "$observed" == "$expected" ]] || {
    echo "frozen code hash mismatch: $name" >&2
    exit 1
  }
done
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

if [[ ! -f "$RUN_DIR/label_extraction_manifest.json" ]]; then
  "$PY" "$MECH/extract_symbol_label_states.py" --run-dir "$RUN_DIR" \
    --model "$MODEL_ID" --device-map balanced --gpu-memory-gib 44
fi
if [[ ! -f "$RUN_DIR/development_selection.json" ]]; then
  "$PY" "$MECH/analyze_symbol_relational_probe.py" \
    --run-dir "$RUN_DIR" --mode selection-only
fi
echo "$RUN_NAME label-state extraction and development selection complete"
