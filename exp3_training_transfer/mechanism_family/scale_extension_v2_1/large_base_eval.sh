#!/bin/bash

set -euo pipefail

REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
EXT="$FAMILY/scale_extension_v2_1"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FAMILY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0

: "${DISCLOSURE:?DISCLOSURE is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"

case "$DISCLOSURE" in disclosed|undisclosed) ;; *) exit 2;; esac
case "$MODEL_KEY" in
  qwen3_14b)
    [ "$MODEL" = Qwen/Qwen3-14B ] || exit 2
    TP=1
    ;;
  qwen3_32b)
    [ "$MODEL" = Qwen/Qwen3-32B ] || exit 2
    TP=2
    ;;
  llama3_1_70b)
    [ "$MODEL" = meta-llama/Llama-3.1-70B-Instruct ] || exit 2
    TP=4
    ;;
  *) exit 2;;
esac

DATA="$FAMILY/data/${DISCLOSURE}_causal_family/heldout"
OUT="$FAMILY/reports/scale_v2_1_base_${DISCLOSURE}_${MODEL_KEY}_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAMILY/logs"
cp "$EXT/PROTOCOL.md" "$OUT/PROTOCOL.md"

"$PY" "$EXT/evaluate_endpoint_tp.py" \
  --model "$MODEL" --data "$DATA" --template "$PROMPT_TEMPLATE" \
  --temperature 0 --n 1 --seed 20260814 \
  --max-tokens 192 --max-model-len 3072 \
  --tensor-parallel-size "$TP" --structured-output forecast_array \
  --output "$OUT/greedy.json"
"$PY" "$EXT/evaluate_endpoint_tp.py" \
  --model "$MODEL" --data "$DATA" --template "$PROMPT_TEMPLATE" \
  --temperature 0.7 --n 5 --seed 20260815 \
  --max-tokens 192 --max-model-len 3072 \
  --tensor-parallel-size "$TP" --structured-output forecast_array \
  --output "$OUT/stochastic_n5.json"

"$PY" "$FAMILY/report.py" score "$OUT/greedy.json" \
  --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm base --seed 0
"$PY" "$FAMILY/report.py" score "$OUT/stochastic_n5.json" \
  --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm base --seed 0

echo "complete: $OUT"
