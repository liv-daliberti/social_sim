#!/bin/bash
# Large-model mechanism-family training entry point.  Resources are supplied
# by the hash-locked launcher because they differ by model.

set -euo pipefail

REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
EXT="$FAMILY/scale_extension_v2_1"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0

: "${DISCLOSURE:?DISCLOSURE is required}"
: "${ARM:?ARM is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
: "${SEED:?SEED is required}"
: "${RUN_KIND:?RUN_KIND is required}"

case "$DISCLOSURE" in disclosed|undisclosed) ;; *) exit 2;; esac
case "$ARM" in causal_family|population_prior) ;; *) exit 2;; esac
case "$SEED" in 42|43|44) ;; *) exit 2;; esac

MODEL_COMPAT=()
case "$MODEL_KEY" in
  qwen3_14b)
    [ "$MODEL" = Qwen/Qwen3-14B ] || exit 2
    [ "$PROMPT_TEMPLATE" = auto_no_think ] || exit 2
    GPUS=2; ACTOR_TP=1; ZERO_STAGE=2; PER_LEARNER_BATCH=16; BUFFER=128
    ;;
  qwen3_32b)
    [ "$MODEL" = Qwen/Qwen3-32B ] || exit 2
    [ "$PROMPT_TEMPLATE" = auto_no_think ] || exit 2
    GPUS=4; ACTOR_TP=2; ZERO_STAGE=3; PER_LEARNER_BATCH=8; BUFFER=64
    # Execution-only settings validated by the Exp4 Qwen3-32B replacement canary.
    MODEL_COMPAT=(--disable-custom-all-reduce --gradient-checkpointing-use-reentrant)
    ;;
  llama3_1_70b)
    [ "$MODEL" = meta-llama/Llama-3.1-70B-Instruct ] || exit 2
    [ "$PROMPT_TEMPLATE" = auto ] || exit 2
    GPUS=8; ACTOR_TP=4; ZERO_STAGE=3; PER_LEARNER_BATCH=4; BUFFER=32
    ;;
  *) echo "unregistered model key: $MODEL_KEY" >&2; exit 2;;
esac

if [ "$RUN_KIND" = canary ]; then
  [ "$SEED" = 42 ] || { echo "canaries are seed 42" >&2; exit 2; }
  [ "$ARM" = causal_family ] || { echo "canaries use causal_family" >&2; exit 2; }
  MAX_TRAIN=160
  EVAL_STEPS=10
  STOCHASTIC_N=1
elif [ "$RUN_KIND" = train ]; then
  MAX_TRAIN=4800
  EVAL_STEPS=25
  STOCHASTIC_N=5
else
  echo "RUN_KIND must be canary or train" >&2
  exit 2
fi

DATA="$FAMILY/data/${DISCLOSURE}_${ARM}"
[ -d "$DATA/train" ] || { echo "missing $DATA/train" >&2; exit 2; }
[ -d "$DATA/heldout" ] || { echo "missing $DATA/heldout" >&2; exit 2; }

TAG="scale_v2_1_${RUN_KIND}_${DISCLOSURE}_${ARM}_${MODEL_KEY}_s${SEED}"
OUT="$FAMILY/reports/${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAMILY/logs"
cp "$EXT/PROTOCOL.md" "$OUT/PROTOCOL.md"
cp "$FAMILY/protocol/c3_mechanism_manifest.json" "$OUT/c3_mechanism_manifest.json"

NO_FUSED=()
LORA_SYNC=()
if [ "$ZERO_STAGE" = 3 ]; then
  NO_FUSED=(--no-use_fused_lm_head)
  LORA_SYNC=(--lora_sync_only)
fi

echo "scale v2.1: kind=$RUN_KIND disclosure=$DISCLOSURE arm=$ARM model=$MODEL_KEY seed=$SEED"
echo "gpus=$GPUS actor_tp=$ACTOR_TP zero=$ZERO_STAGE per_learner_batch=$PER_LEARNER_BATCH"

cd "$REPO/.runtime/oat"
"$PY" "$FAMILY/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus "$GPUS" \
  --num_gpus_per_actor "$ACTOR_TP" \
  --lora_rank 32 \
  --lora_alpha 64 \
  "${LORA_SYNC[@]}" \
  --vllm_gpu_ratio 0.78 \
  "${MODEL_COMPAT[@]}" \
  --max_model_len 3072 \
  --enable_prefix_caching \
  --gradient-checkpointing \
  --flash-attn \
  --bf16 \
  --zero-stage "$ZERO_STAGE" \
  "${NO_FUSED[@]}" \
  --ref_offload \
  --beta 0 \
  --learning_rate 0.000001 \
  --lr_scheduler constant \
  --lr_warmup_ratio 0.03 \
  --num_ppo_epochs 1 \
  --seed "$SEED" \
  --oracle_type reward \
  --oracle math \
  --reward_scale_pts 10 \
  --slope_weight 0.60 \
  --slope_scale_g 4 \
  --pretrain "$MODEL" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --structured_output forecast_array \
  --prompt_data "$DATA/train" \
  --eval_data "$DATA/heldout" \
  --train_split train \
  --input_key input \
  --output_key reference \
  --eval_input_key input \
  --max-train "$MAX_TRAIN" \
  --num_prompt_epoch 1 \
  --prompt_max_length 2304 \
  --num_samples 8 \
  --temperature 1.3 \
  --top_p 1 \
  --generate_max_length 192 \
  --train_batch_size 16 \
  --train_batch_size_per_device 1 \
  --rollout_batch_size 16 \
  --rollout_batch_size_per_device "$PER_LEARNER_BATCH" \
  --pi_buffer_maxlen_per_device "$BUFFER" \
  --eval_batch_size 120 \
  --eval_steps "$EVAL_STEPS" \
  --eval_temperature 0 \
  --eval_n 1 \
  --eval_generate_max_length 192 \
  --save_steps 999999 \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee "$OUT/train.log"

ADAPTER=$(find "$OUT" -type d -path '*/saved_models/step_*' | sort | tail -1)
[ -n "$ADAPTER" ] || { echo "final adapter missing" >&2; exit 3; }

"$PY" "$EXT/evaluate_endpoint_tp.py" \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --data "$DATA/heldout" \
  --template "$PROMPT_TEMPLATE" \
  --temperature 0.7 \
  --n "$STOCHASTIC_N" \
  --seed "$(( SEED + 20260814 ))" \
  --max-tokens 192 \
  --max-model-len 3072 \
  --tensor-parallel-size "$ACTOR_TP" \
  --structured-output forecast_array \
  --output "$OUT/stochastic_n${STOCHASTIC_N}.json"

GREEDY=$(find "$OUT" -type f -path '*/eval_results/*.json' | sort -V | tail -1)
[ -n "$GREEDY" ] || { echo "final greedy evaluation missing" >&2; exit 4; }
"$PY" "$FAMILY/report.py" score "$GREEDY" \
  --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
"$PY" "$FAMILY/report.py" score "$OUT/stochastic_n${STOCHASTIC_N}.json" \
  --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"

echo "complete: $OUT"
