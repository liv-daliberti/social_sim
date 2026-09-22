#!/bin/bash
#SBATCH --job-name=c3_mech
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/logs/train_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/logs/train_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=16:00:00

# Shared training entry point for the disclosed/undisclosed C3 mechanism factorial.
set -euo pipefail

REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
CONTROL="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control"
FROZEN="$CONTROL/frozen_runtime"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FROZEN:$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

DISCLOSURE="${DISCLOSURE:-disclosed}"
ARM="${ARM:-causal_family}"
MODEL_KEY="${MODEL_KEY:-qwen3_4b}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-42}"
DATA="${DATA:-$FAMILY/data/${DISCLOSURE}_${ARM}}"
TAG="${TAG:-${DISCLOSURE}_${ARM}_${MODEL_KEY}_s${SEED}}"
OUT="$CONTROL/reports/${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAMILY/logs"

[ -d "$DATA/train" ] || { echo "missing $DATA/train"; exit 1; }
[ -d "$DATA/heldout" ] || { echo "missing $DATA/heldout"; exit 1; }
case "$DISCLOSURE" in disclosed|undisclosed) ;; *) echo "bad disclosure: $DISCLOSURE"; exit 2;; esac
case "$ARM" in causal_family|population_prior|structureless|shuffled_target) ;; *) echo "bad arm: $ARM"; exit 2;; esac

COLLOCATE="${COLLOCATE:-1}"
if [ "$COLLOCATE" = 1 ]; then
  GPUS="${GPUS:-1}"
  COLLOCATE_FLAG="--collocate"
else
  GPUS="${GPUS:-2}"
  COLLOCATE_FLAG=""
fi

BATCH="${BATCH:-16}"
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"
if [ "$COLLOCATE" = 1 ]; then
  ROLLOUT_WORKERS="$GPUS"
else
  [ $(( GPUS % 2 )) -eq 0 ] || { echo "non-collocated GPUS must be even"; exit 2; }
  ROLLOUT_WORKERS="$(( GPUS / 2 ))"
fi
[ "$ROLLOUT_WORKERS" -gt 0 ] || { echo "no rollout workers"; exit 2; }
[ $(( BATCH % ROLLOUT_WORKERS )) -eq 0 ] || {
  echo "BATCH=$BATCH is not divisible by rollout workers=$ROLLOUT_WORKERS"; exit 2;
}
PROMPTS_PER_ACTOR="$(( BATCH / ROLLOUT_WORKERS ))"
MAX_TRAIN="${MAX_TRAIN:-4800}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-2304}"
GEN_LEN="${GEN_LEN:-192}"
STRUCTURED_OUTPUT="${STRUCTURED_OUTPUT:-forecast_array}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-999999}"
ZERO_STAGE="${ZERO_STAGE:-2}"
NO_FUSED=""; [ "$ZERO_STAGE" = 3 ] && NO_FUSED="--no-use_fused_lm_head"
LORA_SYNC=""; [ "$ZERO_STAGE" = 3 ] && LORA_SYNC="--lora_sync_only"

echo "C3 mechanism: disclosure=$DISCLOSURE arm=$ARM model=$MODEL seed=$SEED"
echo "data=$DATA output=$OUT"
echo "batch=$BATCH rollout_workers=$ROLLOUT_WORKERS prompts_per_actor=$PROMPTS_PER_ACTOR samples_per_prompt=$ROLLOUT_PER_PROMPT buffer_per_learner=$(( PROMPTS_PER_ACTOR * ROLLOUT_PER_PROMPT ))"

cd "$REPO/.runtime/oat"
"$PY" "$FROZEN/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus "$GPUS" \
  --num_gpus_per_actor 1 \
  $COLLOCATE_FLAG \
  ${VLLM_SLEEP:+--vllm_sleep} \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  $LORA_SYNC \
  --vllm_gpu_ratio "${VLLM_RATIO:-0.38}" \
  --max_model_len "$MAX_MODEL_LEN" \
  --enable_prefix_caching \
  --gradient-checkpointing \
  --flash-attn \
  --bf16 \
  --zero-stage "$ZERO_STAGE" \
  $NO_FUSED \
  --ref_offload \
  --beta 0 \
  --learning_rate "${LR:-0.000001}" \
  --lr_scheduler constant \
  --lr_warmup_ratio 0.03 \
  --num_ppo_epochs 1 \
  --seed "$SEED" \
  --oracle_type reward \
  --oracle math \
  --reward_scale_pts 10 \
  --slope_weight "${RESPONSE_W:-0.60}" \
  --slope_scale_g 4 \
  --pretrain "$MODEL" \
  --prompt_template "${PROMPT_TEMPLATE:-biased_news}" \
  --structured_output "$STRUCTURED_OUTPUT" \
  --prompt_data "$DATA/train" \
  --eval_data "$DATA/heldout" \
  --train_split train \
  --input_key input \
  --output_key reference \
  --eval_input_key input \
  --max-train "$MAX_TRAIN" \
  --num_prompt_epoch 1 \
  --prompt_max_length "$PROMPT_MAX_LENGTH" \
  --num_samples "$ROLLOUT_PER_PROMPT" \
  --temperature "${TEMP:-1.3}" \
  --top_p 1 \
  --generate_max_length "$GEN_LEN" \
  --train_batch_size "$BATCH" \
  --train_batch_size_per_device 1 \
  --rollout_batch_size "$BATCH" \
  --rollout_batch_size_per_device "$PROMPTS_PER_ACTOR" \
  --pi_buffer_maxlen_per_device "$(( PROMPTS_PER_ACTOR * ROLLOUT_PER_PROMPT ))" \
  --eval_batch_size "${EVAL_BATCH:-120}" \
  --eval_steps "$EVAL_STEPS" \
  --eval_temperature 0 \
  --eval_n 1 \
  --eval_generate_max_length "$GEN_LEN" \
  --save_steps "$SAVE_STEPS" \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee "$OUT/train.log"

if [ "${STOCHASTIC_N:-5}" -gt 0 ]; then
  ADAPTER=$(find "$OUT" -type d -path '*/saved_models/step_*' | sort | tail -1)
  [ -n "$ADAPTER" ] || { echo "final adapter missing; stochastic endpoint aborted"; exit 3; }
  "$PY" "$FROZEN/evaluate_endpoint.py" \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --data "$DATA/heldout" \
    --template "${PROMPT_TEMPLATE:-biased_news}" \
    --structured-output "$STRUCTURED_OUTPUT" \
    --temperature "${STOCHASTIC_TEMP:-0.7}" \
    --n "${STOCHASTIC_N:-5}" \
    --seed "$(( SEED + 20260814 ))" \
    --max-tokens "$GEN_LEN" \
    --max-model-len "$MAX_MODEL_LEN" \
    --output "$OUT/stochastic_n${STOCHASTIC_N:-5}.json"
fi

GREEDY=$(find "$OUT" -type f -path '*/eval_results/*.json' | sort -V | tail -1)
[ -n "$GREEDY" ] || { echo "final greedy evaluation missing"; exit 4; }
"$PY" "$FROZEN/report.py" score "$GREEDY" --model "$MODEL_KEY" \
  --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
if [ "${STOCHASTIC_N:-5}" -gt 0 ]; then
  "$PY" "$FROZEN/report.py" score "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" \
    --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
fi

echo "complete: $OUT"

# Fresh evaluation is appended after the unchanged registered terminal evaluations.
ADAPTER=$(find "$OUT" -type f -path '*/saved_models/step_00301/adapter_model.safetensors' | head -1)
[ -n "$ADAPTER" ] || { echo "final weights absent"; exit 5; }
ADAPTER=$(dirname "$ADAPTER")
"$PY" "$FROZEN/evaluate_endpoint.py" \
  --model "$MODEL" --adapter "$ADAPTER" \
  --data "$CONTROL/../evidence_use/data/heldout_${DISCLOSURE}" \
  --template "$PROMPT_TEMPLATE" --structured-output forecast_array \
  --temperature 0 --n 1 --seed "$(( SEED + 20260814 ))" \
  --max-tokens "$GEN_LEN" --max-model-len "$MAX_MODEL_LEN" \
  --output "$OUT/fresh_greedy.json" \
  --secondary-output "$OUT/fresh_stochastic_n5.json" --secondary-temperature 0.7 --secondary-n 5
for NAME in fresh_greedy fresh_stochastic_n5; do
  "$PY" "$FROZEN/report.py" score "$OUT/$NAME.json" --model "$MODEL_KEY" \
    --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
done
echo "fresh_complete: $OUT"
