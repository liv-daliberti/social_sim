#!/bin/bash
#SBATCH --job-name=c3_coin
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/train_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/train_%j.err
#SBATCH --partition=cs
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
MECHANISM="$REPO/exp3_training_transfer/mechanism_family"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$MECHANISM:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

ARM="${ARM:-causal}"
MODEL_KEY="${MODEL_KEY:-qwen3_4b}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-42}"
DATA="${DATA:-$FAMILY/data/$ARM}"
TAG="${TAG:-${ARM}_${MODEL_KEY}_s${SEED}}"
OUT="$FAMILY/reports/${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAMILY/logs"

case "$ARM" in causal|population_prior|structureless) ;; *) echo "bad arm: $ARM"; exit 2;; esac
[ -d "$DATA/train" ] || { echo "missing $DATA/train"; exit 2; }
[ -d "$DATA/heldout" ] || { echo "missing $DATA/heldout"; exit 2; }

GPUS="${GPUS:-2}"
BATCH="${BATCH:-16}"
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"
[ $(( GPUS % 2 )) -eq 0 ] || { echo "separated actor/learner requires an even GPU count"; exit 2; }
ROLLOUT_WORKERS="$(( GPUS / 2 ))"
[ $(( BATCH % ROLLOUT_WORKERS )) -eq 0 ] || { echo "batch not divisible by actor count"; exit 2; }
PROMPTS_PER_ACTOR="$(( BATCH / ROLLOUT_WORKERS ))"
MAX_TRAIN="${MAX_TRAIN:-4800}"
GEN_LEN="${GEN_LEN:-192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"

echo "Coin City structural transfer: arm=$ARM model=$MODEL seed=$SEED"
echo "data=$DATA output=$OUT max_train=$MAX_TRAIN"

cd "$REPO/.runtime/oat"
"$PY" "$MECHANISM/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus "$GPUS" \
  --num_gpus_per_actor 1 \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --vllm_gpu_ratio "${VLLM_RATIO:-0.38}" \
  --max_model_len "$MAX_MODEL_LEN" \
  --enable_prefix_caching \
  --gradient-checkpointing \
  --flash-attn \
  --bf16 \
  --zero-stage "${ZERO_STAGE:-2}" \
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
  --structured_output forecast_array \
  --prompt_data "$DATA/train" \
  --eval_data "$DATA/heldout" \
  --train_split train \
  --input_key input \
  --output_key reference \
  --eval_input_key input \
  --max-train "$MAX_TRAIN" \
  --num_prompt_epoch 1 \
  --prompt_max_length "${PROMPT_MAX_LENGTH:-2304}" \
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
  --eval_steps "${EVAL_STEPS:-50}" \
  --eval_temperature 0 \
  --eval_n 1 \
  --eval_generate_max_length "$GEN_LEN" \
  --save_steps 999999 \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee "$OUT/train.log"

ADAPTER=$(find "$OUT" -type d -path '*/saved_models/step_*' | sort -V | tail -1)
[ -n "$ADAPTER" ] || { echo "final adapter missing"; exit 3; }
"$PY" "$MECHANISM/evaluate_endpoint.py" \
  --model "$MODEL" --adapter "$ADAPTER" --data "$DATA/heldout" \
  --template "${PROMPT_TEMPLATE:-biased_news}" --structured-output forecast_array \
  --temperature 0 --n 1 --seed "$(( SEED + 20260818 ))" \
  --max-tokens "$GEN_LEN" --max-model-len "$MAX_MODEL_LEN" \
  --output "$OUT/greedy.json"
"$PY" "$MECHANISM/evaluate_endpoint.py" \
  --model "$MODEL" --adapter "$ADAPTER" --data "$DATA/heldout" \
  --template "${PROMPT_TEMPLATE:-biased_news}" --structured-output forecast_array \
  --temperature "${STOCHASTIC_TEMP:-0.7}" --n "${STOCHASTIC_N:-5}" \
  --seed "$(( SEED + 20260818 ))" --max-tokens "$GEN_LEN" \
  --max-model-len "$MAX_MODEL_LEN" --output "$OUT/stochastic_n${STOCHASTIC_N:-5}.json"

"$PY" "$FAMILY/report.py" score "$OUT/greedy.json" \
  --model "$MODEL_KEY" --arm "$ARM" --seed "$SEED"
"$PY" "$FAMILY/report.py" score "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" \
  --model "$MODEL_KEY" --arm "$ARM" --seed "$SEED"
echo "complete: $OUT"
