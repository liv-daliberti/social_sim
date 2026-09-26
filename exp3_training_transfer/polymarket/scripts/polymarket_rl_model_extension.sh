#!/bin/bash
#SBATCH --job-name=exp3b_ext
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/ext_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/ext_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

# Model-roster extension of the frozen Experiment 3B protocol. The original registered Qwen3-4B
# script is intentionally untouched because its queued jobs resolve that file at run time.
set -euo pipefail

REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
DATA="$POLY/data/exp3b_registered"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
# Launchpad unpickles the actor in a fresh Python process whose working
# directory is the OAT checkout. The extension uses the shared mechanism
# trainer, so both its local output contract and the biased-news scoring helper
# must remain importable in that child process.
export PYTHONPATH="$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
MODEL="${MODEL:?MODEL is required}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
SEED="${SEED:?SEED is required}"
OUT="$POLY/reports/train_${MODEL_KEY}_market_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$POLY/logs"
cp "$DATA/manifest.json" "$OUT/dataset_manifest.json"

COLLOCATE="${COLLOCATE:-0}"
if [ "$COLLOCATE" = 1 ]; then
  GPUS="${GPUS:-1}"; COLLOCATE_FLAG="--collocate"
else
  GPUS="${GPUS:-2}"; COLLOCATE_FLAG=""
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
echo "batch=$BATCH rollout_workers=$ROLLOUT_WORKERS prompts_per_actor=$PROMPTS_PER_ACTOR samples_per_prompt=$ROLLOUT_PER_PROMPT buffer_per_learner=$(( PROMPTS_PER_ACTOR * ROLLOUT_PER_PROMPT ))"

cd "$REPO/.runtime/oat"
"$PY" -c 'from output_contract import FORECAST_ARRAY_GBNF, parse_forecast_array; assert FORECAST_ARRAY_GBNF and callable(parse_forecast_array)'
"$PY" "$REPO/exp3_training_transfer/mechanism_family/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus "$GPUS" \
  --num_gpus_per_actor 1 \
  $COLLOCATE_FLAG \
  --lora_rank 32 \
  --lora_alpha 64 \
  --vllm_gpu_ratio "${VLLM_RATIO:-0.38}" \
  --max_model_len 1920 \
  --enable_prefix_caching \
  --gradient-checkpointing \
  --flash-attn \
  --bf16 \
  --zero-stage 2 \
  --ref_offload \
  --beta 0 \
  --learning_rate 0.000001 \
  --lr_scheduler constant \
  --lr_warmup_ratio 0.03 \
  --num_ppo_epochs 1 \
  --seed "$SEED" \
  --oracle_type reward \
  --oracle math \
  --pretrain "$MODEL" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --prompt_data "$DATA/hf_train_schedule" \
  --eval_data "$DATA/hf_dev" \
  --train_split train \
  --input_key input \
  --output_key reference \
  --eval_input_key input \
  --max-train 4800 \
  --num_prompt_epoch 1 \
  --prompt_max_length 1792 \
  --num_samples "$ROLLOUT_PER_PROMPT" \
  --temperature 1.3 \
  --top_p 1 \
  --generate_max_length 128 \
  --train_batch_size "$BATCH" \
  --train_batch_size_per_device 1 \
  --rollout_batch_size "$BATCH" \
  --rollout_batch_size_per_device "$PROMPTS_PER_ACTOR" \
  --pi_buffer_maxlen_per_device "$(( PROMPTS_PER_ACTOR * ROLLOUT_PER_PROMPT ))" \
  --eval_batch_size 128 \
  --eval_steps 25 \
  --eval_temperature 0 \
  --eval_n 1 \
  --eval_generate_max_length 128 \
  --save_steps 999999 \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee "$OUT/train.log"

echo "results dir: $OUT"
