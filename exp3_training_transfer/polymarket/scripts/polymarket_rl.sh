#!/bin/bash
#SBATCH --job-name=exp3b_grpo
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/grpo_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/grpo_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

# Registered Experiment 3B training. Training-time evaluation uses DEV only;
# the frozen 2026 test set is deliberately absent from this script.
set -euo pipefail

REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
DATA="${DATA:-$POLY/data/exp3b_registered}"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

# Launchpad unpickles the actor in a fresh Python process whose working directory
# is the OAT checkout. Keep the trainer's local scoring helper importable there.
# Without this, the actor exits with ModuleNotFoundError and the learner waits
# until Slurm kills the job at its wall-time limit.
export PYTHONPATH="$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'from forecast_scoring import score_binary_forecast; assert score_binary_forecast("{\"yes_prob\": 1}", True, 0.5)[0] == 1.0'

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-42}"
TAG="${TAG:-market}"
OUT="$POLY/reports/train_${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$POLY/logs"
cp "$DATA/manifest.json" "$OUT/dataset_manifest.json"

GPUS="${GPUS:-1}"
BATCH="${BATCH:-16}"
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"
MAX_TRAIN="${MAX_TRAIN:-4800}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-300}"
GEN_LEN="${GEN_LEN:-128}"
VLLM_RATIO="${VLLM_RATIO:-0.40}"
TEMP="${TEMP:-1.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1920}"

cd "$REPO/.runtime/oat"
"$PY" "$REPO/exp3_training_transfer/biased_news/run_biased_news_rl.py" \
    --critic_type drgrpo \
    --gpus "$GPUS" \
    --num_gpus_per_actor 1 \
    --lora_rank "${LORA_RANK:-32}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --collocate \
    --vllm_gpu_ratio "$VLLM_RATIO" \
    --max_model_len "$MAX_MODEL_LEN" \
    --enable_prefix_caching \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --zero-stage 2 \
    --ref_offload \
    --beta 0 \
    --learning_rate "${LR:-0.000001}" \
    --lr_scheduler constant \
    --lr_warmup_ratio 0.03 \
    --num_ppo_epochs 1 \
    --seed "$SEED" \
    --oracle_type reward \
    --oracle math \
    --pretrain "$MODEL" \
    --prompt_template biased_news \
    --prompt_data "$DATA/hf_train_schedule" \
    --eval_data "$DATA/hf_dev" \
    --train_split train \
    --input_key input \
    --output_key reference \
    --eval_input_key input \
    --max-train "$MAX_TRAIN" \
    --num_prompt_epoch 1 \
    --prompt_max_length 1792 \
    --num_samples "$ROLLOUT_PER_PROMPT" \
    --temperature "$TEMP" \
    --top_p 1 \
    --generate_max_length "$GEN_LEN" \
    --train_batch_size "$BATCH" \
    --train_batch_size_per_device 1 \
    --rollout_batch_size "$BATCH" \
    --rollout_batch_size_per_device "$BATCH" \
    --pi_buffer_maxlen_per_device "$((BATCH * ROLLOUT_PER_PROMPT))" \
    --eval_batch_size 128 \
    --eval_steps "$EVAL_STEPS" \
    --eval_temperature 0 \
    --eval_n 1 \
    --eval_generate_max_length "$GEN_LEN" \
    --save_steps "$SAVE_STEPS" \
    --save_path "$OUT" \
    --no-use-wb \
    2>&1 | tee "$OUT/train.log"

echo "results dir: $OUT"
