#!/bin/bash
#SBATCH --job-name=exp4_qwen32_train
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_train_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_train_%j.err
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=220G
#SBATCH --cpus-per-task=12
#SBATCH --partition=all
#SBATCH --account=mltheory

set -euo pipefail

REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
DATA="$POLY/data/exp3b_registered"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

MODEL_KEY="${MODEL_KEY:-qwen3_32b}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-auto_no_think}"
SEED="${SEED:?SEED is required}"
RUN_KIND="${RUN_KIND:-train}"
MAX_TRAIN="${MAX_TRAIN:-4800}"
EVAL_STEPS="${EVAL_STEPS:-25}"

if [ "$MODEL_KEY" != qwen3_32b ] || [ "$MODEL" != Qwen/Qwen3-32B ]; then
  echo "Qwen3-32B extension is locked to Qwen/Qwen3-32B" >&2
  exit 2
fi
if [ "$PROMPT_TEMPLATE" != auto_no_think ]; then
  echo "Qwen3-32B extension requires the non-thinking chat template" >&2
  exit 2
fi
if [ "$SEED" != 42 ] && [ "$SEED" != 43 ] && [ "$SEED" != 44 ]; then
  echo "Qwen3-32B extension requires seed 42, 43, or 44" >&2
  exit 2
fi
if [ "$RUN_KIND" = train ]; then
  if [ "$MAX_TRAIN" -ne 4800 ] || [ "$EVAL_STEPS" -ne 25 ]; then
    echo "full Qwen3-32B training requires 4,800 presentations and eval_steps=25" >&2
    exit 2
  fi
elif [ "$RUN_KIND" = canary ]; then
  if [ "$MAX_TRAIN" -ne 160 ] || [ "$EVAL_STEPS" -ne 10 ] || [ "$SEED" -ne 42 ]; then
    echo "Qwen3-32B canary requires seed 42, 160 presentations, and eval_steps=10" >&2
    exit 2
  fi
else
  echo "RUN_KIND must be canary or train" >&2
  exit 2
fi

OUT="$POLY/reports/qwen32_${RUN_KIND}_${MODEL_KEY}_market_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$POLY/logs"
cp "$DATA/manifest.json" "$OUT/dataset_manifest.json"
cp "$POLY/data/exp4_scale_registered/manifest.json" "$OUT/scale_holdout_manifest.json"
cp "$POLY/QWEN32_SCALE_PROTOCOL.md" "$OUT/QWEN32_SCALE_PROTOCOL.md"

cd "$REPO/.runtime/oat"
"$PY" -c 'from output_contract import FORECAST_ARRAY_GBNF, parse_forecast_array; assert FORECAST_ARRAY_GBNF and callable(parse_forecast_array)'
"$PY" "$FAMILY/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus 4 \
  --num_gpus_per_actor 2 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_sync_only \
  --vllm_gpu_ratio 0.78 \
  --disable-custom-all-reduce \
  --max_model_len 1920 \
  --enable_prefix_caching \
  --gradient-checkpointing \
  --gradient-checkpointing-use-reentrant \
  --flash-attn \
  --bf16 \
  --zero-stage 3 \
  --no-use_fused_lm_head \
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
  --max-train "$MAX_TRAIN" \
  --num_prompt_epoch 1 \
  --prompt_max_length 1792 \
  --num_samples 8 \
  --temperature 1.3 \
  --top_p 1 \
  --generate_max_length 128 \
  --train_batch_size 16 \
  --train_batch_size_per_device 1 \
  --rollout_batch_size 16 \
  --rollout_batch_size_per_device 8 \
  --pi_buffer_maxlen_per_device 64 \
  --eval_batch_size 128 \
  --eval_steps "$EVAL_STEPS" \
  --eval_temperature 0.7 \
  --eval_top_p 0.8 \
  --eval_top_k 20 \
  --eval_n 5 \
  --eval_generate_max_length 128 \
  --save_steps 999999 \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee "$OUT/train.log"

echo "Qwen3-32B scale training: $OUT"
