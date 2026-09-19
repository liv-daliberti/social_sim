#!/bin/bash
# Scientific copy of polymarket_rl_qwen32_extension.sh for seeds 45/46.

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
EXT="$REPO/exp3_training_transfer/five_seed_extension"
DATA="$POLY/data/exp3b_registered"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

: "${SEED:?SEED is required}"
case "$SEED" in 45|46) ;; *) exit 2;; esac
MODEL_KEY="${MODEL_KEY:-qwen3_32b}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-auto_no_think}"
[ "$MODEL_KEY:$MODEL:$PROMPT_TEMPLATE" = "qwen3_32b:Qwen/Qwen3-32B:auto_no_think" ] || exit 2

OUT="$POLY/reports/qwen32_train_${MODEL_KEY}_market_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$POLY/logs"
cp "$DATA/manifest.json" "$OUT/dataset_manifest.json"
cp "$POLY/data/exp4_scale_registered/manifest.json" "$OUT/scale_holdout_manifest.json"
cp "$POLY/QWEN32_SCALE_PROTOCOL.md" "$OUT/PARENT_PROTOCOL.md"
cp "$EXT/PROTOCOL.md" "$OUT/FIVE_SEED_PROTOCOL.md"

cd "$REPO/.runtime/oat"
"$PY" -c 'from output_contract import FORECAST_ARRAY_GBNF, parse_forecast_array; assert FORECAST_ARRAY_GBNF and callable(parse_forecast_array)'
"$PY" "$FAMILY/run_mechanism_rl.py" \
  --critic_type drgrpo --gpus 4 --num_gpus_per_actor 2 --lora_rank 32 \
  --lora_alpha 64 --lora_sync_only --vllm_gpu_ratio 0.78 --max_model_len 1920 \
  --disable-custom-all-reduce \
  --enable_prefix_caching --gradient-checkpointing \
  --gradient-checkpointing-use-reentrant --flash-attn --bf16 \
  --zero-stage 3 --no-use_fused_lm_head --ref_offload --beta 0 \
  --learning_rate 0.000001 --lr_scheduler constant --lr_warmup_ratio 0.03 \
  --num_ppo_epochs 1 --seed "$SEED" --oracle_type reward --oracle math \
  --pretrain "$MODEL" --prompt_template "$PROMPT_TEMPLATE" \
  --prompt_data "$DATA/hf_train_schedule" --eval_data "$DATA/hf_dev" \
  --train_split train --input_key input --output_key reference --eval_input_key input \
  --max-train 4800 --num_prompt_epoch 1 --prompt_max_length 1792 \
  --num_samples 8 --temperature 1.3 --top_p 1 --generate_max_length 128 \
  --train_batch_size 16 --train_batch_size_per_device 1 --rollout_batch_size 16 \
  --rollout_batch_size_per_device 8 --pi_buffer_maxlen_per_device 64 \
  --eval_batch_size 128 --eval_steps 25 --eval_temperature 0.7 \
  --eval_top_p 0.8 --eval_top_k 20 --eval_n 5 --eval_generate_max_length 128 \
  --save_steps 999999 --max_save_num 1 --save_path "$OUT" --no-use-wb \
  2>&1 | tee "$OUT/train.log"
