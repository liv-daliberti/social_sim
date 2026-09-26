#!/bin/bash
# Dr. GRPO pilot: train Qwen on the Experiment-2 biased-news forecasting reward,
# single GPU, short run (~30 steps). Held-out recovery is read from OAT's eval
# dumps (eval_results/<step>.json) via recovery_from_eval.py -- step 0 is the
# before-training baseline, the final step is after training.
#
# Run ON a GPU node, e.g.:
#   srun -p all --gres=gpu:1 --mem=80G -t 2:00:00 --pty bash
#   bash exp3_training_transfer/biased_news/biased_news_rl.sh
set -euo pipefail

REPO=/n/fs/similarity/social_sim
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

DATA="$REPO/exp3_training_transfer/biased_news/data_v2"
OUT="$REPO/exp3_training_transfer/biased_news/reports/pilot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# ---- knobs (tuned for a ~1h single-GPU pilot) -------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"   # override with MODEL=... if needed
GPUS=1
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"   # group size for Dr. GRPO
BATCH="${BATCH:-16}"                            # prompts per step
MAX_TRAIN="${MAX_TRAIN:-480}"                   # 480/16 = 30 steps (1 prompt epoch)
EVAL_STEPS="${EVAL_STEPS:-30}"
GEN_LEN="${GEN_LEN:-768}"
VLLM_RATIO="${VLLM_RATIO:-0.45}"
TEMP="${TEMP:-1.0}"                              # rollout sampling temperature
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

cd "$REPO/.runtime/oat"     # so `python -m oat.experiment...` and relative paths resolve
"$PY" "$REPO/exp3_training_transfer/biased_news/run_biased_news_rl.py" \
    --critic_type drgrpo \
    --gpus $GPUS \
    --lora_rank ${LORA_RANK:-32} \
    --lora_alpha ${LORA_ALPHA:-64} \
    --collocate \
    --vllm_gpu_ratio $VLLM_RATIO \
    --max_model_len $MAX_MODEL_LEN \
    --enable_prefix_caching \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --zero-stage 2 \
    --ref_offload \
    --beta 0 \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --oracle_type reward \
    --oracle math \
    --pretrain "$MODEL" \
    --prompt_template biased_news \
    --prompt_data "$DATA/train" \
    --eval_data "$DATA/heldout" \
    --train_split train \
    --input_key input \
    --output_key reference \
    --eval_input_key input \
    --max-train $MAX_TRAIN \
    --num_prompt_epoch 1 \
    --prompt_max_length 1024 \
    --num_samples $ROLLOUT_PER_PROMPT \
    --temperature $TEMP \
    --top_p 1 \
    --generate_max_length $GEN_LEN \
    --train_batch_size $BATCH \
    --train_batch_size_per_device 1 \
    --rollout_batch_size $BATCH \
    --rollout_batch_size_per_device $(( BATCH / GPUS )) \
    --pi_buffer_maxlen_per_device $(( BATCH / GPUS * ROLLOUT_PER_PROMPT )) \
    --eval_batch_size 200 \
    --eval_steps $EVAL_STEPS \
    --eval_temperature 0 \
    --eval_n 1 \
    --eval_generate_max_length $GEN_LEN \
    --save_steps -1 \
    --save_path "$OUT" \
    --no-use-wb \
    2>&1 | tee "$OUT/train.log"

echo
echo "==== held-out latent recovery (before vs after) ===="
# OAT writes eval dumps under a debug_* subdir of save_path.
EVDIR=$(find "$OUT" -type d -name eval_results 2>/dev/null | head -1)
"$PY" "$REPO/exp3_training_transfer/biased_news/recovery_from_eval.py" "$EVDIR"
