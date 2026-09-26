#!/bin/bash
#SBATCH --job-name=exp3d_grpo
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/domain_family/logs/grpo_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/domain_family/logs/grpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

# Dr. GRPO on the COIN-* DOMAIN FAMILY (Experiment 3, domain-transfer arm).
#
# The graph is held fixed (the Exp-2 `direct` world) and the DOMAIN varies. Arms form a nested
# ladder over the training pool; every arm is evaluated on the SAME held-out domains
# (CoinBasketball, CoinClinic) that appear in no arm's training data:
#
#   ARM=d1             CoinCity                                    1 training domain
#   ARM=d2             CoinCity + CoinFishing                      2 training domains
#   ARM=d3             CoinCity + CoinFishing + CoinFarm           3 training domains
#   ARM=structureless  same three domains, driver decoupled        format + level only (control)
#
# All arms are row-matched at MAX_TRAIN, so the ladder measures domain DIVERSITY, not data quantity.
# Per-row reward tolerances and clip ranges travel in the dataset (see make_dataset_domains.py), so
# the --reward_scale_pts / --slope_scale_g values below are only the base the builder scaled from.
#
#   sbatch --export=ALL,ARM=d3,SEED=42 domain_rl.sh
set -euo pipefail

REPO=/n/fs/similarity/social_sim
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"     # also pins HF offline: see the 2026-08-11 note there
PY="$REPO/.runtime/oat_conda/bin/python"

FAM="$REPO/exp3_training_transfer/domain_family"
ARM="${ARM:-d3}"
DATA="${DATA:-$FAM/data/domain_$ARM}"
TAG="${TAG:-$ARM}"
OUT="$FAM/reports/exp3d_${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAM/logs"

[ -d "$DATA/train" ] || { echo "missing dataset $DATA/train -- run make_dataset_domains.py"; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
# COLLOCATE=1 (default) puts the vLLM actor and the DeepSpeed learner on ONE GPU, sharing it via
# CUDA IPC. That is the path that deadlocks at init on the post-2026-07-14 driver: every job since
# the node rebuild hangs there with an idle GPU and zero training steps, while torch, NCCL and vLLM
# each work standalone. COLLOCATE=0 gives them a GPU each, which avoids the shared-GPU handshake
# entirely at the cost of a second GPU per job.
COLLOCATE="${COLLOCATE:-1}"
if [ "$COLLOCATE" = "1" ]; then
  GPUS="${GPUS:-1}"; COLLOCATE_FLAG="--collocate"
else
  GPUS="${GPUS:-2}"; COLLOCATE_FLAG=""
fi
NUM_GPUS_PER_ACTOR="${NUM_GPUS_PER_ACTOR:-1}"
ZERO_STAGE="${ZERO_STAGE:-2}"
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"
BATCH="${BATCH:-16}"
MAX_TRAIN="${MAX_TRAIN:-4800}"                    # 4800/16 = 300 steps; identical across arms
# Wall limit is 12h, not 8h: the July reference run needed 6.54h for 302 steps, leaving only 1.5h
# of headroom, and shared-filesystem reads are now far slower than they were then. An 8h request
# risks TIMEOUT at step ~280 -- a wasted run that produces no endpoint.
EVAL_STEPS="${EVAL_STEPS:-25}"
GEN_LEN="${GEN_LEN:-1024}"
VLLM_RATIO="${VLLM_RATIO:-0.40}"
TEMP="${TEMP:-1.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2560}"
SEED="${SEED:-42}"
NO_FUSED=""; [ "$ZERO_STAGE" = "3" ] && NO_FUSED="--no-use_fused_lm_head"
LORA_SYNC=""; [ "$ZERO_STAGE" = "3" ] && LORA_SYNC="--lora_sync_only"
GRAD_CKPT="${GRAD_CKPT:-1}"; GC_FLAG=""
[ "$GRAD_CKPT" = "1" ] && GC_FLAG="--gradient-checkpointing"

echo "=== exp3d arm=$ARM seed=$SEED data=$DATA collocate=$COLLOCATE gpus=$GPUS ==="

cd "$REPO/.runtime/oat"
"$PY" "$REPO/exp3_training_transfer/biased_news/run_biased_news_rl.py" \
    --critic_type drgrpo \
    --gpus $GPUS \
    --num_gpus_per_actor $NUM_GPUS_PER_ACTOR \
    ${VLLM_SLEEP:+--vllm_sleep} \
    --lora_rank ${LORA_RANK:-32} \
    --lora_alpha ${LORA_ALPHA:-64} \
    $LORA_SYNC \
    ${COLLOCATE_FLAG} \
    --vllm_gpu_ratio $VLLM_RATIO \
    --max_model_len $MAX_MODEL_LEN \
    --enable_prefix_caching \
    $GC_FLAG \
    --flash-attn \
    --bf16 \
    --zero-stage $ZERO_STAGE \
    $NO_FUSED \
    --ref_offload \
    --beta ${BETA:-0} \
    --learning_rate ${LR:-0.000001} \
    --lr_scheduler ${LR_SCHED:-constant} \
    --lr_warmup_ratio ${WARMUP:-0.03} \
    --num_ppo_epochs 1 \
    --seed $SEED \
    --oracle_type reward \
    --oracle math \
    --reward_scale_pts ${REWARD_SCALE:-15} \
    --slope_weight ${SLOPE_W:-0.5} \
    --slope_scale_g ${SLOPE_SCALE:-0.5} \
    --pretrain "$MODEL" \
    --prompt_template ${PROMPT_TEMPLATE:-biased_news} \
    --prompt_data "$DATA/train" \
    --eval_data "$DATA/heldout" \
    --train_split train \
    --input_key input \
    --output_key reference \
    --eval_input_key input \
    --max-train $MAX_TRAIN \
    --num_prompt_epoch 1 \
    --prompt_max_length 1280 \
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
echo "==== held-out DOMAIN transfer: arm=$ARM seed=$SEED ===="
EVDIR=$(find "$OUT" -type d -name eval_results 2>/dev/null | head -1)
echo "eval dumps: $EVDIR"
# Score immediately so a finished run is self-describing even if nobody looks at it for a week.
"$PY" "$FAM/transfer_report_domains.py" --step 300 || echo "(scoring skipped)"
echo "results dir: $OUT"
