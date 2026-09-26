#!/bin/bash
#SBATCH --job-name=dag_grpo
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family/logs/grpo_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family/logs/grpo_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

# Dr. GRPO on the LG DAG FAMILY (Experiment 3 transfer arm): train over the 8 TRAINING
# structures; OAT evals the 4 HELD-OUT structures every EVAL_STEPS. Two data formats:
#   * single-shock rows (legacy reward: level accuracy only) -> recovery_from_eval_dag.py
#   * multi-shock rows (2026-07-08 audit fix: level+slope composite reward via `targets`
#     in the reference; a constant 'ignore the news' policy can no longer collect the
#     reward) -> transfer_report.py reads the dumps into the eps/rho/pi transfer table.
# run_biased_news_rl.py branches on the reference schema, so both formats train unchanged.
#
#   sbatch exp3_training_transfer/dag_family/dag_rl.sh
#   sbatch --export=ALL,DATA=$FAM/data/rl_multishock,TAG=ms dag_rl.sh                # slope reward
#   sbatch --export=ALL,DATA=$FAM/data/rl_multishock_control,TAG=ctrl dag_rl.sh     # structureless control
set -euo pipefail

REPO=/n/fs/similarity/social_sim
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

# Launchpad serializes the actor and imports its scoring helper in a fresh worker
# process. Make that helper importable after the worker changes into OAT's runtime
# directory; otherwise the actor dies during cloudpickle deserialization while the
# learner waits indefinitely.
export PYTHONPATH="$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

FAM="$REPO/exp3_training_transfer/dag_family"
DATA="${DATA:-$FAM/data/rl_multishock}"       # canonical FAMILY arm: 8 train / 4 held-out, h* multi-shock probe
                                              # CONTROL arms: DATA=$FAM/data/rl_multishock_control   (structureless: format+level only)
                                              #               DATA=$FAM/data/rl_multishock_priormean (prior-mean: best structure-free)
TAG="${TAG:-fam}"                             # run label baked into the report dir name
OUT="$FAM/reports/pilot_${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAM/logs"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
GPUS="${GPUS:-1}"                                 # multi-GPU for big models (vLLM TP + ZeRO-3 shard)
NUM_GPUS_PER_ACTOR="${NUM_GPUS_PER_ACTOR:-1}"     # vLLM tensor-parallel size (= GPUS for one big actor)
ZERO_STAGE="${ZERO_STAGE:-2}"                     # 3 shards the frozen base across GPUs (32B+)
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"
BATCH="${BATCH:-16}"
MAX_TRAIN="${MAX_TRAIN:-4800}"                    # 4800/16 = 300 steps over the 8 structures (600 eps each)
EVAL_STEPS="${EVAL_STEPS:-25}"                    # eval every 25 steps (step 0 included) -> learning curve
GEN_LEN="${GEN_LEN:-1024}"                        # roomier so concise reasoning still finishes the JSON
VLLM_RATIO="${VLLM_RATIO:-0.40}"                  # audited fix: 0.50 starved the collocated learner
TEMP="${TEMP:-1.3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2560}"            # DAG histories up to ~9 weeks + reasoning
SEED="${SEED:-42}"                                # OAT default 42; override for independent seeds
NO_FUSED=""; [ "$ZERO_STAGE" = "3" ] && NO_FUSED="--no-use_fused_lm_head"   # ZeRO-3 requires this
LORA_SYNC=""; [ "$ZERO_STAGE" = "3" ] && LORA_SYNC="--lora_sync_only"       # ZeRO-3 + LoRA: OAT's merge-and-
                                                                            # broadcast path assumes param.ds_shape
                                                                            # on merged (plain) tensors -> crashes;
                                                                            # sync the adapter via disk + vLLM
                                                                            # LoRARequest instead (skips broadcast)
GRAD_CKPT="${GRAD_CKPT:-1}"; GC_FLAG=""                                     # multi-GPU ZeRO-3 path:
[ "$GRAD_CKPT" = "1" ] && GC_FLAG="--gradient-checkpointing"                # grad-ckpt reentrant recompute
                                                                            # breaks -> set GRAD_CKPT=0

cd "$REPO/.runtime/oat"
"$PY" "$REPO/exp3_training_transfer/biased_news/run_biased_news_rl.py" \
    --critic_type drgrpo \
    --gpus $GPUS \
    --num_gpus_per_actor $NUM_GPUS_PER_ACTOR \
    ${VLLM_SLEEP:+--vllm_sleep} \
    --lora_rank ${LORA_RANK:-32} \
    --lora_alpha ${LORA_ALPHA:-64} \
    $LORA_SYNC \
    --collocate \
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
echo "==== held-out TRANSFER: eps/rho/pi (base -> trained) vs references ===="
EVDIR=$(find "$OUT" -type d -name eval_results 2>/dev/null | head -1)
python3 "$FAM/transfer_report.py" "$EVDIR" || echo "(eval dumps not found at $EVDIR)"
echo "results dir: $OUT"
