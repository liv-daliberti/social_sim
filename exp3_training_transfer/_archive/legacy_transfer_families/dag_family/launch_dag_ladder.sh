#!/bin/bash
# Full ladder retrain on the 13-world catalog, x3 seeds, ALL with the advantage-collapse fix
# (TEMP=1.3, REWARD_SCALE=15). Packs across the whole GPU pool:
#   4B/7B  -> a6000 (all)        8B/14B -> a100 (mltheory)      32B/72B -> 8x a6000 multi-GPU (all)
# 250 steps, eval every 25 -> learning curves. Job ids -> reports/ladder_jobs.txt
#   bash launch_dag_ladder.sh
set -uo pipefail
ROOT=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family
cd "$ROOT"
C="TEMP=1.3,REWARD_SCALE=15,LR=0.000001,EVAL_STEPS=25,GEN_LEN=1024,MAX_MODEL_LEN=2560,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
JOBS="$ROOT/reports/ladder_jobs.txt"; : > "$JOBS"

rec() { echo "$1 $2 seed=$3"; echo "$1 $2 $3" >> "$JOBS"; }

for s in 42 43 44; do
  # --- 4B : a6000, ratio .5, rank 32, BATCH16 x 4000 = 250 steps ---
  j=$(sbatch --parsable --partition=all --gres=gpu:a6000:1 --time=36:00:00 \
      --export=ALL,$C,MODEL=Qwen/Qwen3-4B-Instruct-2507,SEED=$s,VLLM_RATIO=0.5,LORA_RANK=32,LORA_ALPHA=64,BATCH=16,MAX_TRAIN=4000 \
      dag_rl.sh) && rec "$j" Qwen3-4B $s
  # --- 7B : a6000, ratio .45, rank 64 ---
  j=$(sbatch --parsable --partition=all --gres=gpu:a6000:1 --time=36:00:00 \
      --export=ALL,$C,MODEL=Qwen/Qwen2.5-7B-Instruct,SEED=$s,VLLM_RATIO=0.45,LORA_RANK=64,LORA_ALPHA=128,BATCH=16,MAX_TRAIN=4000 \
      dag_rl.sh) && rec "$j" Qwen2.5-7B $s
  # --- 8B : a100/mltheory, ratio .45, rank 64 ---
  j=$(sbatch --parsable --account=mltheory --partition=mltheory --gres=gpu:a100:1 --time=48:00:00 \
      --export=ALL,$C,MODEL=meta-llama/Llama-3.1-8B-Instruct,SEED=$s,VLLM_RATIO=0.45,LORA_RANK=64,LORA_ALPHA=128,BATCH=16,MAX_TRAIN=4000 \
      dag_rl.sh) && rec "$j" Llama-3.1-8B $s
  # --- 14B : a100/mltheory, ratio .35, rank 128, BATCH8 x 2000 = 250 steps ---
  j=$(sbatch --parsable --account=mltheory --partition=mltheory --gres=gpu:a100:1 --time=48:00:00 \
      --export=ALL,$C,MODEL=Qwen/Qwen2.5-14B-Instruct,SEED=$s,VLLM_RATIO=0.35,LORA_RANK=128,LORA_ALPHA=256,BATCH=8,MAX_TRAIN=2000 \
      dag_rl.sh) && rec "$j" Qwen2.5-14B $s
done

for s in 42 43 44; do
  # --- 32B : 8x a6000 multi-GPU (ZeRO-3, no-sleep, ratio .3) ---
  j=$(sbatch --parsable --partition=all --gres=gpu:a6000:8 --nodes=1 --mem=350G --cpus-per-task=32 --time=72:00:00 \
      --export=ALL,$C,MODEL=Qwen/Qwen2.5-32B-Instruct,SEED=$s,GPUS=8,NUM_GPUS_PER_ACTOR=8,ZERO_STAGE=3,GRAD_CKPT=0,VLLM_SLEEP=,VLLM_RATIO=0.3,LORA_RANK=128,LORA_ALPHA=256,BATCH=16,MAX_TRAIN=4000 \
      dag_rl.sh) && rec "$j" Qwen2.5-32B $s
  # --- 72B : 8x a6000 multi-GPU (ZeRO-3, no-sleep, ratio .25) ---
  j=$(sbatch --parsable --partition=all --gres=gpu:a6000:8 --nodes=1 --mem=400G --cpus-per-task=32 --time=96:00:00 \
      --export=ALL,$C,MODEL=Qwen/Qwen2.5-72B-Instruct,SEED=$s,GPUS=8,NUM_GPUS_PER_ACTOR=8,ZERO_STAGE=3,GRAD_CKPT=0,VLLM_SLEEP=,VLLM_RATIO=0.25,LORA_RANK=128,LORA_ALPHA=256,BATCH=16,MAX_TRAIN=4000 \
      dag_rl.sh) && rec "$j" Qwen2.5-72B $s
done

echo "== submitted $(wc -l < "$JOBS") jobs =="; column -t "$JOBS"
