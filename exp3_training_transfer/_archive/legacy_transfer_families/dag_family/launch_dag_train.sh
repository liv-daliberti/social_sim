#!/bin/bash
# 250-step Dr.GRPO with eval every 25 steps -> LEARNING CURVES on the 4 held-out structures.
# Ladder: Qwen3-4B x3 seeds (error band) + Qwen2.5-7B + Qwen2.5-14B (scale ladder).
# steps = MAX_TRAIN / BATCH; eval dumps land at step 0,25,...,250 for curve_from_eval_dag.py.
#   bash launch_dag_train.sh
set -euo pipefail
ROOT=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family
cd "$ROOT"
ES="${EVAL_STEPS:-25}"

# --- Qwen3-4B x3 seeds : a6000, BATCH 16 x 250 steps = MAX_TRAIN 4000 (one full epoch) ---
for s in 42 43 44; do
  jid=$(sbatch --parsable --partition=all --gres=gpu:a6000:1 --time=24:00:00 \
        --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,SEED=$s,MAX_TRAIN=4000,BATCH=16,EVAL_STEPS=$ES,VLLM_RATIO=0.5,GEN_LEN=1024 \
        dag_rl.sh)
  echo "submitted $jid  Qwen3-4B seed=$s  (250 steps, eval@$ES)"
done

# --- Qwen2.5-7B : a6000, BATCH 16 x 250 = MAX_TRAIN 4000 ---
jid=$(sbatch --parsable --partition=all --gres=gpu:a6000:1 --time=30:00:00 \
      --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,SEED=42,MAX_TRAIN=4000,BATCH=16,EVAL_STEPS=$ES,VLLM_RATIO=0.45,GEN_LEN=1024 \
      dag_rl.sh)
echo "submitted $jid  Qwen2.5-7B seed=42  (250 steps, eval@$ES)"

# --- Qwen2.5-14B : a100/mltheory, BATCH 8 x 250 = MAX_TRAIN 2000 ---
jid=$(sbatch --parsable --account=mltheory --partition=mltheory --gres=gpu:a100:1 --time=48:00:00 \
      --export=ALL,MODEL=Qwen/Qwen2.5-14B-Instruct,SEED=42,MAX_TRAIN=2000,BATCH=8,EVAL_STEPS=$ES,VLLM_RATIO=0.35,GEN_LEN=1024,MAX_MODEL_LEN=2560 \
      dag_rl.sh)
echo "submitted $jid  Qwen2.5-14B seed=42  (250 steps, eval@$ES)"
echo "== 5 training jobs submitted (250 steps, eval every $ES) =="
