#!/bin/bash
# UNTRAINED-model DAG-family eval across the full Exp-2 open-weight roster.
# One slurm job per model (each serves its own Ollama), structures=all 20, resumable JSONL.
# 1024-token budget everywhere (fixes the 512-cap parse-miss on verbose models); 70-72B get 4 GPUs.
#   bash launch_dag_eval.sh
set -euo pipefail
ROOT=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family
cd "$ROOT"

N="${N:-20}"
STRUCTURES="${STRUCTURES:-all}"
KS="${KS:-1_3_5}"          # '_' separator survives sbatch --export comma-split
MAXTOK="${MAXTOK:-1024}"   # generation cap: 1024 so verbose models finish the JSON before truncation

# small/mid open-weight: 1x a6000 (48G) is ample at Ollama's default quant
SMALL=(qwen2.5:7b qwen2.5:14b qwen2.5:32b llama3.1:8b mistral-small:24b gemma2:9b gemma2:27b)
# 70-72B: 4 GPUs (Ollama shards the model across them automatically); 4x a5000 = 96G
BIG=(qwen2.5:72b llama3.1:70b llama3.3:70b)

for m in "${SMALL[@]}"; do
  jid=$(sbatch --parsable --partition=all --gres=gpu:a6000:1 \
        --export=ALL,MODEL="$m",N="$N",STRUCTURES="$STRUCTURES",KS="$KS",MAXTOK="$MAXTOK" \
        slurm_dag_eval.sh)
  echo "submitted $jid  $m  (1x a6000, maxtok=$MAXTOK)"
done
for m in "${BIG[@]}"; do
  jid=$(sbatch --parsable --partition=all --gres=gpu:a5000:4 --mem=120G \
        --export=ALL,MODEL="$m",N="$N",STRUCTURES="$STRUCTURES",KS="$KS",MAXTOK="$MAXTOK" \
        slurm_dag_eval.sh)
  echo "submitted $jid  $m  (4x a5000, maxtok=$MAXTOK)"
done
echo "== $(( ${#SMALL[@]} + ${#BIG[@]} )) open-weight eval jobs submitted (maxtok=$MAXTOK, n=$N structures=$STRUCTURES) =="
