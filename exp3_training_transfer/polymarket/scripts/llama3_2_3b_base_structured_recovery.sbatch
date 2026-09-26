#!/bin/bash
#SBATCH --job-name=exp4_l32_base_structured
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/l32_base_structured_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/l32_base_structured_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --account=mltheory
#SBATCH --partition=all
#SBATCH --exclude=node206,node207

set -euo pipefail
REPO=/n/fs/similarity/social_sim
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
MODEL="$REPO/.runtime/hf_home/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/006f5dcd1393c3add266de40994ba96225e9689d"
OUT="$REPO/exp3_training_transfer/polymarket/reports/exp4_llama3_2_3b_base_structured_recovery_j${SLURM_JOB_ID}"
"$PY" "$REPO/exp3_training_transfer/polymarket/scripts/evaluate_llama3_2_3b_base_structured_recovery.py" \
  --model "$MODEL" \
  --output-prefix "$OUT"
