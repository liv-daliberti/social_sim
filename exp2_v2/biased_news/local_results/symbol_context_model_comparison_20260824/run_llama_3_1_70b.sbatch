#!/bin/bash
#SBATCH --job-name=coin_symbol_l70
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/logs/llama_3_1_70b_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/logs/llama_3_1_70b_%j.err
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a5000:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00

set -euo pipefail
repo=/n/fs/similarity/social_sim
. "$repo/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$repo/.runtime/oat_conda/bin/python" \
  "$repo/exp2_v2/biased_news/eval/run_coin_city_symbol_context_open_model.py" \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --model-label Llama-3.1-70B-Instruct \
  --outdir "$repo/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/llama_3_1_70b" \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 8 \
  --chat-template-mode auto \
  --model-license-family "Llama 3.1 Community License open-weight release"
