#!/bin/bash
#SBATCH --job-name=coin_symbol_l8
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/logs/llama_3_1_8b_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/logs/llama_3_1_8b_%j.err
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
repo=/n/fs/similarity/social_sim
. "$repo/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$repo/.runtime/oat_conda/bin/python" \
  "$repo/exp2_v2/biased_news/eval/run_coin_city_symbol_context_open_model.py" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-label Llama-3.1-8B-Instruct \
  --outdir "$repo/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/llama_3_1_8b" \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --chat-template-mode auto \
  --model-license-family "Llama 3.1 Community License open-weight release"
