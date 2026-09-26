#!/bin/bash
#SBATCH --job-name=coin_symbol_q4
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/eval/coin_symbol_q4_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/eval/coin_symbol_q4_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$REPO/.runtime/oat_conda/bin/python" \
  "$REPO/exp2_v2/biased_news/eval/run_coin_city_symbol_context_open_model.py"
