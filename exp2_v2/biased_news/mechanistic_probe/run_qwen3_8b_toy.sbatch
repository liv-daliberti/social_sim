#!/bin/bash
#SBATCH --job-name=coin_q8_probe_toy
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_8b_toy_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_8b_toy_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_8b_toy_v1"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

nvidia-smi

"$PY" exp2_v2/biased_news/mechanistic_probe/build_probe_tasks.py \
  --outdir "$RUN_DIR"

"$PY" exp2_v2/biased_news/mechanistic_probe/extract_qwen_hidden_states.py \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-8B \
  --model-label Qwen3-8B \
  --batch-size 2 \
  --generation-batch-size 2 \
  --max-input-tokens 2304 \
  --max-new-tokens 160 \
  --force

"$PY" exp2_v2/biased_news/mechanistic_probe/analyze_probe.py \
  --run-dir "$RUN_DIR" \
  --permutations 10000

echo "Toy mechanistic probe complete: $RUN_DIR"

