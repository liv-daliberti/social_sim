#!/bin/bash
#SBATCH --job-name=cc_l70_sweep
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
set -euo pipefail
REPO=/n/fs/similarity/social_sim
# Belt and braces: the compute nodes have no route to huggingface.co, and a
# tokenizer load that falls through to the network kills the job in two minutes.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY="$REPO/.runtime/oat_conda/bin/python"
cd "$REPO/exp2_v2/biased_news/mechanistic_probe"
exec "$PY" -u sweep_patch_windows.py "$@"
