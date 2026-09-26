#!/bin/bash
#SBATCH --job-name=fwc_qwen3_14b
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/fwc_qwen3_14b_%A_%a.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/fwc_qwen3_14b_%A_%a.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
# Fixed-window patch confirmation on fresh episodes; every setting is read from
# the committed protocol.json and the runner refuses a changed task or code hash.
# A shard that hits walltime resumes from its saved records when resubmitted.
set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$PY" -u "$MECH/patch_fixed_window_confirmation.py" --model-key qwen3_14b \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 2  --stop-after-seconds 6600
