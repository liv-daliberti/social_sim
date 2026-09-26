#!/bin/bash
#SBATCH --job-name=fwc_llama3_1_70b
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/fwc_llama3_1_70b_%A_%a.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/fwc_llama3_1_70b_%A_%a.err
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --array=0-3
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
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$PY" -u "$MECH/patch_fixed_window_confirmation.py" --model-key llama3_1_70b \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 4 --device-map balanced --gpu-memory-gib 44 --stop-after-seconds 7800
