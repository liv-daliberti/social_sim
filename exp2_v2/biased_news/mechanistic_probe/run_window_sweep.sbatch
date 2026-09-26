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
# oat_env.sh sets HF_HOME and the hub cache paths. Without it the local
# checkpoint is invisible and the tokenizer load reaches for the network, which
# the compute nodes cannot do. The registered probe job sources it for the same
# reason; omitting it is what killed the first two attempts at this sweep.
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
PY="$REPO/.runtime/oat_conda/bin/python"
cd "$REPO/exp2_v2/biased_news/mechanistic_probe"
exec "$PY" -u sweep_patch_windows.py "$@"
