#!/bin/bash
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=16
set -euo pipefail
repo=/n/fs/similarity/social_sim
source "$repo/.runtime/oat_env.sh"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR="/tmp/crfresh_triton_${SLURM_JOB_ID}" VLLM_CACHE_ROOT="/tmp/crfresh_vllm_${SLURM_JOB_ID}" OUTLINES_CACHE_DIR="/tmp/crfresh_outlines_${SLURM_JOB_ID}"
cd "$repo"
exec "$repo/.runtime/oat_conda/bin/python" -u -m "$@"
