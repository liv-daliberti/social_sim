#!/bin/bash
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=16
set -euo pipefail
repo=/n/fs/similarity/social_sim
source "$repo/.runtime/oat_env.sh"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TRITON_CACHE_DIR="/tmp/cr_diag_triton_${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="/tmp/cr_diag_vllm_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/cr_diag_inductor_${SLURM_JOB_ID}"
export OUTLINES_CACHE_DIR="/tmp/cr_diag_outlines_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$OUTLINES_CACHE_DIR"
cd "$repo"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$repo/.runtime/oat_conda/bin/python" -u -m exp1_prospective.context_reversal.run_diagnostic_group \
  --submission "$1" --group "$2"
