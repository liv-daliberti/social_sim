#!/bin/bash
#SBATCH --job-name=exp1_context_reversal
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00

# Submit with explicit --output/--error paths in an already-created log folder.
# Required exports: MODEL_PATH (absolute cached snapshot), MODEL_KEY, INPUT, OUTPUT.
# Optional: TP (default 4), BATCH_SIZE (32), TEMPLATE_MODE (auto), SEED (20260921).
set -euo pipefail
repo=/n/fs/similarity/social_sim
source "$repo/.runtime/oat_env.sh"
: "${MODEL_PATH:?Set MODEL_PATH to an existing local model snapshot}"
: "${MODEL_KEY:?Set MODEL_KEY}"
: "${INPUT:?Set INPUT to compiled units JSONL}"
: "${OUTPUT:?Set OUTPUT to responses JSONL}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Keep runtime compilation/cache files off the shared NFS cache during jobs.
export TRITON_CACHE_DIR="/tmp/exp1_context_triton_${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="/tmp/exp1_context_vllm_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$(dirname "$OUTPUT")"
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_CACHE_DIR="/tmp/exp1_context_inductor_${SLURM_JOB_ID}"
export OUTLINES_CACHE_DIR="/tmp/exp1_context_outlines_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$OUTLINES_CACHE_DIR"
cd "$repo"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$repo/.runtime/oat_conda/bin/python" -u "${RUNNER_SCRIPT:-$repo/exp1_prospective/context_reversal/run_local.py}" \
  --model-path "$MODEL_PATH" --model-key "$MODEL_KEY" \
  --input "$INPUT" --output "$OUTPUT" \
  --tensor-parallel-size "${TP:-4}" --batch-size "${BATCH_SIZE:-32}" \
  --max-model-len 4096 --max-tokens 128 --gpu-memory-utilization 0.90 \
  --dtype bfloat16 --temperature 0.7 --seed "${SEED:-20260921}" \
  --chat-template-mode "${TEMPLATE_MODE:-auto}" --enforce-eager

"$repo/.runtime/oat_conda/bin/python" -u "${ANALYZER_SCRIPT:-$repo/exp1_prospective/context_reversal/analyze.py}" \
  --design "$INPUT" --responses "$OUTPUT" --model-key "$MODEL_KEY" \
  --output "${RESULTS_DIR:-$repo/exp1_prospective/context_reversal/results/$MODEL_KEY}"
