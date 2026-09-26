#!/bin/bash
#SBATCH --job-name=c3m_behavior
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/behavior_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/behavior_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
MECH="$REPO/exp3_training_transfer/coin_city_structural/mechanistic_transfer"
EVAL="$REPO/exp3_training_transfer/mechanism_family/evaluate_endpoint.py"
PY="$REPO/.runtime/oat_conda/bin/python"
: "${SEED:?SEED is required}"
: "${ARM:?ARM is required}"
case "$ARM" in matched|prior|base) ;; *) echo "bad ARM=$ARM"; exit 2;; esac
if [[ "$ARM" != "base" ]]; then : "${ADAPTER:?ADAPTER is required}"; fi
RUN_DIR="$MECH/runs/qwen3_8b_s${SEED}"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$MECH/logs" "$RUN_DIR"
adapter_args=()
if [[ "$ARM" != "base" ]]; then adapter_args=(--adapter "$ADAPTER"); fi
"$PY" "$EVAL" --model Qwen/Qwen3-8B "${adapter_args[@]}" \
  --data "$MECH/data/behavior" --output "$RUN_DIR/behavior_${ARM}.json" \
  --template auto_no_think --temperature 0.7 --n 5 --seed 20260819 \
  --max-tokens 192 --max-model-len 2560 --gpu-memory-utilization 0.88 \
  --structured-output forecast_array
echo "Exp3 mechanism seed $SEED endpoint $ARM behavior complete"
