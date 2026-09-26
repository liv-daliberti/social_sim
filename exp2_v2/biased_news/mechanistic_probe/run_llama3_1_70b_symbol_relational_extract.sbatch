#!/bin/bash
#SBATCH --job-name=cc_l70_rel_extract
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_rel_extract_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_rel_extract_%j.err
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/llama3_1_70b_symbol_relational_v1"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$MECH/logs"
exec 9>"$RUN_DIR/.extract_select.lock"
flock -n 9 || { echo "Llama relational extraction/selection already owned"; exit 0; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

if [[ ! -f "$RUN_DIR/label_extraction_manifest.json" ]]; then
  "$PY" "$MECH/extract_symbol_label_states.py" \
    --run-dir "$RUN_DIR" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --device-map balanced --gpu-memory-gib 44
fi
if [[ ! -f "$RUN_DIR/development_selection.json" ]]; then
  "$PY" "$MECH/analyze_symbol_relational_probe.py" \
    --run-dir "$RUN_DIR" --mode selection-only
fi
echo "Llama-3.1-70B relational extraction and development selection complete"
