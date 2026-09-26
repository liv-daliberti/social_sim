#!/usr/bin/env bash
#SBATCH --job-name=exp1_pull_bigger
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/pull_bigger_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/pull_bigger_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodelist=node302
#
# One-time: pull all 6 bigger models to shared Ollama storage (~165 GB total).
# Run this before submitting the tier forecast jobs.
#
# Usage:
#   sbatch scripts/slurm_pull_bigger_models.sh
#
# Models pulled:
#   Tier 1 (~14B): qwen2.5:14b  (~8.7 GB)   [Llama has no ~14B text model in 3.x family]
#   Tier 2 (~70B): qwen2.5:32b  (~19.8 GB), llama3.1:70b  (~42.5 GB)
#   Tier 3 (~72B): qwen2.5:72b  (~43.7 GB), llama3.3:70b  (~42.5 GB)

set -euo pipefail

OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_HOST="127.0.0.1:11434"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p /n/fs/similarity/social_sim/exp1_prospective/logs

echo "============================================================"
echo " pull bigger models — SLURM job $SLURM_JOB_ID"
echo " Node: $(hostname)   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[warn] no nvidia-smi"

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "[error] Ollama not found at $OLLAMA_BIN — run slurm_setup_ollama.sh first"
    exit 1
fi

echo "[1/2] Starting Ollama server …"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama …"; kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

for i in $(seq 1 20); do
    if curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "[ok] Ollama is up"
        break
    fi
    sleep 3
done

if ! curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "[error] Ollama failed to start"
    exit 1
fi

echo ""
echo "[2/2] Pulling models …"
echo ""

MODELS=(
    # qwen2.5:14b already pulled — skip
    "qwen2.5:32b"   # Tier 2  ~19.8 GB
    "llama3.1:70b"  # Tier 2  ~42.5 GB
    "qwen2.5:72b"   # Tier 3  ~43.7 GB
    "llama3.3:70b"  # Tier 3  ~42.5 GB
)

for MODEL in "${MODELS[@]}"; do
    echo "── pulling $MODEL ──"
    "$OLLAMA_BIN" pull "$MODEL"
    echo ""
done

echo "[ok] All models on disk:"
"$OLLAMA_BIN" list

echo "============================================================"
echo " Pull complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Next step:"
echo "   bash scripts/launch_bigger_models.sh"
echo "   (or submit the three tier scripts individually)"
echo "============================================================"
