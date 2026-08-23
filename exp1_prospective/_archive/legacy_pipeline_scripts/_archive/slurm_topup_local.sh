#!/usr/bin/env bash
#SBATCH --job-name=exp1_topup
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/topup_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/topup_%j.err
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodelist=node302
#
# Tops up local model forecasts to TARGET_K runs per market.
# Reads the existing forecasts_{slug}_*.jsonl and adds only the missing runs.
#
# Override defaults via --export:
#   TARGET_K   target runs per market (default: 5)
#   MODELS     space-separated list (default: "qwen2.5:7b llama3.1:8b")

set -euo pipefail

TARGET_K="${TARGET_K:-5}"
MODELS="${MODELS:-qwen2.5:7b llama3.1:8b}"

REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_HOST="127.0.0.1:11434"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"

echo "============================================================"
echo " exp1 local topup — SLURM job $SLURM_JOB_ID"
echo " Node:    $(hostname)"
echo " Models:  $MODELS"
echo " Target k: $TARGET_K"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "[error] Ollama not found at $OLLAMA_BIN"
    exit 1
fi

if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

echo "[1/2] Starting Ollama server …"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama (pid $OLLAMA_PID)…"; kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

for i in $(seq 1 15); do
    if curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "[ok] Ollama is up"
        break
    fi
    sleep 2
done

if ! curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "[error] Ollama failed to start."
    exit 1
fi

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
echo "[ok] Python: $(which python3)  $(python3 --version)"

cd "$REPO"

if [[ -f "agent/.env" ]]; then
    set -o allexport
    source "agent/.env"
    set +o allexport
fi

echo "[2/2] Topping up forecasts to k=$TARGET_K …"
echo ""

for MODEL in $MODELS; do
    echo "── $MODEL ──────────────────────────────────────────────────"
    python3 agent/topup_local_forecast.py \
        --model "$MODEL" \
        --endpoint "http://$OLLAMA_HOST/v1" \
        --target-k "$TARGET_K" \
        --verbose
    echo ""
done

echo "============================================================"
echo " Job complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
