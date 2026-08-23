#!/usr/bin/env bash
#SBATCH --job-name=exp1_llama31_70b
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/llama31_70b_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/llama31_70b_%j.err
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#
# Stage 2 (k=3, 1 missing market only) + Stage 4 for llama3.1:70b.
# 99/100 Stage 4 markets already done. This runs the 1 missing market (2268861)
# through Stage 2 (k=3) then Stage 4 to reach 100/100.

set -euo pipefail

MODEL="llama3.1:70b"
REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

export OLLAMA_MODELS OLLAMA_HOST OLLAMA_CONTEXT_LENGTH=8192

mkdir -p "$REPO/logs"
echo "============================================================"
echo " Stage 2 + 4: $MODEL (full 100-market run)"
echo " SLURM job $SLURM_JOB_ID   Node: $(hostname)   Port: $OLLAMA_PORT"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
echo ""

[[ -x "$OLLAMA_BIN" ]] || { echo "[error] Ollama not found at $OLLAMA_BIN"; exit 1; }

OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

for i in $(seq 1 20); do
    curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1 && { echo "[ok] Ollama up"; break; }
    sleep 3
done
curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1 || { echo "[error] Ollama failed"; exit 1; }

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
cd "$REPO"
[[ -f "agent/.env" ]] && { set -o allexport; source "agent/.env"; set +o allexport; }

python3 status.py
echo ""

echo "[1/2] Stage 2: $MODEL — 1 missing market (2268861), k=3, timeout=3600s …"
python3 agent/run_local_forecast.py \
    --model "$MODEL" \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --input data/selected_markets/single_2268861.jsonl \
    --k 3 \
    --timeout 3600 \
    --no-third-turn \
    --verbose

echo ""
echo "[2/2] Stage 4: $MODEL — counterfactual updates …"
python3 agent/local_updated_forecast.py \
    --model "$MODEL" \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --timeout 3600

echo ""
echo "============================================================"
echo " Done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
python3 status.py
