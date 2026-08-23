#!/usr/bin/env bash
#SBATCH --job-name=exp1_llama33_fix
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/llama33_fix_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/llama33_fix_%j.err
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --nodelist=node302
#
# Stage 4 ONLY for llama3.3:70b, pointed at the cleaned S2 file (June-17).
# 97/100 markets already done. The cleaned file fixes 3 markets whose
# valid S2 record was being shadowed by a trailing "Connection error" record
# under _load_forecasts' last-wins logic (2108617, 2241674, 540843).
# Dedup skips the 97 done, processes only the 3 fixed markets.

set -euo pipefail

MODEL="llama3.3:70b"
REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

S2_CLEAN="$REPO/data/initial_forecasts/forecasts_llama3.3-70b_2026-06-17.jsonl"

export OLLAMA_MODELS OLLAMA_HOST OLLAMA_CONTEXT_LENGTH=8192

mkdir -p "$REPO/logs"
echo "============================================================"
echo " Stage 4 fixup: $MODEL (3 missing markets)"
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

echo "[1/1] Stage 4: $MODEL — cleaned S2 file, dedup skips 97 done …"
python3 agent/local_updated_forecast.py \
    --model "$MODEL" \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --forecasts "$S2_CLEAN" \
    --timeout 3600 \
    --verbose

echo ""
echo "============================================================"
echo " Done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
python3 status.py
