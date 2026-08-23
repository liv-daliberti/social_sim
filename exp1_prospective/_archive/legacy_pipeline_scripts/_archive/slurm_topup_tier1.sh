#!/usr/bin/env bash
#SBATCH --job-name=exp1_topup_tier1
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/topup_tier1_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/topup_tier1_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --nodelist=node302
#
# Top-up: forecast the 30 June-09 markets that were missing from the June-11 run.
# Appends directly to the existing forecasts_qwen2.5-14b_*.jsonl file.

set -euo pipefail

REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
TOPUP_INPUT="$REPO/data/selected_markets/diverse_topup_30.jsonl"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"
echo "============================================================"
echo " Topup tier1 (qwen2.5:14b) — 30 missing markets"
echo " SLURM job $SLURM_JOB_ID   Node: $(hostname)   Port: $OLLAMA_PORT"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

for i in $(seq 1 20); do
    curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1 && break
    sleep 3
done
curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1 || { echo "[error] Ollama failed"; exit 1; }

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
cd "$REPO"
[[ -f "agent/.env" ]] && { set -o allexport; source "agent/.env"; set +o allexport; }

# Find the existing forecast file for this model and append to it
EXISTING=$(ls -t data/initial_forecasts/forecasts_qwen2.5-14b_*.jsonl 2>/dev/null | grep -v old | head -1)
if [[ -z "$EXISTING" ]]; then
    echo "[error] No existing forecast file for qwen2.5-14b"
    exit 1
fi
echo "Appending to: $EXISTING"

echo "[1/2] Running topup forecasts (30 markets) …"
python3 agent/run_local_forecast.py \
    --model qwen2.5:14b \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --input "$TOPUP_INPUT" \
    --out-file "$EXISTING" \
    --k 3 \
    --no-third-turn \
    --verbose

echo "[2/2] Running stage-2 updated forecasts for all 100 markets …"
python3 agent/local_updated_forecast.py \
    --model qwen2.5:14b \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --verbose

echo "============================================================"
echo " Topup tier1 complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
