#!/usr/bin/env bash
#SBATCH --job-name=exp1_qwen72b
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/qwen72b_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/qwen72b_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --nodelist=node302
#
# Stage 2 topup (30 missing markets) + Stage 4 for qwen2.5:72b.
# Already has 90/100 markets — appends the 30 from diverse_topup_30 to the
# existing forecast file, then runs Stage 4 (auto-resumes from latest file).

set -euo pipefail

MODEL="qwen2.5:72b"
REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"
echo "============================================================"
echo " Stage 2 topup + Stage 4: $MODEL"
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

# Find existing file and append the 30 missing markets to it
EXISTING=$(ls -t data/initial_forecasts/forecasts_qwen2.5-72b_*.jsonl 2>/dev/null | grep -v old | head -1)
[[ -n "$EXISTING" ]] || { echo "[error] No existing qwen2.5-72b forecast file found"; exit 1; }
echo "[1/2] Stage 2 topup: appending 30 markets to $EXISTING …"

python3 agent/run_local_forecast.py \
    --model "$MODEL" \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --input data/selected_markets/diverse_topup_30.jsonl \
    --out-file "$EXISTING" \
    --k 3 \
    --timeout 1800 \
    --no-third-turn \
    --verbose

echo ""
echo "[2/2] Stage 4: $MODEL — counterfactual updates (auto-resumes) …"
python3 agent/local_updated_forecast.py \
    --model "$MODEL" \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --timeout 1800

echo ""
echo "============================================================"
echo " Done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
python3 status.py
