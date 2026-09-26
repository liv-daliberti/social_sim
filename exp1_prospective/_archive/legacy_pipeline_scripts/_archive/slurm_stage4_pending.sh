#!/usr/bin/env bash
#SBATCH --job-name=exp1_s4_pending
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/s4_pending_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/s4_pending_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodelist=node302
#
# Stage 4 (updated forecasts) for all pending local models.
# Auto-resumes from the latest updated_{model}_*.jsonl file for each model,
# so it is safe to re-submit if interrupted.
#
# Current gaps (as of 2026-06-14):
#   llama3.1-8b   96/100  → 4 remaining
#   qwen2.5-14b   38/100  → 62 remaining
#   qwen2.5-32b   84/100  → 16 remaining
#   qwen2.5-72b   54/100  → 46 remaining
#   llama3.1-70b   0/100  → run AFTER Stage 2 completes
#   llama3.3-70b   0/100  → run AFTER Stage 2 completes
#
# Models processed in order of size (smallest first = fastest to finish).
# Override with MODELS env var, e.g.:
#   sbatch --export=MODELS="qwen2.5:72b" slurm_stage4_pending.sh

set -euo pipefail

MODELS="${MODELS:-llama3.1:8b qwen2.5:14b qwen2.5:32b qwen2.5:72b}"

REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"

echo "============================================================"
echo " exp1 Stage 4 — pending local models"
echo " SLURM job $SLURM_JOB_ID   Node: $(hostname)   Port: $OLLAMA_PORT"
echo " Models: $MODELS"
echo " Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo ""

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "[error] Ollama not found at $OLLAMA_BIN"
    exit 1
fi

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

echo "[1/2] Starting Ollama server on port $OLLAMA_PORT …"
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama (pid $OLLAMA_PID)…"; kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

for i in $(seq 1 20); do
    curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1 && { echo "[ok] Ollama up"; break; }
    sleep 3
done
curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1 || { echo "[error] Ollama failed to start"; exit 1; }

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"

cd "$REPO"
[[ -f "agent/.env" ]] && { set -o allexport; source "agent/.env"; set +o allexport; }

echo ""
python3 status.py
echo ""
echo "[2/2] Running Stage 4 …"
echo ""

for MODEL in $MODELS; do
    echo "── $MODEL ──────────────────────────────────────────────────"

    # 72b models need a longer per-request timeout
    TIMEOUT=600
    [[ "$MODEL" == *"72b"* || "$MODEL" == *"70b"* ]] && TIMEOUT=3600

    python3 agent/local_updated_forecast.py \
        --model "$MODEL" \
        --endpoint "http://$OLLAMA_HOST/v1" \
        --timeout "$TIMEOUT" \
        --verbose
    echo ""
done

echo ""
echo "[done] Stage 4 complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
python3 status.py
