#!/usr/bin/env bash
#SBATCH --job-name=exp1_tier1_forecast
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/tier1_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/tier1_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --nodelist=node302
#
# Tier 1 (~14B) forecast: qwen2.5:14b only
# (No Llama 3.x text model exists between 8B and 70B; qwen2.5:14b already pulled.)
#
# Usage:
#   sbatch scripts/slurm_bigger_forecast_tier1.sh
#   sbatch --export=N_MARKETS=5,K_RUNS=1 scripts/slurm_bigger_forecast_tier1.sh  # pilot
#
# Override via --export:
#   N_MARKETS   max markets per model (0 = all)
#   K_RUNS      independent runs per market (default: 3)

set -euo pipefail

N_MARKETS="${N_MARKETS:-0}"
K_RUNS="${K_RUNS:-3}"
MODELS="qwen2.5:14b"

REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"

# Use a job-specific port so parallel tier jobs don't conflict
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"

echo "============================================================"
echo " Tier 1 forecast (qwen2.5:14b)"
echo " SLURM job $SLURM_JOB_ID   Node: $(hostname)"
echo " Port: $OLLAMA_PORT   Markets: ${N_MARKETS:-all}   k=$K_RUNS"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
echo ""

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "[error] Ollama not found at $OLLAMA_BIN"
    exit 1
fi

echo "[1/3] Starting Ollama server on port $OLLAMA_PORT …"
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama (pid $OLLAMA_PID) …"; kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

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

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"

cd "$REPO"

if [[ -f "agent/.env" ]]; then
    set -o allexport
    source "agent/.env"
    set +o allexport
fi

N_FLAG=""
[[ "$N_MARKETS" -gt 0 ]] && N_FLAG="--n $N_MARKETS"

echo "[2/3] Running forecasts …"
echo ""

for MODEL in $MODELS; do
    echo "── $MODEL ──────────────────────────────────────────────────"
    # shellcheck disable=SC2086
    python3 agent/run_local_forecast.py \
        --model "$MODEL" \
        --endpoint "http://$OLLAMA_HOST/v1" \
        --k "$K_RUNS" \
        --input data/selected_markets/diverse_2026-06-09.jsonl \
        --no-third-turn \
        --verbose \
        ${N_FLAG}
    echo ""
done

echo "[3/3] Counterfactual updates …"

if ls data/counterfactuals/counterfactuals_*.jsonl 1>/dev/null 2>&1; then
    for MODEL in $MODELS; do
        echo "── updated forecasts: $MODEL ──"
        # shellcheck disable=SC2086
        python3 agent/local_updated_forecast.py \
            --model "$MODEL" \
            --endpoint "http://$OLLAMA_HOST/v1" \
            --verbose \
            ${N_FLAG}
        echo ""
    done
else
    echo "  No counterfactuals file found — skipping."
fi

echo "============================================================"
echo " Tier 1 complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
