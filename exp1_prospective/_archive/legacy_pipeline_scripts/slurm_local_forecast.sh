#!/usr/bin/env bash
#SBATCH --job-name=exp1_local_forecast
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/slurm_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/slurm_%j.err
#SBATCH --partition=mltheory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodelist=node302
#
# Runs Step 2b (local Qwen/Llama forecasting) on node302.
#
# Usage:
#   sbatch scripts/slurm_local_forecast.sh
#   sbatch --export=N_MARKETS=5,K_RUNS=1 scripts/slurm_local_forecast.sh   # pilot
#
# Override defaults via --export:
#   N_MARKETS   max markets to forecast per model (0 = all)
#   K_RUNS      independent runs per market (default: 3)
#   MODELS      space-separated list (default: "qwen2.5:7b llama3.1:8b")
#   NO_THIRD    set to "1" to skip red/blue-team turn (faster)

set -euo pipefail

# ── configurable parameters (override via --export) ───────────────────────────
N_MARKETS="${N_MARKETS:-0}"
K_RUNS="${K_RUNS:-3}"
MODELS="${MODELS:-qwen2.5:7b llama3.1:8b}"
NO_THIRD="${NO_THIRD:-0}"

# ── paths ──────────────────────────────────────────────────────────────────────
REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_HOST="127.0.0.1:11434"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"

echo "============================================================"
echo " exp1 local forecast — SLURM job $SLURM_JOB_ID"
echo " Node:    $(hostname)"
echo " Models:  $MODELS"
echo " Markets: ${N_MARKETS:-all}  k=$K_RUNS"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── check Ollama is installed ──────────────────────────────────────────────────
if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "[error] Ollama not found at $OLLAMA_BIN"
    echo "  Run setup first: srun -p gpu --gres=gpu:1 -w node302 --pty bash"
    echo "  then: bash $REPO/scripts/setup_ollama.sh"
    exit 1
fi

# ── GPU availability ───────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "[warn] nvidia-smi not found — Ollama will use CPU (slow)"
fi

# ── start Ollama server on this node ──────────────────────────────────────────
echo "[1/3] Starting Ollama server …"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama (pid $OLLAMA_PID)…"; kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

# wait up to 30s for server to accept requests
for i in $(seq 1 15); do
    if curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "[ok] Ollama is up"
        break
    fi
    sleep 2
done

if ! curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "[error] Ollama failed to start. Check logs."
    exit 1
fi

# ── Python: user-installed packages live in ~/.local; make sure they're on path ─
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
echo "[ok] Python: $(which python3)  $(python3 --version)"

cd "$REPO"

# set Tavily key from .env if present
if [[ -f "agent/.env" ]]; then
    set -o allexport
    source "agent/.env"
    set +o allexport
fi

# ── build CLI flags ────────────────────────────────────────────────────────────
N_FLAG=""
[[ "$N_MARKETS" -gt 0 ]] && N_FLAG="--n $N_MARKETS"
THIRD_FLAG=""
[[ "$NO_THIRD" == "1" ]] && THIRD_FLAG="--no-third-turn"

echo "[2/3] Running forecasts …"
echo ""

for MODEL in $MODELS; do
    echo "── $MODEL ──────────────────────────────────────────────────"
    # shellcheck disable=SC2086
    python3 agent/run_local_forecast.py \
        --model "$MODEL" \
        --endpoint "http://$OLLAMA_HOST/v1" \
        --k "$K_RUNS" \
        --verbose \
        ${N_FLAG} \
        ${THIRD_FLAG}
    echo ""
done

# ── Step 4: counterfactual updates (if CFs exist) ─────────────────────────────
echo "[3/3] Counterfactual updates …"

if ls data/counterfactuals/counterfactuals_*.jsonl 1>/dev/null 2>&1; then
    for MODEL in $MODELS; do
        echo "── updated forecasts: $MODEL ──"
        # shellcheck disable=SC2086
        python3 agent/local_updated_forecast.py \
            --model "$MODEL" \
            --endpoint "http://$OLLAMA_HOST/v1" \
            ${N_FLAG} \
            --verbose
        echo ""
    done
else
    echo "  No counterfactuals file found — skipping Step 4."
    echo "  Run build_counterfactuals.py first."
fi

echo "============================================================"
echo " Job complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
