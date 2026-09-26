#!/bin/bash
#SBATCH --job-name=biased_eval
#SBATCH --output=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/biased_eval_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/biased_eval_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio

# ── USAGE ───────────────────────────────────────────────────────────────────
# Generic local (Ollama) eval — replaces slurm_qwen.sh / slurm_llama.sh.
# Pass the model via the MODEL env var (sbatch --export):
#
#   sbatch --export=ALL,MODEL=qwen2.5:7b   eval/slurm_eval.sh
#   sbatch --export=ALL,MODEL=llama3.3:70b eval/slurm_eval.sh
#   sbatch --export=ALL,MODEL=qwen2.5:72b,N=100 eval/slurm_eval.sh
#
# Or submit the whole local roster at once:  bash eval/launch_local.sh
# Extra flags after the script are forwarded to run_batch.py.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news"
cd "$ROOT"
mkdir -p logs data/batch

MODEL="${MODEL:-qwen2.5:7b}"           # override via --export=ALL,MODEL=...
N="${N:-100}"                          # episodes per bias condition
OLLAMA_BIN="/n/fs/similarity/ollama/bin/ollama"
OLLAMA_MODELS_DIR="/n/fs/similarity/ollama/models"

# Port derived from job ID to avoid conflicts if two jobs land on the same node
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 1000)))
ENDPOINT="${OLLAMA_HOST:-http://localhost:${OLLAMA_PORT}/v1}"

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL   N: $N   Endpoint: $ENDPOINT"
echo "Root: $ROOT"

# ── Start Ollama on this node (skip if using external endpoint) ───────────────
if [[ "$ENDPOINT" == "http://localhost:${OLLAMA_PORT}/v1" ]]; then
    echo "Starting Ollama on port ${OLLAMA_PORT}..."
    OLLAMA_MODELS="$OLLAMA_MODELS_DIR" OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
        "$OLLAMA_BIN" serve > /tmp/ollama_${SLURM_JOB_ID}.log 2>&1 &
    OLLAMA_PID=$!
    trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
            echo "Ollama ready (attempt $i)"
            break
        fi
        sleep 2
    done

    OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" "$OLLAMA_BIN" pull "$MODEL" || true
fi

# ── Run batch evaluation ──────────────────────────────────────────────────────
python3 eval/run_batch.py \
    --backend  ollama \
    --model    "$MODEL" \
    --endpoint "$ENDPOINT" \
    --n "$N" \
    --T 12 \
    --variants V0 V1 V2 \
    --delay 0.05 \
    --max-tokens 150 \
    --temperature 0.1 \
    "$@"

echo "Done — results in $ROOT/data/batch/"
