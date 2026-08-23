#!/usr/bin/env bash
#SBATCH --job-name=exp1_s4_q32
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/s4_qwen32b_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/s4_qwen32b_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodelist=node208

set -euo pipefail

MODEL="qwen2.5:32b"
REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

export OLLAMA_MODELS OLLAMA_HOST OLLAMA_CONTEXT_LENGTH=8192

mkdir -p "$REPO/logs"
echo "============================================================"
echo " Stage 4: $MODEL"
echo " SLURM job $SLURM_JOB_ID   Node: $(hostname)   Port: $OLLAMA_PORT"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

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

python3 agent/local_updated_forecast.py \
    --model "$MODEL" \
    --endpoint "http://$OLLAMA_HOST/v1" \
    --verbose

echo "============================================================"
echo " Done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
python3 status.py
