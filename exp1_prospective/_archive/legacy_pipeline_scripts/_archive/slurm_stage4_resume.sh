#!/usr/bin/env bash
#SBATCH --job-name=exp1_stage4_resume
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/stage4_resume_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/stage4_resume_%j.err
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --nodelist=node302
#
# Resume Step 4 for qwen2.5:7b (8 remaining) and llama3.1:8b (79 remaining).
# done_keys mechanism skips already-completed markets automatically.

set -euo pipefail

REPO="/n/fs/similarity/social_sim/exp1_prospective"
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN="$OLLAMA_BASE/bin/ollama"
OLLAMA_MODELS="$OLLAMA_BASE/models"
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 10000))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
MODELS="qwen2.5:7b llama3.1:8b"

export OLLAMA_MODELS OLLAMA_HOST

mkdir -p "$REPO/logs"
echo "============================================================"
echo " Step 4 resume — qwen2.5:7b + llama3.1:8b"
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

for MODEL in $MODELS; do
    echo "── Step 4: $MODEL ──"
    python3 agent/local_updated_forecast.py \
        --model "$MODEL" \
        --endpoint "http://$OLLAMA_HOST/v1" \
        --verbose
    echo ""
done

echo "============================================================"
echo " Step 4 resume complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
