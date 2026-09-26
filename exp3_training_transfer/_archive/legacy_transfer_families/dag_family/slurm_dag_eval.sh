#!/bin/bash
#SBATCH --job-name=dag_eval
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family/logs/dageval_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family/logs/dageval_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=all

# Untrained-model eval on the LG DAG family, via Ollama (mirrors exp2's slurm_recovery.sh).
#   sbatch --export=ALL,MODEL=qwen2.5:7b,N=25,STRUCTURES=test slurm_dag_eval.sh
#   sbatch --export=ALL,MODEL=qwen2.5:7b,N=25,STRUCTURES=all  slurm_dag_eval.sh   # full sweep
set -euo pipefail
ROOT="/n/fs/similarity/social_sim/exp3_training_transfer/dag_family"
cd "$ROOT"; mkdir -p logs data/dag_eval

MODEL="${MODEL:-qwen2.5:7b}"
N="${N:-25}"
STRUCTURES="${STRUCTURES:-test}"
# KS uses '_' as separator to survive sbatch --export comma-splitting; converted to ',' below.
KS="${KS:-1_3_5}"
KS="${KS//_/,}"
OLLAMA_BIN="/n/fs/similarity/ollama/bin/ollama"
OLLAMA_MODELS_DIR="/n/fs/similarity/ollama/models"
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 1000)))
ENDPOINT="http://localhost:${OLLAMA_PORT}/v1"

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  Model $MODEL  N $N  Structures $STRUCTURES  Endpoint $ENDPOINT"

OLLAMA_MODELS="$OLLAMA_MODELS_DIR" OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
    "$OLLAMA_BIN" serve > /tmp/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1 && { echo "Ollama ready ($i)"; break; }
    sleep 2
done
OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" "$OLLAMA_BIN" pull "$MODEL" || true

echo "Warming $MODEL into VRAM ..."
OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" OLLAMA_KEEP_ALIVE=-1 \
    timeout 1800 "$OLLAMA_BIN" run "$MODEL" "Reply with: ok" \
    && echo "warmup done" || echo "warmup returned nonzero (continuing)"

python3 run_dag_eval.py --backend ollama --model "$MODEL" --endpoint "$ENDPOINT" \
    --structures "$STRUCTURES" --n "$N" --ks "$KS" \
    --temperature 0.1 --max-tokens "${MAXTOK:-512}" --delay 0.02 "$@"

echo "Done — results in $ROOT/data/dag_eval/"
