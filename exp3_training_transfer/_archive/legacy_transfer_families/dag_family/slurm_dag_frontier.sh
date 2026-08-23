#!/bin/bash
#SBATCH --job-name=dag_front
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family/logs/dagfront_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family/logs/dagfront_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs

# Untrained FRONTIER (Azure Foundry) eval on the LG DAG family. No GPU — API only.
# Mirrors exp2's slurm_recovery_azure.sh: sources the exp1 .env for AZURE_AI_API_KEY,
# then run_dag_eval.py resolves the per-model endpoint (claude -> /anthropic, gpt/deepseek
# -> liv-forecast /openai/v1) and picks the right client. Resumable JSONL.
#   sbatch --export=ALL,MODEL=gpt-5.4,N=12,STRUCTURES=all slurm_dag_frontier.sh
set -euo pipefail
ROOT=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family
cd "$ROOT"; mkdir -p logs data/dag_eval

# AZURE_AI_API_KEY / CLAUDE_AZURE_API_KEY / DEEPSEEK_AZURE_API_KEY
set +u; . /n/fs/similarity/social_sim/exp1_prospective/agent/.env; set -u

MODEL="${MODEL:?set MODEL}"
N="${N:-12}"
STRUCTURES="${STRUCTURES:-all}"
KS="${KS:-1_3_5}"; KS="${KS//_/,}"

echo "Job $SLURM_JOB_ID  Frontier $MODEL  N $N  Structures $STRUCTURES  KS $KS"
python3 run_dag_eval.py --backend azure --azure-auth --model "$MODEL" \
    --structures "$STRUCTURES" --n "$N" --ks "$KS" \
    --temperature 0.1 --max-tokens "${MAXTOK:-1024}" --delay "${DELAY:-0.2}" \
    --timeout "${TIMEOUT:-120}" "$@"
echo "Done — results in $ROOT/data/dag_eval/"
