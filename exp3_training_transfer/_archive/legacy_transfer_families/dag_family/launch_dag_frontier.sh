#!/bin/bash
# Fan out the UNTRAINED FRONTIER eval across the 3 Exp-2 Azure models.
#   N=12 STRUCTURES=all bash launch_dag_frontier.sh     # roster breadth (default)
#   N=20 STRUCTURES=test bash launch_dag_frontier.sh    # held-out only, cheaper-per-call but denser
set -euo pipefail
ROOT=/n/fs/similarity/social_sim/exp3_training_transfer/dag_family
cd "$ROOT"

N="${N:-12}"
STRUCTURES="${STRUCTURES:-all}"
KS="${KS:-1_3_5}"

MODELS=(gpt-5.4 claude-opus-4-8 DeepSeek-V4-Pro)
for m in "${MODELS[@]}"; do
  jid=$(sbatch --parsable \
        --export=ALL,MODEL="$m",N="$N",STRUCTURES="$STRUCTURES",KS="$KS" \
        slurm_dag_frontier.sh)
  echo "submitted $jid  $m  (n=$N structures=$STRUCTURES ks=$KS)"
done
echo "== 3 frontier eval jobs submitted =="
