#!/bin/bash
#SBATCH --job-name=cc_l70_rel_final
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_rel_final_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_rel_final_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/llama3_1_70b_symbol_relational_v1"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
exec 9>"$RUN_DIR/.finalize.lock"
flock -n 9 || { echo "Llama relational final analysis already owned"; exit 0; }
if [[ -f "$RUN_DIR/relational_probe_results.json" ]]; then
  echo "Llama relational result already complete"; exit 0
fi
"$PY" "$MECH/analyze_symbol_relational_probe.py" \
  --run-dir "$RUN_DIR" --mode final --permutations 10000
echo "Llama-3.1-70B prospective relational replication complete"
