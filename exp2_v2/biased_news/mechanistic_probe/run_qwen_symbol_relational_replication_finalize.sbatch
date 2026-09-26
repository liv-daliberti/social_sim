#!/bin/bash
#SBATCH --job-name=cc_qwen_rel_final
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_rel_final_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_rel_final_%j.err
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00

set -euo pipefail
: "${RUN_NAME:?RUN_NAME is required}"
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
DATA="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe"
RUN_DIR="$DATA/$RUN_NAME"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
exec 9>"$RUN_DIR/.finalize.lock"
flock -n 9 || { echo "$RUN_NAME final analysis already owned"; exit 0; }
if [[ -f "$RUN_DIR/relational_probe_results.json" ]]; then
  echo "$RUN_NAME relational result already complete"
  exit 0
fi
for name in symbol_relational_common.py extract_symbol_label_states.py \
  analyze_symbol_relational_probe.py patch_symbol_label_activations.py \
  freeze_symbol_relational_protocol.py; do
  expected=$(jq -r --arg name "$name" '.code_sha256[$name]' \
    "$RUN_DIR/relational_protocol_manifest.json")
  observed=$(sha256sum "$MECH/$name" | cut -d' ' -f1)
  [[ "$observed" == "$expected" ]] || {
    echo "frozen code hash mismatch: $name" >&2
    exit 1
  }
done
"$PY" "$MECH/analyze_symbol_relational_probe.py" \
  --run-dir "$RUN_DIR" --mode final --permutations 10000
echo "$RUN_NAME prospective relational replication complete"
