#!/bin/bash
# Seven-seed finalization, chained behind the training array and the endpoint
# resume so nothing has to be driven by hand.
#
# It COMPUTES but does not PUBLISH: it writes results_ext4748.json and
# RESULTS_EXT4748.md and stops. freeze_extension.json says seven-seed estimates
# are "reported alongside, never in place of" the frozen five-seed roster, so
# whether the appendix gains a sentence is a decision for the author, not a
# side effect of a batch job. results.json and analyze.py are never touched.
#SBATCH --job-name=mech_ext_finalize
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
CONTROL=/n/fs/similarity/social_sim/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control
PY=/n/fs/similarity/social_sim/.runtime/oat_conda/bin/python
cd "$CONTROL"

echo "=== cell inventory ==="
missing=0
for disclosure in disclosed undisclosed; do
  for seed in 42 43 44 45 46 47 48; do
    for arm in shuffled_target matched; do
      case "$arm:$seed" in
        matched:47|matched:48) pat="${disclosure}_matched_ext_qwen3_4b_s${seed}_*" ;;
        matched:45|matched:46) pat="${disclosure}_matched_hardware_qwen3_4b_s${seed}_*" ;;
        matched:*)             pat="${disclosure}_matched_recovery_qwen3_4b_s${seed}_*" ;;
        *)                     pat="${disclosure}_shuffled_target_qwen3_4b_s${seed}_*" ;;
      esac
      n=$(ls -d reports/$pat 2>/dev/null | wc -l)
      have=$(ls reports/$pat/fresh_stochastic_n5.scores.jsonl 2>/dev/null | wc -l)
      [ "$have" -ge 1 ] || { echo "  MISSING $disclosure $arm s$seed (dirs=$n)"; missing=$((missing+1)); }
    done
  done
done
echo "  missing cells: $missing"

echo "=== seven-seed contrasts ==="
"$PY" analyze_extension.py || echo "analysis reported incomplete cells; see results_ext4748.json"

echo "=== frozen five-seed roster untouched ==="
"$PY" - <<'PYEOF'
import hashlib, json
from pathlib import Path
r = Path('results.json')
print('  results.json sha:', hashlib.sha256(r.read_bytes()).hexdigest()[:16] if r.exists() else 'ABSENT')
d = json.loads(r.read_text()) if r.exists() else {}
print('  five-seed complete:', d.get('complete'), ' cells:', len(d.get('cells', [])))
PYEOF
echo "finalize_complete"
