#!/bin/bash
#SBATCH --job-name=c3_current_paper
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/logs/finalize_current_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/logs/finalize_current_%j.err
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp3_training_transfer/mechanism_family"
POLY="$REPO/exp3_training_transfer/polymarket"
mkdir -p "$REPO/exp3_training_transfer/logs"

"$PY" "$MECH/make_paper_outputs.py" \
  --ledger "$MECHANISM_LEDGER" \
  --greedy-latex "$REPO/paper/tables/exp3c_structural_ood_greedy.tex" \
  --stochastic-latex "$REPO/paper/tables/exp3c_structural_ood_stochastic.tex" \
  --overall-latex "$REPO/paper/tables/exp3c_structural_ood_overall.tex" \
  --secondary-latex "$REPO/paper/tables/exp3c_structural_ood_secondary.tex"

"$PY" "$POLY/scripts/render_model_extension_table.py" \
  --extension-ledger "$POLY_EXTENSION_LEDGER" \
  --latex "$REPO/paper/tables/exp3b_model_roster_results.tex" \
  --qwen8-main-latex "$REPO/paper/tables/exp3b_qwen3_8b_endpoint_results.tex"

(
  cd "$REPO/paper"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)
echo "Current structural-OOD and Polymarket-roster tables rendered; ICLR PDF compiled"
