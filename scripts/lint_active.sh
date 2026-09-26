#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for command in black flake8; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: $command is required; install requirements-dev.txt" >&2
    exit 1
  fi
done

is_active_path() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  case "$path" in
    .runtime/*|paper/ICLR/*|*/_archive/*|*/data/*|*/logs/*|*/reports/*|*/results/*|*/runs/*)
      return 1
      ;;
  esac
  return 0
}

python_files=()
while IFS= read -r path; do
  if is_active_path "$path"; then
    python_files+=("$path")
  fi
done < <(git ls-files --cached --others --exclude-standard '*.py')

shell_files=()
while IFS= read -r path; do
  if is_active_path "$path"; then
    shell_files+=("$path")
  fi
done < <(git ls-files --cached --others --exclude-standard '*.sh' '*.sbatch')

# Freeze-registered experiment files intentionally retain their recorded bytes.
# Enforce formatting on review-facing tooling, while applying correctness lint
# (syntax, undefined names, and invalid control flow) to every maintained module.
formatted_python=(
  exp1_prospective/agent/evaluate_clustered_uncertainty.py
  exp1_prospective/agent/tests/test_clustered_uncertainty.py
  hf_release/build_release.py
  hf_release/push_release.py
  hf_release/validate_release.py
  paper/generate_coin_city_appendix_tables.py
  paper/generate_coin_city_split_figures.py
  paper/generate_figures.py
  paper/generate_paper_figures.py
  paper/tests/test_layout.py
  tests/test_repository_boundaries.py
  viewer/__init__.py
  viewer/server.py
)

echo "==> Black formatting (${#formatted_python[@]} review-facing files)"
black --check "${formatted_python[@]}"

echo "==> Flake8 correctness (${#python_files[@]} maintained files)"
flake8 --select=E9,F63,F7,F82 --show-source --statistics "${python_files[@]}"

echo "==> Bash syntax (${#shell_files[@]} maintained files)"
for path in "${shell_files[@]}"; do
  bash -n "$path"
done
