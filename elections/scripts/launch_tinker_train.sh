#!/usr/bin/env bash
# Launch a Tinker GRPO training run for the elections environment.
# Sources TINKER_API_KEY from the kalshi .env, uses the kalshi tinker_py311_venv.
#
# Usage:
#   cd /n/fs/similarity/social_sim/elections
#   bash scripts/launch_tinker_train.sh [--config configs/elections_smoke.yml] [extra args...]
#
# Override config:
#   bash scripts/launch_tinker_train.sh --config configs/elections_smoke.yml --n-epochs 20
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELECTIONS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

KALSHI_ENV_FILE="${KALSHI_ENV_FILE:-/n/fs/similarity/kalshi/agentic_forecasting/.env}"
TINKER_VENV="${TINKER_VENV:-/n/fs/similarity/social_sim/.runtime/tinker_venv}"
PYTHON_BIN="$TINKER_VENV/bin/python"
TRAIN_SCRIPT="$ELECTIONS_DIR/scripts/run_tinker_elections_grpo.py"
DEFAULT_CONFIG="$ELECTIONS_DIR/configs/elections_smoke.yml"

if [[ ! -f "$KALSHI_ENV_FILE" ]]; then
    echo "ERROR: kalshi .env not found at $KALSHI_ENV_FILE" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: tinker venv Python not found at $PYTHON_BIN" >&2
    exit 1
fi

# Source .env but only export TINKER_API_KEY and WANDB_API_KEY to avoid
# polluting the environment with unrelated kalshi service vars.
_load_key() {
    local key="$1"
    local val
    val="$(grep -E "^${key}=" "$KALSHI_ENV_FILE" | head -1 | cut -d= -f2- || true)"
    if [[ -n "$val" ]]; then
        export "${key}=${val}"
    fi
}
_load_key TINKER_API_KEY
_load_key WANDB_API_KEY

if [[ -z "${TINKER_API_KEY:-}" ]]; then
    echo "WARNING: TINKER_API_KEY not set and not found in $KALSHI_ENV_FILE" >&2
fi

# Default config, overridable by first arg if it's --config
ARGS=("--config" "$DEFAULT_CONFIG")
if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
    ARGS=("--config" "$2")
    shift 2
fi
ARGS+=("$@")

cd "$ELECTIONS_DIR"
echo "Working dir : $ELECTIONS_DIR"
echo "Python      : $PYTHON_BIN  ($("$PYTHON_BIN" --version 2>&1))"
echo "Script      : $TRAIN_SCRIPT"
echo "Args        : ${ARGS[*]}"
echo ""

exec "$PYTHON_BIN" "$TRAIN_SCRIPT" "${ARGS[@]}"
