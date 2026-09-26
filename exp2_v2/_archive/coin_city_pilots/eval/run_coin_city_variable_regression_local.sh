#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
RUN_ROOT="$PROJECT_ROOT/data/coin_city_variable_regression_v1"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
CACHE_DIR="$RUN_ROOT/cache"
MODELS=("claude-opus-4-8" "DeepSeek-V4-Pro" "gpt-5.4")
ARMS=("baseline" "abc_no_context" "abc_context")
PLANNED_CALLS=3150
HARD_CALL_CAP=5000

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/responses" "$LOG_DIR" "$PID_DIR" \
    "$CACHE_DIR/tmp" "$CACHE_DIR/matplotlib" "$RUN_ROOT/live"
export TMPDIR="$CACHE_DIR/tmp"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
if (( PLANNED_CALLS != 3150 || PLANNED_CALLS > HARD_CALL_CAP )); then
    echo "variable-regression call cap invariant failed" >&2
    exit 1
fi
if [[ -f "$PID_DIR/controller.pid" ]]; then
    existing="$(<"$PID_DIR/controller.pid")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
        echo "variable-regression controller already running pid=$existing" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$PID_DIR/controller.pid"
pids=()
monitor=""
cleanup() {
    if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; fi
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

python3 eval/validate_coin_city_variable_regression.py || exit 1
set -a
. "$ENV_FILE"
set +a
for model in "${MODELS[@]}"; do
    for arm in "${ARMS[@]}"; do
        python3 eval/run_coin_city_variable_regression.py \
            --model "$model" --arm "$arm" --max-attempts 1 --dry-run \
            > "$LOG_DIR/dry_run_${model}_${arm}.json" || exit 1
    done
done
python3 analysis/render_coin_city_variable_regression.py \
    --watch --interval 60 > "$LOG_DIR/live_monitor.log" 2>&1 &
monitor=$!
printf '%s\n' "$monitor" > "$PID_DIR/live_monitor.pid"
labels=()
for model in "${MODELS[@]}"; do
    for arm in "${ARMS[@]}"; do
        label="${model}_${arm}"
        python3 eval/run_coin_city_variable_regression.py \
            --model "$model" --arm "$arm" --temperature 0 --max-tokens 512 \
            --timeout 1200 --max-attempts 1 --delay 0.05 \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        pids+=("$pid")
        labels+=("$label")
        printf '%s\n' "$pid" > "$PID_DIR/${label}.pid"
        echo "started $label pid=$pid"
    done
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "completed ${labels[$index]}"
    else
        echo "failed ${labels[$index]}" >&2
        status=1
    fi
done
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
monitor=""
python3 analysis/render_coin_city_variable_regression.py || status=1
printf '%s\n' "$status" > "$RUN_ROOT/controller_exit_status.txt"
trap - INT TERM EXIT
exit "$status"
