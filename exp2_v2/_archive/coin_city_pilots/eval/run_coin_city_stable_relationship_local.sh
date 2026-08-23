#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
PYTHON="/usr/bin/python3.9"
RUN_ROOT="$PROJECT_ROOT/data/coin_city_stable_relationship_v2"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
CACHE_DIR="$RUN_ROOT/cache"
MODELS=("claude-opus-4-8" "DeepSeek-V4-Pro" "gpt-5.4")
ARMS=("baseline" "abc_no_context" "abc_context")
PLANNED_CALLS=2250
HARD_CALL_CAP=5000

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/responses" "$LOG_DIR" "$PID_DIR" \
    "$CACHE_DIR/tmp" "$CACHE_DIR/matplotlib" "$RUN_ROOT/live"
export TMPDIR="$CACHE_DIR/tmp"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
if (( PLANNED_CALLS != 2250 || PLANNED_CALLS > HARD_CALL_CAP )); then
    echo "stable-response call-cap invariant failed" >&2
    exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
    echo "missing API environment file: $ENV_FILE" >&2
    exit 1
fi
if [[ -f "$PID_DIR/controller.pid" ]]; then
    existing="$(<"$PID_DIR/controller.pid")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
        echo "stable-response controller already running pid=$existing" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$PID_DIR/controller.pid"
pids=()
echo "execution_mode=local_login_node host=$(hostname) controller=$$"
monitor=""
cleanup() {
    if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; fi
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

"$PYTHON" eval/validate_coin_city_stable_relationship.py || exit 1
set -a
. "$ENV_FILE"
set +a
for model in "${MODELS[@]}"; do
    for arm in "${ARMS[@]}"; do
        "$PYTHON" eval/run_coin_city_stable_relationship.py \
            --model "$model" --arm "$arm" --max-attempts 1 --dry-run \
            > "$LOG_DIR/dry_run_${model}_${arm}.json" || exit 1
    done
done
(
    while true; do
        "$PYTHON" analysis/render_coin_city_stable_relationship.py || true
        sleep 60
    done
) > "$LOG_DIR/live_monitor.log" 2>&1 &
monitor=$!
printf '%s\n' "$monitor" > "$PID_DIR/live_monitor.pid"

labels=()
for model in "${MODELS[@]}"; do
    for arm in "${ARMS[@]}"; do
        label="${model}_${arm}"
        "$PYTHON" eval/run_coin_city_stable_relationship.py \
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
"$PYTHON" analysis/render_coin_city_stable_relationship.py || status=1
printf '%s\n' "$status" > "$RUN_ROOT/controller_exit_status.txt"
trap - INT TERM EXIT
exit "$status"
