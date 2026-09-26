#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
PARENT_ROOT="$PROJECT_ROOT/data/coin_city_llm_pilot_v1"
RUN_ROOT="$PARENT_ROOT/context100_rerun"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
CACHE_DIR="$RUN_ROOT/cache"
MODELS=("claude-opus-4-8" "DeepSeek-V4-Pro" "gpt-5.4")
NEW_CALLS=600
CUMULATIVE_CALLS=2400
HARD_CALL_CAP=5000

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/responses" "$LOG_DIR" "$PID_DIR" "$CACHE_DIR/tmp" "$CACHE_DIR/matplotlib" "$RUN_ROOT/live"
export TMPDIR="$CACHE_DIR/tmp"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
if (( NEW_CALLS != 600 || CUMULATIVE_CALLS > HARD_CALL_CAP )); then
    echo "aligned-context call cap invariant failed" >&2
    exit 1
fi
if [[ -f "$PID_DIR/controller.pid" ]]; then
    existing="$(<"$PID_DIR/controller.pid")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
        echo "aligned-context controller already running pid=$existing" >&2
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

python3 eval/validate_coin_city_context100_rerun.py || exit 1
set -a
. "$ENV_FILE"
set +a
for model in "${MODELS[@]}"; do
    python3 eval/run_coin_city_context100_rerun.py \
        --model "$model" --max-attempts 1 --dry-run \
        > "$LOG_DIR/dry_run_${model}.json" || exit 1
done
python3 analysis/render_coin_city_context100.py --watch --interval 60 \
    --output "$PARENT_ROOT/live/live_structure.png" \
    > "$LOG_DIR/live_monitor.log" 2>&1 &
monitor=$!
printf '%s\n' "$monitor" > "$PID_DIR/live_monitor.pid"
labels=()
for model in "${MODELS[@]}"; do
    label="${model}_abc_context_100pct"
    python3 eval/run_coin_city_context100_rerun.py \
        --model "$model" --temperature 0 --max-tokens 512 \
        --timeout 1200 --max-attempts 1 --delay 0.05 \
        > "$LOG_DIR/${label}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    labels+=("$label")
    printf '%s\n' "$pid" > "$PID_DIR/${label}.pid"
    echo "started $label pid=$pid"
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
python3 analysis/render_coin_city_context100.py \
    --output "$PARENT_ROOT/live/live_structure.png" || status=1
python3 analysis/render_coin_city_context100.py \
    --output "$RUN_ROOT/live/context100_structure.png" || status=1
printf '%s\n' "$status" > "$RUN_ROOT/controller_exit_status.txt"
trap - INT TERM EXIT
exit "$status"
