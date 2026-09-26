#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
PYTHON="/usr/bin/python3.9"
RUN_ROOT="$PROJECT_ROOT/data/coin_city_stable_relationship_claude_n250_v4"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
MODEL="DeepSeek-V4-Pro"
ARMS=("baseline" "abc_no_context" "abc_context")
C_LEVELS=(0 1 2 3 4)
PLANNED_CALLS=3750
HARD_CALL_CAP=5000

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/responses" "$LOG_DIR" "$PID_DIR"
export PYTHONDONTWRITEBYTECODE=1
if (( PLANNED_CALLS != 3750 || PLANNED_CALLS > HARD_CALL_CAP )); then
    echo "DeepSeek replication call-cap invariant failed" >&2
    exit 1
fi
if [[ -z "${DEEPSEEK_V4_PRO_AZURE_API_KEY:-}" ]]; then
    read -r -s -p "DeepSeek project API key: " DEEPSEEK_V4_PRO_AZURE_API_KEY
    printf '\n'
    export DEEPSEEK_V4_PRO_AZURE_API_KEY
fi

controller_file="$PID_DIR/DeepSeek-V4-Pro_controller.pid"
if [[ -f "$controller_file" ]]; then
    existing="$(<"$controller_file")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
        echo "DeepSeek controller already running pid=$existing" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$controller_file"
pids=()
labels=()
echo "execution_mode=local_login_node host=$(hostname) controller=$$"

cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT

"$PYTHON" eval/validate_coin_city_stable_relationship_claude_n250.py || exit 1
for arm in "${ARMS[@]}"; do
    "$PYTHON" eval/run_coin_city_stable_relationship_deepseek_v4_pro.py \
        --model "$MODEL" --arm "$arm" --max-attempts 1 --dry-run \
        > "$LOG_DIR/dry_run_${MODEL}_${arm}.json" || exit 1
done

for arm in "${ARMS[@]}"; do
    for c_level in "${C_LEVELS[@]}"; do
        label="${MODEL}_${arm}_c${c_level}"
        "$PYTHON" eval/run_coin_city_stable_relationship_deepseek_v4_pro.py \
            --model "$MODEL" --arm "$arm" --prefixes "$c_level" \
            --temperature 0 --max-tokens 512 --timeout 1200 \
            --max-attempts 1 --delay 0.05 \
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

for arm in "${ARMS[@]}"; do
    if ! "$PYTHON" eval/run_coin_city_stable_relationship_deepseek_v4_pro.py \
        --model "$MODEL" --arm "$arm" --temperature 0 --max-tokens 512 \
        --timeout 1200 --max-attempts 1 --delay 0.05 \
        > "$LOG_DIR/${MODEL}_${arm}_finalize.log" 2>&1; then
        status=1
    fi
done
"$PYTHON" eval/validate_coin_city_stable_relationship_deepseek_v4_pro.py \
    > "$LOG_DIR/${MODEL}_validation.json" 2>&1 || status=1
printf '%s\n' "$status" > "$RUN_ROOT/deepseek_v4_pro_controller_exit_status.txt"
trap - INT TERM EXIT
exit "$status"
