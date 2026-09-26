#!/usr/bin/env bash
set -euo pipefail

project_root="/n/fs/similarity/social_sim"
result_root="$project_root/exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824"
runner="$project_root/exp2_v2/biased_news/eval/run_coin_city_symbol_context.py"
python_bin="/usr/local/anaconda3/2024.02/bin/python"
model="FW-Kimi-K3"
arm="abc_symbol_context"
shard_root="$result_root/kimi_k3_shards"
final_root="$result_root/hosted"
log_root="$result_root/logs"
final_file="$final_root/responses_${model}_${arm}.jsonl"
temporary_file="$final_file.pending"

if [[ -z "${AZURE_AI_API_KEY:-}" ]]; then
    echo "AZURE_AI_API_KEY is required" >&2
    exit 1
fi
export PYTHONPATH="/tmp/coin_city_kimi_sdk${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$final_file" ]]; then
    echo "Refusing to overwrite existing final response file: $final_file" >&2
    exit 1
fi

mkdir -p "$shard_root" "$final_root" "$log_root"
pids=()
for prefix in 0 1 2 3 4; do
    shard_dir="$shard_root/k${prefix}"
    mkdir -p "$shard_dir"
    "$python_bin" "$runner" \
        --model "$model" \
        --arm "$arm" \
        --prefixes "$prefix" \
        --outdir "$shard_dir" \
        --temperature 0 \
        --max-tokens 512 \
        --timeout 1200 \
        --max-attempts 1 \
        --delay 0.05 \
        > "$log_root/${model}_${arm}_c${prefix}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "At least one Kimi shard failed; final artifact was not assembled" >&2
    exit "$status"
fi

: > "$temporary_file"
for prefix in 0 1 2 3 4; do
    shard_file="$shard_root/k${prefix}/responses_${model}_${arm}.jsonl"
    if [[ "$(wc -l < "$shard_file")" -ne 250 ]]; then
        echo "Unexpected record count in $shard_file" >&2
        exit 1
    fi
    jq -e . "$shard_file" >/dev/null
    /bin/cat "$shard_file" >> "$temporary_file"
done
if [[ "$(wc -l < "$temporary_file")" -ne 1250 ]]; then
    echo "Merged Kimi file does not contain 1,250 records" >&2
    exit 1
fi
if [[ "$(jq -r .task_id "$temporary_file" | sort -u | wc -l)" -ne 1250 ]]; then
    echo "Merged Kimi file does not contain 1,250 unique task IDs" >&2
    exit 1
fi
mv "$temporary_file" "$final_file"

"$python_bin" "$runner" \
    --model "$model" \
    --arm "$arm" \
    --outdir "$final_root" \
    --temperature 0 \
    --max-tokens 512 \
    --timeout 1200 \
    --max-attempts 1 \
    --delay 0.05 \
    > "$log_root/${model}_${arm}_finalize.log" 2>&1

echo "Completed $final_file"
