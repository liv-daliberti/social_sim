#!/usr/bin/env bash
set -euo pipefail

result_dir="exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/qwen2_5_72b"
model_label="Qwen2.5-72B-Instruct"
expected_rows=1250
complete=1

for arm in abc_no_context abc_context abc_symbol_context; do
    response_file="${result_dir}/responses_${model_label}_${arm}.jsonl"
    if [[ ! -f "${response_file}" ]] || [[ "$(wc -l < "${response_file}")" -ne "${expected_rows}" ]]; then
        complete=0
    fi
done

if [[ "${complete}" -eq 1 ]]; then
    echo "All three 72B arms already have ${expected_rows} rows; guarded continuation has nothing to do."
    exit 0
fi

. .runtime/oat_env.sh
export COIN_CITY_TP=8
python -c 'import os,runpy,sys,vllm; base=vllm.LLM; vllm.LLM=lambda *a,**kw: base(*a,**(kw|{"tensor_parallel_size":int(os.environ["COIN_CITY_TP"])})); sys.argv=["run_coin_city_symbol_context_open_model.py","--model","Qwen/Qwen2.5-72B-Instruct","--model-label","Qwen2.5-72B-Instruct","--outdir","exp2_v2/biased_news/local_results/symbol_context_model_comparison_20260824/qwen2_5_72b","--gpu-memory-utilization","0.90"]; runpy.run_path("exp2_v2/biased_news/eval/run_coin_city_symbol_context_open_model.py",run_name="__main__")'
