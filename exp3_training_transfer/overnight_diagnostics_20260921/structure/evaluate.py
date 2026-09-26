#!/usr/bin/env python3
"""One model load; frozen diagnostic rows and sequential existing LoRAs."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EXP3 = ROOT.parents[1]
sys.path.insert(0, str(EXP3 / 'mechanism_family'))
from output_contract import FORECAST_ARRAY_GBNF
from score import write_summary

SELECTION_GRAMMAR = 'root ::= "{" ws "\\\"reference\\\"" ws ":" ws [12] ws "}"\nws ::= [ \\t\\n\\r]*\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--roster', type=Path, default=ROOT / 'roster.json')
    parser.add_argument('--keys', nargs='*')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for filename, expected in manifest['sha256'].items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen artifact differs: {filename}')
    rows = [json.loads(line) for line in (ROOT / 'diagnostic.jsonl').read_text().splitlines()]
    roster = [entry for entry in json.loads(args.roster.read_text()) if entry['model'] == args.model and (not args.keys or entry['key'] in args.keys)]
    roster = [entry for entry in roster if not (args.output_dir / (entry['key'] + '.jsonl')).exists()]
    if not roster:
        print('All requested model outputs already exist.', flush=True)
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams
    has_lora = any(entry.get('adapter') for entry in roster)
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=.88,
              tensor_parallel_size=args.tensor_parallel_size, seed=20260921,
              enable_prefix_caching=True, enable_lora=has_lora, max_lora_rank=32,
              max_loras=1, max_cpu_loras=8, guided_decoding_backend='xgrammar',
              disable_custom_all_reduce=args.tensor_parallel_size > 1)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': row['input']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    assert max(lengths) + 192 <= 4096, max(lengths)
    print(f'Prompt tokens: min={min(lengths)} max={max(lengths)}; models={len(roster)}', flush=True)
    params = [SamplingParams(temperature=0, top_p=1, n=1, max_tokens=32 if row['reference']['interface'] == 'selection_only' else 192,
              seed=20260921, guided_decoding=GuidedDecodingParams(grammar=SELECTION_GRAMMAR if row['reference']['interface'] == 'selection_only' else FORECAST_ARRAY_GBNF, backend='xgrammar')) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_index, entry in enumerate(roster, 1):
        print(f'Beginning {entry["key"]} at {datetime.now(timezone.utc).isoformat()}', flush=True)
        adapter = entry.get('adapter')
        if adapter and entry.get('adapter_weights_sha256'):
            assert hashlib.sha256((Path(adapter) / 'adapter_model.safetensors').read_bytes()).hexdigest() == entry['adapter_weights_sha256']
        request = LoRARequest(entry['key'], model_index, adapter) if adapter else None
        generated = llm.generate(prompts, params, lora_request=request)
        records = [{**row, 'output': result.outputs[0].text, 'model_key': entry['key'],
                    'model': args.model, 'adapter': adapter, 'temperature': 0, 'decode_seed': 20260921,
                    'finish_reason': result.outputs[0].finish_reason} for row, result in zip(rows, generated)]
        output = args.output_dir / (entry['key'] + '.jsonl')
        temporary = output.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(record, sort_keys=True) + '\n' for record in records))
        temporary.replace(output)
        write_summary(output)
        print(f'Completed {entry["key"]} at {datetime.now(timezone.utc).isoformat()}', flush=True)

if __name__ == '__main__':
    main()
