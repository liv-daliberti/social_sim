#!/usr/bin/env python3
"""One base load; frozen-checkpoint sequential greedy inference, resumable by arm."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
FAMILY=REPO/'exp3_training_transfer/mechanism_family'
sys.path.insert(0,str(FAMILY))
from output_contract import FORECAST_ARRAY_GBNF
from evaluate_endpoint import formatted_prompts

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--disclosure',required=True,choices=['disclosed','undisclosed'])
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--checkpoint-id',action='append')
    parser.add_argument('--suites',nargs='+',choices=['gain_pairs','heldout'],default=['gain_pairs','heldout'])
    args=parser.parse_args()
    manifest_path=args.data/'frozen_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    inventory_path=args.data/'checkpoint_inventory.json'
    assert sha(inventory_path)==manifest['checkpoint_inventory_sha256']
    inventory=json.loads(inventory_path.read_text())
    checkpoints=[c for c in inventory['available'] if c['disclosure']==args.disclosure and
                 (not args.checkpoint_id or c['id'] in args.checkpoint_id)]
    if not checkpoints: raise SystemExit('No checkpoint selected')
    todo=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for c in checkpoints:
        if c['adapter']:
            assert sha(Path(c['adapter'])/'adapter_model.safetensors')==c['adapter_sha256']
        for suite in args.suites:
            dataset=manifest['artifacts'][f'{suite}_{args.disclosure}']
            assert sha(dataset['jsonl'])==dataset['sha256']
            output=args.output/f'{c["id"]}_{suite}.json'
            if output.exists():
                saved=json.loads(output.read_text())
                assert saved['manifest_sha256']==sha(manifest_path) and saved['dataset_sha256']==dataset['sha256']
                assert saved['checkpoint']==c and len(saved['records'])==dataset['rows']
                print(f'complete existing {output}',flush=True)
            else: todo.append((c,suite,dataset,output))
    if not todo: return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams
    llm=LLM(model=inventory['model'],max_model_len=3072,gpu_memory_utilization=.82,
            tensor_parallel_size=1,seed=20260921,enable_prefix_caching=True,
            enable_lora=True,max_lora_rank=32,max_loras=1,max_cpu_loras=8,
            guided_decoding_backend='xgrammar')
    tokenizer=llm.get_tokenizer()
    sampling=SamplingParams(temperature=0,top_p=1,n=1,max_tokens=192,seed=20260921,
            guided_decoding=GuidedDecodingParams(grammar=FORECAST_ARRAY_GBNF,backend='xgrammar'))
    lora_ids={c['id']:i+1 for i,c in enumerate(checkpoints)}
    for c,suite,dataset,output in todo:
        start=time.time()
        rows=[json.loads(line) for line in Path(dataset['jsonl']).read_text().splitlines()]
        prompts=formatted_prompts(tokenizer,[r['input'] for r in rows],'biased_news')
        lengths=[len(tokenizer.encode(p)) for p in prompts]
        if max(lengths)+192>3072: raise ValueError(f'Prompt would truncate: max length={max(lengths)}')
        request=LoRARequest(c['id'],lora_ids[c['id']],c['adapter']) if c['adapter'] else None
        generated=llm.generate(prompts,sampling,lora_request=request)
        records=[dict(reference=row['reference'],output=[p.text for p in result.outputs])
                 for row,result in zip(rows,generated)]
        payload=dict(protocol=manifest['protocol'],manifest_sha256=sha(manifest_path),dataset_sha256=dataset['sha256'],
            model=inventory['model'],checkpoint=c,suite=suite,temperature=0,n=1,decode_seed=20260921,
            structured_output='forecast_array',prompt_template='biased_news',max_tokens=192,max_model_len=3072,
            max_prompt_tokens=max(lengths),records=records,elapsed_seconds=time.time()-start,
            completed_at=datetime.now(timezone.utc).isoformat(),runner_sha256=sha(__file__))
        temporary=output.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,sort_keys=True)+'\n')
        temporary.replace(output)
        print(f'COMPLETE {c["id"]} {suite}: {len(rows)} prompts in {time.time()-start:.1f}s -> {output}',flush=True)
    print('ALL_REQUESTED_CHECKPOINTS_COMPLETE',flush=True)

if __name__=='__main__': main()
