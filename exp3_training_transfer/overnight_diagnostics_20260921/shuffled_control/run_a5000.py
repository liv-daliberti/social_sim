#!/usr/bin/env python3
"""Apply only the predeclared equivalent nominal actor-memory allocation."""
import argparse
import json
import os
import subprocess
from launch import ROOT,REPO,verify,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('control','recovery'),required=True);p.add_argument('--index',type=int,required=True);args=p.parse_args()
    verify()
    integrity=json.loads((ROOT/'extra_integrity_manifest.json').read_text())
    for name,expected in integrity['file_sha256'].items():
        assert sha(name)==expected,f'source dataset or model reference changed: {name}'
    row=json.loads((ROOT/f'{args.kind}_roster.json').read_text())[args.index]
    env=os.environ.copy();env.update(row['environment']);env.pop('VLLM_SLEEP',None)
    assert env['VLLM_RATIO']=='0.38';assert env['COLLOCATE']=='0';assert env['GPUS']=='2'
    env['VLLM_RATIO']='0.76'
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True)
    log={'array_id':env.get('SLURM_ARRAY_JOB_ID'),'array_index':args.index,'effective_environment':{k:env[k] for k in row['environment']},
         'device_info':devices,'nominal_actor_memory_gb':24*.76,'frozen_nominal_actor_memory_gb':48*.38,
         'hardware_amendment_sha256':sha(ROOT/'HARDWARE_AMENDMENT.md'),'memory_amendment_sha256':sha(ROOT/'MEMORY_AMENDMENT.md'),
         'freeze_sha256':sha(ROOT/'freeze.json'),'run_wrapper_sha256':sha(__file__)}
    print(json.dumps(log,indent=2,sort_keys=True),flush=True)
    subprocess.run(['bash',str(ROOT/'train.sh')],env=env,cwd=REPO,check=True)

if __name__=='__main__':main()
