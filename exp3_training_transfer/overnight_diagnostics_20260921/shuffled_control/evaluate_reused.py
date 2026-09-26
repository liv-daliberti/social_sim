#!/usr/bin/env python3
"""Matched seeds45/46 evaluated exactly like the new shuffled checkpoints."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from launch import ROOT,REPO,sha,verify

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int);parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    verify()
    roster=[r for r in json.loads((ROOT/'checkpoint_audit.json').read_text())['checkpoints']
            if r['arm']=='causal_family' and r['reusable_on_fresh_eval']]
    assert len(roster)==4
    if args.index is None:
        command=['sbatch','--parsable','--account=mltheory','--partition=all','--qos=none','--gres=gpu:a5000:1',
                 '--cpus-per-task=8','--mem=64G','--time=03:00:00','--exclude=node206','--array=0-3%4',
                 '--job-name=mech_fresh_matched',f'--output={ROOT}/logs/reused_%A_%a.out',
                 f'--error={ROOT}/logs/reused_%A_%a.err',str(ROOT/'evaluate_reused.sh')]
        print(command,flush=True)
        if args.submit:
            if list((ROOT/'runs').glob('reused_*.json')):raise SystemExit('Already submitted reused evals')
            out=subprocess.check_output(command,text=True,cwd=REPO).strip();jobid=out.split(';')[0]
            scheduler=subprocess.check_output(['scontrol','show','job',jobid,'-o'],text=True)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            (ROOT/'runs'/f'reused_{stamp}.json').write_text(json.dumps({'job_id':jobid,'command':command,'roster':roster,
                'scheduler_record':scheduler,'freeze_sha256':sha(ROOT/'freeze.json')},indent=2)+'\n');print(jobid)
        return
    row=roster[args.index];disclosure=row['disclosure'];seed=row['seed'];frozen=ROOT/'frozen_runtime'
    out=ROOT/'reports'/f'reused_{disclosure}_causal_family_qwen3_4b_s{seed}';out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(frozen/'evaluate_endpoint.py'),'--model','Qwen/Qwen3-4B-Instruct-2507',
        '--adapter',row['adapter'],'--data',str(ROOT.parent/'evidence_use/data'/f'heldout_{disclosure}'),
        '--template','biased_news','--structured-output','forecast_array','--temperature','0','--n','1',
        '--seed',str(seed+20260814),'--max-tokens','192','--max-model-len','3072',
        '--output',str(out/'fresh_greedy.json'),'--secondary-output',str(out/'fresh_stochastic_n5.json'),
        '--secondary-temperature','0.7','--secondary-n','5'],check=True,cwd=REPO)
    for name in ('fresh_greedy','fresh_stochastic_n5'):
        subprocess.run([sys.executable,str(frozen/'report.py'),'score',str(out/f'{name}.json'),
            '--model','qwen3_4b','--disclosure',disclosure,'--arm','causal_family','--seed',str(seed)],check=True)

if __name__=='__main__':main()
