#!/usr/bin/env python3
"""Fail-closed launcher: dry-run by default; write exact submission ledger."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for mapping in ('source_sha256','runtime_sha256'):
        for path,expected in freeze[mapping].items():
            if sha(path)!=expected: raise RuntimeError(f'frozen file changed: {path}')
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for record in manifest['datasets']:
        path=Path(record['path'])
        for rel,expected in record['files_sha256'].items():
            if sha(path/rel)!=expected: raise RuntimeError(f'frozen dataset changed: {path/rel}')
        if sha(path/'donor_map.json')!=record['donor_map_sha256']:raise RuntimeError(f'donor map changed: {path}')
    return freeze

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=('control','recovery'),default='control')
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--run-cell',choices=('control','recovery'))
    parser.add_argument('--index',type=int)
    parser.add_argument('--indices',help='Infrastructure retry indices, e.g. 0,4; no result-based selection')
    args=parser.parse_args()
    freeze=verify()
    kind=args.run_cell or args.kind
    roster=json.loads((ROOT/f'{kind}_roster.json').read_text())
    if args.run_cell:
        if args.index is None or not 0<=args.index<len(roster):raise SystemExit('Invalid array index')
        row=roster[args.index]
        env=os.environ.copy();env.update(row['environment']);env.pop('VLLM_SLEEP',None)
        print(json.dumps({'job_id':env.get('SLURM_JOB_ID'),'array_id':env.get('SLURM_ARRAY_JOB_ID'),
                          'array_index':args.index,'record':row,'freeze_sha256':sha(ROOT/'freeze.json')},indent=2),flush=True)
        subprocess.run(['bash',str(ROOT/'train.sh')],env=env,cwd=REPO,check=True)
        return
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
    array=args.indices or f'0-{len(roster)-1}'
    command=['sbatch','--parsable','--account=allcs','--partition=cs','--qos=none','--gres=gpu:a6000:2',
             '--cpus-per-task=8','--mem=100G','--time=20:00:00','--exclude=node206',
             f'--nice={1000 if kind=="control" else 2000}',f'--array={array}%{len(roster)}',
             f'--job-name=mech_{"shuffle" if kind=="control" else "restore"}',
             f'--output={ROOT}/logs/{kind}_%A_%a.out',f'--error={ROOT}/logs/{kind}_%A_%a.err',
             str(ROOT/'run_array.sh'),kind]
    print(shlex.join(command),flush=True)
    if not args.submit:return
    prior=list((ROOT/'runs').glob(f'{kind}_*.json'))
    if prior and not args.indices:raise SystemExit('Already submitted: use explicit infrastructure retry indices')
    out=subprocess.check_output(command,cwd=REPO,text=True).strip();job_id=out.split(';')[0]
    if not job_id.isdigit():raise RuntimeError(out)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    scheduler=subprocess.check_output(['scontrol','show','job',job_id,'-o'],text=True)
    ledger={'scheduler_record':scheduler,'submitted_at':stamp,'kind':kind,'job_id':job_id,'command':command,'roster':roster,
            'freeze_sha256':sha(ROOT/'freeze.json'),'indices':args.indices,
            'failure_retries_only':bool(args.indices)}
    (ROOT/'runs'/f'{kind}_{stamp}.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'job_id':job_id,'kind':kind,'count':len(roster),'freeze_sha256':ledger['freeze_sha256']}))

if __name__=='__main__':main()
