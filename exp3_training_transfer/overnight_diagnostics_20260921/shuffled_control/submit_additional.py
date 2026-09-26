#!/usr/bin/env python3
"""Operational A5000 rescheduling under the documented hardware amendment."""
import argparse
from datetime import datetime,timezone
import json
import subprocess
from launch import ROOT,REPO,verify,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('additional',),required=True)
    p.add_argument('--indices',required=True);p.add_argument('--submit',action='store_true');args=p.parse_args()
    verify()
    roster=json.loads((ROOT/'additional_matched_roster.json').read_text())['runs']
    selected=[int(v) for v in args.indices.split(',')]
    assert all(0<=v<len(roster) for v in selected)
    cmd=['sbatch','--parsable','--account=allcs','--partition=cs','--qos=none',
         '--gres=gpu:a5000:2','--cpus-per-task=8','--mem=100G','--time=30:00:00','--exclude=node206',
         f'--nice={1000 if args.kind=="control" else 2000}',f'--array={args.indices}%{len(roster)}',
         f'--job-name=mech_{"shuffle" if args.kind=="control" else "restore"}_a5',
         f'--output={ROOT}/logs/{args.kind}_%A_%a.out',f'--error={ROOT}/logs/{args.kind}_%A_%a.err',
         str(ROOT/'run_array_additional.sh'),args.kind]
    print(cmd,flush=True)
    if args.submit:
        out=subprocess.check_output(cmd,text=True,cwd=REPO).strip();jobid=out.split(';')[0]
        scheduler=subprocess.check_output(['scontrol','show','job',jobid,'-o'],text=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        ledger={'job_id':jobid,'kind':args.kind,'indices':selected,'roster':[roster[v] for v in selected],
                'command':cmd,'scheduler_record':scheduler,'freeze_sha256':sha(ROOT/'freeze.json'),
                'hardware_amendment_sha256':sha(ROOT/'HARDWARE_AMENDMENT.md'),'submitter_sha256':sha(__file__),
                'submitted_at':stamp,'numerical_training_settings_changed':False,'vllm_gpu_ratio':0.76,'memory_amendment_sha256':sha(ROOT/'MEMORY_AMENDMENT.md'),'run_wrapper_sha256':sha(ROOT/'run_additional.py')}
        (ROOT/'runs'/f'{args.kind}_a5000_{stamp}.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
        print(jobid)

if __name__=='__main__':main()
