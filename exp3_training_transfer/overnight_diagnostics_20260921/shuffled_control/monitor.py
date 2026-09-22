#!/usr/bin/env python3
"""Read-only status for the exact twenty-cell same-hardware campaign."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
from audit_checkpoints import EXPECTED,effective_args
from launch import ROOT,verify

ROSTERS=[('control','31445281',[2]),('control','31445452',[0,1,3,4,5,6,7,8,9]),
         ('recovery','31445453',[0,1,2,3,4]),('recovery','31445454',[5]),('additional','31445455',[0,1,2,3])]

def main():
    verify()
    ids=','.join(jid for _,jid,_ in ROSTERS)
    queue=subprocess.check_output(['squeue','-h','-r','-j',ids,'-o','%i|%T|%M|%P|%N|%R'],text=True)
    states={line.split('|')[0]:line.split('|') for line in queue.splitlines() if line.strip()}
    reports=[]
    for kind,jid,indices in ROSTERS:
        roster=(json.loads((ROOT/'additional_matched_roster.json').read_text())['runs'] if kind=='additional'
                else json.loads((ROOT/f'{kind}_roster.json').read_text()))
        for index in indices:
            row=roster[index];key=f'{jid}_{index}';entry={'job_id':key,'kind':kind,'disclosure':row['disclosure'],'seed':row['seed']}
            state=states.get(key)
            if state:entry.update(state=state[1],elapsed=state[2],partition=state[3],node=state[4],reason=state[5])
            else:
                entry['state']='NOT_IN_QUEUE'
                entry['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j',key,'--format=JobID,State,Elapsed,ExitCode'],text=True).strip()
            path=ROOT/'logs'/f'{kind}_{jid}_{index}.out';err=path.with_suffix('.err')
            text=path.read_text(errors='replace') if path.is_file() else ''
            error_text=err.read_text(errors='replace') if err.is_file() else ''
            entry['log_mtime']=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat() if path.is_file() else None
            rounds=re.findall(r"'train/learning_round': ([0-9.]+)",text);sgd=re.findall(r"'misc/policy_sgd_step': ([0-9.]+)",text)
            entry['last_learning_round']=float(rounds[-1]) if rounds else None
            entry['last_optimizer_step']=float(sgd[-1]) if sgd else None
            entry['first_update_verified']=bool(rounds and sgd and float(rounds[-1])>=1 and float(sgd[-1])>=8)
            entry['errors']=[v for v in ['Traceback (most recent call last)','CUDA out of memory','AssertionError:',
                'No available memory for the cache blocks','vllm cannot load the model'] if v in text or v in error_text]
            entry['actor_ratio_076_verified']='"VLLM_RATIO": "0.76"' in text
            entry['a5000_verified']='NVIDIA RTX A5000' in text
            if path.is_file():
                args,_=effective_args(path)
                entry['runtime_arg_drift']={k:{'expected':v,'actual':args[k]} for k,v in EXPECTED.items() if k in args and args[k]!=v}
                entry['runtime_args_verified']=len(set(EXPECTED)&set(args))
                assert not entry['runtime_arg_drift'],(key,entry['runtime_arg_drift'])
            reports.append(entry)
    assert len(reports)==20
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'source_freeze_verified':True,'jobs':reports}
    (ROOT/'monitor_status.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    counts={}
    for r in reports:counts[r['state']]=counts.get(r['state'],0)+1
    print(json.dumps({'checked_at':result['checked_at'],'states':counts,'verified_first_updates':sum(r['first_update_verified'] for r in reports),
                      'runtime_arg_checks':sum(r.get('runtime_args_verified',0)>=40 for r in reports),
                      'errors':[{k:r[k] for k in ('job_id','state','errors')} for r in reports if r['errors']]},indent=2))

if __name__=='__main__':main()
