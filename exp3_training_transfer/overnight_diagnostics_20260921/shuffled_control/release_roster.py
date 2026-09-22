#!/usr/bin/env python3
"""Release the fixed19 remaining cells only after a real actor+learner update."""
import argparse
from datetime import datetime,timezone
import json
import re
import subprocess
import sys
from launch import ROOT,verify,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');args=p.parse_args();verify()
    canary=ROOT/'logs/control_31445281_2.out'
    raw=canary.read_text(errors='replace') if canary.is_file() else ''
    rounds=[float(v) for v in re.findall(r"'train/learning_round': ([0-9.]+)",raw)]
    steps=[float(v) for v in re.findall(r"'misc/policy_sgd_step': ([0-9.]+)",raw)]
    passed=bool(rounds and steps and max(rounds)>=1 and max(steps)>=8)
    print(json.dumps({'canary':'31445281_2','learning_round':max(rounds,default=0),'optimizer_steps':max(steps,default=0),'passed':passed}),flush=True)
    if not passed:raise SystemExit(3)
    if not args.submit:return
    if (ROOT/'released_roster.json').is_file():raise SystemExit('Roster already released')
    queued=subprocess.check_output(['squeue','-h','-r','-j','31445018','-o','%i|%T'],text=True)
    states=[line.split('|')[1] for line in queued.splitlines() if line.strip()]
    if any(state!='PENDING' for state in states):raise SystemExit('Original hardware task started; inspect before replacement')
    evidence={'released_at':datetime.now(timezone.utc).isoformat(),'canary_job':'31445281_2',
              'max_learning_round':max(rounds),'max_optimizer_steps':max(steps),'canary_log_sha256':sha(canary),
              'previous_pending_scheduler_rows':queued,'numerical_parameters_changed':False,
              'primary_roster':'10shuffled+10matched,allA5000,matchingactorallocation0.76'}
    (ROOT/'release_gate.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if states:subprocess.run(['scancel','31445018'],check=True)
    selections=[('control','0,1,3,4,5,6,7,8,9','submit_alternate.py','allcs'),
                ('recovery','0,1,2,3,4','submit_alternate.py','mltheory'),
                ('recovery','5','submit_alternate.py','allcs'),
                ('additional','0,1,2,3','submit_additional.py',None)]
    released=[]
    for kind,indices,script,account in selections:
        wanted=[int(v) for v in indices.split(',')]
        existing=[]
        for path in (ROOT/'runs').glob(f'{kind}_a5000_*.json'):
            row=json.loads(path.read_text())
            if row['indices']==wanted:existing.append(row)
        if existing:
            released.append({'kind':kind,'existing_job_id':existing[-1]['job_id']});continue
        cmd=[sys.executable,str(ROOT/script),'--kind',kind,'--indices',indices,'--submit']
        if account:cmd += ['--account',account]
        out=subprocess.check_output(cmd,text=True)
        print(out,flush=True);released.append({'kind':kind,'submission_output':out})
    (ROOT/'released_roster.json').write_text(json.dumps({'gate':evidence,'releases':released},indent=2)+'\n')

if __name__=='__main__':main()
