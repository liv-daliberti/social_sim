"""Run both predeclared robustness tasks and grouped analysis on one local model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify(submission):
    root=Path(__file__).resolve().parent
    for name,want in submission['code_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:raise ValueError('Frozen code changed: '+name)
    for key in ('probability_design','direction_design'):
        if hashlib.sha256(Path(submission[key]).read_bytes()).hexdigest()!=submission[key+'_sha256']:raise ValueError('Frozen design changed: '+key)
    if hashlib.sha256(Path(submission['protocol_path']).read_bytes()).hexdigest()!=submission['protocol_sha256']:raise ValueError('Frozen protocol changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--submission',type=Path,required=True);p.add_argument('--group',required=True);a=p.parse_args()
    submission=json.loads(a.submission.read_text());group=submission['groups'][a.group];failures=[]
    for task in ('probability','direction'):
        verify(submission)
        print('Starting '+a.group+'/'+task,flush=True)
        result=subprocess.run(group[task+'_command'],check=False)
        if result.returncode:failures.append({'task':task,'returncode':result.returncode})
    verify(submission)
    result=subprocess.run(group['analysis_command'],check=False)
    if result.returncode:failures.append({'task':'analysis','returncode':result.returncode})
    print(json.dumps({'group':a.group,'process_failures':failures}),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
