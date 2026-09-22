"""Finalize all predeclared robustness models, retaining incomplete arms."""
import argparse
import json
from pathlib import Path
import subprocess
from . import run_local as common
from .run_robustness_group import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--submission',type=Path,required=True);a=p.parse_args()
    s=json.loads(a.submission.read_text());verify(s)
    report={'status':'finalizing','created_at':common.utc_now(),'submission_sha256':common.file_sha256(a.submission),'models':{},'frontier_api_calls':0,'new_human_review_requested':False,'expected_records':6720}
    for name,g in s['groups'].items():
        verify(s);command=list(g['analysis_command']);absent=[]
        for task in ('probability','direction'):
            path=Path(g[task+'_responses'])
            if not path.exists():
                absent.append(task);empty=a.submission.parent/'missing_responses'/f'{name}_{task}.jsonl'
                if empty.exists() and empty.stat().st_size:raise ValueError('Nonempty missing placeholder')
                common.atomic_write(empty,'');command[command.index('--'+task+'-responses')+1]=str(empty)
        result=subprocess.run(command,text=True,capture_output=True,check=False)
        item={'analysis_returncode':result.returncode,'missing_response_files':absent,'analysis_stdout':result.stdout,'analysis_stderr':result.stderr,'job_id':g['job_id']}
        if result.returncode==0:
            path=Path(g['results_dir'])/'summary.json';item.update(summary_path=str(path),summary_sha256=common.file_sha256(path),metrics=json.loads(path.read_text()))
        report['models'][name]=item
    report['record_complete']=all(m.get('metrics',{}).get('record_complete',False) for m in report['models'].values())
    report['all_valid']=all(m.get('metrics',{}).get('all_valid',False) for m in report['models'].values())
    report['analysis_complete']=all(m['analysis_returncode']==0 for m in report['models'].values())
    report['status']='complete' if report['record_complete'] and report['all_valid'] else 'complete_with_errors' if report['record_complete'] else 'incomplete'
    common.atomic_write(a.submission.parent/'completion.json',json.dumps(report,indent=2)+'\n')
    lines=['# Controlled robustness development','',f"Status: {report['status']}; full results retain all 20 parent families and four variants per model.",'']
    for name,m in report['models'].items():
        lines += [f"- {name}: analysis return code {m['analysis_returncode']}; summary {m.get('summary_path','unavailable')}."]
    common.atomic_write(a.submission.parent/'summary.md','\n'.join(lines)+'\n');print(json.dumps({k:report[k] for k in ('status','record_complete','all_valid','analysis_complete')}))
    if not report['analysis_complete']:raise SystemExit(1)

if __name__=='__main__':main()
