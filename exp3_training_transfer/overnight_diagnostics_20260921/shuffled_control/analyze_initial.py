#!/usr/bin/env python3
"""Summarize fixed fresh paired contrasts; fail closed on incomplete or duplicate cells."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SEEDS=(42,43,44,45,46)
TCRIT={1:12.706205,2:4.302653,3:3.182446,4:2.776445}
WEBB=np.array([-math.sqrt(1.5),-1.,-math.sqrt(.5),math.sqrt(.5),1.,math.sqrt(1.5)])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize_values(differences,metadata,selected_tasks):
    """Each independent world/trajectory remains bundled across its evidence depths."""
    seeds=sorted(differences)
    array=np.asarray([[differences[s][t] for t in selected_tasks] for s in seeds])
    means=array.mean(axis=1);estimate=float(means.mean());n=len(seeds)
    out={'seed_effects':dict(zip(map(str,seeds),map(float,means))),'effect':estimate,'training_seeds':n,
         'tasks_per_seed':len(selected_tasks),'complete_five_seed_roster':n==5}
    if n<2:return out
    sd=float(means.std(ddof=1));half=TCRIT[n-1]*sd/math.sqrt(n)
    out['student_t_95ci']=[estimate-half,estimate+half]
    positives=int((means>0).sum())
    out['sign_test']={'positive':positives,'seeds':n,'p_one_sided':sum(math.comb(n,k) for k in range(positives,n+1))/2**n}
    rng=np.random.default_rng(20260818)
    star=rng.choice(WEBB,size=(9999,n))*means
    star_sd=star.std(axis=1,ddof=1)
    if sd==0:
        wild_p=1. if estimate==0 else float((np.abs(star.mean(axis=1)[star_sd==0])>=abs(estimate)).sum()+1)/10000
    else:
        with np.errstate(divide='ignore',invalid='ignore'):
            tstar=star.mean(axis=1)/(star_sd/math.sqrt(n))
        tstar=tstar[np.isfinite(tstar)]
        wild_p=float((np.abs(tstar)>=abs(estimate/(sd/math.sqrt(n)))).sum()+1)/(len(tstar)+1)
    out['wild_cluster_webb_p_two_sided']=wild_p
    # Resample training seeds, and within each selected seed sample60 independent
    # trajectory ids inside each world, retaining all k values of each trajectory.
    groups=defaultdict(lambda:defaultdict(list))
    for i,task in enumerate(selected_tasks):
        meta=metadata[task];groups[meta['world']][meta['seed']].append(i)
    indices=[list(world.values()) for world in groups.values()]
    rng=np.random.default_rng(20260818);draws=[]
    for _ in range(5000):
        sample_means=[]
        for selected_seed in rng.integers(n,size=n):
            strata=[]
            for clusters in indices:
                choices=rng.integers(len(clusters),size=len(clusters))
                selected=np.concatenate([clusters[i] for i in choices])
                strata.append(float(array[selected_seed,selected].mean()))
            sample_means.append(float(np.mean(strata)))
        draws.append(float(np.mean(sample_means)))
    out['hierarchical_seed_trajectory_95ci']=np.quantile(draws,[.025,.975]).tolist()
    return out

def read_complete(path,expected,draws):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups=defaultdict(list)
    for row in rows:groups[row['task_id']].append(row)
    assert set(groups)==set(expected),f'Wrong fresh tasks in {path}'
    assert all(len(v)==draws for v in groups.values()),f'Incomplete draws in {path}'
    assert all(len({r['draw'] for r in v})==draws for v in groups.values()),f'Duplicate draws in {path}'
    return groups

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    report={'created_at':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(ROOT/'freeze.json'),
            'analysis_code_sha256':sha(__file__),'primary':'stochastic5 shuffled minus matched response MAE',
            'incomplete_cells':[],'cells':[],'contrasts':[],'sources':{},'notes':[
              'Five training seeds are the target replication count; decoding draws are averaged within task.',
              'Trajectory seeds remain clustered across k=3,6,9 in the bootstrap.',
              'Recovery seeds42-44 use newly restored matched checkpoints; matched45-46 are reused.',
              'Partial effects are descriptive and explicitly labeled; final five-seed claims require the full fixed roster.']}
    reports=ROOT/'reports'
    for disclosure in ('disclosed','undisclosed'):
        fresh=ROOT.parent/'evidence_use/data'/f'heldout_{disclosure}.jsonl'
        metadata={}
        for line in fresh.read_text().splitlines():
            row=json.loads(line);ref=json.loads(row['reference']);metadata[ref['task_id']]=ref
        for decode,stem,draws in [('greedy','fresh_greedy',1),('stochastic','fresh_stochastic_n5',5)]:
            paired={};paired_level={}
            for seed in SEEDS:
                loaded={}
                for arm in ('causal_family','shuffled_target'):
                    if arm=='shuffled_target':pattern=f'{disclosure}_shuffled_target_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
                    elif seed in (45,46):pattern=f'reused_{disclosure}_causal_family_qwen3_4b_s{seed}/{stem}.scores.jsonl'
                    else:pattern=f'{disclosure}_matched_recovery_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
                    paths=list(reports.glob(pattern))
                    if len(paths)>1:raise RuntimeError(f'Multiple complete attempts, resolve provenance before analysis: {paths}')
                    if not paths:
                        report['incomplete_cells'].append({'disclosure':disclosure,'decode':decode,'seed':seed,'arm':arm});continue
                    path=paths[0];groups=read_complete(path,metadata,draws);loaded[arm]=groups;report['sources'][str(path)]=sha(path)
                    report['cells'].append({'disclosure':disclosure,'decode':decode,'seed':seed,'arm':arm,'rows':len(groups),
                      'response_mae':float(np.mean([r['response_mae'] for v in groups.values() for r in v])),
                      'level_mae':float(np.mean([r['level_mae'] for v in groups.values() for r in v])),
                      'parse_rate':float(np.mean([r['parsed'] for v in groups.values() for r in v]))})
                if len(loaded)==2:
                    for metric,output in [('response_mae',paired),('level_mae',paired_level)]:
                        output[seed]={task:float(np.mean([r[metric] for r in loaded['shuffled_target'][task]]))-
                                              float(np.mean([r[metric] for r in loaded['causal_family'][task]])) for task in metadata}
            if paired:
                scopes=[('overall',list(metadata))]
                scopes += [(f'block:{block}',[t for t,m in metadata.items() if m['block']==block]) for block in sorted({m['block'] for m in metadata.values()})]
                scopes += [(f'k:{k}',[t for t,m in metadata.items() if m['k']==k]) for k in (3,6,9)]
                for scope,tasks in scopes:
                    result=summarize_values(paired,metadata,tasks)
                    result.update(disclosure=disclosure,decode=decode,scope=scope,metric='response_mae')
                    report['contrasts'].append(result)
                result=summarize_values(paired_level,metadata,list(metadata));result.update(disclosure=disclosure,decode=decode,scope='overall',metric='level_mae');report['contrasts'].append(result)
    report['complete']=not report['incomplete_cells']
    (ROOT/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# Shuffled-target control status','',f'Updated {report["created_at"]}.',
           f'Completed endpoint/decode cells: {len(report["cells"])}/40; missing: {len(report["incomplete_cells"])}.',
           'Final five-seed comparison complete.' if report['complete'] else 'The fixed five-seed comparison is still incomplete.','',
           '| Disclosure | Decode | Seeds | Shuffled − matched response MAE | 95% Student-t interval |',
           '|---|---|---:|---:|---|']
    for result in report['contrasts']:
        if result['scope']=='overall' and result['metric']=='response_mae':
            interval=result.get('student_t_95ci');formatted=f'[{interval[0]:+.4f}, {interval[1]:+.4f}]' if interval else 'not estimable'
            lines.append(f'| {result["disclosure"]} | {result["decode"]} | {result["training_seeds"]}/5 | {result["effect"]:+.4f} | {formatted} |')
    lines += ['','Positive favors matched training. All comparisons use the frozen fresh test; no historical test scores substitute for missing endpoints.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if args.require_complete and not report['complete']:raise SystemExit(2)

if __name__=='__main__':main()
