#!/usr/bin/env python3
"""Analyze frozen gain-pair changes and fresh ordinary transfer predictions."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO/'exp3_training_transfer/mechanism_family'))
from output_contract import parse_forecast_array
from worlds import response_vector

def pair_metrics(low,high,predlow,predhigh):
    true_low=np.asarray(low['targets']); true_high=np.asarray(high['targets'])
    true_delta=response_vector(true_high)-response_vector(true_low)
    true_level_delta=true_high-true_low
    valid=predlow is not None and predhigh is not None
    delta=(response_vector(predhigh)-response_vector(predlow)) if valid else np.zeros(8)
    leveldelta=(np.asarray(predhigh)-predlow) if valid else np.zeros(10)
    nonzero=np.abs(true_delta)>1e-8
    return dict(pair_id=low['pair_id'],world=low['world'],split=low['split'],block=low['block'],valid=valid,
        change_mae=float(np.abs(delta-true_delta).mean()),no_change_mae=float(np.abs(true_delta).mean()),
        level_change_mae=float(np.abs(leveldelta-true_level_delta).mean()),
        level_no_change_mae=float(np.abs(true_level_delta).mean()),
        tracking_numerator=float(np.dot(delta,true_delta)),tracking_denominator=float(np.dot(true_delta,true_delta)),
        direction_correct=int(np.sum((np.sign(delta)==np.sign(true_delta))&nonzero)),direction_n=int(nonzero.sum()),
        simulator_response_delta=true_delta.tolist(),predicted_response_delta=delta.tolist())

def metrics(rows):
    denom=sum(r['tracking_denominator'] for r in rows)
    m=lambda key:float(np.mean([r[key] for r in rows]))
    baseline=m('no_change_mae')
    return dict(n_pairs=len(rows),valid_pair_rate=m('valid'),change_mae=m('change_mae'),no_change_mae=baseline,
        improvement_over_no_change=baseline-m('change_mae'),relative_error_reduction=1-m('change_mae')/baseline,
        tracking_slope=sum(r['tracking_numerator'] for r in rows)/denom if denom else None,
        direction_accuracy=sum(r['direction_correct'] for r in rows)/sum(r['direction_n'] for r in rows),
        level_change_mae=m('level_change_mae'),level_no_change_mae=m('level_no_change_mae'))

def strata(rows):
    return {'all':rows,**{s:[r for r in rows if r['split']==s] for s in ('train','test')},
            **{w:[r for r in rows if r['world']==w] for w in sorted({r['world'] for r in rows})}}

def ci(values): return [float(x) for x in np.quantile(values,[.025,.975])]

def contrasts(pairrows,repetitions):
    rng=np.random.default_rng(20260921)
    output=[]
    for disclosure in ('disclosed','undisclosed'):
        for group in ('all','train','test'):
            chosen=[r for r in pairrows if r['disclosure']==disclosure and (group=='all' or r['split']==group)]
            index=defaultdict(dict)
            for r in chosen: index[(r['arm'],r['training_seed'])][r['pair_id']]=r
            trained_seeds=sorted(s for arm,s in index if arm=='causal_family' and ('population_prior',s) in index)
            for comparator in ('population_prior','base'):
                seeds=[s for s in trained_seeds if (comparator,None if comparator=='base' else s) in index]
                if not seeds: continue
                maps=[]
                for s in seeds:
                    matched=index[('causal_family',s)]
                    other=index[(comparator,None if comparator=='base' else s)]
                    if set(matched)!=set(other): raise ValueError('Unpaired comparison')
                    byworld=defaultdict(list)
                    for pid in sorted(matched):
                        byworld[matched[pid]['world']].append(other[pid]['change_mae']-matched[pid]['change_mae'])
                    maps.append({w:np.asarray(v) for w,v in byworld.items()})
                estimates=[float(np.mean(np.concatenate(list(m.values())))) for m in maps]
                boot=[]
                for _ in range(repetitions):
                    sampled=[]
                    for si in rng.integers(0,len(maps),len(maps)):
                        sampled.append(np.mean(np.concatenate([v[rng.integers(0,len(v),len(v))] for v in maps[si].values()])))
                    boot.append(float(np.mean(sampled)))
                output.append(dict(disclosure=disclosure,group=group,contrast=f'{comparator}_minus_causal_family_change_mae',
                    training_seeds=seeds,individual_seed_effects=dict(zip(map(str,seeds),estimates)),estimate=float(np.mean(estimates)),
                    ci95=ci(boot),bootstrap_repetitions=repetitions,
                    uncertainty='paired training seeds and world-stratified episodes; two-seed results are exploratory'))
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis')
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    args=parser.parse_args()
    pairs=[]; summary=[]; ordinary=[]; provenance=[]
    frozen=json.loads((ROOT/'data/frozen_manifest.json').read_text())
    manifest_sha=hashlib.sha256((ROOT/'data/frozen_manifest.json').read_bytes()).hexdigest()
    for path in sorted(args.results.glob('*.json')):
        data=json.loads(path.read_text())
        if 'records' not in data: continue
        assert data['manifest_sha256']==manifest_sha
        checkpoint=data['checkpoint']
        common=dict(checkpoint=checkpoint['id'],arm=checkpoint['arm'],training_seed=checkpoint['seed'],disclosure=checkpoint['disclosure'])
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        refs=[]
        for rec in data['records']:
            ref=json.loads(rec['reference'])
            predicted=parse_forecast_array(rec['output'][0],10,ref['clip'])
            refs.append((ref,predicted))
        if data['suite']=='gain_pairs':
            bypair=defaultdict(dict)
            for ref,pred in refs: bypair[ref['pair_id']][ref['side']]=(ref,pred)
            checkpoint_pairs=[]
            for pid,sides in sorted(bypair.items()):
                assert set(sides)=={'low','high'}
                low,pl=sides['low'];high,ph=sides['high']
                checkpoint_pairs.append({**common,**pair_metrics(low,high,pl,ph)})
            pairs.extend(checkpoint_pairs)
            for group,rows in strata(checkpoint_pairs).items():
                if not rows: continue
                valid=[r for r in rows if r['valid']]
                summary.append({**common,'group':group,**metrics(rows),
                    'valid_only':metrics(valid) if valid else None})
        elif data['suite']=='heldout':
            for group in ['all',*sorted({r['world'] for r,p in refs})]:
                subset=[(r,p) for r,p in refs if group=='all' or r['world']==group]
                parsed=[(r,np.asarray(p)) for r,p in subset if p is not None]
                ordinary.append({**common,'group':group,'n':len(subset),'parse_rate':len(parsed)/len(subset),
                    'forecast_mae_valid_only':float(np.mean([np.abs(p-r['targets']).mean() for r,p in parsed])) if parsed else None,
                    'response_mae_valid_only':float(np.mean([np.abs(response_vector(p)-response_vector(r['targets'])).mean() for r,p in parsed])) if parsed else None})
    args.output.mkdir(parents=True,exist_ok=True)
    expected=[c['id'] for c in json.loads((ROOT/'data/checkpoint_inventory.json').read_text())['available']]
    observed=sorted({r['checkpoint'] for r in pairs})
    ordinary_observed=sorted({r['checkpoint'] for r in ordinary})
    result=dict(protocol=frozen['protocol'],analyzed_at=datetime.now(timezone.utc).isoformat(),frozen_manifest_sha256=manifest_sha,
        expected_checkpoint_ids=expected,observed_checkpoint_ids=observed,
        ordinary_observed_checkpoint_ids=ordinary_observed,
        gain_complete=set(expected)==set(observed),ordinary_complete=set(expected)==set(ordinary_observed),
        complete=set(expected)==set(observed)==set(ordinary_observed),
        missing_checkpoint_ids=sorted(set(expected)-set(observed)),gain_summary=summary,fresh_ordinary_summary=ordinary,
        paired_contrasts=contrasts(pairs,args.bootstrap_repetitions),prediction_sources=provenance,
        limitations=['Only seeds45 and46 have retained mechanism adapter weights.','Gain pairs are 16 episodes per mechanism at k9.',
                    'This is a gain sensitivity diagnostic, not a graph intervention.','Positive change-MAE contrasts favor matched training.'])
    # Disclosure jobs may finish concurrently; unique temporary files avoid shared temp races.
    import os
    def atomic(path,value):
        temp=path.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(value);temp.replace(path)
    atomic(args.output/'summary.json',json.dumps(result,indent=2,sort_keys=True)+'\n')
    atomic(args.output/'pair_metrics.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in pairs))
    lines=['# Mechanism evidence-use diagnostic', '',f'Complete: {result["complete"]}. Frozen manifest: `{manifest_sha}`.',
        '', 'Only two retained training seeds (45, 46) are available; these results are exploratory.', '',
        '| Disclosure | Arm | Seed | Mechanisms | Delta MAE | No-change MAE | Tracking slope | Parse pairs |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in summary:
        if r['group'] in ('train','test'):
            lines.append(f'| {r["disclosure"]} | {r["arm"]} | {r["training_seed"]} | {r["group"]} | {r["change_mae"]:.3f} | {r["no_change_mae"]:.3f} | {r["tracking_slope"]:.3f} | {r["valid_pair_rate"]:.3f} |')
    lines.extend(['','The tracking slope is 1 for the simulator and 0 for a model with no response to changed evidence. Delta MAE measures high-minus-low response changes after subtracting the zero-shock forecast.','', '| Disclosure | Mechanisms | Comparison (positive favors matched) | Effect | 95% interval |','|---|---|---|---:|---|'])
    for r in result['paired_contrasts']:
        lines.append(f'| {r["disclosure"]} | {r["group"]} | {r["contrast"]} | {r["estimate"]:.3f} | [{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}] |')
    atomic(args.output/'RESULTS.md','\n'.join(lines)+'\n')
    print(json.dumps({'complete':result['complete'],'checkpoints':len(observed),'gain_pairs':len(pairs),'output':str(args.output)},sort_keys=True))

if __name__=='__main__': main()
