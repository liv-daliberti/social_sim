#!/usr/bin/env python3
"""Auxiliary diagnostic: known mechanism, posterior inference of the displayed gain.

Extra mechanism information is supplied even for the undisclosed condition.
This is an observation-adequacy reference, not an LM or a fair undisclosed oracle.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from build_frozen import ROOT, CATALOG, expected_series, forecast_scenarios
from analyze import pair_metrics,metrics,strata

def posterior_prediction(ref):
    world=next(w for w in CATALOG if w.name==ref['world'])
    inputs=np.asarray(ref['target_inputs'])
    # The LM sees two decimal places, not the saved four-decimal observations.
    observed=np.asarray([float(f'{v:.2f}') for v in ref['target_observed']])
    gains=np.linspace(world.gain_lo,world.gain_hi,201)
    means=np.stack([expected_series(world,inputs,float(g))[0] for g in gains])
    ll=-.5*np.sum(((observed[None,:]-means)/world.noise)**2,axis=1)
    weights=np.exp(ll-ll.max());weights/=weights.sum()
    forecast=np.stack([forecast_scenarios(world,inputs,float(g)) for g in gains])
    return np.sum(forecast*weights[:,None],axis=0),float(np.dot(gains,weights))

def main():
    path=ROOT/'data/gain_pairs_disclosed.jsonl'
    rows=[json.loads(json.loads(s)['reference']) for s in path.read_text().splitlines()]
    result=[]
    for lo,hi in zip(rows[::2],rows[1::2]):
        lp,lg=posterior_prediction(lo);hp,hg=posterior_prediction(hi)
        result.append({**pair_metrics(lo,hi,lp,hp),'posterior_gain_low':lg,'posterior_gain_high':hg,
            'true_gain_low':lo['g'],'true_gain_high':hi['g']})
    output=ROOT/'analysis';output.mkdir(exist_ok=True)
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        purpose='Auxiliary known-mechanism posterior observation-adequacy reference; extra mechanism information supplied in undisclosed case.',
        primary_lm_protocol_unchanged=True,prior='uniform over each native gain support, 201-point grid',
        observation_likelihood='Gaussian using exactly displayed two-decimal outcomes; no outcome clipping in these frozen observations.',
        summaries=[{'group':g,**metrics(r)} for g,r in strata(result).items() if r],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert all(0 < v < 100 for r in rows for v in r['target_observed'])
    (output/'known_mechanism_baseline.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (output/'known_mechanism_pair_metrics.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in result))
    for r in summary['summaries']:
        print(r['group'], 'change_MAE',round(r['change_mae'],3),'no_change_MAE',round(r['no_change_mae'],3),'slope',round(r['tracking_slope'],3))

if __name__=='__main__': main()
