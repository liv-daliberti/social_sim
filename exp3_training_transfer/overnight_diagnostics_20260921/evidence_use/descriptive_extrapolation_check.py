#!/usr/bin/env python3
"""Post-hoc descriptive subset; never replaces the registered all-heldout endpoint."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from analyze import ROOT, metrics

def main():
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    if not summary['complete']:raise SystemExit('Wait for the complete frozen roster.')
    rows=[json.loads(s) for s in (ROOT/'analysis/pair_metrics.jsonl').read_text().splitlines()]
    oracle=[json.loads(s) for s in (ROOT/'analysis/known_mechanism_pair_metrics.jsonl').read_text().splitlines()]
    excluded='feedback_saturating_extrap'
    results=[]
    for disclosure in ['disclosed','undisclosed']:
        for arm in ['base','causal_family','population_prior']:
            for scope in ['all_heldout','three_heldout_excluding_noisy_extrapolation']:
                selected=[r for r in rows if r['disclosure']==disclosure and r['arm']==arm and r['split']=='test' and (scope=='all_heldout' or r['world']!=excluded)]
                individual=[]
                for seed in sorted({r['training_seed'] for r in selected},key=lambda x:-1 if x is None else x):
                    individual.append({'training_seed':seed,**metrics([r for r in selected if r['training_seed']==seed])})
                results.append(dict(disclosure=disclosure,arm=arm,scope=scope,individual_seeds=individual,**metrics(selected)))
    refs=[]
    for scope in ['all_heldout','three_heldout_excluding_noisy_extrapolation']:
        refs.append({'scope':scope,**metrics([r for r in oracle if r['split']=='test' and (scope=='all_heldout' or r['world']!=excluded)])})
    output=dict(created_at=datetime.now(timezone.utc).isoformat(),analysis_status='post-hoc descriptive, no hypothesis test',
        primary_endpoint='all four heldout worlds; unchanged',excluded_world_for_descriptive_subset=excluded,
        motivation='Known-mechanism posterior adequacy reference, frozen before LM predictions were inspected, recovers only about 0.13 of the gain effect in the high-noise extrapolation world.',
        requested_after_partial_model_results=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metrics=results,known_mechanism_reference=refs)
    (ROOT/'analysis/descriptive_extrapolation_check.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    lines=['# Descriptive check of the noisy extrapolation world', '',
        'The registered primary endpoint remains all four heldout worlds. The three-world subset below was requested after partial model results and is descriptive, with no new hypothesis test. Its exclusion is motivated by the separately frozen known-mechanism reference.', '',
        '| Disclosure | Arm | Worlds | Change MAE | No-change MAE | Tracking slope |',
        '|---|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["disclosure"]} | {r["arm"]} | {4 if r["scope"]=="all_heldout" else 3} | {r["change_mae"]:.3f} | {r["no_change_mae"]:.3f} | {r["tracking_slope"]:.3f} |')
    for r in refs:
        lines.append(f'| Extra mechanism information | Numerical reference | {4 if r["scope"]=="all_heldout" else 3} | {r["change_mae"]:.3f} | {r["no_change_mae"]:.3f} | {r["tracking_slope"]:.3f} |')
    (ROOT/'analysis/DESCRIPTIVE_EXTRAPOLATION_CHECK.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
