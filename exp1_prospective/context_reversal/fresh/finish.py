"""Audit completed fresh responses and produce results; never make model calls."""
import argparse,json,math,types
from collections import Counter
from pathlib import Path
import numpy as np
from . import analyze,reasoning,freeze
from .. import run_local as c

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'data/fresh_evaluation_v1/frozen_v1'
# The enabled arm was collected under a recorded amendment that raised only the
# output-token allowance; see runs/fresh_evaluation_v1/TOKEN_BUDGET_AMENDMENT.md.
# Each arm is verified against the freeze it was actually produced under.
AMENDED=ROOT/'data/fresh_evaluation_v1/frozen_v1_amended/local_freeze_max_tokens_15360.json'
ARM_FREEZE={'enabled':AMENDED,'disabled':None}
RESULTS=ROOT/'results/fresh_evaluation_v1'


def save(name,value):c.atomic_write(RESULTS/name,json.dumps(value,indent=2)+'\n')

def independent_primary(units,rows,summary):
    by={c.record_key(r):r for r in rows};checks=0;pair_values={}
    for v,reported in summary['by_variant'].items():
        parents=sorted({u['parent_family_id'] for u in units if u['variant']==v});pairs=[];signs=[];broken=[];controls={x:[] for x in ('no_news','repeated_news')};endpoint=[]
        for f in parents:
            valid_signs=[]
            for ctx in ('positive','negative','broken'):
                unit=next(u for u in units if u['parent_family_id']==f and u['variant']==v and u['context_id']==ctx);t=unit['trial_id'];b=by[t,'baseline','baseline'];new=by[t,'update','new_news']
                d=100*(new['probability']-b['probability']) if b['status']=='ok' and new['status']=='ok' else None
                if ctx=='broken':
                    if d is not None:broken.append(d)
                else:
                    ok=d is not None and (d>0 if ctx=='positive' else d<0);signs.append(ok);valid_signs.append(ok)
                for arm in controls:
                    r=by[t,'update',arm]
                    if b['status']=='ok' and r['status']=='ok':controls[arm].append(100*abs(r['probability']-b['probability']))
            pair=all(valid_signs);pairs.append(pair);pair_values[f,v]=int(pair)
        for key,vals in [('paired_reversal',pairs),('sign_accuracy',signs)]:
            assert reported[key]['n_success']==sum(vals) and reported[key]['denominator']==len(vals)
            assert math.isclose(reported[key]['estimate'],sum(vals)/len(vals),abs_tol=1e-12);checks+=3
        if broken:
            assert math.isclose(reported['broken_signed_pp']['estimate'],math.fsum(broken)/len(broken),abs_tol=1e-10)
            assert math.isclose(reported['broken_absolute_pp']['estimate'],math.fsum(map(abs,broken))/len(broken),abs_tol=1e-10);checks+=2
        assert reported['broken_absolute_pp']['n_observed']==len(broken);checks+=1
        if len(broken)!=len(parents):assert reported['broken_stability']['status']=='indeterminate';checks+=1
        for arm,vals in controls.items():
            if vals:assert math.isclose(reported['controls_absolute_pp'][arm]['estimate'],math.fsum(vals)/len(vals),abs_tol=1e-10);checks+=1
    return checks,pair_values


def finish():
    fp=FROZEN/'local_freeze.json';f=json.loads(fp.read_text())
    for spec in [*f['code'].values(),*f['artifacts'].values(),*f['plans']]:freeze.check_artifact(spec)
    full_units=c.read_units(Path(f['artifacts']['plan']['path']));missing=[];progress={}
    for mode in f['model_keys']:
        progress[mode]={'planned_records':3840,'recorded':0,'complete_shards':0}
        for spec in f['plans']:
            plan=Path(spec['path']);out=Path(f['responses_dir'])/(mode+'_'+plan.stem+'.jsonl');mp=Path(str(out)+'.manifest.json')
            man=json.loads(mp.read_text()) if mp.exists() else {}
            progress[mode]['recorded']+=man.get('records',0)
            if man.get('status') in ('complete','complete_with_errors') and man.get('records')==480:progress[mode]['complete_shards']+=1
            else:missing.append(str(out))
    if missing:
        frontier_manifest=Path(f['responses_dir'])/'frontier_gpt56_sol.jsonl.manifest.json'
        frontier=json.loads(frontier_manifest.read_text()) if frontier_manifest.exists() else {}
        result={'status':'waiting_for_local_collection','progress':progress,'incomplete_files':missing,'frontier_status':frontier.get('status','not_started'),'frontier_records':frontier.get('records',0),'frontier_cost_upper_bound_usd':frontier.get('exposure_usd'),'recorded_includes_durable_inflight_intents':True,'updated_at':c.utc_now()};save('status.json',result);return result
    # Reconstruct the exact effective chat formatting and count the actual tokenized input.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f['local_model']['path'],local_files_only=True,trust_remote_code=False)
    template_hash=c.text_sha256(c.canonical_json(tok.chat_template));summaries={};all_rows={};checks=0;hashes={};pair_by_mode={}
    for mode,key in f['model_keys'].items():
        rows=[]
        for spec in f['plans']:
            plan=Path(spec['path']);units=c.read_units(plan);out=Path(f['responses_dir'])/(mode+'_'+plan.stem+'.jsonl');mp=Path(str(out)+'.manifest.json');man=json.loads(mp.read_text());cfg=man['config']
            fz=ARM_FREEZE.get(mode) or fp
            args=types.SimpleNamespace(design_freeze=fz,input=plan,output=out,model_key=key,thinking=mode)
            freeze.verify_local(args,cfg)
            assert man['design_freeze_sha256']==c.file_sha256(fz)
            assert cfg['runner_sha256']==c.file_sha256(Path(reasoning.__file__)) and cfg['helper_runner_sha256']==c.file_sha256(Path(c.__file__))
            assert man['tokenizer_chat_template_sha256']==template_hash
            indexed=reasoning.load_existing(out,mp,cfg,units);assert len(indexed)==len(units)*4;checks+=5
            for row in indexed.values():
                if row['prompt'] is not None and row.get('prompt_token_count') is not None:
                    chat,opened=reasoning.render_chat(tok,row['prompt'],mode)
                    assert c.text_sha256(chat)==row['chat_prompt_sha256'] and len(tok.encode(chat,add_special_tokens=False))==row['prompt_token_count'] and opened==row['reasoning_open_in_prompt'];checks+=3
                rows.append(row)
            hashes[str(out)]=c.file_sha256(out);hashes[str(mp)]=c.file_sha256(mp)
        summary=analyze.analyze(full_units,rows,key);assert summary['all_planned_received'];summary['provenance']={'local_freeze':freeze.artifact(fp),'amended_freeze':freeze.artifact(AMENDED),'arm_freezes':{m:str((ARM_FREEZE.get(m) or fp)) for m in ('disabled','enabled')},'scorer':freeze.artifact(Path(analyze.__file__)),'response_sha256':dict(hashes)}
        n,values=independent_primary(full_units,rows,summary);checks+=n;pair_by_mode[mode]=values;summaries[mode]=summary;all_rows[mode]=rows;save(mode+'.json',summary)
    comparison=analyze.compare(summaries['disabled'],summaries['enabled']);save('reasoning_comparison.json',comparison)
    parents=sorted({u['parent_family_id'] for u in full_units});indices=np.random.default_rng(20260922).integers(0,len(parents),(2000,len(parents)))
    for v in ('base','names','paraphrase','resample'):
        values=np.array([100*(pair_by_mode['enabled'][p,v]-pair_by_mode['disabled'][p,v]) for p in parents]);m=comparison['paired_reversal_on_minus_off_pp'][v]
        assert math.isclose(float(values.mean()),m['estimate'],abs_tol=1e-12)
        assert np.allclose(np.quantile(values[indices].mean(axis=1),(.025,.975)),m['ci95'],atol=1e-12);checks+=2
    paid=c.read_units(FROZEN/'frontier_plan.jsonl');paid_ids={u['trial_id'] for u in paid}
    for mode,key in f['model_keys'].items():
        subset=analyze.analyze(paid,[r for r in all_rows[mode] if r['trial_id'] in paid_ids],key);save(mode+'_frontier_subset.json',subset)
    frontier_path=Path(f['responses_dir'])/'frontier_gpt56_sol.jsonl';fm=Path(str(frontier_path)+'.manifest.json');frontier_status='not_complete'
    if fm.exists():
        manifest=json.loads(fm.read_text())
        if manifest.get('status') in ('complete','complete_with_errors') and manifest.get('records')==576:
            from ..run_frozen_frontier_batch import execute
            preflight=execute(FROZEN/'frontier_generation_freeze.json') # offline: verifies durable provider/own-prior evidence
            assert preflight['api_calls']==0 and preflight['credential_reads']==0
            records=analyze.read(frontier_path);result=analyze.analyze(paid,records,'gpt56_sol_fresh');n,_=independent_primary(paid,records,result);checks+=n
            result['provenance']={'freeze':freeze.artifact(FROZEN/'frontier_generation_freeze.json'),'responses':freeze.artifact(frontier_path),'offline_transport_validation':preflight['status'],'cost_upper_bound_usd':manifest.get('exposure_usd')};save('frontier.json',result);frontier_status='complete'
    summary_names=['enabled.json','disabled.json','reasoning_comparison.json','enabled_frontier_subset.json','disabled_frontier_subset.json']+(['frontier.json'] if frontier_status=='complete' else [])
    audit={'summary_artifacts':{name:freeze.artifact(RESULTS/name) for name in summary_names},'status':'passed','checks':checks,'raw_completion_reparsed':True,'context_priors_verified':True,'prompt_chat_hash_and_input_token_checks':True,'primary_counts_and_means_independently_recomputed':True,'paired_difference_ci_independently_recomputed':True,'no_model_calls':True,'local_freeze':freeze.artifact(fp),'response_sha256':hashes,'audit_code':freeze.artifact(Path(__file__)),'scorer':freeze.artifact(Path(analyze.__file__)),'frontier_status':frontier_status,'updated_at':c.utc_now()};save('technical_audit.json',audit)
    lines=['# Fresh Exp1 evaluation','', '80 finite-rule instances; eight shared mechanism classes. Model screening plus formal author adjudication, not human validation.','', '| Deployment | Wording | Paired reversal | Sign accuracy | Broken mean absolute pp | Valid/planned |','|---|---|---:|---:|---:|---:|']
    for mode,s in summaries.items():
        for v,m in s['by_variant'].items():
            pair=m['paired_reversal'];sign=m['sign_accuracy'];broken=m['broken_absolute_pp'];value='NA' if broken['estimate'] is None else f"{broken['estimate']:.2f}"
            lines.append(f"| Qwen3 {mode} | {v} | {pair['n_success']}/{pair['n_planned']} | {sign['n_success']}/{sign['n_planned']} | {value} | {s['coverage'].get('ok',0)}/{s['coverage']['planned']} |")
    effect=comparison['paired_reversal_on_minus_off_pp']['base'];lines.extend(['',f"Base wording, thinking enabled minus disabled: {effect['estimate']:.2f} percentage points; descriptive 95% interval {effect['ci95']}.",'',f'Frontier collection: {frontier_status}. Detailed control drift, failures, normative-effect bins and mechanism sensitivity are in the JSON artifacts.'])
    if frontier_status=='complete':
        lines.extend(['','| Frontier wording | Paired reversal | Sign accuracy | Broken mean absolute pp |','|---|---:|---:|---:|'])
        for v,m in result['by_variant'].items():
            lines.append(f"| {v} | {m['paired_reversal']['n_success']}/24 | {m['sign_accuracy']['n_success']}/48 | {m['broken_absolute_pp']['estimate']:.3f} |")
        lines.extend(['',f"Frontier usage-based cost upper bound: ${manifest['exposure_usd']}; 576 planned records.",'', 'All figures use the frozen denominators. Missing, zero and invalid signed updates count as failures. Broken-link stability is indeterminate if any planned measurement is missing. The eight shared mechanisms and common outcome wrapper limit generalization to natural news; the original lexical relevance shortcut remains a limitation.'])
    lines.extend(['','| Local mode | Recorded status | Records |','|---|---|---:|'])
    for mode,s in summaries.items():
        for name,n in s['coverage'].items():
            if name not in ('planned','missing'):lines.append(f'| {mode} | {name} | {n} |')
    c.atomic_write(RESULTS/'summary.md','\n'.join(lines)+'\n')
    status={'status':'complete' if frontier_status=='complete' else 'local_complete_frontier_pending','audit_checks':checks,'local_planned_records':7680,'frontier_status':frontier_status,'updated_at':c.utc_now()};save('status.json',status);return status

if __name__=='__main__':print(json.dumps(finish(),indent=2))
