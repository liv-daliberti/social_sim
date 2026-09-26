"""Explicit preparation and verification of immutable fresh-study artifacts."""
import json
from pathlib import Path
from .. import run_local as c


def artifact(path):return {'path':str(Path(path).resolve()),'sha256':c.file_sha256(Path(path))}

def check_artifact(spec):
    p=Path(spec['path'])
    if not p.is_absolute() or c.file_sha256(p)!=spec['sha256']:raise ValueError('frozen artifact changed: '+str(p))
    return p


def verify_local(args,config):
    if args.design_freeze is None:raise ValueError('fresh target inference requires --design-freeze')
    f=json.loads(args.design_freeze.read_text())
    if f.get('status')!='frozen' or f.get('schema_version')!='fresh_local_factorial_v1' or f.get('target_inference_started_before_freeze') is not False:raise ValueError('invalid design freeze')
    for item in f['artifacts'].values():check_artifact(item)
    for item in f['code'].values():check_artifact(item)
    reviews=json.loads(check_artifact(f['artifacts']['review']).read_text())
    if reviews.get('status')!='complete' or reviews.get('unresolved_findings')!=0 or reviews.get('target_model_outputs_used') is not False:raise ValueError('screening/adjudication incomplete')
    if str(args.input.resolve()) not in {s['path'] for s in f['plans']}:raise ValueError('unfrozen local shard')
    for item in f['plans']:check_artifact(item)
    d=config['decode'];e=config['engine'];expected=f['local_settings']
    fields={'temperature':d['temperature'],'top_p':d['top_p'],'max_tokens':d['max_tokens'],'seed':d['seed'],'max_model_len':e['max_model_len'],'batch_size':e['batch_size'],'tensor_parallel_size':e['tensor_parallel_size'],'dtype':e['dtype']}
    if fields!=expected:raise ValueError('local settings differ from freeze')
    if config['local_model']!=f['local_model']:raise ValueError('local checkpoint differs')
    if args.model_key!=f['model_keys'][args.thinking]:raise ValueError('incorrect reasoning arm identity')
    expected_output=(Path(f['responses_dir'])/(args.thinking+'_'+args.input.stem+'.jsonl')).resolve()
    if args.output.resolve()!=expected_output:raise ValueError('unexpected response destination')
    return f


def prepare(adjudication_path):
    from .materials import OUT,ROOT
    from .. import frontier_budget as budget,run_frozen_frontier as frontier
    import random,shutil
    adjudication_path=Path(adjudication_path).resolve();adj=json.loads(adjudication_path.read_text())
    if adj.get('status')!='complete' or adj.get('unresolved_findings')!=0 or adj.get('target_model_outputs_used') is not False:raise ValueError('adjudication is not complete')
    if adj['candidate_plan_sha256']!=c.file_sha256(OUT/'probability_plan.jsonl'):raise ValueError('reviewed candidate differs')
    for item in adj['review_artifacts']:check_artifact(item)
    run=ROOT/'runs/fresh_evaluation_v1';dest=ROOT/'data/fresh_evaluation_v1/frozen_v1';responses=ROOT/'responses/fresh_evaluation_v1'
    if dest.exists():raise ValueError('preserve existing freeze; refusing overwrite')
    if responses.exists() and list(responses.glob('*.jsonl')):raise ValueError('target outputs already exist before design freeze')
    protocol=(run/'PROTOCOL_DRAFT.md').read_text().replace('protocol awaiting screening and freeze','frozen protocol').replace('Before any fresh target inference, finalize','Before any fresh target inference, finalize')
    dest.mkdir(parents=True)
    def save(name,value):
        path=dest/name;c.atomic_write(path,json.dumps(value,indent=2)+'\n');return path
    def jl(name,rows):
        path=dest/name;c.atomic_write(path,''.join(c.canonical_json(x)+'\n' for x in rows));return path
    protocol_path=dest/'PROTOCOL.md';protocol_path.write_text(protocol)
    source=c.read_units(OUT/'probability_plan.jsonl');units=[dict(u,material_status='model_screened_finite_rule_evaluation') for u in source]
    plan=jl('probability_plan.jsonl',units)
    paid_ids={u['trial_id'] for u in c.read_units(OUT/'frontier_plan.jsonl')};paid=[u for u in units if u['trial_id'] in paid_ids];paid_plan=jl('frontier_plan.jsonl',paid)
    family_path=jl('families.jsonl',[dict(json.loads(line),review_status='model_screened_formal_author_adjudication_not_human_validated') for line in (OUT/'families.jsonl').read_text().splitlines()])
    parents=sorted({u['parent_family_id'] for u in units});random.Random(20260922).shuffle(parents)
    plans=[]
    for k in range(8):
        selected=set(parents[k*10:(k+1)*10]);plans.append(artifact(jl(f'shard{k:02d}.jsonl',[u for u in units if u['parent_family_id'] in selected])))
    development=[artifact(ROOT/'results/local_diagnostics_v1/reasoning_comparison.json'),artifact(ROOT/'results/robustness_development_v2/qwen3_32b/summary.json')]
    # Locate the recorded Qwen3 development summary without guessing a model-key spelling.
    if not Path(development[-1]['path']).exists():raise ValueError('missing development result')
    exclusions=[artifact(ROOT/'runs'/x) for x in ('frontier_gpt56_pilot_v1/plan.jsonl','local_diagnostics_v1/probability_plan.jsonl','robustness_development_v2/probability_plan.jsonl')]
    old_units=[u for item in exclusions for u in c.read_units(Path(item['path']))];old_families={u['family_id'] for u in old_units};old_prompts={u['baseline_prompt'] for u in old_units}|{t for u in old_units for t in u['update_templates'].values()}
    if any(u['family_id'] in old_families or u['baseline_prompt'] in old_prompts or any(t in old_prompts for t in u['update_templates'].values()) for u in units):raise ValueError('nonfresh target plan')
    artifacts_by_plan={}
    for name,up,path in [('local',units,plan),('frontier',paid,paid_plan)]:
        families=sorted({u['family_id'] for u in up})
        review={'status':'complete','plan_sha256':c.file_sha256(path),'family_ids':families,'reviewer':'Blinded Llama-3.1-8B-Instruct and Qwen3-14B screening; Codex author adjudication against independent exact-rule audit; not human validation','unresolved_findings':0,'frontier_model_outputs_used':False,'target_model_outputs_used':False,'outcome_based_family_filtering':False,'local_development_complete':True,'development_results':development,'adjudication':artifact(adjudication_path),'limitations':adj['limitations'],'review_artifacts':adj['review_artifacts'],'candidate_plan_sha256':c.file_sha256(OUT/'probability_plan.jsonl')}
        rp=save(name+'_review.json',review)
        fp=save(name+'_freshness.json',{'status':'passed','plan_sha256':c.file_sha256(path),'family_ids':families,'excluded_plans':exclusions,'scope':'No exact family IDs or prompts overlap old evaluated plans; not a claim of statistical independence among mechanisms.'})
        artifacts_by_plan[name]={'plan':artifact(path),'protocol':artifact(protocol_path),'scoring':artifact(Path(__file__).with_name('analyze.py')),'review':artifact(rp),'freshness':artifact(fp),'adjudication':artifact(adjudication_path),'material_audit':artifact(run/'material_audit.json'),'families':artifact(family_path),'candidate_plan':artifact(OUT/'probability_plan.jsonl')}
    model_path=next((ROOT.parents[1]/'.runtime/hf_home/hub').glob('models--Qwen--Qwen3-32B/snapshots/*/config.json')).parent
    local_code={str(p.relative_to(ROOT)):artifact(p) for p in [Path(__file__),Path(__file__).with_name('reasoning.py'),Path(__file__).with_name('analyze.py'),Path(__file__).with_name('materials.py'),Path(__file__).with_name('run_pair.py'),Path(__file__).with_name('run.sbatch'),ROOT/'run_local.py',ROOT/'analyze.py']}
    settings={'temperature':.7,'top_p':1.,'max_tokens':8192,'seed':20260922,'max_model_len':16384,'batch_size':16,'tensor_parallel_size':2,'dtype':'bfloat16'}
    lf={'schema_version':'fresh_local_factorial_v1','status':'frozen','created_at':c.utc_now(),'target_inference_started_before_freeze':False,'artifacts':artifacts_by_plan['local'],'code':local_code,'plans':plans,'local_model':c.inspect_local_model(model_path),'local_settings':settings,'model_keys':{'enabled':'qwen3_32b_fresh_thinking','disabled':'qwen3_32b_fresh_nonthinking'},'responses_dir':str(responses),'screening_not_human_validation':True}
    local_path=save('local_freeze.json',lf)
    code_files=list(frontier.CODE_FILES)+['collect_frontier_token_counts.py','run_frozen_frontier_batch.py']
    ff={'schema_version':'frozen_frontier_evaluation_v1','status':'frozen','created_at':lf['created_at'],'freeze_stage':'after_local_development_before_frontier','evaluation_id':'fresh_finite_context_reversal_v1','target_inference_started_before_freeze':False,'authorization_scope':budget.AUTHORIZATION_SCOPE,'model':frontier.MODEL,'model_key':'gpt56_sol_fresh','max_output_tokens':2048,'reasoning_effort':'low','max_retries':0,'concurrency':4,'pricing':'batch','prices_nusd_per_token':budget.PRICES_NUSD_PER_TOKEN['batch'],'pricing_valid_through':budget.PRICE_VALID_THROUGH,'budget_usd':'24','contexts':['positive','negative','broken'],'family_order':sorted({u['family_id'] for u in paid}),'output_path':str(responses/'frontier_gpt56_sol.jsonl'),'artifacts':artifacts_by_plan['frontier'],'code_sha256':{name:c.file_sha256(ROOT/name) for name in code_files},'token_certificate_output':str((run/'frontier_token_certificate.json').resolve()),'local_design_freeze':artifact(local_path)}
    fpath=save('frontier_design_freeze.json',ff)
    validated=frontier.validate_freeze(fpath)
    save('freeze_validation.json',{'status':'passed','created_at':c.utc_now(),'local_freeze':artifact(local_path),'frontier_freeze':artifact(fpath),'local_units':len(units),'parent_instances':80,'local_records':7680,'frontier_parent_instances':24,'frontier_records':576,'whole_cohort_conservative_budget':validated[3]['budget']})
    print(json.dumps({'local_freeze':str(local_path),'frontier_design_freeze':str(fpath),'budget':validated[3]['budget']['cohort_worst_case_usd']},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--adjudication',type=Path,required=True);a=p.parse_args();prepare(a.adjudication)
