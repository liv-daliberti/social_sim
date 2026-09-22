"""Controlled-edit development analysis, clustering all variants by parent family."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from . import analyze as num, analyze_direction as cat, direction_design, run_local as common

VARIANTS = ('repaired_base', 'name_only', 'paraphrase', 'resample')
CONTEXTS = ('positive', 'negative', 'broken', 'masked')
ARMS = ('new_news', 'no_news', 'repeated_news')
TOLERANCE_PP = 2.0


def validate_plans(probability_units, direction_units):
    units = num.validate_design(probability_units)
    direction_design.validate_units(direction_units)
    directions = {u['trial_id']: u for u in direction_units}
    if set(units) != set(directions):
        raise ValueError('Probability and direction units differ')
    cells, parents = {}, set()
    for trial, unit in units.items():
        parent, variant = unit.get('parent_family_id'), unit.get('variant')
        if not parent or unit.get('sampling_family_id') != parent or variant not in VARIANTS:
            raise ValueError('Missing or inconsistent parent/variant metadata')
        key = (parent, variant, unit['context_id'])
        if key in cells:
            raise ValueError('Duplicate robustness cell')
        cells[key] = unit
        parents.add(parent)
        direction = directions[trial]
        for field in ('parent_family_id', 'sampling_family_id', 'variant'):
            if direction.get(field) != unit[field]:
                raise ValueError('Direction parent/variant metadata differs')
        if direction['parent_unit_sha256'] != common.text_sha256(common.canonical_json(unit)):
            raise ValueError('Direction parent unit hash differs')
        visible, messages = direction_design.extract_visible_text(unit)
        if visible != direction['scenario_text'] or messages != direction['messages']:
            raise ValueError('Direction and numeric texts differ')
    expected = {(f,v,c) for f in parents for v in VARIANTS for c in CONTEXTS}
    if set(cells) != expected:
        raise ValueError('Every parent requires all variants and contexts')
    for family in parents:
        for context in CONTEXTS:
            base, replica = (cells[family,v,context] for v in ('repaired_base','resample'))
            if any(base[field] != replica[field] for field in ('baseline_prompt','update_templates')):
                raise ValueError('Resample must repeat identical visible prompts')
    return units, directions, cells, sorted(parents)


def analyze(probability_units, direction_units, probability_rows, direction_rows, model_key, *, draws=2000, seed=20260921):
    units, directions, cells, parents = validate_plans(probability_units, direction_units)
    numeric, ignored_n = num.index_responses(probability_rows, units, model_key)
    categorical, ignored_d = cat.index_responses(direction_rows, directions, model_key)
    for (trial, condition), row in categorical.items():
        unit = directions[trial]
        required = {'prompt': unit['direction_prompts'][condition],
                    'prompt_sha256': unit['direction_prompt_sha256'][condition],
                    'parent_plan_sha256': unit['parent_plan_sha256']}
        if any(row.get(field) != expected for field, expected in required.items()):
            raise ValueError('Direction response requires its exact prompt/hash/parent-plan provenance')
        if row.get('prior_probability') is not None or row.get('probability') is not None:
            raise ValueError('Direction response contains a numeric forecast or prior')
        if row['status'] != 'ok' and row.get('direction') is not None:
            raise ValueError('Failed direction response has a nonnull classification')
    bootstrap = num.FamilyBootstrap(parents, draws, seed)
    coverage = {'probability':Counter(), 'direction':Counter()}
    values, labels, rows = {}, {}, []
    for (family, variant, context), unit in cells.items():
        trial = unit['trial_id']
        probabilities = {}
        for condition in ('baseline', *ARMS):
            row = numeric.get((trial,condition))
            p, reason = num.probability(row)
            probabilities[condition] = p
            count = coverage['probability'];count.update(planned=1,present=int(row is not None),valid=int(p is not None),missing=int(row is None),invalid=int(row is not None and p is None));count['reason:'+reason]+=1
            if row is not None:
                baseline_row = numeric.get((trial, 'baseline'))
                prior = num.probability(baseline_row)[0]
                if row['status'] != 'ok' and row.get('probability') is not None:
                    raise ValueError('Failed numeric response has a nonnull probability')
                if condition == 'baseline':
                    if row['status'] == 'blocked_baseline':
                        raise ValueError('A numeric baseline cannot be blocked on itself')
                    if row.get('prior_probability') is not None:
                        raise ValueError('A numeric baseline cannot receive a prior forecast')
                else:
                    if baseline_row is None:
                        raise ValueError('Numeric update lacks its own baseline record')
                    if (row['status'] == 'blocked_baseline') != (prior is None):
                        raise ValueError('Numeric update/baseline status mismatch')
                    if row['status'] == 'blocked_baseline' and (row.get('response_received') or row.get('raw')):
                        raise ValueError('A blocked update cannot contain a received completion')
                prompt = unit['baseline_prompt'] if condition=='baseline' else common.render_update(unit,condition,prior) if prior is not None else None
                if row.get('prompt') != prompt or row.get('prompt_sha256') != (common.text_sha256(prompt) if prompt is not None else None):
                    raise ValueError('Numeric response prompt/own-context prior mismatch')
                if condition!='baseline' and row.get('prior_probability') != prior:
                    raise ValueError('Numeric update uses the wrong prior')
                if p is not None and common.parse_probability(row['raw']) != p:
                    raise ValueError('Numeric raw output and probability differ')
        baseline = probabilities['baseline']
        values[family,variant,context,'baseline'] = None if baseline is None else 100*baseline
        for condition in ARMS:
            p = probabilities[condition]
            delta = None if baseline is None or p is None else 100*(p-baseline)
            values[family,variant,context,condition] = delta
            row = categorical.get((trial,condition)); label, reason = cat.direction_value(row)
            labels[family,variant,context,condition] = label
            count=coverage['direction'];count.update(planned=1,present=int(row is not None),valid=int(label is not None),missing=int(row is None),invalid=int(row is not None and label is None));count['reason:'+reason]+=1
            rows.append({'parent_family_id':family,'variant':variant,'context_id':context,'condition':condition,'trial_id':trial,'baseline_probability':baseline,'update_probability':p,'delta_pp':delta,'direction':label,'expected_direction':cat.expected_direction(context,condition)})
    def metric(items, binary=False):
        return bootstrap.summarize([num.Sample(f,v) for f,v in items],binary=binary)
    def move(items):
        return {'signed_mean_pp':metric(items),'mean_absolute_pp':metric([(f,None if v is None else abs(v)) for f,v in items])}
    def correct(value, context):
        return None if value is None else float(value>0 if context=='positive' else value<0)
    variant_results = {}
    for variant in VARIANTS:
        signed=[(f,correct(values[f,variant,c,'new_news'],c)) for f in parents for c in ('positive','negative')]
        pair=[];direction_pair=[]
        for f in parents:
            pos,neg=(values[f,variant,c,'new_news'] for c in ('positive','negative'))
            pair.append((f,None if pos is None or neg is None else float(pos>0 and neg<0)))
            pl,nl=(labels[f,variant,c,'new_news'] for c in ('positive','negative'))
            direction_pair.append((f,None if pl is None or nl is None else float(pl=='increase' and nl=='decrease')))
        broken=move([(f,values[f,variant,'broken','new_news']) for f in parents])
        controls=[(f,None if values[f,variant,c,a] is None else float(abs(values[f,variant,c,a])<=TOLERANCE_PP+1e-9)) for f in parents for c in CONTEXTS for a in ('no_news','repeated_news')]
        variant_results[variant]={'numeric_sign_correct':metric(signed,True),'numeric_paired_reversal':metric(pair,True),'broken_movement':broken,'broken_stability':num.stability_diagnostic(broken['signed_mean_pp'],broken['mean_absolute_pp']),'numeric_controls_within_2pp':metric(controls,True),'direction_sign_correct':metric([(f,None if labels[f,variant,c,'new_news'] is None else float(labels[f,variant,c,'new_news']==cat.expected_direction(c,'new_news'))) for f in parents for c in ('positive','negative')],True),'direction_paired_reversal':metric(direction_pair,True),'direction_broken_unchanged':metric([(f,None if labels[f,variant,'broken','new_news'] is None else float(labels[f,variant,'broken','new_news']=='unchanged')) for f in parents],True),'direction_controls_unchanged':metric([(f,None if labels[f,variant,c,a] is None else float(labels[f,variant,c,a]=='unchanged')) for f in parents for c in CONTEXTS for a in ('no_news','repeated_news')],True)}
        variant_results[variant]['controls_by_condition'] = {}
        for condition in ('no_news', 'repeated_news'):
            changes = [(f, values[f, variant, c, condition]) for f in parents for c in CONTEXTS]
            variant_results[variant]['controls_by_condition'][condition] = {
                'numeric_within_2pp': metric([(f, None if value is None else float(abs(value) <= TOLERANCE_PP + 1e-9))
                                             for f, value in changes], True),
                'numeric_movement': move(changes),
                'direction_unchanged': metric([(f, None if labels[f, variant, c, condition] is None else
                                                float(labels[f, variant, c, condition] == 'unchanged'))
                                               for f in parents for c in CONTEXTS], True),
            }
    invariance={}
    for variant in VARIANTS[1:]:
        comparisons={}
        for condition in ('baseline',*ARMS):
            differences=[];successes=[];excess=[];agreements=[];joint=[]
            for f in parents:
                for c in CONTEXTS:
                    base,edited,replica=(values[f,v,c,condition] for v in ('repaired_base',variant,'resample'))
                    diff=None if base is None or edited is None else edited-base
                    differences.append((f,diff));successes.append((f,None if diff is None else float(abs(diff)<=TOLERANCE_PP+1e-9)))
                    excess.append((f,None if diff is None or replica is None else abs(diff)-abs(replica-base)))
                    if condition!='baseline':
                        bl,el=(labels[f,v,c,condition] for v in ('repaired_base',variant))
                        agreements.append((f,None if bl is None or el is None else float(bl==el)))
                        target=cat.expected_direction(c,condition)
                        if target is not None:
                            joint.append((f,None if bl is None or el is None else float(bl==target and el==target)))
            comparisons[condition]={'change_difference':move(differences),'within_2pp':metric(successes,True),'absolute_difference_minus_resample_pp':metric(excess),'direction_agreement':metric(agreements,True) if agreements else None,'both_direction_judgments_correct':metric(joint,True) if joint else None}
        invariance[variant]=comparisons
    return {'schema_version':'controlled_robustness_analysis_v1','status':'exploratory_development','model_key':model_key,'n_sampling_families':len(parents),'n_variants':len(VARIANTS),'n_units':len(units),'tolerance_pp':TOLERANCE_PP,'bootstrap':{'unit':'parent_family_id','draws':draws,'seed':seed,'variants_are_not_independent_families':True},'coverage':{k:dict(v) for k,v in coverage.items()},'record_complete':all(v['missing']==0 for v in coverage.values()),'all_valid':all(v['valid']==v['planned'] for v in coverage.values()),'ignored_other_model_records':{'probability':ignored_n,'direction':ignored_d},'by_variant':variant_results,'invariance_vs_repaired_base':invariance,'trial_rows':rows,'interpretation':['All planned binary outcomes retain missing and invalid records as failures; magnitude summaries never impute missing values.','Invariance alone is not correctness: agreement and joint correctness are reported separately.','Identical-text resample uses another request seed; excess discrepancy compares edits with this sampling reference.','All variants cluster within the original parent families; no confirmatory or causal identification claim.','Repaired material outcomes do not overwrite or replace original-pilot outcomes.']}


def markdown(report):
    lines = ['# Controlled-edit robustness development', '',
             f"Model: {report['model_key']}; {report['n_sampling_families']} parent families, {report['n_variants']} variants. Complete: {report['record_complete']}; all valid: {report['all_valid']}.", '',
             '| Task | Planned | Present | Valid | Missing | Invalid present |',
             '|---|---:|---:|---:|---:|---:|']
    for task, coverage in report['coverage'].items():
        lines.append('| ' + task + ' | ' + ' | '.join(str(coverage[k]) for k in ('planned', 'present', 'valid', 'missing', 'invalid')) + ' |')
    lines += ['', '| Variant | Numeric sign | Numeric paired reversal | Direction sign | Direction paired reversal | Broken unchanged |',
              '|---|---:|---:|---:|---:|---:|']
    def count(metric):
        return f"{metric['n_success']}/{metric['n_planned']}"
    def observed(metric):
        return f"{metric['n_observed']}/{metric['n_planned']}"
    def estimate(metric):
        return 'unavailable' if metric['estimate'] is None else f"{metric['estimate']:.3f}"
    for variant, result in report['by_variant'].items():
        lines.append('| ' + variant + ' | ' + ' | '.join(count(result[key]) for key in (
            'numeric_sign_correct', 'numeric_paired_reversal', 'direction_sign_correct',
            'direction_paired_reversal', 'direction_broken_unchanged')) + ' |')
    lines += ['', '| Edit vs base | New-news update within 2 pp | Mean absolute update difference (pp) | Observed/planned differences | Direction agreement | Both direction judgments correct |',
              '|---|---:|---:|---:|---:|---:|']
    for variant, result in report['invariance_vs_repaired_base'].items():
        metric = result['new_news']
        magnitude = metric['change_difference']['mean_absolute_pp']
        lines.append(f"| {variant} | {count(metric['within_2pp'])} | {estimate(magnitude)} | {observed(magnitude)} | {count(metric['direction_agreement'])} | {count(metric['both_direction_judgments_correct'])} |")
    lines += ['', 'Agreement includes masked new-news judgments; joint correctness excludes masked new news, which has no assigned target. Missing and invalid pairs remain failures in both planned binary denominators.', '',
              '| Variant | Control message | Numeric within 2 pp | Mean absolute movement (pp) | Observed/planned movements | Direction unchanged |',
              '|---|---|---:|---:|---:|---:|']
    for variant, result in report['by_variant'].items():
        for condition, control in result['controls_by_condition'].items():
            magnitude = control['numeric_movement']['mean_absolute_pp']
            lines.append(f"| {variant} | {condition} | {count(control['numeric_within_2pp'])} | {estimate(magnitude)} | {observed(magnitude)} | {count(control['direction_unchanged'])} |")
    lines += ['', *report['interpretation'], '',
              'Full JSON includes intervals, resampling-adjusted discrepancies, missingness and trial-level values.', '']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probability-design','direction-design','probability-responses','direction-responses','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-key',required=True);a=p.parse_args(argv)
    artifacts={name:{'path':str(getattr(a,name).resolve()),'sha256':common.file_sha256(getattr(a,name))} for name in ('probability_design','direction_design','probability_responses','direction_responses')}
    plans={task:num.read_jsonl(getattr(a,task+'_design')) for task in ('probability','direction')}
    if any(unit.get('parent_plan_sha256') != artifacts['probability_design']['sha256'] for unit in plans['direction']):
        raise ValueError('Direction parent-plan hash differs from actual probability design')
    responses={task:num.read_jsonl(getattr(a,task+'_responses')) for task in ('probability','direction')}
    for task in ('probability','direction'):
        if any(r.get('model_key')==a.model_key and r.get('input_sha256')!=artifacts[task+'_design']['sha256'] for r in responses[task]):raise ValueError('Response design hash differs')
    report=analyze(plans['probability'],plans['direction'],responses['probability'],responses['direction'],a.model_key)
    report['provenance']=artifacts|{'analysis_code':{'path':str(Path(__file__).resolve()),'sha256':common.file_sha256(Path(__file__))}}
    common.atomic_write(a.output/'summary.json',json.dumps(report,indent=2,allow_nan=False)+'\n');common.atomic_write(a.output/'summary.md',markdown(report))
    print(json.dumps({'record_complete':report['record_complete'],'all_valid':report['all_valid'],'output':str(a.output)}))

if __name__=='__main__':main()
