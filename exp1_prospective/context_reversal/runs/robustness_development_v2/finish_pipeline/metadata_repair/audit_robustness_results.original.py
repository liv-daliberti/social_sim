"""Independent CPU audit of completed robustness outputs; no model inference.

Uses raw JSON, standalone counters and parent-weighted resampling, without
importing the execution or analysis modules. Technical audit success is not a
scientific readiness gate. Run with the local transformers/NumPy environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
VARIANTS = ('repaired_base', 'name_only', 'paraphrase', 'resample')
CONTEXTS = ('positive', 'negative', 'broken', 'masked')
CONDITIONS = ('new_news', 'no_news', 'repeated_news')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def audit(run_dir, model_key):
    from transformers import AutoTokenizer
    errors, checks, sources = [], Counter(), {}
    def require(condition, description):
        checks[description] += 1
        if not condition:
            errors.append(description)
    def read(path, jsonl=False):
        path = Path(path)
        payload = path.read_bytes()
        sources[str(path)] = sha(payload)
        return [json.loads(line) for line in payload.splitlines() if line.strip()] if jsonl else json.loads(payload)
    submission = read(run_dir / 'submission.json')
    group = submission['groups'][model_key]
    result_dir = Path(group['results_dir'])
    summary_path = result_dir / 'summary.json'
    if not summary_path.exists():
        raise ValueError('Per-model summary has not been written; wait for both tasks to finish')
    summary = read(summary_path)
    require(summary['record_complete'], 'summary_record_complete')
    require(summary['model_key'] == model_key, 'summary_model_identity')
    require(summary['bootstrap'] == {'unit': 'parent_family_id', 'draws': 2000, 'seed': 20260921, 'variants_are_not_independent_families': True}, 'bootstrap_parent_contract')
    for filename, expected in submission['code_sha256'].items():
        require(sha((run_dir / 'code' / filename).read_bytes()) == expected, 'frozen_code_hashes')
    require(summary['provenance']['analysis_code']['sha256'] == submission['code_sha256']['analyze_robustness.py'], 'executed_analysis_code_hash')
    plans, rows, manifests, indexed = {}, {}, {}, {}
    for task, expected in [('probability', 1280), ('direction', 960)]:
        plan_path = Path(submission[task + '_design'])
        plans[task] = read(plan_path, True)
        require(sources[str(plan_path)] == submission[task + '_design_sha256'], 'frozen_design_hashes')
        response_path = Path(group[task + '_responses'])
        rows[task] = read(response_path, True)
        manifests[task] = read(Path(str(response_path) + '.manifest.json'))
        manifest = manifests[task]
        if manifest['status'] != 'complete' or len(rows[task]) != expected:
            raise ValueError(f'{task} is not complete: {len(rows[task])}/{expected}; manifest={manifest["status"]}')
        require(manifest['records'] == manifest['expected_records'] == expected, 'manifest_complete_coverage')
        require(manifest['counts'] == dict(Counter(row['status'] for row in rows[task])), 'manifest_status_counts')
        require(manifest['config']['input_sha256'] == sources[str(plan_path)], 'manifest_design_hash')
        require(manifest['config']['local_model']['path'] == group['model_path'], 'model_checkpoint_identity')
        require(manifest['config']['decode']['chat_template_mode'] in ('auto', 'qwen-no-thinking'), 'categorical_and_numeric_thinking_disabled')
        require(manifest['config']['decode']['temperature'] == .7 and manifest['config']['decode']['top_p'] == 1 and manifest['config']['decode']['max_tokens'] == 128, 'fixed_decode_settings')
        for field, path in [('design', plan_path), ('responses', response_path)]:
            require(summary['provenance'][task + '_' + field]['sha256'] == sources[str(path)], 'summary_input_hashes')
        index = {(row['trial_id'], row['condition']): row for row in rows[task]}
        require(len(index) == expected, 'unique_response_keys')
        expected_keys = {(unit['trial_id'], condition) for unit in plans[task] for condition in (('baseline', *CONDITIONS) if task == 'probability' else CONDITIONS)}
        require(set(index) == expected_keys, 'exact_planned_response_keys')
        indexed[task] = index
    units = {unit['trial_id']: unit for unit in plans['probability']}
    directions = {unit['trial_id']: unit for unit in plans['direction']}
    parents = sorted({unit['parent_family_id'] for unit in units.values()})
    require(len(units) == len(directions) == 320 and set(units) == set(directions), 'all_320_matching_context_units')
    require(len(parents) == summary['n_sampling_families'] == 20, 'all_20_parent_clusters')
    cells = {(unit['parent_family_id'], unit['variant'], unit['context_id']): unit for unit in units.values()}
    require(set(cells) == {(f, v, c) for f in parents for v in VARIANTS for c in CONTEXTS}, 'complete_parent_variant_context_grid')
    tokenizer = AutoTokenizer.from_pretrained(group['model_path'], local_files_only=True, trust_remote_code=False)
    for task, manifest in manifests.items():
        config = manifest['config']
        require(sha(canonical(tokenizer.chat_template)) == manifest['tokenizer_chat_template_sha256'], 'tokenizer_template_hash')
        require(sha(canonical(config)) == manifest['run_signature'], 'manifest_run_signature')
        for filename, expected in config['local_model']['metadata_sha256'].items():
            require(sha((Path(group['model_path']) / filename).read_bytes()) == expected, 'local_checkpoint_metadata_hashes')
    for trial, unit in units.items():
        direction = directions[trial]
        require(unit['sampling_family_id'] == unit['parent_family_id'] and unit['repeat'] == 0, 'parent_cluster_and_single_repeat_metadata')
        require(direction['parent_plan_sha256'] == submission['probability_design_sha256'], 'direction_exact_parent_plan')
        require(direction['parent_unit_sha256'] == sha(canonical(unit)), 'direction_exact_parent_unit')
        for field in ('trial_id', 'family_id', 'domain', 'context_id', 'repeat', 'material_status', 'parent_family_id', 'sampling_family_id', 'variant'):
            require(direction[field] == unit[field], 'matching_plan_metadata')
        require(unit['baseline_prompt'].startswith(direction['scenario_text'] + '\n\nEstimate the probability'), 'same_visible_scenario_across_tasks')
        for condition in CONDITIONS:
            prefix = direction['scenario_text'] + '\n\nYOUR PREVIOUS FORECAST\n__PRIOR_PROBABILITY__\n\nMESSAGE\n' + direction['messages'][condition]
            require(unit['update_templates'][condition].startswith(prefix + '\n\nRevise your previous probability'), 'same_visible_message_across_tasks')
    parsed, coverage = {'probability': {}, 'direction': {}}, {}
    for task in ('probability', 'direction'):
        manifest = manifests[task]
        counter = Counter(planned=len(rows[task]), present=len(rows[task]), valid=0, missing=0, invalid=0)
        for row in rows[task]:
            trial, condition = row['trial_id'], row['condition']
            unit = units[trial]
            stage = 'direction' if task == 'direction' else 'baseline' if condition == 'baseline' else 'update'
            require(row['model_key'] == model_key and row['stage'] == stage, 'response_task_and_model_identity')
            for field in ('family_id', 'domain', 'context_id', 'repeat', 'material_status'):
                require(row[field] == unit[field], 'response_plan_metadata')
            require(row['input_sha256'] == submission[task + '_design_sha256'], 'response_design_hash')
            require(row['run_signature'] == manifest['run_signature'], 'response_run_signature')
            expected_seed = int(sha(canonical([manifest['config']['decode']['seed'], trial, stage, condition]))[:16], 16) % (2**31 - 1)
            require(row['seed'] == expected_seed, 'deterministic_request_seed')
            value = None
            if row['status'] == 'ok':
                raw = json.loads(row['raw'])
                if task == 'probability':
                    require(set(raw) == {'probability'} and type(raw['probability']) in (int, float) and math.isfinite(raw['probability']) and 0 <= raw['probability'] <= 1, 'strict_numeric_raw_object')
                    require(re.fullmatch(r'\{"probability": (?:0(?:\.[0-9]{1,6})?|1(?:\.0{1,6})?)\}', row['raw']) is not None, 'numeric_raw_grammar')
                    value = float(raw['probability'])
                else:
                    require(set(raw) == {'direction'} and raw['direction'] in ('increase', 'decrease', 'unchanged', 'unclear'), 'strict_direction_raw_object')
                    require(re.fullmatch(r'\{"direction": "(?:increase|decrease|unchanged|unclear)"\}', row['raw']) is not None, 'direction_raw_grammar')
                    value = raw['direction']
                require(value == row[task], 'raw_output_matches_stored_value')
            else:
                require(row.get(task) is None, 'failed_output_not_imputed')
            parsed[task][trial, condition] = value
            counter['valid' if value is not None else 'invalid'] += 1
            counter['reason:' + row['status']] += 1
            if task == 'direction':
                prompt = directions[trial]['direction_prompts'][condition]
                require(row['parent_plan_sha256'] == submission['probability_design_sha256'], 'direction_response_parent_hash')
                require('YOUR PREVIOUS FORECAST' not in prompt and '__PRIOR_PROBABILITY__' not in prompt and row.get('probability') is None and row.get('prior_probability') is None, 'standalone_direction_without_numeric_prior')
            elif condition == 'baseline':
                prompt = unit['baseline_prompt']
                require(row['prior_probability'] is None and row['status'] != 'blocked_baseline', 'baseline_independent_of_other_forecasts')
            else:
                prior_row = indexed['probability'][trial, 'baseline']
                prior = float(json.loads(prior_row['raw'])['probability']) if prior_row['status'] == 'ok' else None
                require(row['prior_probability'] == prior and (row['status'] == 'blocked_baseline') == (prior is None), 'update_uses_only_own_context_baseline')
                prompt = None if prior is None else unit['update_templates'][condition].replace('__PRIOR_PROBABILITY__', json.dumps(prior))
            require(row['prompt'] == prompt and row['prompt_sha256'] == (None if prompt is None else sha(prompt)), 'exact_prompt_and_hash')
            if prompt is not None:
                options = {'enable_thinking': False} if manifest['config']['decode']['chat_template_mode'] == 'qwen-no-thinking' else {}
                chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, **options)
                require(row['chat_prompt_sha256'] == sha(chat), 'one_user_turn_chat_without_cross_branch_answers')
                require(row['prompt_token_count'] == len(tokenizer.encode(chat, add_special_tokens=False)), 'recorded_prompt_token_count')
        coverage[task] = dict(counter)
        require(coverage[task] == summary['coverage'][task], 'independent_coverage_matches_summary')
    for family in parents:
        for context in CONTEXTS:
            base, resample = [cells[family, variant, context] for variant in ('repaired_base', 'resample')]
            require(base['baseline_prompt'] == resample['baseline_prompt'] and base['update_templates'] == resample['update_templates'], 'resample_visible_text_is_identical')
            for task, conditions in [('probability', ('baseline', *CONDITIONS)), ('direction', CONDITIONS)]:
                for condition in conditions:
                    require(indexed[task][base['trial_id'], condition]['seed'] != indexed[task][resample['trial_id'], condition]['seed'], 'resample_request_seed_differs')
    indices = np.random.default_rng(20260921).integers(0, len(parents), size=(2000, len(parents)))
    weights = np.column_stack([(indices == i).sum(axis=1) for i in range(len(parents))])
    def metric(samples, binary=False):
        observed = [(f, x) for f, x in samples if x is not None]
        sums = np.array([sum(x for f, x in observed if f == family) for family in parents])
        counts = np.array([sum(f == family for f, x in (samples if binary else observed)) for family in parents])
        denominator = len(samples) if binary else len(observed)
        result = {'n_planned': len(samples), 'n_observed': len(observed), 'n_missing': len(samples) - len(observed), 'n_families_observed': len({f for f, x in observed}), 'denominator': denominator, 'estimate': sum(x for f, x in observed) / denominator if denominator else None, 'ci95': None, 'ci90': None}
        if binary:
            result.update(n_success=int(sum(x for f, x in observed)), n_failure_including_missing=len(samples) - int(sum(x for f, x in observed)))
        denominators = weights @ counts
        if result['n_families_observed'] >= 2 and (denominators > 0).all():
            boot = (weights @ sums) / denominators
            result['ci95'] = np.quantile(boot, [.025, .975]).tolist()
            result['ci90'] = np.quantile(boot, [.05, .95]).tolist()
        return result
    def compare(expected, recomputed):
        for field, value in recomputed.items():
            old = expected[field]
            equal = old is None if value is None else old == value if isinstance(value, int) else old is not None and bool(np.allclose(old, value, atol=1e-9, rtol=1e-9))
            require(equal, 'independent_metric_' + field)
    def number(f, v, c, condition):
        trial = cells[f, v, c]['trial_id']
        initial = parsed['probability'][trial, 'baseline']
        value = parsed['probability'][trial, condition]
        return None if initial is None or value is None else 100 * (value if condition == 'baseline' else value - initial)
    def label(f, v, c, condition):
        return parsed['direction'][cells[f, v, c]['trial_id'], condition]
    def target(c, condition):
        return 'unchanged' if condition != 'new_news' else {'positive': 'increase', 'negative': 'decrease', 'broken': 'unchanged', 'masked': None}[c]
    def score(value, truth):
        return None if value is None else int(value == truth)
    recomputed_variants, saturation = {}, {}
    for variant in VARIANTS:
        stored = summary['by_variant'][variant]
        sign, pair, dsign, dpair = [], [], [], []
        blocked, endpoints, cross = [], Counter(), Counter()
        for family in parents:
            deltas = [number(family, variant, c, 'new_news') for c in ('positive', 'negative')]
            labels = [label(family, variant, c, 'new_news') for c in ('positive', 'negative')]
            pair.append((family, None if None in deltas else int(deltas[0] > 0 and deltas[1] < 0)))
            dpair.append((family, None if None in labels else int(labels == ['increase', 'decrease'])))
            for context in CONTEXTS:
                prior = number(family, variant, context, 'baseline')
                endpoints['baseline_zero'] += prior == 0
                endpoints['baseline_one'] += prior == 100
                if context in ('positive', 'negative'):
                    value = number(family, variant, context, 'new_news')
                    correct = None if value is None else int(value > 0 if context == 'positive' else value < 0)
                    categorical = score(label(family, variant, context, 'new_news'), target(context, 'new_news'))
                    sign.append((family, correct)); dsign.append((family, categorical))
                    is_blocked = (context == 'positive' and prior == 100) or (context == 'negative' and prior == 0)
                    blocked.append({'parent_family_id': family, 'context': context, 'baseline_percent': prior, 'blocked': is_blocked, 'numeric_correct': bool(correct), 'direction_correct': bool(categorical)})
                    cross['both_correct' if correct and categorical else 'direction_only' if categorical else 'numeric_only' if correct else 'both_incorrect'] += 1
        result = {'numeric_sign_correct': metric(sign, True), 'numeric_paired_reversal': metric(pair, True), 'direction_sign_correct': metric(dsign, True), 'direction_paired_reversal': metric(dpair, True), 'direction_broken_unchanged': metric([(f, score(label(f, variant, 'broken', 'new_news'), 'unchanged')) for f in parents], True)}
        changes = [(f, number(f, variant, 'broken', 'new_news')) for f in parents]
        result['broken_movement'] = {'signed_mean_pp': metric(changes), 'mean_absolute_pp': metric([(f, None if x is None else abs(x)) for f, x in changes])}
        for key in ('numeric_sign_correct', 'numeric_paired_reversal', 'direction_sign_correct', 'direction_paired_reversal', 'direction_broken_unchanged'):
            compare(stored[key], result[key])
        for key in ('signed_mean_pp', 'mean_absolute_pp'):
            compare(stored['broken_movement'][key], result['broken_movement'][key])
        signed, absolute = result['broken_movement']['signed_mean_pp'], result['broken_movement']['mean_absolute_pp']
        status = ('indeterminate' if signed['n_missing'] or signed['ci90'] is None or absolute['ci95'] is None else
                  'criterion_met' if -2 < signed['ci90'][0] and signed['ci90'][1] < 2 and absolute['ci95'][1] < 2 else 'criterion_not_met')
        require(stored['broken_stability']['status'] == status, 'broken_stability_criterion_matches')
        result['broken_stability'] = status
        result['controls_by_condition'] = {}
        pooled_numeric, pooled_direction = [], []
        for condition in ('no_news', 'repeated_news'):
            changes = [(f, number(f, variant, c, condition)) for f in parents for c in CONTEXTS]
            numeric_binary = [(f, None if x is None else int(abs(x) <= 2 + 1e-9)) for f, x in changes]
            categorical = [(f, score(label(f, variant, c, condition), 'unchanged')) for f in parents for c in CONTEXTS]
            pooled_numeric.extend(numeric_binary); pooled_direction.extend(categorical)
            control = {'numeric_within_2pp': metric(numeric_binary, True), 'direction_unchanged': metric(categorical, True), 'numeric_movement': {'signed_mean_pp': metric(changes), 'mean_absolute_pp': metric([(f, None if x is None else abs(x)) for f, x in changes])}}
            for key in ('numeric_within_2pp', 'direction_unchanged'):
                compare(stored['controls_by_condition'][condition][key], control[key])
            for key in ('signed_mean_pp', 'mean_absolute_pp'):
                compare(stored['controls_by_condition'][condition]['numeric_movement'][key], control['numeric_movement'][key])
            result['controls_by_condition'][condition] = control
        compare(stored['numeric_controls_within_2pp'], metric(pooled_numeric, True)); compare(stored['direction_controls_unchanged'], metric(pooled_direction, True))
        saturation[variant] = {'baseline_endpoints': dict(endpoints), 'signed_trials_planned': 40, 'direction_blocked_count': sum(row['blocked'] for row in blocked), 'signed_cross_tab': dict(cross), 'signed_trials': blocked}
        recomputed_variants[variant] = result
    recomputed_invariance = {}
    for variant in VARIANTS[1:]:
        recomputed_invariance[variant] = {}
        for condition in ('baseline', *CONDITIONS):
            differences, excess, agreement, joint, relation_split = [], [], [], [], Counter()
            for family in parents:
                for context in CONTEXTS:
                    base, edit, replica = [number(family, v, context, condition) for v in ('repaired_base', variant, 'resample')]
                    difference = None if base is None or edit is None else edit - base
                    differences.append((family, difference)); excess.append((family, None if difference is None or replica is None else abs(difference) - abs(replica - base)))
                    if condition != 'baseline':
                        before, after = [label(family, v, context, condition) for v in ('repaired_base', variant)]
                        agreement.append((family, None if before is None or after is None else int(before == after)))
                        truth = target(context, condition)
                        if truth is not None:
                            joint.append((family, None if before is None or after is None else int(before == after == truth)))
                        if difference is not None and before is not None and after is not None:
                            relation_split[('numeric_difference_over_2pp' if abs(difference) > 2 + 1e-9 else 'numeric_difference_within_2pp') + ('_labels_agree' if before == after else '_labels_disagree')] += 1
            result = {'change_difference': {'signed_mean_pp': metric(differences), 'mean_absolute_pp': metric([(f, None if x is None else abs(x)) for f, x in differences])}, 'within_2pp': metric([(f, None if x is None else int(abs(x) <= 2 + 1e-9)) for f, x in differences], True), 'absolute_difference_minus_resample_pp': metric(excess), 'direction_agreement': metric(agreement, True) if agreement else None, 'both_direction_judgments_correct': metric(joint, True) if joint else None, 'numeric_vs_categorical_discrepancies': dict(relation_split)}
            stored = summary['invariance_vs_repaired_base'][variant][condition]
            for key in ('signed_mean_pp', 'mean_absolute_pp'):
                compare(stored['change_difference'][key], result['change_difference'][key])
            for key in ('within_2pp', 'absolute_difference_minus_resample_pp', 'direction_agreement', 'both_direction_judgments_correct'):
                if result[key] is not None:
                    compare(stored[key], result[key])
            recomputed_invariance[variant][condition] = result
    for path, before in sources.items():
        if Path(path).name != 'submission.json':
            require(sha(Path(path).read_bytes()) == before, 'audited_artifacts_unchanged_during_audit')
    report = {'schema_version': 'independent_robustness_audit_v1', 'model_key': model_key, 'generated_at': datetime.now(timezone.utc).isoformat(), 'audit_pass': not errors, 'audit_scope': 'Technical execution and independently recomputed descriptive metrics; not a scientific readiness gate.', 'no_model_inference': True, 'sources_sha256': sources, 'audit_code_sha256': sha(Path(__file__).read_bytes()), 'checks': dict(checks), 'errors': dict(Counter(errors)), 'coverage': coverage, 'sampling_families': parents, 'bootstrap_verification': 'Independent parent multiplicity-weighted resampling matched point estimates and 90/95% intervals.', 'by_variant': recomputed_variants, 'invariance_vs_repaired_base': recomputed_invariance, 'saturation_and_sign_diagnostics': saturation}
    def fmt(value, digits=2):
        return 'unavailable' if value is None else f'{value:.{digits}f}'
    review = ['# Independent robustness review: ' + model_key, '', 'Technical audit: **' + ('PASS' if report['audit_pass'] else 'FAIL') + '**. This is not a scientific pass/fail gate or permission to select a model.', '', 'All 20 parent families and four variants remain grouped; numeric coverage is 1,280 records and categorical coverage 960 records. Raw answers, exact prompts, own-context priors, separately serialized one-user-turn categorical chats, deterministic seeds, checkpoint metadata, and frozen source hashes were checked without model inference. Parent-bootstrap point estimates and intervals were recomputed independently.', '', '| Variant | Numeric sign / 40 | Numeric pair / 20 | Direction sign / 40 | Direction pair / 20 | Broken unchanged / 20 | Broken mean absolute (pp) | Blocked signed priors / 40 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant, result in recomputed_variants.items():
        values = [result[k]['n_success'] for k in ('numeric_sign_correct', 'numeric_paired_reversal', 'direction_sign_correct', 'direction_paired_reversal', 'direction_broken_unchanged')]
        review.append('| ' + variant + ' | ' + ' | '.join(str(x) for x in values) + f" | {fmt(result['broken_movement']['mean_absolute_pp']['estimate'])} | {saturation[variant]['direction_blocked_count']} |")
    review += ['', '| Edit vs repaired base | New-news update discrepancy mean absolute (pp) | Within 2 pp / 80 | Label agreement / 80 | Both labels correct / 60 | Excess discrepancy vs resample (pp; 95% interval) |', '|---|---:|---:|---:|---:|---|']
    for variant, result in recomputed_invariance.items():
        value = result['new_news']; excess = value['absolute_difference_minus_resample_pp']
        review.append(f"| {variant} | {fmt(value['change_difference']['mean_absolute_pp']['estimate'])} | {value['within_2pp']['n_success']} | {value['direction_agreement']['n_success']} | {value['both_direction_judgments_correct']['n_success']} | {fmt(excess['estimate'])}; {excess['ci95']} |")
    review += ['', '| Variant | No-news absolute drift (pp) | Repeated-news absolute drift (pp) |', '|---|---:|---:|']
    for variant, result in recomputed_variants.items():
        review.append('| ' + variant + ' | ' + ' | '.join(f"{fmt(result['controls_by_condition'][c]['numeric_movement']['mean_absolute_pp']['estimate'], 3)}" for c in ('no_news', 'repeated_news')) + ' |')
    review += ['', '## Numerical instability versus categorical judgments', '',
               '| Comparison | Update difference over 2 pp / 80 | Of these, categorical labels agree | Of these, categorical labels disagree |',
               '|---|---:|---:|---:|']
    for variant, result in recomputed_invariance.items():
        cross = result['new_news']['numeric_vs_categorical_discrepancies']
        agrees = cross.get('numeric_difference_over_2pp_labels_agree', 0)
        disagrees = cross.get('numeric_difference_over_2pp_labels_disagree', 0)
        review.append(f'| {variant} | {agrees + disagrees} | {agrees} | {disagrees} |')
    review += ['', 'These counts separate numerical update instability from changes in the standalone categorical judgment; they do not condition away failures in the primary outcomes. Numerical differences with agreeing labels can include consistently wrong interpretations, which the joint-correctness column exposes. Compare every edit with the identical-text resample before attributing discrepancy to wording.', '',
               'Broken-link label accuracy and broken numerical movement provide separate evidence about relevance judgment. If categorical judgments also fail while no-news drift is small, ordinary numerical re-elicitation alone does not explain the observed pattern. Endpoint blocking can contribute to signed numeric failures but cannot mechanically force an erroneous standalone label. An excess-discrepancy interval crossing zero establishes neither an edit effect nor equivalence.', '']
    review += ['', '## Scientific interpretation limits', '', '- A stable wrong label is agreement, not successful interpretation; the table keeps joint correctness separate and excludes masked new news only from correctness.', '- The resample repeats the exact visible text with another request seed. Its nonzero discrepancy is a descriptive sampling reference. A single resample and descriptive family intervals do not isolate a causal naming or paraphrase effect.', '- Endpoint-blocked priors remain unconditional numeric failures. Their counts quantify one mechanical restriction, not a reason to remove difficult items. Standalone categorical errors persist independently of numerical endpoint constraints.', '- Compare broken-link categorical failures with broken numerical movement and control drift before attributing failures to probability elicitation alone. Agreement can coexist with unstable numerical updates; the JSON retains that cross-tab with observed denominators.', '- Repairs and all edit variants are development observations within the original 20 families. They are not 80 independent families, do not replace original results, and cannot validate the proposed fresh cohort. Future freezing criteria concern materials and execution, not these observed success rates.', '']
    if errors:
        review += ['Technical discrepancies: ' + json.dumps(dict(Counter(errors)), sort_keys=True), '']
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / 'independent_audit.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (result_dir / 'review.md').write_text('\n'.join(review))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=HERE / 'runs/robustness_development_v2')
    parser.add_argument('--model-key', required=True)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.model_key)
    print(json.dumps({'audit_pass': result['audit_pass'], 'model_key': result['model_key'], 'errors': result['errors'], 'checks': sum(result['checks'].values())}, indent=2))
    return 0 if result['audit_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
