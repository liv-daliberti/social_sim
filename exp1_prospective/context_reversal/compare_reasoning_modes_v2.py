#!/usr/bin/env python3
"""Offline v2 paired comparison after symmetric, format-only Qwen3 recovery.

Definitions were specified before either new response file was inspected. This
script cannot call a model. A complete report requires every planned record and
terminal runner manifests; invalid completed responses remain in the analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

try:
    from . import analyze, run_reasoning_local_v2 as runner, run_reasoning_local as strict_runner
except ImportError:
    import analyze
    import run_reasoning_local_v2 as runner
    import run_reasoning_local as strict_runner

SOURCE = Path(__file__).resolve().parent
common = runner.common
MODES = ('disabled', 'enabled')
DEFINITIONS = {
    'version': 'qwen3_paired_reasoning_comparison_v1',
    'frozen_before_new_response_inspection': True,
    'design': 'All original 20 families, four contexts, repeat 0; 80 units and 320 records per mode',
    'sign_accuracy': 'Strict expected sign of new_news minus own context baseline, pooled over 40 positive/negative trials; zero and invalid measurements fail',
    'paired_reversal': 'Both positive increase and negative decrease within each of 20 families; zero and invalid measurements fail',
    'broken_mean_absolute_movement_pp': '100 * abs(new_news minus own broken-context baseline); only valid measurements, missingness explicit',
    'contrast': 'Enabled minus disabled; binary rates in percentage points, movement in probability points',
    'paired_binary_denominator': 'All 20 families; each contributes its mean of two direction successes and its one paired-reversal success',
    'paired_magnitude_denominator': 'Families with valid broken movement in both modes only; report both marginal and common-pair means; never impute a forecast or movement',
    'bootstrap': {'draws': 2000, 'seed': 20260921, 'unit': 'family_id',
                  'interval': 'Percentile 95%; identical whole-family draws preserve contexts, arms, and modes; any undefined resample makes the magnitude interval unavailable'},
    'endpoint_blocking': 'Exact baseline 1 in positive context or exact baseline 0 in negative context; separate missing baselines; no post-hoc threshold',
    'reference': 'Existing constrained Qwen3 repeat0 from matched comparison JSON; descriptive reference only, excluded from paired contrasts',
    'interpretation': 'Exploratory development diagnostic; no confirmatory inference, model ranking, or new materials review',
}


def artifact(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': common.file_sha256(path)}


def assert_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: {actual!r} != {expected!r}')


def metric_values(units: list[dict], records: dict) -> tuple[list[str], dict]:
    families = sorted({u['family_id'] for u in units})
    deltas = {family: {} for family in families}
    for unit in units:
        def probability(condition):
            row = records[(unit['trial_id'], 'baseline' if condition == 'baseline' else 'update', condition)]
            return row['probability'] if row['status'] == 'ok' else None
        baseline, updated = probability('baseline'), probability('new_news')
        deltas[unit['family_id']][unit['context_id']] = None if baseline is None or updated is None else 100 * (updated - baseline)
    values = {name: [] for name in ('sign_accuracy', 'paired_reversal', 'broken_mean_absolute_movement_pp')}
    for family in families:
        positive, negative, broken = (deltas[family][c] for c in ('positive', 'negative', 'broken'))
        pos_ok, neg_ok = positive is not None and positive > 0, negative is not None and negative < 0
        values['sign_accuracy'].append(50.0 * (int(pos_ok) + int(neg_ok)))
        values['paired_reversal'].append(100.0 * (pos_ok and neg_ok))
        values['broken_mean_absolute_movement_pp'].append(None if broken is None else abs(broken))
    return families, values


def summarize(values: list, indices: np.ndarray) -> dict:
    array = np.array([np.nan if v is None else v for v in values], dtype=float)
    present = np.isfinite(array)
    resampled = array[indices]
    counts = np.isfinite(resampled).sum(axis=1)
    defined = counts > 0
    result = {'estimate': float(array[present].mean()) if present.any() else None,
              'n_planned_families': len(values), 'n_observed_families': int(present.sum()),
              'n_missing_families': int((~present).sum()), 'ci95': None,
              'bootstrap_draws': len(indices), 'bootstrap_draws_defined': int(defined.sum()),
              'interval_status': 'insufficient_observed_families'}
    if present.sum() >= 2:
        result['interval_status'] = 'undefined_resamples_due_to_missingness'
        if defined.all():
            estimates = np.nansum(resampled, axis=1) / counts
            result.update(ci95=[float(v) for v in np.quantile(estimates, (.025, .975))],
                          interval_status='descriptive_paired_whole_family_bootstrap')
    return result


def paired_contrasts(units: list[dict], by_mode: dict) -> dict:
    vectors = {mode: metric_values(units, by_mode[mode]) for mode in MODES}
    families = vectors['disabled'][0]
    assert_equal(vectors['enabled'][0], families, 'Mode family alignment')
    rng = np.random.default_rng(DEFINITIONS['bootstrap']['seed'])
    indices = rng.integers(0, len(families), (DEFINITIONS['bootstrap']['draws'], len(families)))
    result = {}
    for name in vectors['disabled'][1]:
        disabled, enabled = (vectors[mode][1][name] for mode in MODES)
        paired = [None if d is None or e is None else e - d for d, e in zip(disabled, enabled)]
        item = {'disabled': summarize(disabled, indices), 'enabled': summarize(enabled, indices),
                'enabled_minus_disabled': summarize(paired, indices),
                'paired_family_values': [{'family_id': family, 'disabled': d, 'enabled': e, 'difference': diff}
                                         for family, d, e, diff in zip(families, disabled, enabled, paired)],
                'units': 'percentage_points' if name != 'broken_mean_absolute_movement_pp' else 'probability_points'}
        if name == 'broken_mean_absolute_movement_pp':
            item['missing_policy'] = 'Observed common pairs only; no zero imputation'
            for mode, values in (('disabled', disabled), ('enabled', enabled)):
                item[f'{mode}_on_common_pairs'] = summarize([v if diff is not None else None for v, diff in zip(values, paired)], indices)
        else:
            item['missing_policy'] = 'Invalid measurements count as binary failures; every planned family retained'
            item['n_planned_trials'] = len(families) * (2 if name == 'sign_accuracy' else 1)
        result[name] = item
    return result


def diagnostics(units: list[dict], records: dict) -> dict:
    rows = list(records.values())
    generated = [r for r in rows if r['response_received']]
    def distribution(field, selection):
        values = [r[field] for r in selection if r.get(field) is not None]
        return {'n': len(values), 'sum': sum(values), 'mean': float(np.mean(values)) if values else None,
                'median': float(np.median(values)) if values else None, 'max': max(values) if values else None}
    endpoints = []
    for unit in units:
        if unit['context_id'] not in ('positive', 'negative'):
            continue
        base = records[(unit['trial_id'], 'baseline', 'baseline')]
        p = base['probability'] if base['status'] == 'ok' else None
        blocked = None if p is None else p == (1 if unit['context_id'] == 'positive' else 0)
        endpoints.append({'trial_id': unit['trial_id'], 'family_id': unit['family_id'],
                          'context_id': unit['context_id'], 'baseline_probability': p, 'blocked': blocked})
    endpoint_summary = {}
    for context in ('positive', 'negative', 'pooled'):
        selected = [r for r in endpoints if context == 'pooled' or r['context_id'] == context]
        endpoint_summary[context] = {'n_planned': len(selected), 'n_valid_baselines': sum(r['blocked'] is not None for r in selected),
                                     'n_missing_baselines': sum(r['blocked'] is None for r in selected),
                                     'n_direction_blocked': sum(r['blocked'] is True for r in selected)}
    return {'status_counts': dict(Counter(r['status'] for r in rows)),
            'failure_counts': dict(Counter(r['failure_kind'] for r in rows if r.get('failure_kind'))),
            'n_records': len(rows), 'n_responses_received': len(generated),
            'n_reasoning_present': sum(r['reasoning_present'] for r in generated),
            'n_reasoning_closed': sum(r['reasoning_closed'] for r in generated),
            'n_reasoning_empty': sum(r['reasoning_empty'] for r in generated),
            'n_truncated': sum(r['truncated'] for r in generated),
            'max_tokens': sorted({r['max_tokens'] for r in rows}),
            'max_model_len': sorted({r['max_model_len'] for r in rows}),
            'output_tokens_received': distribution('output_token_count', generated),
            'prompt_tokens_received': distribution('prompt_token_count', generated),
            'reasoning_characters_received': distribution('reasoning_char_count', generated),
            'tokens_by_stage': {stage: distribution('output_token_count', [r for r in generated if r['stage'] == stage]) for stage in ('baseline', 'update')},
            'truncation_by_stage': {stage: sum(r['truncated'] for r in generated if r['stage'] == stage) for stage in ('baseline', 'update')},
            'endpoint_blocking': endpoint_summary, 'endpoint_trials': endpoints}


def reference_report(path: Path, units: list[dict], design_hash: str) -> dict:
    reference = json.loads(path.read_text())
    assert_equal(reference['selection']['repeat'], 0, 'Reference repeat')
    assert_equal(reference['selection']['trial_ids_in_plan_order'], [u['trial_id'] for u in units], 'Reference trial order')
    assert_equal(reference['provenance']['frontier_plan']['sha256'], design_hash, 'Reference matched design')
    audit = reference['provenance']['local_repeat0_audit']
    assert_equal(common.file_sha256(Path(audit['path'])), audit['sha256'], 'Reference audit hash')
    original = reference['models']['qwen3_32b']
    summary = original['summary_report']
    return {'role': 'Separate constrained nonthinking repeat0 reference; excluded from paired-mode contrasts',
            'provenance': {'matched_comparison': artifact(path), 'matched_design': reference['provenance']['frontier_plan'],
                           'local_repeat0_audit': audit}, 'deployment': original['deployment'],
            'coverage': summary['coverage']['totals'],
            'sign_accuracy': summary['primary']['direction_correct']['pooled'],
            'paired_reversal': summary['primary']['paired_reversal']['both_directions_correct'],
            'broken_mean_absolute_movement_pp': summary['raw_updates']['broken']['new_news']['mean_absolute_pp']}


def strict_source_report(manifest: dict, records: dict, units: list[dict], model_key: str) -> dict:
    recovery = manifest.get('recovery')
    if not recovery:
        raise ValueError('V2 comparison requires a seeded, source-linked format recovery')
    source_path = Path(recovery['source_response']['path'])
    source_manifest_path = Path(recovery['source_manifest']['path'])
    assert_equal(common.file_sha256(source_path), recovery['source_response']['sha256'], 'Immutable strict source response hash')
    assert_equal(common.file_sha256(source_manifest_path), recovery['source_manifest']['sha256'], 'Immutable strict source manifest hash')
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest['status'] not in ('complete', 'complete_with_errors'):
        raise ValueError('Strict source must be terminal before recovery comparison')
    source_config = source_manifest['config']
    assert_equal(source_config['runner_sha256'], common.file_sha256(Path(strict_runner.__file__)), 'Frozen strict runner hash')
    source_records = strict_runner.load_existing(source_path, source_manifest_path, source_config, units)
    assert_equal(len(source_records), 320, 'Strict source record count')
    assert_equal(dict(Counter(r['status'] for r in source_records.values())), recovery['source_status_counts'], 'Strict source status counts')
    allowed = {tuple(item['key']) for item in recovery['previously_blocked_unattempted_updates']}
    origins = Counter()
    for key, row in records.items():
        original = source_records[key]
        assert_equal(row.get('source_record_sha256'), common.text_sha256(common.canonical_json(original)), 'Source record lineage')
        assert_equal(row.get('source_response_path'), str(source_path), 'Source response lineage path')
        assert_equal(row.get('source_response_sha256'), recovery['source_response']['sha256'], 'Source response lineage hash')
        origin = row.get('recovery_origin')
        origins[origin] += 1
        if origin == 'previously_blocked_unattempted_update':
            if key not in allowed or original['response_received'] or original['status'] != 'blocked_baseline' or key[1] != 'update':
                raise ValueError('Recovery generated a response outside the previously blocked update set')
        elif origin in ('reparsed_received_completion', 'retained_unattempted_failure'):
            for field in ('raw', 'prompt', 'prompt_sha256', 'seed', 'prior_probability', 'prompt_token_count',
                          'output_token_count', 'finish_reason', 'stop_reason', 'created_at', 'response_received'):
                assert_equal(row.get(field), original.get(field), f'Unchanged reused sample {key}/{field}')
        else:
            raise ValueError('Unrecognized recovery lineage')
    summary = analyze.analyze(units, list(source_records.values()), model_key, bootstrap_draws=2000, seed=20260921)
    return {'role': 'Original strict-format v1 results preserved before the symmetric formatting amendment',
            'source_response': recovery['source_response'], 'source_manifest': recovery['source_manifest'],
            'source_run_signature': recovery['source_run_signature'], 'coverage': summary['coverage']['totals'],
            'primary': summary['primary'], 'broken_mean_absolute_movement_pp': summary['raw_updates']['broken']['new_news']['mean_absolute_pp'],
            'diagnostics': diagnostics(units, source_records),
            'format_recovery': {key: recovery[key] for key in ('source_parser_version', 'target_parser_version',
                'reused_received_completions', 'recovered_format_records', 'additional_generation_requests', 'baseline_generation_requests')},
            'validated_v2_lineage_counts': dict(origins)}


def inference_arguments(job: dict) -> argparse.Namespace:
    """Extract the actual v2 CLI from either a direct job or its recovery wrapper."""
    command = job.get('recovery_inference_command', job['inference_command'])
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError('V2 inference command must be a list of strings')
    if command.count('-m') != 1 or command.count('--model-path') != 1:
        raise ValueError('V2 inference command requires one module and one model-path argument')
    module_index = command.index('-m')
    model_index = command.index('--model-path')
    if module_index + 1 >= len(command) or command[module_index + 1] != 'exp1_prospective.context_reversal.run_reasoning_local_v2' or model_index <= module_index + 1:
        raise ValueError('Actual recovery inference command must invoke run_reasoning_local_v2')
    return runner.build_parser().parse_args(command[model_index:])


def compare(submission_path: Path, reference_path: Path, amendment_path: Path | None = None) -> dict:
    submission = json.loads(submission_path.read_text())
    amendment_path = amendment_path or submission_path.parent / 'FORMAT_RECOVERY_AMENDMENT.md'
    for name in ('run_reasoning_local_v2.py', 'run_local.py', 'analyze.py'):
        expected = submission['code_sha256'][name]
        assert_equal(common.file_sha256(SOURCE / name), expected, f'Live dependency {name}')
        assert_equal(common.file_sha256(submission_path.parent / 'code' / name), expected, f'Frozen dependency {name}')
    design_path = Path(submission['probability_design'])
    design_hash = common.file_sha256(design_path)
    assert_equal(design_hash, submission['probability_design_sha256'], 'Design hash')
    units = common.read_units(design_path)
    analyze.validate_design(units)
    assert_equal((len(units), len({u['family_id'] for u in units}), {u['repeat'] for u in units}), (80, 20, {0}), 'Frozen design size')
    jobs = [job for job in submission['jobs'] if job['task'] == 'probability']
    assert_equal(len(jobs), 2, 'Probability arm count')
    assert_equal({job['thinking'] for job in jobs}, set(MODES), 'Probability modes')
    report = {'schema_version': DEFINITIONS['version'], 'status': 'incomplete', 'record_complete': False,
              'created_at': common.utc_now(), 'definitions': DEFINITIONS,
              'comparison_version': 'qwen3_paired_reasoning_comparison_v2',
              'formatting_amendment': {'provenance': artifact(amendment_path), 'parser_version': runner.PARSER_VERSION,
                  'application': 'Both modes accept only one optional surrounding Markdown json or unlabeled fence around strict final JSON; enabled still requires the thinking close',
                  'sampling_policy': 'Reuse every already-received completion; generate only previously blocked unattempted updates; no new baselines or retries of malformed/truncated outputs',
                  'comparison_definitions_changed': False},
              'definitions_sha256': common.text_sha256(common.canonical_json(DEFINITIONS)),
              'provenance': {'comparison_script': artifact(Path(__file__)), 'submission': artifact(submission_path),
                             'design': artifact(design_path), 'dependency_sha256': {name: submission['code_sha256'][name] for name in ('run_reasoning_local_v2.py', 'run_local.py', 'analyze.py')}},
              'expected_records_per_mode': 320, 'arms': {}, 'paired_comparisons': None,
              'original_constrained_reference': reference_report(reference_path, units, design_hash)}
    by_mode, configurations = {}, {}
    for job in jobs:
        mode = job['thinking']
        args = inference_arguments(job)
        for actual, expected, label in ((args.thinking, mode, 'Mode'), (args.model_key, job['model_key'], 'Model key'),
                                       (args.input.resolve(), design_path.resolve(), 'Input'),
                                       (args.output.resolve(), Path(job['response_path']).resolve(), 'Output')):
            assert_equal(actual, expected, label)
        config = runner.make_config(args, units)
        configurations[mode] = config
        output = Path(job['response_path'])
        manifest_path = Path(str(output) + '.manifest.json')
        records = runner.load_existing(output, manifest_path, config, units)
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        expected_keys = {(u['trial_id'], stage, condition) for u in units
                         for stage, condition in [('baseline', 'baseline'), *[('update', arm) for arm in runner.CONDITIONS]]}
        complete = set(records) == expected_keys
        if complete:
            assert_equal(manifest.get('records'), 320, f'{mode} manifest record count')
            assert_equal(manifest.get('counts'), dict(Counter(r['status'] for r in records.values())), f'{mode} manifest status counts')
        by_mode[mode] = records
        arm = {'model_key': job['model_key'], 'present_records': len(records), 'missing_records': 320 - len(records),
               'record_complete': complete, 'manifest_status': manifest.get('status', 'absent'),
               'response': artifact(output) if output.exists() else {'path': str(output), 'sha256': None},
               'manifest': artifact(manifest_path) if manifest_path.exists() else {'path': str(manifest_path), 'sha256': None},
               'run_signature': manifest.get('run_signature'), 'decode': config['decode'],
               'status_counts': dict(Counter(r['status'] for r in records.values()))}
        if manifest.get('recovery'):
            arm['original_strict_format'] = strict_source_report(manifest, records, units, job['model_key'])
        if complete:
            if 'original_strict_format' not in arm:
                raise ValueError('Complete v2 arm lacks strict-source recovery provenance')
            summary = analyze.analyze(units, list(records.values()), job['model_key'], bootstrap_draws=2000, seed=20260921)
            arm.update(diagnostics=diagnostics(units, records), coverage=summary['coverage'], primary=summary['primary'],
                       raw_updates=summary['raw_updates'], broken_stability=summary['broken_stability'])
        report['arms'][mode] = arm
    # Compare everything that can affect inference, except mode/key and their signatures.
    configs = [json.loads(common.canonical_json(configurations[mode])) for mode in MODES]
    for config in configs:
        config.pop('model_key')
        config['decode'].pop('thinking')
        config['decode'].pop('enable_thinking')
    assert_equal(configs[0], configs[1], 'Matched inference settings')
    report['record_complete'] = all(arm['record_complete'] for arm in report['arms'].values())
    if report['record_complete'] and all(arm['manifest_status'] in ('complete', 'complete_with_errors') for arm in report['arms'].values()):
        report['all_valid'] = all(r['status'] == 'ok' for rows in by_mode.values() for r in rows.values())
        report['status'] = 'complete' if report['all_valid'] else 'complete_with_invalid_responses'
        report['paired_comparisons'] = paired_contrasts(units, by_mode)
    else:
        report['incomplete_reason'] = 'Both modes require all 320 validated records and terminal runner manifests before paired reporting'
    return report


def metric_text(metric: dict, scale=1) -> str:
    if metric['estimate'] is None:
        return 'unavailable'
    value = f"{scale * metric['estimate']:.2f}"
    return value + (' [' + ', '.join(f'{scale * x:.2f}' for x in metric['ci95']) + ']' if metric['ci95'] is not None else ' [interval unavailable]')


def markdown(report: dict) -> str:
    lines = ['# Qwen3 reasoning-mode comparison', '', f"Status: **{report['status']}**.", '',
             'Exploratory development diagnostic. New-news minus each mode’s own context baseline defines all primary outcomes. Binary failures retain planned denominators; missing magnitudes are never replaced by zero.', '',
             'Enabled minus disabled contrasts use 2,000 paired whole-family bootstrap draws, seed 20260921. Values and 95% intervals are in percentage/probability points.', '',
             '| Metric | Disabled | Enabled | Enabled minus disabled | Paired families |', '|---|---:|---:|---:|---:|']
    if report['paired_comparisons']:
        for name, item in report['paired_comparisons'].items():
            difference = item['enabled_minus_disabled']
            lines.append(f"| {name} | {metric_text(item['disabled'])} | {metric_text(item['enabled'])} | {metric_text(difference)} | {difference['n_observed_families']} / {difference['n_planned_families']} |")
        broken = report['paired_comparisons']['broken_mean_absolute_movement_pp']
        lines += ['', f"Broken common-pair means: disabled {metric_text(broken['disabled_on_common_pairs'])}; enabled {metric_text(broken['enabled_on_common_pairs'])}. Marginal missing families: disabled {broken['disabled']['n_missing_families']}, enabled {broken['enabled']['n_missing_families']}; missing pairs {broken['enabled_minus_disabled']['n_missing_families']}."]
    else:
        lines += ['', report['incomplete_reason']]
    lines += ['', '| Mode | Present / planned | Status counts | Reasoning present / received | Empty reasoning | Truncated | Output tokens | Mean output tokens | Direction-blocked endpoints / valid baselines |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    diagnostic_notes = []
    for mode in MODES:
        arm = report['arms'][mode]
        d = arm.get('diagnostics')
        counts = ', '.join(f'{key}={value}' for key, value in sorted(arm['status_counts'].items())) or 'none'
        if d:
            endpoint = d['endpoint_blocking']['pooled']
            lines.append(f"| {mode} | {arm['present_records']} / 320 | {counts} | {d['n_reasoning_present']} / {d['n_responses_received']} | {d['n_reasoning_empty']} | {d['n_truncated']} | {d['output_tokens_received']['sum']} | {d['output_tokens_received']['mean']} | {endpoint['n_direction_blocked']} / {endpoint['n_valid_baselines']} |")
            diagnostic_notes += [f"{mode} failure kinds: {json.dumps(d['failure_counts'], sort_keys=True)}; missing directional baselines: {endpoint['n_missing_baselines']}; reasoning closed: {d['n_reasoning_closed']}; mean reasoning characters: {d['reasoning_characters_received']['mean']}."]
        else:
            lines.append(f"| {mode} | {arm['present_records']} / 320 | {counts} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
    for note in diagnostic_notes:
        lines += ['', note]
    reference = report['original_constrained_reference']
    lines += ['', 'The original constrained nonthinking Qwen3 repeat-0 result is a separate descriptive reference (128 output tokens and probability grammar):', '',
              f"Sign accuracy {metric_text(reference['sign_accuracy'], 100)}%; paired reversal {metric_text(reference['paired_reversal'], 100)}%; broken mean absolute movement {metric_text(reference['broken_mean_absolute_movement_pp'])} probability points. It is excluded from the paired contrast.", '',
              f"Reference: `{reference['provenance']['matched_comparison']['path']}`; SHA-256 `{reference['provenance']['matched_comparison']['sha256']}`.", '',
              f"Definition SHA-256: `{report['definitions_sha256']}`. JSON retains exact response/manifest/code provenance, all paired family values, coverage, failures, endpoints, token usage, and truncation diagnostics.", '']
    lines += ['', '## Symmetric formatting amendment', '',
              report['formatting_amendment']['application'] + '.',
              report['formatting_amendment']['sampling_policy'] + '.',
              'All original paired-comparison definitions, denominators, and bootstrap draws/seed are unchanged.', '',
              '| Original strict v1 arm | Valid / planned | Status counts | Strict sign accuracy | Strict paired reversal | Fenced outputs recovered | Newly unblocked update requests |',
              '|---|---:|---|---:|---:|---:|---:|']
    for mode in MODES:
        original = report['arms'][mode].get('original_strict_format')
        if not original:
            lines.append(f'| {mode} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |')
            continue
        coverage, primary, recovery = original['coverage'], original['primary'], original['format_recovery']
        counts = ', '.join(f'{key}={value}' for key, value in sorted(original['diagnostics']['status_counts'].items()))
        lines.append(f"| {mode} | {coverage['valid']} / 320 | {counts} | {metric_text(primary['direction_correct']['pooled'], 100)}% | {metric_text(primary['paired_reversal']['both_directions_correct'], 100)}% | {recovery['recovered_format_records']} | {recovery['additional_generation_requests']} |")
    amendment = report['formatting_amendment']['provenance']
    lines += ['', f"Amendment: `{amendment['path']}`; SHA-256 `{amendment['sha256']}`.", '']
    return '\n'.join(lines)


def self_test():
    units, records = [], {mode: {} for mode in MODES}
    for family in ('a', 'b'):
        for context in ('positive', 'negative', 'broken', 'masked'):
            unit = {'trial_id': f'{family}:{context}:r0', 'family_id': family, 'context_id': context}
            units.append(unit)
            for mode in MODES:
                baseline = .5
                updated = (.7 if context == 'positive' else .3) if mode == 'enabled' else .5
                if context in ('broken', 'masked'):
                    updated = .6 if mode == 'enabled' else .5
                for condition, p in (('baseline', baseline), ('new_news', updated)):
                    records[mode][(unit['trial_id'], 'baseline' if condition == 'baseline' else 'update', condition)] = {'status': 'ok', 'probability': p}
    result = paired_contrasts(units, records)
    assert result['sign_accuracy']['enabled_minus_disabled']['estimate'] == 100
    assert result['paired_reversal']['enabled_minus_disabled']['ci95'] == [100, 100]
    invalid = records['enabled'][('a:positive:r0', 'update', 'new_news')]
    invalid.update(status='parse_error', probability=None)
    broken = records['enabled'][('a:broken:r0', 'update', 'new_news')]
    broken.update(status='parse_error', probability=None)
    result = paired_contrasts(units, records)
    assert result['sign_accuracy']['enabled_minus_disabled']['estimate'] == 75
    assert result['paired_reversal']['enabled_minus_disabled']['estimate'] == 50
    magnitude = result['broken_mean_absolute_movement_pp']['enabled_minus_disabled']
    assert magnitude['n_missing_families'] == 1 and abs(magnitude['estimate'] - 10) < 1e-10
    assert magnitude['ci95'] is None and magnitude['bootstrap_draws_defined'] < 2000
    print('Paired bootstrap, planned failures, and missing-magnitude checks passed.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, default=SOURCE / 'runs/local_diagnostics_v1/submission.json')
    parser.add_argument('--reference', type=Path, default=SOURCE / 'results/frontier_gpt56_pilot_v1/comparison.json')
    parser.add_argument('--output', type=Path, default=SOURCE / 'results/local_diagnostics_v1/reasoning_comparison', help='Output filename stem')
    parser.add_argument('--amendment', type=Path, default=SOURCE / 'runs/local_diagnostics_v1/FORMAT_RECOVERY_AMENDMENT.md')
    parser.add_argument('--definitions-only', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Synthetic offline verification; reads no run outputs')
    args = parser.parse_args()
    if args.definitions_only:
        print(json.dumps(DEFINITIONS, sort_keys=True, indent=2))
        return 0
    if args.self_test:
        self_test()
        return 0
    report = compare(args.submission.resolve(), args.reference.resolve(), args.amendment.resolve())
    common.atomic_write(Path(str(args.output) + '.json'), json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    common.atomic_write(Path(str(args.output) + '.md'), markdown(report))
    print(json.dumps({'status': report['status'], 'record_complete': report['record_complete'], 'output': str(args.output)}))
    return 0 if report['status'].startswith('complete') else 2


if __name__ == '__main__':
    sys.exit(main())
