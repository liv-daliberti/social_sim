import copy
import json
from pathlib import Path
import pytest
from exp1_prospective.context_reversal import analyze_robustness as analysis, direction_design, run_local, run_direction_local


def jsonl_payload(rows):
    return ''.join(run_local.canonical_json(row) + '\n' for row in rows)


def fixture_rows():
    path = Path(__file__).parents[1] / 'runs/frontier_gpt56_pilot_v1/plan.jsonl'
    source = run_local.read_units(path)
    families = sorted({u['family_id'] for u in source})[:2]
    units = []
    for original in source:
        if original['family_id'] not in families:
            continue
        for variant in analysis.VARIANTS:
            unit = copy.deepcopy(original)
            parent = original['family_id']
            unit.update(parent_family_id=parent, sampling_family_id=parent, variant=variant,
                        family_id=parent + '_' + variant,
                        trial_id=parent + '_' + variant + ':' + original['context_id'] + ':r0')
            units.append(unit)
    probability_hash = run_local.text_sha256(jsonl_payload(units))
    directions = direction_design.compile_plan(units, probability_hash)
    by_id = {u['trial_id']: u for u in units}
    for unit in directions:
        for key in ('parent_family_id', 'sampling_family_id', 'variant'):
            unit[key] = by_id[unit['trial_id']][key]
    configs = {task: {'model_key': 'test', 'input_sha256': run_local.text_sha256(jsonl_payload(rows)),
                      'decode': {'seed': 20260921}}
               for task, rows in [('probability', units), ('direction', directions)]}
    numbers, labels = [], []
    for unit in units:
        for condition in ('baseline', *analysis.ARMS):
            value = (.5 if condition != 'new_news' or unit['context_id'] in ('broken', 'masked') else
                     .6 if unit['context_id'] == 'positive' else .4)
            baseline = condition == 'baseline'
            prompt = unit['baseline_prompt'] if baseline else run_local.render_update(unit, condition, .5)
            row = run_local.base_record(unit, 'baseline' if baseline else 'update', condition,
                                        prompt, configs['probability'], None if baseline else .5)
            row.update(status='ok', probability=value, raw=json.dumps({'probability': value}))
            numbers.append(row)
    for unit in directions:
        for condition in analysis.ARMS:
            label = analysis.cat.expected_direction(unit['context_id'], condition) or 'unchanged'
            row = run_direction_local.base_record(unit, condition, configs['direction'])
            row.update(status='ok', direction=label, raw=json.dumps({'direction': label}))
            labels.append(row)
    return units, directions, numbers, labels


def test_variants_are_clustered_and_correct_invariance_reported():
    u,d,n,c=fixture_rows();r=analysis.analyze(u,d,n,c,'test',draws=100)
    assert r['n_sampling_families']==2 and r['n_variants']==4
    assert r['coverage']['probability']['planned']==128
    assert r['by_variant']['name_only']['numeric_sign_correct']['n_success']==4
    assert r['invariance_vs_repaired_base']['paraphrase']['new_news']['within_2pp']['n_success']==8
    assert r['invariance_vs_repaired_base']['resample']['new_news']['absolute_difference_minus_resample_pp']['estimate']==0


def test_missing_output_stays_in_unconditional_denominator():
    u,d,n,c=fixture_rows()
    n=[row for row in n if not(row['family_id']=='dev_01_name_only' and row['context_id']=='positive' and row['condition']=='new_news')]
    r=analysis.analyze(u,d,n,c,'test',draws=100)
    metric=r['by_variant']['name_only']['numeric_sign_correct']
    assert metric['n_planned']==4 and metric['n_success']==3 and metric['n_missing']==1
    assert not r['record_complete']
    assert r['invariance_vs_repaired_base']['name_only']['new_news']['within_2pp']['n_planned']==8


def test_consistent_wrong_labels_are_not_jointly_correct():
    u,d,n,c=fixture_rows()
    for row in c:
        if row['context_id']=='positive' and row['condition']=='new_news':
            row.update(direction='decrease',raw=json.dumps({'direction':'decrease'}))
    r=analysis.analyze(u,d,n,c,'test',draws=100)
    metric=r['invariance_vs_repaired_base']['name_only']['new_news']
    assert metric['direction_agreement']['n_success']==8
    assert metric['both_direction_judgments_correct']['n_success']==4
    assert r['by_variant']['name_only']['direction_paired_reversal']['n_success']==0


def test_resample_must_keep_identical_visible_text():
    u,d,n,c=fixture_rows()
    next(row for row in u if row['variant']=='resample')['baseline_prompt']+=' Altered.'
    with pytest.raises(ValueError):analysis.analyze(u,d,n,c,'test',draws=10)


@pytest.mark.parametrize('field', ['prompt', 'prompt_sha256', 'parent_plan_sha256'])
@pytest.mark.parametrize('mode', ['missing', 'incorrect'])
def test_direction_requires_exact_prompt_hash_and_parent_provenance(field, mode):
    units, directions, numbers, labels = fixture_rows()
    if mode == 'missing':
        labels[0].pop(field)
    else:
        labels[0][field] = 'incorrect'
    with pytest.raises(ValueError, match='Direction response|provenance'):
        analysis.analyze(units, directions, numbers, labels, 'test', draws=10)


@pytest.mark.parametrize('field', ['prior_probability', 'probability'])
def test_direction_rejects_nonnull_numeric_forecasts_but_allows_null_fields(field):
    units, directions, numbers, labels = fixture_rows()
    labels[0][field] = None
    assert analysis.analyze(units, directions, numbers, labels, 'test', draws=10)['all_valid']
    labels[0][field] = .5
    with pytest.raises(ValueError, match='numeric forecast or prior'):
        analysis.analyze(units, directions, numbers, labels, 'test', draws=10)


@pytest.mark.parametrize('defect', ['self_blocked_baseline', 'missing_baseline', 'blocked_valid_baseline', 'ok_update_after_failed_baseline'])
def test_impossible_baseline_update_records_fail_closed(defect):
    units, directions, numbers, labels = fixture_rows()
    baseline = numbers[0]
    update = next(row for row in numbers if row['trial_id'] == baseline['trial_id'] and row['condition'] == 'new_news')
    if defect == 'self_blocked_baseline':
        baseline.update(status='blocked_baseline', probability=None, raw='')
    elif defect == 'missing_baseline':
        numbers.remove(baseline)
    elif defect == 'blocked_valid_baseline':
        update.update(status='blocked_baseline', probability=None, raw='', prompt=None, prompt_sha256=None, prior_probability=None)
    else:
        baseline.update(status='generation_error', probability=None, raw='')
        update.update(prompt=None, prompt_sha256=None, prior_probability=None)
    with pytest.raises(ValueError, match='baseline'):
        analysis.analyze(units, directions, numbers, labels, 'test', draws=10)


def test_actual_blocked_records_are_retained_as_failures_with_complete_coverage():
    units, directions, numbers, labels = fixture_rows()
    trial = numbers[0]['trial_id']
    for row in numbers:
        if row['trial_id'] != trial:
            continue
        if row['condition'] == 'baseline':
            row.update(status='parse_error', probability=None, raw='invalid output')
        else:
            row.update(status='blocked_baseline', probability=None, raw='', prompt=None,
                       prompt_sha256=None, prior_probability=None, response_received=False)
    report = analysis.analyze(units, directions, numbers, labels, 'test', draws=10)
    assert report['record_complete'] and not report['all_valid']
    assert report['coverage']['probability']['reason:blocked_baseline'] == 3
    assert report['by_variant']['repaired_base']['numeric_sign_correct']['n_planned'] == 4
    assert report['by_variant']['repaired_base']['numeric_sign_correct']['n_missing'] == 1


def test_wrong_own_context_prior_rejected():
    units, directions, numbers, labels = fixture_rows()
    update = next(row for row in numbers if row['condition'] == 'new_news')
    update['prior_probability'] = .3
    with pytest.raises(ValueError, match='wrong prior'):
        analysis.analyze(units, directions, numbers, labels, 'test', draws=10)


def test_controls_are_separate_and_markdown_shows_missingness_and_joint_correctness():
    units, directions, numbers, labels = fixture_rows()
    trial = numbers[0]['trial_id']
    numbers = [row for row in numbers if (row['trial_id'], row['condition']) != (trial, 'no_news')]
    repeated = next(row for row in numbers if (row['trial_id'], row['condition']) == (trial, 'repeated_news'))
    repeated.update(probability=.55, raw=json.dumps({'probability': .55}))
    repeated_label = next(row for row in labels if (row['trial_id'], row['condition']) == (trial, 'repeated_news'))
    repeated_label.update(direction='increase', raw=json.dumps({'direction': 'increase'}))
    report = analysis.analyze(units, directions, numbers, labels, 'test', draws=10)
    controls = report['by_variant']['repaired_base']['controls_by_condition']
    assert set(controls) == {'no_news', 'repeated_news'}
    assert controls['no_news']['numeric_within_2pp']['n_planned'] == 8
    assert controls['no_news']['numeric_within_2pp']['n_missing'] == 1
    assert controls['no_news']['numeric_movement']['mean_absolute_pp']['n_observed'] == 7
    assert controls['repeated_news']['numeric_within_2pp']['n_success'] == 7
    assert controls['repeated_news']['direction_unchanged']['n_success'] == 7
    text = analysis.markdown(report)
    assert 'Observed/planned differences' in text and 'Both direction judgments correct' in text
    assert 'masked new-news' in text and '| probability | 128 | 127 | 127 | 1 | 0 |' in text
    assert '| repaired_base | no_news | 7/8 | 0.000 | 7/8 | 8/8 |' in text
    assert '| repaired_base | repeated_news | 7/8 |' in text


def test_cli_rejects_direction_parent_hash_mismatching_actual_probability_file(tmp_path):
    units, directions, numbers, labels = fixture_rows()
    for row in directions:
        row['parent_plan_sha256'] = 'b' * 64
    arguments = []
    for name, rows in [('probability-design', units), ('direction-design', directions),
                       ('probability-responses', numbers), ('direction-responses', labels)]:
        path = tmp_path / (name + '.jsonl')
        path.write_text(jsonl_payload(rows))
        arguments.extend(['--' + name, str(path)])
    arguments.extend(['--model-key', 'test', '--output', str(tmp_path / 'results')])
    with pytest.raises(ValueError, match='actual probability design'):
        analysis.main(arguments)
    assert not (tmp_path / 'results' / 'summary.json').exists()
