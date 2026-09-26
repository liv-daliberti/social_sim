#!/usr/bin/env python3
"""Freeze a paired, component-specific inference diagnostic from existing worlds."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP3 = ROOT.parents[1]
SPEC = importlib.util.spec_from_file_location('selection_generator', EXP3 / 'coin_city_structure_selection/make_dataset.py')
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)
from worlds import DOMAINS, forecast_scenarios, response_vector

INTERFACES = ('selection_only', 'original', 'oracle_both_patterns', 'oracle_selected_pattern')
SEED_BASE = 171_000_000
REPLICATES = 24


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pattern(reference, domain):
    forecasts = forecast_scenarios(domain, reference['structure'], [], reference['parameters'])
    # Scenarios D and I each have one positive shock unit at horizons 1 and 3.
    return [(float(forecasts[index]) - domain.baseline) / domain.shock_unit for index in (3, 8)]


def summary(index, reference, domain):
    h1, h3 = pattern(reference, domain)
    return (f'REFERENCE SYSTEM {index}: starting from the resting level, a one-time +1 unit '
            f'{domain.driver} changes {domain.outcome} by {h1:.8f} at horizon 1 and '
            f'{h3:.8f} at horizon 3. The change is proportional to the shock, including '
            'negative shocks; with zero shock the outcome remains at its resting level.')


def selection_prompt(raw_prompt):
    """Same selection interface for diagnostic inference and optional supervised training."""
    prefix = raw_prompt.split('COUNTERFACTUAL FORECASTS', 1)[0]
    return (prefix.replace('Forecasting task', 'Reference-identification task', 1)
            + 'REFERENCE IDENTIFICATION\nWhich reference system does the target background '
              'name? Identify the reference whose background matches the target background. '
              'Return only strict JSON: {"reference":1} or {"reference":2}.')


def build_rows():
    rows = []
    for domain_index, domain_name in enumerate(('coin_city', 'coin_harbor')):
        domain = DOMAINS[domain_name]
        for label_index, label_kind in enumerate(('semantic', 'arbitrary')):
            for replicate in range(REPLICATES):
                seed = SEED_BASE + domain_index * 1_000_000 + label_index * 100_000 + replicate
                pair_id = f'{domain_name}:{label_kind}:r{replicate:02d}'
                for structure in GEN.STRUCTURES:
                    ep = GEN._episode(domain, seed, 0, structure, 'correct', label_kind)
                    row = GEN._row(ep, None, pair_id, f'{pair_id}:{structure}')
                    reference = json.loads(row['reference'])
                    selected = 1 + [ref['structure'] for ref in ep['references']].index(structure)
                    prefix, forecast_suffix = row['input'].split('COUNTERFACTUAL FORECASTS', 1)
                    oracle_prefix = prefix.replace('no response equation or coefficient is provided.', 'verified response summaries are supplied below.')
                    generic = ('VERIFIED RESPONSE SUMMARY\nThese summaries describe the exact response '
                               'patterns. The target starts from its resting level. Use the cue to '
                               'identify its reference when selection is not supplied; use the '
                               'corresponding response pattern to calculate each forecast.')
                    prompts = {
                        'original': row['input'],
                        'selection_only': selection_prompt(row['input']),
                        'oracle_both_patterns': oracle_prefix + generic + '\n\n'
                            + '\n\n'.join(summary(i, ref, domain) for i, ref in enumerate(ep['references'], 1))
                            + '\n\nCOUNTERFACTUAL FORECASTS' + forecast_suffix,
                        'oracle_selected_pattern': oracle_prefix + generic + '\n\n'
                            + f'The intended reference for the target is REFERENCE SYSTEM {selected}.\n'
                            + summary(selected, ep['references'][selected - 1], domain)
                            + '\n\nCOUNTERFACTUAL FORECASTS' + forecast_suffix,
                    }
                    for interface in INTERFACES:
                        metadata = {**reference, 'interface': interface, 'selected_reference': selected,
                                    'latent_seed': seed, 'task_id': f'{pair_id}:{structure}:{interface}',
                                    'diagnostic_protocol': 'structure_components_v1',
                                    'unit_responses': {str(i): pattern(ref, domain)
                                                       for i, ref in enumerate(ep['references'], 1)}}
                        rows.append({'input': prompts[interface], 'reference': metadata})
    return rows


def validate(rows):
    assert len(rows) == 2 * 2 * REPLICATES * 2 * len(INTERFACES)
    by_pair = {}
    for row in rows:
        ref = row['reference']
        assert ref['k'] == 0 and ref['target_inputs'] == [] and ref['target_observed'] == []
        assert 'direct_a' not in row['input'] and 'mediated_b' not in row['input']
        key = (ref['pair_id'], ref['interface'])
        by_pair.setdefault(key, []).append(row)
        if ref['interface'] != 'selection_only':
            predicted = []
            for shock, horizon in zip(ref['scenario_shocks'], ref['scenario_horizons']):
                coef = ref['unit_responses'][str(ref['selected_reference'])][(horizon - 1) // 2]
                predicted.append(DOMAINS[ref['domain']].baseline + shock * float(f'{coef:.8f}'))
            assert np.max(np.abs(np.array(predicted) - ref['truth_targets'])) < 1e-6
    for pair in by_pair.values():
        assert len(pair) == 2
        left, right = (r['reference'] for r in pair)
        assert left['numeric_hash'] == right['numeric_hash']
        assert left['reference_structures'] == right['reference_structures']
        assert left['unit_responses'] == right['unit_responses']
        assert left['selected_reference'] != right['selected_reference']
        assert np.allclose(left['truth_response'][:4], right['truth_response'][:4])
        if left['interface'] in ('original', 'selection_only', 'oracle_both_patterns'):
            a, b = [r['input'].split('TARGET SYSTEM', 1) for r in pair]
            assert a[0] == b[0]
            assert a[1].split('No target-system observations', 1)[1] == b[1].split('No target-system observations', 1)[1]
    return {'rows': len(rows), 'latent_pairs': len(by_pair) // len(INTERFACES),
            'pairs_by_interface': dict(Counter(k[1] for k in by_pair)),
            'checks': ['same numeric evidence across cue flips', 'opposite cued references',
                       'same horizon-1 responses', 'oracle arithmetic reconstructs simulator targets',
                       'only target background changes in original paired prompts', 'no internal structure names']}


def roster():
    entries = [{'key': 'qwen3_8b_base', 'model': 'Qwen/Qwen3-8B', 'arm': 'base', 'seed': 0, 'adapter': None}]
    for arm in ('causal', 'population_prior'):
        for seed in (42, 43, 44):
            paths = list((EXP3 / 'coin_city_structural/reports').glob(
                f'sel_{arm}_qwen3_8b_s{seed}_*/debug_*/saved_models/step_00301/adapter_config.json'))
            if len(paths) != 1:
                raise RuntimeError(f'Expected one final adapter for {arm}/{seed}: {paths}')
            config = json.loads(paths[0].read_text())
            assert config['base_model_name_or_path'] == 'Qwen/Qwen3-8B' and config['r'] == 32
            entries.append({'key': f'qwen3_8b_{arm}_s{seed}', 'model': 'Qwen/Qwen3-8B',
                            'arm': arm, 'seed': seed, 'adapter': str(paths[0].parent),
                            'adapter_config_sha256': digest(paths[0]),
                            'adapter_weights_sha256': digest(paths[0].parent / 'adapter_model.safetensors')})
    entries.append({'key': 'qwen3_32b_base', 'model': 'Qwen/Qwen3-32B', 'arm': 'base', 'seed': 0, 'adapter': None})
    return entries


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--validate-only', action='store_true')
    opts = args.parse_args()
    if opts.validate_only:
        rows = [json.loads(line) for line in (ROOT / 'diagnostic.jsonl').read_text().splitlines()]
        print(json.dumps(validate(rows), indent=2))
        return
    if (ROOT / 'manifest.json').exists():
        raise SystemExit('Frozen manifest exists; refuse to overwrite. Use --validate-only.')
    rows = build_rows()
    checks = validate(rows)
    (ROOT / 'diagnostic.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    (ROOT / 'roster.json').write_text(json.dumps(roster(), indent=2) + '\n')
    paths = ['build_diagnostic.py', 'PROTOCOL.md', 'diagnostic.jsonl', 'roster.json', 'evaluate.py', 'score.py', 'run.sbatch']
    manifest = {'protocol': 'structure_components_v1', 'frozen_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                'seed_base': SEED_BASE, 'replicates_per_domain_label': REPLICATES,
                'validation': checks, 'sha256': {name: digest(ROOT / name) for name in paths},
                'upstream_sha256': {str(p): digest(p) for p in [EXP3 / 'coin_city_structure_selection/make_dataset.py', EXP3 / 'coin_city_structural/worlds.py', EXP3 / 'coin_city_structural/prompt.py']}}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
