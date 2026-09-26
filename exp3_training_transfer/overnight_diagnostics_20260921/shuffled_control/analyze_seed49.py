#!/usr/bin/env python3
"""Eight-seed shuffled-target contrasts (seeds 42-49), reported ALONGSIDE the
frozen five-seed roster and the seven-seed extension, never in place of either.

freeze_seed49.json records that seed 49 was added after the seven-seed result
was known. Its reporting rule is binding here: the eight-seed estimate is an
outcome-informed estimate, its intervals do not carry nominal coverage, and no
eight-seed p-value or interval may be presented as a confirmatory test. This
script writes results_ext49.json and RESULTS_EXT49.md and leaves analyze.py,
analyze_extension.py, results.json and results_ext4748.json byte-identical.

Every statistic is imported from analyze.py through analyze_extension.py; the
only additions are the seed-49 directory pattern (it shares the `_matched_ext_`
naming of seeds 47-48) and the df=7 Student-t critical value.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import numpy as np

import analyze
import analyze_extension as ext
from analyze import ROOT, sha, read_complete, summarize_values

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)
# scipy.stats.t.ppf(.975, 7) = 2.364624
analyze.TCRIT.setdefault(7, 2.364624)


def matched_pattern(disclosure, seed, stem):
    if seed == 49:
        return f'{disclosure}_matched_ext_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
    return ext.matched_pattern(disclosure, seed, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    freeze49 = json.loads((ROOT / 'freeze_seed49.json').read_text())
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'freeze_sha256': sha(ROOT / 'freeze.json'),
              'extension_freeze_sha256': sha(ROOT / 'freeze_extension.json'),
              'seed49_freeze_sha256': sha(ROOT / 'freeze_seed49.json'),
              'analysis_code_sha256': sha(__file__),
              'extension_analysis_code_sha256': sha(ROOT / 'analyze_extension.py'),
              'parent_analysis_code_sha256': sha(ROOT / 'analyze.py'),
              'primary': 'stochastic5 shuffled minus matched response MAE',
              'seeds': list(SEEDS), 'incomplete_cells': [], 'cells': [], 'contrasts': [], 'sources': {},
              'decision_is_outcome_informed': True,
              'inference_caveat': freeze49['inference_caveat'],
              'reporting_rule': freeze49['reporting_rule'],
              'notes': ['Eight-seed extension; seed 49 was declared after the seven-seed result was known.',
                        'Reported alongside the frozen five-seed roster and the seven-seed extension, never in place of them.',
                        'Statistics are imported from analyze.py; only cell discovery and the df=7 t critical value are added.',
                        'No eight-seed interval or p-value is confirmatory; see freeze_seed49.json.']}
    reports = ROOT / 'reports'
    for disclosure in ('disclosed', 'undisclosed'):
        fresh = ROOT.parent / 'evidence_use/data' / f'heldout_{disclosure}.jsonl'
        metadata = {}
        for line in fresh.read_text().splitlines():
            row = json.loads(line); ref = json.loads(row['reference']); metadata[ref['task_id']] = ref
        for decode, stem, draws in [('greedy', 'fresh_greedy', 1), ('stochastic', 'fresh_stochastic_n5', 5)]:
            paired = {}; paired_level = {}
            for seed in SEEDS:
                loaded = {}
                for arm in ('causal_family', 'shuffled_target'):
                    pattern = (f'{disclosure}_shuffled_target_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
                               if arm == 'shuffled_target' else matched_pattern(disclosure, seed, stem))
                    paths = list(reports.glob(pattern))
                    if len(paths) > 1:
                        raise RuntimeError(f'Multiple complete attempts, resolve provenance before analysis: {paths}')
                    if not paths:
                        report['incomplete_cells'].append({'disclosure': disclosure, 'decode': decode,
                                                           'seed': seed, 'arm': arm})
                        continue
                    path = paths[0]; groups = read_complete(path, metadata, draws)
                    loaded[arm] = groups; report['sources'][str(path)] = sha(path)
                    report['cells'].append({'disclosure': disclosure, 'decode': decode, 'seed': seed, 'arm': arm,
                        'rows': len(groups),
                        'response_mae': float(np.mean([r['response_mae'] for v in groups.values() for r in v])),
                        'level_mae': float(np.mean([r['level_mae'] for v in groups.values() for r in v])),
                        'parse_rate': float(np.mean([r['parsed'] for v in groups.values() for r in v]))})
                if len(loaded) == 2:
                    for metric, output in [('response_mae', paired), ('level_mae', paired_level)]:
                        output[seed] = {task: float(np.mean([r[metric] for r in loaded['shuffled_target'][task]])) -
                                              float(np.mean([r[metric] for r in loaded['causal_family'][task]]))
                                        for task in metadata}
            if paired:
                scopes = [('overall', list(metadata))]
                scopes += [(f'k:{k}', [t for t, m in metadata.items() if m['k'] == k]) for k in (3, 6, 9)]
                for scope, tasks in scopes:
                    result = summarize_values(paired, metadata, tasks)
                    result.update(disclosure=disclosure, decode=decode, scope=scope, metric='response_mae')
                    report['contrasts'].append(result)
                result = summarize_values(paired_level, metadata, list(metadata))
                result.update(disclosure=disclosure, decode=decode, scope='overall', metric='level_mae')
                report['contrasts'].append(result)
    report['complete'] = not report['incomplete_cells']
    (ROOT / 'results_ext49.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    lines = ['# Shuffled-target control, eight-seed extension (seed 49)', '',
             'Reported alongside the frozen five-seed roster (results.json) and the seven-seed',
             'extension (results_ext4748.json), never in place of either. Seed 49 was added after',
             'the seven-seed result was known (freeze_seed49.json); the eight-seed estimate is',
             'outcome-informed, its intervals do not carry nominal coverage, and nothing here is a',
             'confirmatory test.', '',
             f'Updated {report["created_at"]}.',
             f'Completed endpoint/decode cells: {len(report["cells"])}/64; missing: {len(report["incomplete_cells"])}.',
             '', '| Disclosure | Decode | Seeds | Shuffled − matched response MAE | 95% Student-t interval '
             '| Hierarchical 95% | Webb p | Seeds > 0 |',
             '|---|---|---:|---:|---|---|---:|---:|']
    for result in report['contrasts']:
        if result['scope'] == 'overall' and result['metric'] == 'response_mae':
            interval = result.get('student_t_95ci')
            formatted = f'[{interval[0]:+.4f}, {interval[1]:+.4f}]' if interval else 'not estimable'
            hier = result.get('hierarchical_seed_trajectory_95ci')
            hier_f = f'[{hier[0]:+.4f}, {hier[1]:+.4f}]' if hier else 'not estimable'
            webb = result.get('wild_cluster_webb_p_two_sided')
            sign = result.get('sign_test', {})
            lines.append(f'| {result["disclosure"]} | {result["decode"]} | {result["training_seeds"]}/8 '
                         f'| {result["effect"]:+.4f} | {formatted} | {hier_f} '
                         f'| {webb if webb is None else f"{webb:.4f}"} '
                         f'| {sign.get("positive", "")}/{sign.get("seeds", "")} |')
    lines += ['', 'Positive favors matched training.', '',
              'Seed-level effects (stochastic, response MAE):']
    for result in report['contrasts']:
        if result['scope'] == 'overall' and result['metric'] == 'response_mae' and result['decode'] == 'stochastic':
            effects = ', '.join(f's{s}: {v:+.4f}' for s, v in result['seed_effects'].items())
            lines.append(f'- {result["disclosure"]}: {effects}')
    (ROOT / 'RESULTS_EXT49.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if args.require_complete and not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
