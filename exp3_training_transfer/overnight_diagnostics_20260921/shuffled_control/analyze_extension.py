#!/usr/bin/env python3
"""Seven-seed shuffled-target contrasts, reported ALONGSIDE the frozen five-seed roster.

freeze_extension.json is explicit that seeds 47-48 are "reported alongside, never
in place of, the frozen five-seed roster", so this writes results_ext4748.json and
never touches results.json. analyze.py is left byte-identical: its output hash is
referenced as parent_results_sha256, and it hardcodes SEEDS=(42..46) and a t-table
that stops at df=4, so it cannot express a seven-seed estimate.

Every statistic is imported from analyze.py rather than restated, so the two
analyses cannot diverge in how an effect, an interval or a bootstrap is computed.
What is re-implemented here is only cell discovery, because the matched arm's
directory naming differs by seed generation:

    seeds 42-44   {disclosure}_matched_recovery_qwen3_4b_s{seed}_*
    seeds 45-46   {disclosure}_matched_hardware_qwen3_4b_s{seed}_*
    seeds 47-48   {disclosure}_matched_ext_qwen3_4b_s{seed}_*      <- the extension
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

import analyze
from analyze import ROOT, sha, read_complete, summarize_values

SEEDS = (42, 43, 44, 45, 46, 47, 48)
# analyze.TCRIT stops at df=4 because five seeds was the frozen target. Seven
# seeds needs df=6; scipy.stats.t.ppf(.975, 6) = 2.446912.
analyze.TCRIT = dict(analyze.TCRIT)
analyze.TCRIT.setdefault(5, 2.570582)
analyze.TCRIT.setdefault(6, 2.446912)


def matched_pattern(disclosure, seed, stem):
    if seed in (47, 48):
        return f'{disclosure}_matched_ext_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
    if seed in (45, 46):
        return f'{disclosure}_matched_hardware_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
    return f'{disclosure}_matched_recovery_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'freeze_sha256': sha(ROOT / 'freeze.json'),
              'extension_freeze_sha256': sha(ROOT / 'freeze_extension.json'),
              'analysis_code_sha256': sha(__file__),
              'parent_analysis_code_sha256': sha(ROOT / 'analyze.py'),
              'primary': 'stochastic5 shuffled minus matched response MAE',
              'seeds': list(SEEDS), 'incomplete_cells': [], 'cells': [], 'contrasts': [], 'sources': {},
              'notes': ['Seven-seed extension of the frozen five-seed roster; reported alongside, never in place of it.',
                        'Statistics are imported from analyze.py; only cell discovery differs.',
                        'Student-t uses df=n-1, so df=6 at the full seven-seed roster.']}
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
    (ROOT / 'results_ext4748.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    lines = ['# Shuffled-target control, seven-seed extension', '',
             'Reported alongside the frozen five-seed roster in results.json, never in place of it.', '',
             f'Updated {report["created_at"]}.',
             f'Completed endpoint/decode cells: {len(report["cells"])}/56; missing: {len(report["incomplete_cells"])}.',
             '', '| Disclosure | Decode | Seeds | Shuffled − matched response MAE | 95% Student-t interval |',
             '|---|---|---:|---:|---|']
    for result in report['contrasts']:
        if result['scope'] == 'overall' and result['metric'] == 'response_mae':
            interval = result.get('student_t_95ci')
            formatted = f'[{interval[0]:+.4f}, {interval[1]:+.4f}]' if interval else 'not estimable'
            lines.append(f'| {result["disclosure"]} | {result["decode"]} | {result["training_seeds"]}/7 '
                         f'| {result["effect"]:+.4f} | {formatted} |')
    lines += ['', 'Positive favors matched training.']
    (ROOT / 'RESULTS_EXT4748.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if args.require_complete and not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
