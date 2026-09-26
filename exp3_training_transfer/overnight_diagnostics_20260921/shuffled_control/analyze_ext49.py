#!/usr/bin/env python3
"""Eight-seed (42-49) shuffled-target contrasts, reusing analyze.py estimators via analyze_extension.

Mirrors analyze_extension.main() exactly except: SEEDS adds 49, TCRIT adds df=7,
matched_pattern maps seed 49 to the *_matched_ext_* directory, and outputs go to the
scratchpad (never into shuffled_control/).
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
SC = Path('/n/fs/similarity/social_sim/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control')
sys.path.insert(0, str(SC))
import analyze
import analyze_extension as ext
from analyze import ROOT, sha, read_complete, summarize_values

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)
analyze.TCRIT.setdefault(7, 2.364624)  # scipy.stats.t.ppf(.975, 7)

def matched_pattern(disclosure, seed, stem):
    if seed == 49:
        return f'{disclosure}_matched_ext_qwen3_4b_s{seed}_*/{stem}.scores.jsonl'
    return ext.matched_pattern(disclosure, seed, stem)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results_ext49.json'

def main():
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'freeze_sha256': sha(ROOT / 'freeze.json'),
              'extension_freeze_sha256': sha(ROOT / 'freeze_extension.json'),
              'seed49_freeze_sha256': sha(ROOT / 'freeze_seed49.json'),
              'analysis_code_sha256': sha(__file__),
              'ext4748_analysis_code_sha256': sha(ROOT / 'analyze_extension.py'),
              'parent_analysis_code_sha256': sha(ROOT / 'analyze.py'),
              'primary': 'stochastic5 shuffled minus matched response MAE',
              'seeds': list(SEEDS), 'incomplete_cells': [], 'cells': [], 'contrasts': [], 'sources': {},
              'notes': ['Eight-seed estimate (outcome-informed amendment, freeze_seed49.json); reported alongside, never in place of, the frozen five-seed roster.',
                        'Statistics are imported from analyze.py; only cell discovery differs.',
                        'Student-t uses df=n-1, so df=7 at the full eight-seed roster.']}
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
                        raise RuntimeError(f'Multiple complete attempts: {paths}')
                    if not paths:
                        report['incomplete_cells'].append({'disclosure': disclosure, 'decode': decode, 'seed': seed, 'arm': arm})
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
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('complete:', report['complete'], 'cells:', len(report['cells']), 'missing:', report['incomplete_cells'])
    for r in report['contrasts']:
        if r['scope'] == 'overall':
            ci = r['student_t_95ci']; hb = r['hierarchical_seed_trajectory_95ci']
            print(f"{r['metric']:13s} {r['disclosure']:12s} {r['decode']:11s} n={r['training_seeds']} eff={r['effect']:+.4f} "
                  f"t=[{ci[0]:+.4f},{ci[1]:+.4f}] hb=[{hb[0]:+.4f},{hb[1]:+.4f}] wild_p={r['wild_cluster_webb_p_two_sided']:.4f} "
                  f"pos={r['sign_test']['positive']}/{r['sign_test']['seeds']} sign_p={r['sign_test']['p_one_sided']:.4f} "
                  f"s49={r['seed_effects']['49']:+.4f}")
main()
