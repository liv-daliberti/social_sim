#!/usr/bin/env python3
"""Unconditional accounting for every planned call in the frozen cohort.

Every planned record is classified and every category counts in the denominator.
The split that matters for reading the results is model failure versus instrument
failure: a refusal or unparseable answer is the system's, while a truncation at
the output budget or a request killed mid-batch by an allocation boundary is
ours. Both are failures, neither is dropped, and the distinction is reported
rather than folded away.

Offline: reads frozen plans and saved records, verifies hashes, writes results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path

from exp1_prospective.context_reversal import run_local as common

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'data/fresh_evaluation_v1/frozen_v1/probability_plan.jsonl'
FRONTIER_PLAN = HERE / 'data/fresh_evaluation_v1/frozen_v1/frontier_plan.jsonl'
RESPONSES = HERE / 'responses/fresh_evaluation_v1'
OUT = HERE / 'results/fresh_evaluation_v1'

ARMS = [
    ('GPT-5.6', ['frontier_gpt56_sol.jsonl'], FRONTIER_PLAN),
    ('Claude Opus 5', ['frontier_claude_opus5.jsonl'], FRONTIER_PLAN),
    ('Qwen3-32B thinking off', ['disabled_shard*.jsonl'], PLAN),
    ('Qwen3-32B thinking on', ['enabled_shard*.jsonl'], PLAN),
]
# category -> (attribution, human label)
TAXONOMY = {
    'ok': ('none', 'Valid answer'),
    'model_refusal': ('model', 'Declined by a safety classifier'),
    'model_unparseable': ('model', 'Answer not a valid probability'),
    'model_framing': ('model', 'Reasoning frame malformed'),
    'instrument_truncation': ('instrument', 'Hit the output token budget'),
    'instrument_harness_kill': ('instrument', 'Killed mid-batch by an allocation boundary'),
    'cascade_blocked': ('cascade', 'Blocked by its own failed baseline'),
    'not_attempted': ('cascade', 'Never dispatched'),
    'pending': ('pending', 'Run still in progress; not yet attributable'),
}


def classify(row):
    if row is None:
        return 'not_attempted'
    status, kind = row.get('status'), row.get('failure_kind')
    error_kind = (row.get('error') or {}).get('kind') if isinstance(row.get('error'), dict) else None
    if status == 'ok':
        return 'ok'
    if status == 'blocked_baseline' or kind == 'blocked_baseline' or error_kind == 'baseline_has_no_valid_probability':
        return 'cascade_blocked'
    if error_kind == 'not_attempted_after_fatal_error':
        return 'not_attempted'
    if kind == 'truncated' or error_kind == 'truncated_output_budget' or row.get('finish_reason') == 'length':
        return 'instrument_truncation'
    if kind == 'interrupted_unknown':
        return 'instrument_harness_kill'
    if error_kind == 'model_refusal' or row.get('response_status') == 'refusal':
        return 'model_refusal'
    if kind in {'missing_reasoning_close', 'missing_reasoning_open', 'invalid_reasoning_frame'}:
        return 'model_framing'
    return 'model_unparseable'


def run_in_progress(files):
    """A record can only be a kill once its run has stopped writing.

    reasoning.py writes terminal `interrupted_unknown` placeholders before it
    dispatches a batch and overwrites them when the batch returns, so a live run
    always shows one batch in that state. Counting those as failures would
    misreport an arm that is merely still going.
    """
    for path in files:
        manifest = Path(str(path) + '.manifest.json')
        if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'running':
            return True
    return False


def score(label, patterns, plan):
    units = [json.loads(l) for l in plan.read_text().splitlines() if l.strip()]
    files = sorted(p for pattern in patterns for p in glob.glob(str(RESPONSES / pattern)))
    records = defaultdict(dict)
    for path in files:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                records[row['trial_id']][row['condition']] = row
    planned = [(u['trial_id'], c) for u in units for c in ('baseline', *common.CONDITIONS)]
    counts = Counter(classify(records.get(t, {}).get(c)) for t, c in planned)
    in_progress = run_in_progress(files)
    if in_progress:
        # Reclassify: nothing is attributable until the run stops writing.
        for category in ('instrument_harness_kill', 'not_attempted'):
            if counts.pop(category, 0):
                counts['pending'] = counts.get('pending', 0) + 0
        counts['pending'] = len(planned) - counts.get('ok', 0) - sum(
            v for k, v in counts.items() if k not in ('ok', 'pending'))
    attributed = Counter()
    for category, n in counts.items():
        attributed[TAXONOMY.get(category, ('pending',))[0]] += n
    return {'arm': label, 'planned_records': len(planned), 'categories': dict(counts),
            'in_progress': in_progress, 'by_attribution': dict(attributed),
            'sources': [{'path': p, 'sha256': common.file_sha256(Path(p))} for p in files],
            'plan': {'path': str(plan), 'sha256': common.file_sha256(plan)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'unconditional_accounting.json')
    args = parser.parse_args()
    arms = [score(label, patterns, plan) for label, patterns, plan in ARMS]
    report = {'schema_version': 'context_reversal_unconditional_accounting_v1',
              'rule': ('Every planned call appears in the denominator. Model failures, instrument failures and '
                       'failures cascading from a failed baseline are reported separately and none is dropped.'),
              'taxonomy': {k: {'attribution': v[0], 'label': v[1]} for k, v in TAXONOMY.items()},
              'arms': arms, 'created_at': common.utc_now(), 'code_sha256': common.file_sha256(Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_write(args.out, json.dumps(report, indent=2) + '\n')

    order = list(TAXONOMY)
    header = f"{'arm':26s} {'planned':>8s} " + ' '.join(f'{k.split("_",1)[-1][:9]:>10s}' for k in order)
    print(header)
    for a in arms:
        row = ' '.join(f"{a['categories'].get(k, 0):>10d}" for k in order)
        print(f"{a['arm']:26s} {a['planned_records']:>8d} {row}")
    print()
    for a in arms:
        att = a['by_attribution']
        n = a['planned_records']
        flag = '  [run in progress]' if a['in_progress'] else ''
        print(f"{a['arm']:26s} valid {a['categories'].get('ok',0):4d}/{n} | "
              f"model {att.get('model',0):4d} | instrument {att.get('instrument',0):4d} | "
              f"cascade {att.get('cascade',0):4d} | pending {att.get('pending',0):4d}{flag}")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
