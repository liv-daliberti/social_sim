#!/usr/bin/env python3
"""Triple-level scoring of the frozen context-reversal cohort.

The design groups positive, negative, and irrelevant contexts into triples with
identical news, exact baseline probability, and named actor. A constant revision
cannot satisfy both signed targets, so it cannot win a whole triple. Because the
unchanged tolerance overlaps either signed target, a tiny nonzero revision can
win two of three directional items. Exact posterior agreement has a separate
one-third bound when the three tolerance intervals do not overlap.

Offline: reads frozen plans and saved response records, verifies their hashes,
writes results. No model is called and no record is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import glob
import json
from pathlib import Path
import re
import statistics as st

from exp1_prospective.context_reversal import run_local as common

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'data/fresh_evaluation_v1/frozen_v1/probability_plan.jsonl'
RESPONSES = HERE / 'responses/fresh_evaluation_v1'
OUT = HERE / 'results/fresh_evaluation_v1'
# Exact agreement to four decimals: the granularity the oracles are stated at.
EXACT = 1e-4
# The protocol's prespecified broken-link stability criterion, in probability units.
STABILITY = 0.02
AUDIT = re.compile(r'probability of X=1 is (\d+/\d+), replacing the initial (\d+/\d+)')

ARMS = [
    ('gpt56_sol', 'GPT-5.6', ['frontier_gpt56_sol.jsonl']),
    ('claude_opus5', 'Claude Opus 5', ['frontier_claude_opus5.jsonl']),
    ('qwen3_32b_off', 'Qwen3-32B thinking off', ['disabled_shard*.jsonl']),
    ('qwen3_32b_on', 'Qwen3-32B thinking on', ['enabled_shard*.jsonl']),
]


def frac(value):
    return float(Fraction(value)) if isinstance(value, str) else float(value)


def load_units():
    units = {}
    for line in PLAN.read_text().splitlines():
        if line.strip():
            unit = json.loads(line)
            units[unit['trial_id']] = unit
    return units


def build_triples(units):
    """Group into (parent, variant) triples and verify the held-constant design."""
    groups = defaultdict(list)
    for unit in units.values():
        groups[(unit['parent_family_id'], unit['variant'])].append(unit)
    triples = {}
    for key, members in groups.items():
        if len(members) != 3:
            raise ValueError(f'Triple {key} does not have three relations')
        for field in ('oracle_baseline', 'reported_entity'):
            if len({m[field] for m in members}) != 1:
                raise ValueError(f'Triple {key} does not hold {field} constant')
        if len({m['messages']['new_news'] for m in members}) != 1:
            raise ValueError(f'Triple {key} does not hold the evidence text constant')
        if sorted(m['expected_sign'] for m in members) != [-1, 0, 1]:
            raise ValueError(f'Triple {key} does not cover the three relations')
        if len({m['oracle_update'] for m in members}) != 3:
            raise ValueError(f'Triple {key} has a repeated correct answer; the bound would not hold')
        triples[key] = sorted(members, key=lambda m: m['expected_sign'])
    return triples


def read_records(patterns):
    records = defaultdict(dict)
    files = sorted(p for pattern in patterns for p in glob.glob(str(RESPONSES / pattern)))
    for path in files:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                records[row['trial_id']][row['condition']] = row
    return records, [{'path': p, 'sha256': common.file_sha256(Path(p))} for p in files]


def valid(row):
    return bool(row) and row.get('status') == 'ok' and isinstance(row.get('probability'), (int, float))


def item_outcomes(unit, records):
    """Per-item result, or None when the arm produced no usable pair."""
    pair = records.get(unit['trial_id'], {})
    baseline, update = pair.get('baseline'), pair.get('new_news')
    if not valid(baseline) or not valid(update):
        return None
    prior, posterior = baseline['probability'], update['probability']
    oracle_b, oracle_u = frac(unit['oracle_baseline']), frac(unit['oracle_update'])
    change, oracle_change = posterior - prior, oracle_u - oracle_b
    sign = unit['expected_sign']
    return {
        'baseline_error': abs(prior - oracle_b), 'update_error': abs(posterior - oracle_u),
        'revision_error': abs(change - oracle_change), 'change': change, 'oracle_change': oracle_change,
        'exact': abs(posterior - oracle_u) <= EXACT,
        'sign_correct': (abs(change) <= EXACT) if sign == 0 else (change * sign > 0),
        'stable': abs(change) <= STABILITY if sign == 0 else None,
        'exact_zero': change == 0.0 if sign == 0 else None,
        'expected_sign': sign,
    }


def shortcut_predictors(triple):
    """Predictors that read only evidence, prior and entity: constant in a triple."""
    prior = frac(triple[0]['oracle_baseline'])
    news = triple[0]['messages']['new_news']
    match = AUDIT.search(news)
    typical = st.median([abs(frac(m['oracle_update']) - prior) for m in triple])
    direction = 0.0
    if match:
        direction = typical if frac(match.group(1)) > frac(match.group(2)) else -typical
    return {
        'no_change': 0.0,
        'evidence_direction': direction,
        'always_up': typical,
    }


def score_arm(label, patterns, units, triples):
    records, sources = read_records(patterns)
    covered = {k: v for k, v in triples.items() if all(m['trial_id'] in records for m in v)}
    per_item, triple_rows = [], []
    for key, members in covered.items():
        outcomes = [item_outcomes(m, records) for m in members]
        per_item.extend(o for o in outcomes if o)
        complete = all(o is not None for o in outcomes)
        triple_rows.append({
            'key': list(key), 'complete': complete,
            'sign_all_correct': complete and all(o['sign_correct'] for o in outcomes),
            'exact_all_correct': complete and all(o['exact'] for o in outcomes),
            'items_valid': sum(o is not None for o in outcomes),
        })
    n_triples = len(triple_rows)
    complete_rows = [t for t in triple_rows if t['complete']]

    def dist(values, label_):
        if not values:
            return {'metric': label_, 'n': 0}
        values = sorted(values)
        pick = lambda p: values[min(len(values) - 1, int(p * len(values)))]
        return {'metric': label_, 'n': len(values), 'mean_pp': 100 * st.mean(values),
                'median_pp': 100 * values[len(values) // 2], 'p90_pp': 100 * pick(.9),
                'max_pp': 100 * values[-1]}

    inert = [o for o in per_item if o['expected_sign'] == 0]
    relevant = [o for o in per_item if o['expected_sign'] != 0]
    return {
        'arm': label, 'sources': sources,
        'triples_planned': n_triples, 'triples_complete': len(complete_rows),
        'triple_sign_unconditional': sum(t['sign_all_correct'] for t in triple_rows),
        'triple_exact_unconditional': sum(t['exact_all_correct'] for t in triple_rows),
        'triple_sign_complete_case': sum(t['sign_all_correct'] for t in complete_rows),
        'items_valid': len(per_item), 'items_planned': 3 * n_triples,
        'item_sign_correct': sum(o['sign_correct'] for o in per_item),
        'item_exact_correct': sum(o['exact'] for o in per_item),
        'errors': [dist([o['baseline_error'] for o in per_item], 'baseline absolute error'),
                   dist([o['update_error'] for o in per_item], 'update absolute error'),
                   dist([o['revision_error'] for o in per_item], 'revision-size error')],
        'inert': {'n': len(inert), 'exact_zero': sum(bool(o['exact_zero']) for o in inert),
                  'within_stability': sum(bool(o['stable']) for o in inert),
                  'movement': dist([abs(o['change']) for o in inert], 'inert absolute movement')},
        'relevant_movement': dist([abs(o['change']) for o in relevant], 'relevant absolute movement'),
        'triple_rows': triple_rows,
    }


def score_shortcuts(triples):
    """Selected constant revisions, including the best of the oracle revisions."""
    results = {}
    names = ['no_change', 'evidence_direction', 'always_up', 'best_constant_with_oracle']
    for name in names:
        item_sign = item_exact = triple_all = 0
        for members in triples.values():
            prior = frac(members[0]['oracle_baseline'])
            if name == 'best_constant_with_oracle':
                # Choose among the three oracle revisions. This candidate set
                # does not optimize over every tolerance-admissible revision.
                options = [frac(m['oracle_update']) - prior for m in members]
                best = max(options, key=lambda c: sum(
                    (abs(c) <= EXACT) if m['expected_sign'] == 0 else (c * m['expected_sign'] > 0)
                    for m in members))
                change = best
            else:
                change = shortcut_predictors(members)[name]
            signs = [(abs(change) <= EXACT) if m['expected_sign'] == 0 else (change * m['expected_sign'] > 0)
                     for m in members]
            exacts = [abs((prior + change) - frac(m['oracle_update'])) <= EXACT for m in members]
            item_sign += sum(signs)
            item_exact += sum(exacts)
            triple_all += all(signs)
        results[name] = {'items': 3 * len(triples), 'item_sign_correct': item_sign,
                         'item_exact_correct': item_exact, 'triples': len(triples),
                         'triple_sign_all_correct': triple_all}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'triple_analysis.json')
    args = parser.parse_args()
    units = load_units()
    triples = build_triples(units)
    report = {
        'schema_version': 'context_reversal_triple_analysis_v1',
        'plan': {'path': str(PLAN), 'sha256': common.file_sha256(PLAN)},
        'units': len(units), 'triples': len(triples),
        'design_invariants_verified': ['identical evidence text', 'identical prior', 'identical entity',
                                       'relations cover {-1, 0, +1}', 'three distinct correct answers'],
        'exact_tolerance': EXACT, 'stability_criterion': STABILITY,
        'impossibility_bound': {
            'statement': ('A constant revision cannot satisfy both signed targets. '
                          'The unchanged tolerance allows a tiny nonzero revision to satisfy '
                          'two of three directional targets, but never a whole triple.'),
            'max_item_accuracy': '2/3', 'max_triple_accuracy': '0',
        },
        'shortcuts': score_shortcuts(triples),
        'arms': [score_arm(label, patterns, units, triples) for _, label, patterns in ARMS],
        'created_at': common.utc_now(), 'code_sha256': common.file_sha256(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_write(args.out, json.dumps(report, indent=2) + '\n')
    for arm in report['arms']:
        if not arm['items_valid']:
            continue
        print(f"{arm['arm']:26s} triples {arm['triple_sign_unconditional']:3d}/{arm['triples_planned']:3d} sign-correct "
              f"(complete {arm['triples_complete']:3d}) | items {arm['item_sign_correct']}/{arm['items_planned']} sign, "
              f"{arm['item_exact_correct']}/{arm['items_planned']} exact")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
