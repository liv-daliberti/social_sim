#!/usr/bin/env python3
"""Score the ladder calibration by rung: does difficulty rise with binding depth?

Same measures as the frozen cohort -- whole triples, item direction, exact
agreement with the oracle -- computed per rung so the rungs are comparable to
each other and to the cohort's L0-equivalent result. Denominators are
unconditional. Development material: the negative outcome is reported in
App. app:exp1-ladder as an exploratory probe, and supports no capability claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import glob
import json
from pathlib import Path
import statistics as st

from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal.ladder import materials as ladder

HERE = Path(__file__).resolve().parents[1]
PLAN = HERE / 'data/ladder_dev_v1/probability_plan.jsonl'
RESPONSES = HERE / 'responses/ladder_dev_v1'
OUT = HERE / 'results/ladder_dev_v1'
EXACT = 1e-4


def frac(value):
    return float(Fraction(value)) if isinstance(value, str) else float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'ladder_calibration.json')
    args = parser.parse_args()
    units = {u['trial_id']: u for u in common.read_units(PLAN)}
    rows = defaultdict(dict)
    files = sorted(glob.glob(str(RESPONSES / '*.jsonl')))
    for path in files:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row['trial_id']][row['condition']] = row

    by_rung = defaultdict(lambda: {'items': 0, 'valid': 0, 'sign': 0, 'exact': 0, 'errors': [],
                                   'triples': set(), 'triple_ok': set(), 'triple_seen': defaultdict(list)})
    for trial, unit in units.items():
        rung = unit['variant']
        cell = by_rung[rung]
        cell['items'] += 1
        key = (unit['parent_family_id'], rung)
        cell['triples'].add(key)
        pair = rows.get(trial, {})
        baseline, update = pair.get('baseline'), pair.get('new_news')
        ok = (baseline and baseline.get('status') == 'ok' and update and update.get('status') == 'ok')
        if not ok:
            cell['triple_seen'][key].append(False)
            continue
        cell['valid'] += 1
        change = update['probability'] - baseline['probability']
        oracle_change = frac(unit['oracle_update']) - frac(unit['oracle_baseline'])
        sign = unit['expected_sign']
        correct = (abs(change) <= EXACT) if sign == 0 else (change * sign > 0)
        exact = abs(update['probability'] - frac(unit['oracle_update'])) <= EXACT
        cell['sign'] += correct
        cell['exact'] += exact
        cell['errors'].append(abs(update['probability'] - frac(unit['oracle_update'])))
        cell['triple_seen'][key].append(bool(correct))

    report = {'schema_version': 'ladder_calibration_v1', 'development_only': True,
              'plan': {'path': str(PLAN), 'sha256': common.file_sha256(PLAN)},
              'responses': [{'path': p, 'sha256': common.file_sha256(Path(p))} for p in files],
              'exact_tolerance': EXACT, 'rungs': [], 'created_at': common.utc_now(),
              'code_sha256': common.file_sha256(Path(__file__))}
    for rung in ladder.RUNGS:
        cell = by_rung.get(rung)
        if not cell:
            continue
        triples = len(cell['triples'])
        won = sum(1 for k in cell['triples']
                  if len(cell['triple_seen'][k]) == 3 and all(cell['triple_seen'][k]))
        errors = sorted(cell['errors'])
        report['rungs'].append({
            'rung': rung, 'depth': ladder.RUNGS.index(rung), 'triples': triples, 'triples_won': won,
            'items': cell['items'], 'valid': cell['valid'], 'sign_correct': cell['sign'],
            'exact_correct': cell['exact'],
            'posterior_error_mean_pp': 100 * st.mean(errors) if errors else None,
            'posterior_error_median_pp': 100 * errors[len(errors) // 2] if errors else None})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_write(args.out, json.dumps(report, indent=2) + '\n')
    print(f"{'rung':24s} {'triples':>10s} {'items sign':>12s} {'items exact':>13s} {'err mean pp':>12s}")
    for r in report['rungs']:
        err = '--' if r['posterior_error_mean_pp'] is None else f"{r['posterior_error_mean_pp']:.3f}"
        print(f"{r['rung']:24s} {r['triples_won']:4d}/{r['triples']:<5d} "
              f"{r['sign_correct']:5d}/{r['items']:<6d} {r['exact_correct']:5d}/{r['items']:<7d} {err:>12s}")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
