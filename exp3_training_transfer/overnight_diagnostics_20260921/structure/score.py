#!/usr/bin/env python3
"""Paired diagnostic scoring; invalid generations are explicit, not silently dropped."""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np


def forecasts(text):
    try:
        obj = json.loads(text)
        values = obj['forecasts']
        if set(obj) != {'forecasts'} or len(values) != 10:
            return None
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
            return None
        return np.asarray(values, dtype=float)
    except (ValueError, TypeError, KeyError):
        return None


def response(pred):
    return np.array([pred[i] - pred[z] for z, indices in ((2, [0, 1, 3, 4]), (7, [5, 6, 8, 9])) for i in indices])


def mean(values):
    return float(np.mean(values)) if values else None


def ci(values, seed=171000000):
    if not values:
        return None
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def score(records):
    cells = defaultdict(list)
    for record in records:
        ref = record['reference']
        cells[(ref['domain'], ref['label_kind'], ref['interface'])].append(record)
    summaries = []
    for (domain, label, interface), items in sorted(cells.items()):
        pairs = defaultdict(list)
        parsed = 0
        maes, rmaes, acc = [], [], []
        for record in items:
            ref = record['reference']
            row = {'reference': ref, 'valid': False}
            if interface == 'selection_only':
                try:
                    obj = json.loads(record['output'])
                    valid = set(obj) == {'reference'} and type(obj['reference']) is int and obj['reference'] in (1, 2)
                except (ValueError, TypeError):
                    valid = False
                    obj = {}
                correct = valid and obj['reference'] == ref['selected_reference']
                acc.append(float(correct))
                row.update(valid=valid, correct=float(correct), predicted=obj.get('reference'))
            else:
                pred = forecasts(record['output'])
                valid = pred is not None
                span = ref['clip'][1] - ref['clip'][0]
                if valid:
                    pred = np.clip(pred, *ref['clip'])
                    truth = np.asarray(ref['truth_targets'])
                    pred_response = response(pred)
                    row.update(pred=pred_response,
                               mae=float(np.mean(np.abs(pred - truth))),
                               response_mae=float(np.mean(np.abs(pred_response - ref['truth_response']))))
                    direct = np.mean(np.abs(pred_response - ref['direct_template_response']))
                    mediated = np.mean(np.abs(pred_response - ref['mediated_template_response']))
                    predicted_structure = 'direct_a' if direct < mediated else ('mediated_b' if mediated < direct else 'tie')
                    row['structure_correct'] = float(predicted_structure == ref['target_structure'])
                else:
                    row.update(mae=float(span), response_mae=float(span), structure_correct=0.)
                row['valid'] = valid
                maes.append(row['mae'])
                rmaes.append(row['response_mae'])
                acc.append(row['structure_correct'])
            parsed += int(valid)
            pairs[ref['pair_id']].append(row)
        summary = {'domain': domain, 'label_kind': label, 'interface': interface,
                   'n': len(items), 'n_pairs': len(pairs), 'parse_rate': parsed / len(items)}
        paired_accuracy = [mean([r['correct'] if interface == 'selection_only' else r['structure_correct'] for r in pair]) for pair in pairs.values()]
        summary.update(accuracy=mean(acc), accuracy_pair_bootstrap_ci95=ci(paired_accuracy))
        if interface == 'selection_only':
            summary['both_cues_correct_rate'] = mean([float(all(r['correct'] for r in pair)) for pair in pairs.values()])
        else:
            pair_mae = [mean([r['response_mae'] for r in pair]) for pair in pairs.values()]
            calibrated, delta_error, direction = [], [], []
            for pair in pairs.values():
                assert len(pair) == 2
                direct = next(r for r in pair if r['reference']['target_structure'] == 'direct_a')
                mediated = next(r for r in pair if r['reference']['target_structure'] == 'mediated_b')
                if not all(r['valid'] for r in pair):
                    continue
                delta = mediated['pred'] - direct['pred']
                expected = np.array(mediated['reference']['truth_response']) - direct['reference']['truth_response']
                projection = float(delta @ expected / (expected @ expected))
                calibrated.append(projection)
                delta_error.append(float(np.mean(np.abs(delta - expected))))
                direction.append(float(projection > 0))
            summary.update(forecast_mae=mean(maes), response_mae=mean(rmaes),
                           response_mae_pair_bootstrap_ci95=ci(pair_mae),
                           valid_cue_pairs=len(calibrated),
                           cue_change_calibration=mean(calibrated),
                           cue_change_calibration_ci95=ci(calibrated),
                           cue_change_mae=mean(delta_error), cue_direction_accuracy=mean(direction))
        summaries.append(summary)
    return summaries


def write_summary(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    summary = {'model_key': records[0]['model_key'], 'model': records[0]['model'],
               'n': len(records), 'cells': score(records)}
    output = path.with_suffix('.summary.json')
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Scored {len(records)} rows: {output}', flush=True)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    write_summary(parser.parse_args().path)
