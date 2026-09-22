#!/usr/bin/env python3
"""Render completed diagnostic outputs without changing frozen metrics or data."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    summaries = [json.loads(p.read_text()) for p in sorted((ROOT / 'results').glob('*.summary.json'))]
    expected = json.loads((ROOT / 'roster.json').read_text())
    keys = {s['model_key'] for s in summaries}
    text = ['# Structure component diagnostic results', '',
            f'Updated {datetime.now(timezone.utc).isoformat()}. {len(summaries)}/{len(expected)} frozen model endpoints complete.', '',
            'Fresh fixed evaluation: 96 paired latent episodes, 24 per domain × label cell; '
            'zero target observations; 768 prompts per model. Greedy JSON-constrained inference. '
            'Selection accuracy tests cue-to-reference identification. Forecast response MAE measures '
            'numerical forecast quality. Cue-change calibration is 0 for cue-invariant forecasts '
            'and 1 for the simulator-predicted paired change. These are diagnostic results, not '
            'the registered multi-draw endpoint or a seed-level significance claim.', '',
            '## Aggregate overview', '',
            '| Model | Interface | Parse | Accuracy | Response MAE | Cue-change calibration |',
            '|---|---|---:|---:|---:|---:|']
    for summary in summaries:
        for interface in ('selection_only','original','oracle_both_patterns','oracle_selected_pattern'):
            cells = [c for c in summary['cells'] if c['interface']==interface]
            def avg(field):
                values=[c[field] for c in cells if c.get(field) is not None]
                return sum(values)/len(values) if values else None
            text.append(f'| {summary["model_key"]} | {interface} | {fmt(avg("parse_rate"))} | {fmt(avg("accuracy"))} | {fmt(avg("response_mae"))} | {fmt(avg("cue_change_calibration"))} |')
    text += ['', 'Accuracy is reference identification for selection-only; nearest-template structure accuracy for forecasts. '
             'Overview averages the four equally sized domain × label cells. See below for transfer boundaries.']
    for summary in summaries:
        text += ['', '## '+summary['model_key'], '',
                 '| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |',
                 '|---|---|---|---:|---:|---:|---:|---:|']
        for c in summary['cells']:
            text.append(f'| {c["domain"]} | {c["label_kind"]} | {c["interface"]} | {fmt(c["parse_rate"])} | {fmt(c["accuracy"])} | {fmt(c.get("response_mae"))} | {fmt(c.get("cue_change_calibration"))} | {c.get("valid_cue_pairs", c["n_pairs"])} |')
    text += ['', '## Coverage and limitations', '',
             'Completed: '+', '.join(sorted(keys))+'.',
             'Remaining: '+(', '.join(e['key'] for e in expected if e['key'] not in keys) or 'none')+'.', '',
             'The oracle gives exact latent per-unit effects, while original trajectories are noisy. '
             'The diagnostic therefore tests whether numerical assistance removes a bottleneck; '
             'it does not claim those coefficients can be estimated exactly from fourteen rows. '
             'There are no target observations, so target-evidence integration remains outside this diagnostic. '
             'Cell CIs and invalid-generation penalties are recorded in the JSON summaries. '
             'See PROTOCOL.md and amendment_01_oracle_clarity.json for the frozen design and preregeneration wording correction.', '']
    (ROOT/'REPORT.md').write_text('\n'.join(text))
    fields=['model_key','domain','label_kind','interface','n','n_pairs','parse_rate','accuracy','response_mae','forecast_mae','cue_change_calibration','cue_change_mae','cue_direction_accuracy','valid_cue_pairs','both_cues_correct_rate']
    with (ROOT/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader()
        for s in summaries:
            for cell in s['cells']:writer.writerow({'model_key':s['model_key'],**cell})
    print(f'Rendered {len(summaries)} model endpoints to {ROOT / "REPORT.md"}')

if __name__=='__main__':main()
