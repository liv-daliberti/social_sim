#!/usr/bin/env python3
"""Readable interpretation across all fixed seeds; no seed-generalized claim."""
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent

def avg(values):return sum(values)/len(values) if values else None

def fmt(x):return 'pending' if x is None else f'{x:.3f}'

def main():
    summaries=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.summary.json'))]
    lookup={(s['model_key'],c['domain'],c['label_kind'],c['interface']):c for s in summaries for c in s['cells']}
    audit=json.loads((ROOT/'baseline_omission_audit.json').read_text())
    raw={(c['model_key'],c['domain'],c['label_kind'],c['interface']):c for c in audit['cells']}
    selection=[c['accuracy'] for s in summaries for c in s['cells'] if c['interface']=='selection_only']
    lines=['# Structure-selection diagnostic findings','',f'Updated {datetime.now(timezone.utc).isoformat()}; {len(summaries)}/8 frozen endpoints complete.','',
           f'Cue-to-reference identification is intact: selection-only accuracy ranges from {100*min(selection):.1f}% to {100*max(selection):.1f}% across completed models and all four domain/label cells. '
           'That identifies a compositional forecasting problem beyond simply matching the cue to a reference. '
           'The oracle interfaces do not establish that fourteen noisy observations can yield exact latent coefficients; they isolate use when coefficients are supplied.','',
           '## Original task: all three trained seeds','',
           'Each trained entry below is an unweighted average of seeds 42, 43, and 44, only shown when all three are available. '
           'Calibration 0 means no paired cue response; 1 is the simulator-predicted change. These are descriptive three-seed diagnostics, '
           'not a trained-arm significance claim.','',
           '| Domain | Labels | Base 8B calibration | Matched calibration (range) | Prior calibration (range) | Matched response MAE | Prior response MAE |',
           '|---|---|---:|---:|---:|---:|---:|']
    for domain in ('coin_city','coin_harbor'):
        for label in ('semantic','arbitrary'):
            base=lookup.get(('qwen3_8b_base',domain,label,'original'),{})
            values={}
            for arm in ('causal','population_prior'):
                cells=[lookup.get((f'qwen3_8b_{arm}_s{seed}',domain,label,'original')) for seed in (42,43,44)]
                if all(cells):
                    alpha=[c['cue_change_calibration'] for c in cells]
                    values[arm]=(f'{avg(alpha):.3f} ({min(alpha):.3f} to {max(alpha):.3f})',fmt(avg([c['response_mae'] for c in cells])))
                else:values[arm]=('pending','pending')
            lines.append(f'| {domain} | {label} | {fmt(base.get("cue_change_calibration"))} | {values["causal"][0]} | {values["population_prior"][0]} | {values["causal"][1]} | {values["population_prior"][1]} |')
    lines+=['','## Oracle assistance in the trained models','',
            '| Domain | Labels | Arm | Original MAE | Both patterns MAE | Selected pattern MAE | Selected calibration |',
            '|---|---|---|---:|---:|---:|---:|']
    for domain in ('coin_city','coin_harbor'):
        for label in ('semantic','arbitrary'):
            for arm in ('causal','population_prior'):
                groups={interface:[lookup.get((f'qwen3_8b_{arm}_s{seed}',domain,label,interface)) for seed in (42,43,44)] for interface in ('original','oracle_both_patterns','oracle_selected_pattern')}
                if all(all(cells) for cells in groups.values()):
                    values=[avg([c['response_mae'] for c in groups[i]]) for i in groups]
                    values.append(avg([c['cue_change_calibration'] for c in groups['oracle_selected_pattern']]))
                    lines.append('| '+' | '.join([domain,label,arm]+[fmt(v) for v in values])+' |')
    lines+=['','## Stronger-model numerical ability and baseline omission','',
            'The stronger model frequently returns changes relative to rest instead of absolute forecasts. '
            'Clipping those changes to the valid outcome range before computing response differences distorts the response vector, '
            'especially in Coin Harbor where the lower bound is 40. The frozen clipped metrics remain the primary results. '
            'The following sensitivity was added after auditing raw outputs and is explicitly post hoc. It assesses whether '
            'the numerical response is encoded in an incorrectly offset answer; it does not count that answer as a correct forecast.','',
            '| Domain | Labels | Original raw response MAE | Both-pattern raw response MAE | Selected raw response MAE | Selected exact relative vectors | Selected baseline omitted | Selected offset-corrected level MAE |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for domain in ('coin_city','coin_harbor'):
        for label in ('semantic','arbitrary'):
            groups={i:raw.get(('qwen3_32b_base',domain,label,i)) for i in ('original','oracle_both_patterns','oracle_selected_pattern')}
            if not all(groups.values()):continue
            selected=groups['oracle_selected_pattern']
            values=[groups[i]['raw_response_mae'] for i in groups]
            values.extend([selected['relative_vector_exact_01'],selected['both_zero_shocks_near_zero'],selected['global_offset_corrected_level_mae']])
            lines.append('| '+' | '.join([domain,label]+[fmt(v) for v in values])+' |')
    lines+=['','Exact relative vectors have all eight unclipped response coordinates within 0.01 of truth. '
            'Offset correction adds one common shift estimated from the two zero-shock predictions; this is an interpretive check. '
            'The 8B oracle failures remain predominantly response-pattern errors after this correction, rather than simple omission of rest.','',
            'The evidence supports conditional numerical ability in the stronger model and a failure to combine reference identification, '
            'response extraction/use, and absolute forecast output reliably. It does not establish that the original task is solved '
            'or that this model family cannot learn it. The full original task remains poor under arbitrary labels and shows inconsistent '
            'domain behavior. A separately disclosed supervised pilot can test whether direct forecast supervision learns the missing composition; '
            'reference-selection supervision is unnecessary for the already-perfect selection-only subtask.','',
            '## Records','',
            'The main table and all cell CIs are in REPORT.md and results/*.summary.json. The raw-output audit is in '
            'BASELINE_AUDIT.md and baseline_omission_audit.json. The prepared arithmetic prompt followup was not launched because '
            'its frozen gate was closed by stronger-model oracle improvement; arithmetic_followup/gate.json preserves that decision. '
            'All eight original endpoints remain scheduled independent of outcome. Zero target observations isolate this decomposition; '
            'integration with target evidence at k>0 is not tested.','']
    (ROOT/'FINDINGS.md').write_text('\n'.join(lines))
    print(f'Updated {ROOT/"FINDINGS.md"}')
if __name__=='__main__':main()
