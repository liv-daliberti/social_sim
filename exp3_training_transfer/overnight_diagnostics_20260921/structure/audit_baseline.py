#!/usr/bin/env python3
"""Outcome-motivated raw-response sensitivity; frozen clipped metrics stay unchanged."""
import collections
import datetime
import json
from pathlib import Path
import numpy as np
from score import forecasts,response,ci

ROOT=Path(__file__).resolve().parent


def main():
    output={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'Post hoc sensitivity motivated by stronger-model forecasts omitting the resting baseline. Frozen primary metrics unchanged.',
            'definitions':{
                'raw_response':'Unclipped prediction minus its zero-shock prediction at the corresponding horizon.',
                'relative_vector_exact_01':'All eight raw response coordinates within 0.01 of simulator response.',
                'global_offset_corrected_level_mae':'Add baseline minus mean(predicted zero-shock values at h1 and h3) to every forecast before MAE.',
                'horizon_offset_corrected_level_mae':'Add baseline minus predicted zero-shock value separately within each horizon; equals .8 times raw-response MAE.',
                'zero_shock_offset':'Predicted zero-shock forecast minus true domain resting level; negative baseline indicates omission.',
                'selection':'Selection-only uses no forecasts and is excluded from this numerical audit.'},'cells':[]}
    for path in sorted((ROOT/'results').glob('*.jsonl')):
        cells=collections.defaultdict(list)
        for line in path.read_text().splitlines():
            row=json.loads(line);m=row['reference']
            if m['interface']=='selection_only':continue
            pred=forecasts(row['output'])
            if pred is None:continue
            baseline=50. if m['domain']=='coin_city' else 180.
            resp=response(pred);truth=np.array(m['truth_targets']);response_error=np.abs(resp-m['truth_response'])
            offsets=pred[[2,7]]-baseline
            corrected=pred-float(offsets.mean())
            horizon_corrected=pred-np.repeat(offsets,5)
            cell={'reference':m,'predicted_response':resp,
                  'raw_response_mae':float(response_error.mean()),
                  'raw_forecast_mae':float(np.abs(pred-truth).mean()),
                  'relative_vector_exact_01':float(response_error.max()<=.01),
                  'relative_coordinate_exact_01':float((response_error<=.01).mean()),
                  'global_offset_corrected_level_mae':float(np.abs(corrected-truth).mean()),
                  'horizon_offset_corrected_level_mae':float(np.abs(horizon_corrected-truth).mean()),
                  'h1_zero_shock_offset':float(offsets[0]),'h3_zero_shock_offset':float(offsets[1]),
                  'zero_shock_absolute_offset':float(np.abs(offsets).mean()),
                  'both_zero_shocks_near_rest':float(np.max(np.abs(offsets))<=.01),
                  'both_zero_shocks_near_zero':float(np.max(np.abs(pred[[2,7]]))<=.01),
                  'out_of_range_fraction':float(((pred<m['clip'][0])|(pred>m['clip'][1])).mean())}
            cells[(m['domain'],m['label_kind'],m['interface'])].append(cell)
        for (domain,label,interface),rows in sorted(cells.items()):
            pairs=collections.defaultdict(list)
            for row in rows:pairs[row['reference']['pair_id']].append(row)
            alpha=[]
            for pair in pairs.values():
                if len(pair)!=2:continue
                direct=next(r for r in pair if r['reference']['target_structure']=='direct_a')
                mediated=next(r for r in pair if r['reference']['target_structure']=='mediated_b')
                pred=mediated['predicted_response']-direct['predicted_response']
                truth=np.array(mediated['reference']['truth_response'])-direct['reference']['truth_response']
                alpha.append(float(pred@truth/(truth@truth)))
            fields=[k for k in rows[0] if k not in ('reference','predicted_response')]
            summary={'model_key':path.stem,'domain':domain,'label_kind':label,'interface':interface,'n_valid':len(rows),
                     'valid_pairs':len(alpha),'unclipped_cue_change_calibration':float(np.mean(alpha)),
                     'unclipped_cue_change_calibration_ci95':ci(alpha),
                     **{key:float(np.mean([row[key] for row in rows])) for key in fields}}
            output['cells'].append(summary)
    (ROOT/'baseline_omission_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# Post hoc baseline and raw-response sensitivity','',output['status'],'',
           'This audit subtracts each horizon\'s predicted zero-shock output before calculating response error, '
           'without clipping predictions first. It separates an omitted resting level from incorrect response patterns. '
           'Exact relative accuracy means all eight response coordinates are within 0.01. Offset-corrected MAE adds '
           'one common shift estimated from the two zero-shock predictions; it is an interpretive sensitivity, '
           'not a valid submitted forecast or replacement primary outcome.','',
           '| Model | Domain | Labels | Interface | Raw response MAE | Relative exact | Baseline omitted | Offset-corrected MAE | Raw cue calibration |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for cell in output['cells']:
        lines.append('| '+' | '.join([cell['model_key'],cell['domain'],cell['label_kind'],cell['interface'],
                     f'{cell["raw_response_mae"]:.3f}',f'{cell["relative_vector_exact_01"]:.3f}',
                     f'{cell["both_zero_shocks_near_zero"]:.3f}',f'{cell["global_offset_corrected_level_mae"]:.3f}',
                     f'{cell["unclipped_cue_change_calibration"]:.3f}'])+' |')
    lines+=['','Baseline omitted means both zero-shock predictions lie within 0.01 of zero, while true resting '
            'levels are 50 (Coin City) and 180 (Coin Harbor). JSON includes signed/absolute baseline offsets, '
            'per-coordinate relative accuracy, raw forecast MAE, horizon-specific correction, and paired-bootstrap '
            'calibration intervals. Every completed model/interface is reported.','']
    (ROOT/'BASELINE_AUDIT.md').write_text('\n'.join(lines))
    print(f'Audited {len(output["cells"])} completed model×domain×label×forecast-interface cells.')

if __name__=='__main__':main()
