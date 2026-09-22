#!/usr/bin/env python3
"""Keep both fixed SFT recipes in a diagnostic comparison with base and RL42."""
import json
from datetime import datetime,timezone
from pathlib import Path
LEARN=Path(__file__).resolve().parent
ROOT=LEARN.parent/'structure'


def main():
    descriptors=[('Base 8B',ROOT/'results/qwen3_8b_base.summary.json'),
                 ('Matched RL seed42',ROOT/'results/qwen3_8b_causal_s42.summary.json'),
                 ('Matched RL seed43 (descriptive)',ROOT/'results/qwen3_8b_causal_s43.summary.json'),
                 ('Matched RL seed44 (descriptive)',ROOT/'results/qwen3_8b_causal_s44.summary.json')]
    for recipe in ('forecast_lr1e6','forecast_lr2e5'):
        descriptor=(f'SFT {recipe} seed42',LEARN/'runs'/f'{recipe}_s42'/'component_results'/f'qwen3_8b_sft_{recipe}_s42.summary.json')
        descriptors.append(descriptor)
    summaries={label:json.loads(path.read_text()) for label,path in descriptors if path.exists()}
    lines=['# Fixed supervised learnability comparison','',f'Updated {datetime.now(timezone.utc).isoformat()}.','',
           'The primary diagnostic comparison uses matched RL seed42 versus both fixed one-pass SFT recipes and the same untrained base on the '
           'frozen paired component diagnostic. SFT training used 4,800 original prompts, seed42, rank32/alpha64, '
           'one pass and 300 optimizer updates; RL sampled eight completions per prompt and used about 2,400 '
           'optimizer updates. The recipes match prompt exposures, not compute or update count. The second '
           'SFT learning rate differs from RL. These are outcome-motivated exploratory diagnostics, not a '
           'population-level or isolated universal SFT-versus-RL effect. Both fixed SFT recipes are retained. Matched RL seeds43/44 are included as descriptive context.','',
           'The original SFT launch gate was closed. A disclosed amendment justified the pilot from component '
           'evidence of stronger-model conditional arithmetic and an absolute-level interface error. See '
           'PROTOCOL.md and its gate/amendment records. Selection-only was already perfect '
           'for base/RL; any SFT improvement here concerns full forecasting/composition, not newly acquired '
           'cue-name identification.','',
           '## Original task','',
           '| Domain | Labels | Model/recipe | Response MAE | Cue calibration | Calibration 95% episode CI | Structure accuracy | Parse |',
           '|---|---|---|---:|---:|---|---:|---:|']
    for domain in ('coin_city','coin_harbor'):
        for label in ('semantic','arbitrary'):
            for model,_ in descriptors:
                if model not in summaries:
                    lines.append(f'| {domain} | {label} | {model} | pending | pending | pending | pending | pending |')
                    continue
                c=next(x for x in summaries[model]['cells'] if x['domain']==domain and x['label_kind']==label and x['interface']=='original')
                ci=c['cue_change_calibration_ci95']
                ci_text=f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else 'undefined'
                lines.append(f'| {domain} | {label} | {model} | {c["response_mae"]:.3f} | {c["cue_change_calibration"]:.3f} | {ci_text} | {c["accuracy"]:.3f} | {c["parse_rate"]:.3f} |')
    lines+=['','## Every component interface','',
            '| Model/recipe | Domain | Labels | Interface | Accuracy | Response MAE | Cue calibration |',
            '|---|---|---|---|---:|---:|---:|']
    for model,_ in descriptors:
        if model not in summaries:continue
        for c in summaries[model]['cells']:
            mae=f'{c["response_mae"]:.3f}' if c.get('response_mae') is not None else '—'
            alpha=f'{c["cue_change_calibration"]:.3f}' if c.get('cue_change_calibration') is not None else '—'
            lines.append(f'| {model} | {c["domain"]} | {c["label_kind"]} | {c["interface"]} | {c["accuracy"]:.3f} | {mae} | {alpha} |')
    lines+=['','Calibration CIs resample paired episodes within a cell. They do not account for training-seed '
            'variation and must not be used for a seed-generalized method claim. The original scoring clips '
            'forecast values to the domain range before forming response vectors; baseline/raw sensitivities '
            'should be examined if an SFT output omits the resting level.','']
    (LEARN/'COMPONENT_COMPARISON.md').write_text('\n'.join(lines))
    (LEARN/'component_comparison.json').write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),
         'available':list(summaries),'pending':[name for name,_ in descriptors if name not in summaries],
         'summaries':summaries},indent=2)+'\n')
    # Reuse the post hoc baseline audit on a temporary input directory. Frozen v1
    # outputs and summaries are read only; preserve separate supervised artifacts.
    import sys,tempfile
    sys.path.insert(0,str(ROOT))
    import audit_baseline
    with tempfile.TemporaryDirectory(prefix='sft_component_audit_',dir='/tmp') as temporary:
        temp=Path(temporary);(temp/'results').mkdir()
        for label,path in descriptors:
            raw_path=path.with_name(path.name.replace('.summary.json','.jsonl'))
            if path.exists() and raw_path.exists():
                (temp/'results'/raw_path.name).symlink_to(raw_path.resolve())
        previous=audit_baseline.ROOT
        audit_baseline.ROOT=temp
        try:audit_baseline.main()
        finally:audit_baseline.ROOT=previous
        for name in ('baseline_omission_audit.json','BASELINE_AUDIT.md'):
            (LEARN/name).write_bytes((temp/name).read_bytes())
    print(f'SFT comparison updated: {len(summaries)}/{len(descriptors)} endpoints available.')

if __name__=='__main__':main()
