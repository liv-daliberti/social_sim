#!/usr/bin/env python3
"""One disclosed followup to test arithmetic with an explicit calculation rule."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if (ROOT/'manifest.json').exists():raise SystemExit('Already frozen; refuse overwrite')
    rows=[]
    source=[json.loads(x) for x in (PARENT/'diagnostic.jsonl').read_text().splitlines()]
    for row in source:
        m=row['reference']
        if m['interface']!='oracle_selected_pattern' or int(m['pair_id'].rsplit(':r',1)[1])>=8:continue
        coeff=m['unit_responses'][str(m['selected_reference'])]
        baseline=50. if m['domain']=='coin_city' else 180.
        formula=(f'EXPLICIT CALCULATION RULE\nThe selected reference is REFERENCE SYSTEM {m["selected_reference"]}. '
                 f'Set coefficient(1)={coeff[0]:.8f} and coefficient(3)={coeff[1]:.8f}. '
                 f'For EACH listed scenario calculate: forecast = {baseline:.1f} + shock * coefficient(horizon). '
                 'Use that scenario\'s signed shock and its horizon. In particular, if coefficient(horizon) '
                 'is zero, the forecast equals the resting level. Do not carry the horizon-1 response '
                 'to horizon 3 unless its stated coefficient makes that correct.')
        for interface in ('selected_pattern_explicit_formula','calculator_only_explicit_formula'):
            if interface=='selected_pattern_explicit_formula':
                prompt=row['input']
            else:
                intro=row['input'].split('REFERENCE SYSTEM 1',1)[0]
                # Keep the reference background labels and target cue, remove observed trajectories.
                contexts=[]
                for index in (1,2):
                    ref_block=row['input'].split(f'REFERENCE SYSTEM {index}\n\n',1)[1]
                    background=ref_block.split('\n\n',1)[0]
                    contexts.append(f'REFERENCE SYSTEM {index}\n\n{background}')
                target_part='TARGET SYSTEM'+row['input'].split('TARGET SYSTEM',1)[1]
                prompt=intro.replace('The reference systems are complete examples.', 'Reference labels and an exact selected response summary are provided.')+'\n\n'.join(contexts)+'\n\n'+target_part
                assert ' | ' not in prompt
            output_marker='Return only strict JSON with exactly ten finite numbers in scenario order:'
            assert prompt.count(output_marker)==1
            prompt=prompt.replace(output_marker,formula+'\n\n'+output_marker)
            metadata={**m,'interface':interface,'source_v1_task_id':m['task_id'],
                      'task_id':m['task_id']+':'+interface,'diagnostic_protocol':'structure_arithmetic_followup_v1'}
            rows.append({'input':prompt,'reference':metadata})
    assert len(rows)==128
    for row in rows:
        m=row['reference'];co=m['unit_responses'][str(m['selected_reference'])]
        baseline=50 if m['domain']=='coin_city' else 180
        predicted=[baseline+shock*round(co[(horizon-1)//2],8) for shock,horizon in zip(m['scenario_shocks'],m['scenario_horizons'])]
        assert max(abs(x-y) for x,y in zip(predicted,m['truth_targets']))<1e-6
    (ROOT/'diagnostic.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    wanted={'qwen3_8b_base','qwen3_8b_causal_s42','qwen3_32b_base'}
    roster=[r for r in json.loads((PARENT/'roster.json').read_text()) if r['key'] in wanted]
    (ROOT/'roster.json').write_text(json.dumps(roster,indent=2)+'\n')
    paths=['PROTOCOL.md','build.py','diagnostic.jsonl','roster.json','evaluate_supplement.py','run.sbatch','../evaluate.py','../score.py']
    manifest={'protocol':'structure_arithmetic_followup_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
              'rows':len(rows),'paired_episodes':32,'source_manifest_sha256':sha(PARENT/'manifest.json'),
              'trigger_observation':'Qwen3-8B base selection-only 100% but selected-pattern forecasts ignore supplied zero horizon-3 effects; oracle assistance worsens response MAE in every domain/label cell.',
              'sha256':{name:sha(ROOT/name) for name in paths}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
