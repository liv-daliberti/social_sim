#!/usr/bin/env python3
"""Evaluate the predeclared component gate; does not submit GPU work."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT.parent/'structure/results'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(key):
 p=RESULTS/(key+'.summary.json')
 if not p.exists(): return None
 value=json.loads(p.read_text()); assert value['n']==768
 return value

def cell(result,interface):
 return next(x for x in result['cells'] if x['domain']=='coin_city' and x['label_kind']=='semantic' and x['interface']==interface)
rl=read('qwen3_8b_causal_s42')
base=read('qwen3_8b_base')
strong=read('qwen3_32b_base')
if rl is None or base is None:
 print('Waiting for complete base and matched seed42 component outputs')
 sys.exit(2)
a=cell(rl,'original')
oracles=[cell(rl,k) for k in ['oracle_both_patterns','oracle_selected_pattern']]
oracle_improvement=1-min(x['response_mae'] for x in oracles)/max(a['response_mae'],1e-12)
strong_cal=cell(strong,'original')['cue_change_calibration'] if strong else None
forecast=bool(a['cue_change_calibration']<.25 and (oracle_improvement>=.25 or (strong_cal is not None and strong_cal>.5)))
if not forecast and strong is None:
 print('Oracle branch did not open gate; waiting for stronger-model branch')
 sys.exit(2)
sources={p.name:sha(p) for p in RESULTS.glob('*.summary.json') if p.name in ['qwen3_8b_base.summary.json','qwen3_8b_causal_s42.summary.json','qwen3_32b_base.summary.json']}
result={'created_utc':datetime.now(timezone.utc).isoformat(),'launch_forecast_sft':forecast,
 'rl_original_calibration':a['cue_change_calibration'],'rl_original_response_mae':a['response_mae'],
 'best_oracle_fractional_mae_improvement':oracle_improvement,'stronger_original_calibration':strong_cal,
 'rl_selection_accuracy':cell(rl,'selection_only')['accuracy'],
 'base_selection_accuracy':cell(base,'selection_only')['accuracy'],
 'selection_supervision_consideration':cell(rl,'selection_only')['accuracy']<.9,
 'diagnostic_summary_sha256':sources,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
 'config_sha256':{n:sha(ROOT/(n+'.json')) for n in ['forecast_lr1e6','forecast_lr2e5']},
 'code_sha256':{n:sha(ROOT/n) for n in ['train_sft.py','prepare.py','run.sbatch','evaluate.sbatch','check_gate.py']}}
out=ROOT/'gate.json'
if out.exists(): raise SystemExit('Gate is already frozen; refusing to replace')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
