#!/usr/bin/env python3
"""Freeze source, job roster and evaluation files before any sbatch submission."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from build import ROOT, REPO, MECH, sha

def main():
    if (ROOT/'freeze.json').exists(): raise SystemExit('Freeze already exists; refusing overwrite')
    data=json.loads((ROOT/'data/manifest.json').read_text())
    audit=json.loads((ROOT/'checkpoint_audit.json').read_text())
    evidence=ROOT.parent/'evidence_use'
    for disclosure in ('disclosed','undisclosed'):
        assert (evidence/'data'/f'heldout_{disclosure}'/'dataset_dict.json').is_file(), 'fresh eval not ready'
    frozen=ROOT/'frozen_runtime';frozen.mkdir(exist_ok=True)
    for name in ('run_mechanism_rl.py','output_contract.py','evaluate_endpoint.py','report.py','contrasts.py','worlds.py','prompt.py'):
        shutil.copyfile(MECH/name,frozen/name)
    shutil.copyfile(REPO/'exp3_training_transfer/biased_news/forecast_scoring.py',frozen/'forecast_scoring.py')
    wrapper=(MECH/'mechanism_rl.sh').read_text()
    wrapper=wrapper.replace('FAMILY="$REPO/exp3_training_transfer/mechanism_family"',
                            'FAMILY="$REPO/exp3_training_transfer/mechanism_family"\nCONTROL="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control"\nFROZEN="$CONTROL/frozen_runtime"')
    wrapper=wrapper.replace('export PYTHONPATH="$FAMILY:', 'export PYTHONPATH="$FROZEN:$FAMILY:')
    wrapper=wrapper.replace('OUT="$FAMILY/reports/', 'OUT="$CONTROL/reports/')
    wrapper=wrapper.replace('causal_family|population_prior|structureless)', 'causal_family|population_prior|structureless|shuffled_target)')
    for name in ('run_mechanism_rl.py','evaluate_endpoint.py','report.py'):
        wrapper=wrapper.replace(f'"$FAMILY/{name}"',f'"$FROZEN/{name}"')
    wrapper += '''\n# Fresh evaluation is appended after the unchanged registered terminal evaluations.\nADAPTER=$(find "$OUT" -type f -path '*/saved_models/step_00301/adapter_model.safetensors' | head -1)\n[ -n "$ADAPTER" ] || { echo "final weights absent"; exit 5; }\nADAPTER=$(dirname "$ADAPTER")\n"$PY" "$FROZEN/evaluate_endpoint.py" \\\n  --model "$MODEL" --adapter "$ADAPTER" \\\n  --data "$CONTROL/../evidence_use/data/heldout_${DISCLOSURE}" \\\n  --template "$PROMPT_TEMPLATE" --structured-output forecast_array \\\n  --temperature 0 --n 1 --seed "$(( SEED + 20260814 ))" \\\n  --max-tokens "$GEN_LEN" --max-model-len "$MAX_MODEL_LEN" \\\n  --output "$OUT/fresh_greedy.json" \\\n  --secondary-output "$OUT/fresh_stochastic_n5.json" --secondary-temperature 0.7 --secondary-n 5\nfor NAME in fresh_greedy fresh_stochastic_n5; do\n  "$PY" "$FROZEN/report.py" score "$OUT/$NAME.json" --model "$MODEL_KEY" \\\n    --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"\ndone\necho "fresh_complete: $OUT"\n'''
    (ROOT/'train.sh').write_text(wrapper)
    parent_ledger=REPO/'exp3_training_transfer/five_seed_extension/runs/mechanism_qwen3_4b_five_seed_20260916T214457Z.json'
    parents=json.loads(parent_ledger.read_text())['training_jobs']
    base={row['disclosure']:row['environment'] for row in parents if row['arm']=='causal_family' and row['seed']==45}
    roster=[]
    for disclosure in ('disclosed','undisclosed'):
        for seed in (45,46,42,43,44):  # Existing matched endpoints permit the first usable comparisons.
            env=dict(base[disclosure]);env.update(SEED=str(seed),ARM='shuffled_target',
                DATA=str(ROOT/'data'/f'{disclosure}_s{seed}'),TAG=f'{disclosure}_shuffled_target_qwen3_4b_s{seed}')
            roster.append({'kind':'control','disclosure':disclosure,'seed':seed,'arm':'shuffled_target','environment':env})
    recoveries=[]
    for disclosure in ('disclosed','undisclosed'):
        for seed in (42,43,44):
            record=next(r for r in audit['checkpoints'] if r['arm']=='causal_family' and r['disclosure']==disclosure and r['seed']==seed)
            assert not record['weights_exist'], 'unnecessary recovery'
            env=dict(base[disclosure]);env.update(SEED=str(seed),DATA=str(MECH/'data'/f'{disclosure}_causal_family'),
                TAG=f'{disclosure}_matched_recovery_qwen3_4b_s{seed}')
            recoveries.append({'kind':'matched_recovery','disclosure':disclosure,'seed':seed,'arm':'causal_family','environment':env,
                               'missing_original_adapter':record['adapter']})
    (ROOT/'control_roster.json').write_text(json.dumps(roster,indent=2,sort_keys=True)+'\n')
    (ROOT/'recovery_roster.json').write_text(json.dumps(recoveries,indent=2,sort_keys=True)+'\n')
    paths=[ROOT/name for name in ('PROTOCOL.md','build.py','test_build.py','test_reward_path.py','evaluate_reused.sh','evaluate_reused.py','audit_checkpoints.py','checkpoint_audit.json','prepare_launch.py','train.sh','launch.py','run_array.sh','control_roster.json','recovery_roster.json')]
    paths += list(frozen.glob('*.py'))+[ROOT/'data/manifest.json']
    paths += [p for p in (evidence/'data').rglob('*') if p.is_file() and ('heldout_' in str(p))]
    # Runtime OAT and environment are hashed so later unrelated edits fail closed before training.
    runtime_paths=list((REPO/'.runtime/oat/oat').rglob('*.py'))+[REPO/'.runtime/oat_env.sh']
    payload={'protocol':'mechanism_shuffled_target_20260921_v1','frozen_at':datetime.now(timezone.utc).isoformat(),
             'source_sha256':{str(p):sha(p) for p in paths},
             'runtime_sha256':{str(p):sha(p) for p in runtime_paths},
             'training_data_manifest_sha256':sha(ROOT/'data/manifest.json'),
             'control_runs':10,'matched_recovery_runs':6,'fresh_evaluation':str(evidence/'data'),
             'run_order_note':'Seeds45/46 controls come first because their matched weights already exist.'}
    (ROOT/'freeze.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if not k.endswith('sha256')},indent=2))

if __name__=='__main__':main()
