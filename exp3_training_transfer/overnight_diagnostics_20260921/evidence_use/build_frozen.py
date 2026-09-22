#!/usr/bin/env python3
"""Freeze fresh ordinary transfer evaluation and paired gain interventions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FAMILY = REPO / 'exp3_training_transfer/mechanism_family'
sys.path.insert(0, str(FAMILY))
from make_dataset import make_row, numeric_hash
from prompt import gold_response, render_prompt
from worlds import CATALOG, TEST, balanced_inputs, expected_series, forecast_scenarios, population_prior_targets, response_pairs, response_vector, scenario_grid

DISCLOSURES = ('disclosed', 'undisclosed')
PAIR_SEED_BASE = 950_000_000
FRESH_SEED_BASE = 921_000_000

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def series(world, seed, length, gain=None):
    rng = np.random.default_rng(seed)
    sampled_gain = float(rng.uniform(world.gain_lo, world.gain_hi))
    gain = sampled_gain if gain is None else float(gain)
    inputs = balanced_inputs(world, length, rng)
    noise = rng.normal(0.0, world.noise, length)
    expected, state = expected_series(world, inputs, gain)
    observed = np.clip(expected + noise, *world.clip)
    return dict(inputs=inputs.tolist(), observed=observed.round(4).tolist(),
                expected=expected.tolist(), noise=noise.tolist(), gain=gain,
                terminal_state=list(state), seed=seed)

def make_gain_pair(world, seed, disclosure):
    calibrations = [series(world, seed + 100_000*(i+1), 10) for i in range(3)]
    pair_id = f'gain:{world.name}:{seed}:9'
    rows = []
    for side, quantile in [('low', .2), ('high', .8)]:
        gain = world.gain_lo + quantile*(world.gain_hi-world.gain_lo)
        target = series(world, seed, 9, gain)
        bundle = dict(calibrations=calibrations, target_inputs=target['inputs'], target_observed=target['observed'])
        targets = forecast_scenarios(world, np.asarray(target['inputs']), gain)
        prior = population_prior_targets(world, target['observed'][-1])
        ref = dict(protocol='mechanism_evidence_gain_20260921_v1', task_id=f'{pair_id}:{side}', pair_id=pair_id,
            side=side, intervention='target_gain', world=world.name, block=world.block, split=world.split,
            seed=seed, k=9, disclosure=disclosure, mode='evaluation', g=gain,
            targets=targets.tolist(), truth_targets=targets.tolist(), response_targets=response_vector(targets).tolist(),
            truth_response=response_vector(targets).tolist(), prior_targets=prior.tolist(), prior_response=response_vector(prior).tolist(),
            response_pairs=response_pairs(), clip=world.clip, scenario_labels=[s['label'] for s in scenario_grid()],
            scenario_shocks=[s['shock'] for s in scenario_grid()], scenario_horizons=[s['horizon'] for s in scenario_grid()],
            shocks=[s['shock'] for s in scenario_grid()], reward_scale=10., response_scale=4., numeric_hash=numeric_hash(bundle),
            calibrations=calibrations, target_inputs=target['inputs'], target_observed=target['observed'],
            target_expected=target['expected'], target_exogenous_noise=target['noise'], terminal_state=target['terminal_state'],
            world_parameters=asdict(world), gold=gold_response(targets))
        rows.append(dict(input=render_prompt(world, bundle, disclosure), reference=json.dumps(ref, sort_keys=True)))
    return rows

def validate_pairs(rows):
    for lo, hi in zip(rows[::2], rows[1::2]):
        a,b = [json.loads(r['reference']) for r in (lo,hi)]
        assert a['pair_id'] == b['pair_id'] and a['side']=='low' and b['side']=='high'
        assert a['calibrations']==b['calibrations'] and a['target_inputs']==b['target_inputs']
        assert a['target_exogenous_noise']==b['target_exogenous_noise']
        assert a['g'] < b['g']
        for ref in (a,b):
            w = next(w for w in CATALOG if w.name==ref['world'])
            assert w.gain_lo <= ref['g'] <= w.gain_hi
            means, state = expected_series(w,np.asarray(ref['target_inputs']),ref['g'])
            obs = np.clip(means+np.asarray(ref['target_exogenous_noise']),*w.clip).round(4)
            np.testing.assert_array_equal(obs,ref['target_observed'])
            np.testing.assert_allclose(forecast_scenarios(w,np.asarray(ref['target_inputs']),ref['g']),ref['targets'],rtol=0,atol=1e-12)
        assert lo['input'] != hi['input']
        assert np.max(np.abs(np.asarray(a['truth_response'])-b['truth_response'])) > 0

def inventory():
    available, missing = [], []
    for disclosure in DISCLOSURES:
        available.append(dict(id=f'{disclosure}_base', disclosure=disclosure, arm='base', seed=None, adapter=None))
        for arm in ['causal_family','population_prior']:
            for seed in range(42,47):
                candidates = sorted(FAMILY.glob(f'reports/{disclosure}_{arm}_qwen3_4b_s{seed}_*/debug_*/saved_models/step_00301/adapter_model.safetensors'))
                entry = dict(id=f'{disclosure}_{arm}_s{seed}', disclosure=disclosure, arm=arm, seed=seed)
                if not candidates:
                    missing.append({**entry,'reason':'No final adapter_model.safetensors; metadata-only directory is not a checkpoint.'})
                    continue
                weights = candidates[-1]
                cfg = json.loads((weights.parent/'adapter_config.json').read_text())
                assert cfg['r']==32 and cfg['lora_alpha']==64
                available.append({**entry,'adapter':str(weights.parent.resolve()),'adapter_sha256':sha(weights),
                    'adapter_config_sha256':sha(weights.parent/'adapter_config.json')})
    return dict(model='Qwen/Qwen3-4B-Instruct-2507', available=available, missing=missing)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    if (args.output/'frozen_manifest.json').exists():
        raise SystemExit('Refusing to replace frozen data; choose another output directory.')
    from datasets import Dataset, DatasetDict
    args.output.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for disclosure in DISCLOSURES:
        pairs=[row for wi,w in enumerate(CATALOG) for i in range(16)
            for row in make_gain_pair(w,PAIR_SEED_BASE+wi*1_000_000+i,disclosure)]
        validate_pairs(pairs)
        heldout=[make_row(w,FRESH_SEED_BASE+wi*1_000_000+i,k,disclosure,'causal_family',evaluation=True)
            for wi,w in enumerate(TEST) for i in range(60) for k in (3,6,9)]
        for name,rows in [(f'gain_pairs_{disclosure}',pairs),(f'heldout_{disclosure}',heldout)]:
            path=args.output/f'{name}.jsonl'
            path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
            DatasetDict({'train':Dataset.from_list(rows)}).save_to_disk(str(args.output/name))
            artifacts[name]={'jsonl':str(path.resolve()),'dataset':str((args.output/name).resolve()),'sha256':sha(path),'rows':len(rows)}
    checkpoint_inventory=inventory()
    write_json(args.output/'checkpoint_inventory.json',checkpoint_inventory)
    manifest=dict(protocol='mechanism_evidence_gain_20260921_v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        artifacts=artifacts, pair_seed_base=PAIR_SEED_BASE, fresh_holdout_seed_base=FRESH_SEED_BASE,
        gain_quantiles=[.2,.8],pair_n_per_world=16,pair_k=9,fresh_n_per_world=60,fresh_k=[3,6,9],
        model=checkpoint_inventory['model'],checkpoint_inventory_sha256=sha(args.output/'checkpoint_inventory.json'),
        source_sha256={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__), ROOT/'PROTOCOL.md',FAMILY/'worlds.py',FAMILY/'make_dataset.py',FAMILY/'prompt.py',FAMILY/'output_contract.py']})
    write_json(args.output/'frozen_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
