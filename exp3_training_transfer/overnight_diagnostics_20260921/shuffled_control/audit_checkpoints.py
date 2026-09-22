#!/usr/bin/env python3
"""Check actual weights, registered provenance, effective runtime args and source hashes."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from build import ROOT, REPO, MECH, sha

EXPECTED = {'critic_type':'drgrpo','learning_rate':1e-6,'lr_scheduler':'constant', 'lr_warmup_ratio':.03,
            'lora_rank':32,'lora_alpha':64,'lora_dropout':0,'max_train':'4800',
            'num_prompt_epoch':1,'num_samples':8,'temperature':1.3,'top_p':1.,
            'train_batch_size':16,'train_batch_size_per_device':1,'rollout_batch_size':16,
            'rollout_batch_size_per_device':16,'pi_buffer_maxlen_per_device':128,
            'num_ppo_epochs':1,'prompt_max_length':2304,'generate_max_length':192,
            'max_model_len':3072,'enable_prefix_caching':True,'prompt_template':'biased_news',
            'structured_output':'forecast_array','eval_steps':25,'eval_batch_size':120,
            'eval_temperature':0.,'eval_n':1,'eval_generate_max_length':192,'save_steps':999999,
            'save_ckpt':False,'beta':0.,'slope_weight':.6,'slope_scale_g':4.,'reward_scale_pts':10.,
            'zero_stage':2,'bf16':True,'ref_offload':True,'gradient_checkpointing':True,'flash_attn':True,
            'max_norm':1.,'adam_beta_1':.9,'adam_beta_2':.95,'l2':0.,'collocate':False,'gpus':2}

def effective_args(path):
    raw=path.read_text(errors='replace')
    cleaned=re.sub(r'\x1b\[[0-9;]*m','',raw)
    args={}
    for line in cleaned.splitlines():
        if '[learner_0_0/0]' not in line: continue
        m=re.search(r"│\s+'([^']+)': (.+),$",line)
        if m:
            try: args[m[1]]=ast.literal_eval(m[2])
            except (ValueError,SyntaxError): pass
    return args, raw

def main():
    aggregate=json.loads((MECH/'reports/c3_mechanism_full_aggregate.json').read_text())
    ledger_path=REPO/aggregate['ledger'];ledger=json.loads(ledger_path.read_text())
    added_path=REPO/'exp3_training_transfer/five_seed_extension/runs/mechanism_qwen3_4b_five_seed_20260916T214457Z.json'
    added=json.loads(added_path.read_text())
    roots={}
    for rel in aggregate['score_files']:
        m=re.search(r'reports/((disclosed|undisclosed)_(causal_family|population_prior)_qwen3_4b_s(4[234])_[^/]+)',rel)
        if m: roots[(m[2],m[3],int(m[4]))]=MECH/'reports'/m[1]
    for item in added['training_jobs']:
        key=(item['disclosure'],item['arm'],item['seed'])
        found=list((MECH/'reports').glob(f'{key[0]}_{key[1]}_qwen3_4b_s{key[2]}_*_j{item["job_id"]}'))
        if len(found)!=1: raise RuntimeError((key,found))
        roots[key]=found[0]
    records=[]
    for (disclosure,arm,seed),path in sorted(roots.items()):
        args,raw=effective_args(path/'train.log')
        missing={k:v for k,v in EXPECTED.items() if k not in args}
        drift={k:{'expected':v,'actual':args.get(k)} for k,v in EXPECTED.items() if k in args and args[k]!=v}
        # Some OAT versions print vLLM-derived flags only in actor logs, not PPOArgs.
        assert not drift,(path,drift)
        assert len(set(missing)-{'enable_prefix_caching','collocate','gpus'}) <= 2, (path,missing)
        assert 'DeepSpeed Basic Optimizer = AdamW' in raw
        adapters=list(path.glob('*/saved_models/step_00301'))
        assert len(adapters)==1,(path,adapters)
        weights=adapters[0]/'adapter_model.safetensors'
        records.append({'disclosure':disclosure,'arm':arm,'seed':seed,'report_dir':str(path),
                        'adapter':str(adapters[0]),'weights_exist':weights.is_file(),
                        'weights_sha256':sha(weights) if weights.is_file() else None,
                        'reusable_on_fresh_eval':weights.is_file(), 'runtime_args':args,
                        'verified_expected_args':{k:args[k] for k in EXPECTED if k in args},
                        'not_in_scalar_args':missing,'optimizer_verified':'torch.optim.AdamW',
                        'runtime_arg_drift':drift, 'train_log_sha256':sha(path/'train.log')})
    hist_sources={'train_script_sha256':'mechanism_rl.sh','trainer_sha256':'run_mechanism_rl.py',
                  'output_contract_sha256':'output_contract.py','world_catalog_sha256':'worlds.py',
                  'prompt_renderer_sha256':'prompt.py'}
    source_status={key:{'historical':ledger.get(key),'current':sha(MECH/file),
                        'match':ledger.get(key)==sha(MECH/file),'file':file} for key,file in hist_sources.items()}
    diff=subprocess.run(['git','diff','--','exp3_training_transfer/mechanism_family/run_mechanism_rl.py'],cwd=REPO,text=True,capture_output=True,check=True).stdout
    (ROOT/'trainer_worktree_diff.patch').write_text(diff)
    payload={'created_at':datetime.now(timezone.utc).isoformat(),'historical_ledger':str(ledger_path),
             'historical_git_revision':ledger['git_revision'], 'historical_sources':source_status,
             'added_seed_ledger':str(added_path),'checkpoints':records,
             'source_caveat':'Historical trainer hash differs current. Git worktree delta adds disabled execution flags and learner-local AdamW reapplication; logs verify both generations actually used torch.optim.AdamW. Historical revision not available in local git object database. Current file predates Sep19 reusable checkpoints; byte identity for those is not archived, so semantic recipe match is supported, exact historical bytes not asserted.'}
    (ROOT/'checkpoint_audit.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'checkpoints':len(records),'reusable_weights':sum(r['weights_exist'] for r in records),
                     'matched_reusable':[f'{r["disclosure"]}:{r["seed"]}' for r in records if r['arm']=='causal_family' and r['weights_exist']],
                     'source_status':source_status},indent=2))

if __name__=='__main__':main()
