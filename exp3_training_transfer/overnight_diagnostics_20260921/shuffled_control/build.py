#!/usr/bin/env python3
"""Freeze whole-vector within-mechanism derangements without changing any prompt."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MECH = REPO / 'exp3_training_transfer/mechanism_family'
DISCLOSURES = ('disclosed', 'undisclosed')
SEEDS = (42, 43, 44, 45, 46)
DONOR_KEYS = ('targets', 'response_targets', 'gold')
COMPAT_KEYS = ('world', 'k', 'scenario_labels', 'scenario_shocks', 'scenario_horizons',
               'response_pairs', 'clip', 'reward_scale', 'response_scale')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def group_key(ref):
    return json.dumps({key: ref[key] for key in COMPAT_KEYS}, sort_keys=True)

def shuffled_rows(rows, seed):
    refs = [json.loads(row['reference']) for row in rows]
    groups = defaultdict(list)
    for i, ref in enumerate(refs):
        groups[group_key(ref)].append(i)
    donors = list(range(len(rows)))
    for key, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError('Cannot derange singleton compatibility group')
        # Sattolo gives one uniform random cycle, hence no episode retains itself.
        local = indices.copy()
        rng = random.Random(int(hashlib.sha256(f'20260921:{seed}:{key}'.encode()).hexdigest(), 16))
        for i in range(len(local) - 1, 0, -1):
            j = rng.randrange(i)
            local[i], local[j] = local[j], local[i]
        for recipient, donor in zip(indices, local):
            donors[recipient] = donor
    result = []
    for i, donor in enumerate(donors):
        ref = dict(refs[i])
        for key in DONOR_KEYS:
            ref[key] = refs[donor][key]
        ref.update(mode='shuffled_target', shuffle_seed=seed,
                   target_source_task_id=refs[donor]['task_id'])
        result.append({'input': rows[i]['input'], 'reference': json.dumps(ref, sort_keys=True)})
    audit = validate(rows, result, donors)
    return result, donors, audit

def validate(source, shuffled, donors):
    assert len(source) == len(shuffled) == len(donors)
    a = [json.loads(row['reference']) for row in source]
    b = [json.loads(row['reference']) for row in shuffled]
    assert sorted(donors) == list(range(len(source)))
    equal_values = 0
    before, after = defaultdict(Counter), defaultdict(Counter)
    for i, (old, new, donor) in enumerate(zip(a, b, donors)):
        assert donor != i, 'self assignment'
        assert source[i]['input'] == shuffled[i]['input'], 'prompt changed'
        assert group_key(old) == group_key(a[donor]), 'incompatible donor'
        assert new['target_source_task_id'] == a[donor]['task_id']
        for key in DONOR_KEYS:
            assert new[key] == a[donor][key], f'non-vector target transfer: {key}'
        for key, value in old.items():
            if key not in (*DONOR_KEYS, 'mode'):
                assert new[key] == value, f'recipient metadata changed: {key}'
        key = group_key(old)
        before[key][json.dumps({k: old[k] for k in DONOR_KEYS}, sort_keys=True)] += 1
        after[key][json.dumps({k: new[k] for k in DONOR_KEYS}, sort_keys=True)] += 1
        if old['targets'] == new['targets']:
            equal_values += 1
        derived = [new['targets'][j] - new['targets'][k] for j, k in new['response_pairs']]
        assert max(abs(x-y) for x,y in zip(derived,new['response_targets'])) <= 0.000101
    assert before == after, 'target multiset not exactly conserved by stratum'
    return {'rows': len(source), 'groups': len(before), 'group_sizes': sorted(sum(c.values()) for c in before.values()),
            'self_assignments': 0, 'numerically_unchanged_vectors': equal_values,
            'exact_target_multiset_preserved': True, 'prompts_byte_identical': True,
            'prompt_sequence_sha256': digest([r['input'] for r in source]),
            'permutation_sha256': digest(donors),
            'target_multiset_sha256': digest(sorted(json.dumps({k:r[k] for k in DONOR_KEYS},sort_keys=True) for r in a))}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    from datasets import Dataset, DatasetDict, load_from_disk
    if (args.output/'manifest.json').exists():
        raise SystemExit('Frozen data already exists; refuse overwrite')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': 'mechanism_shuffled_target_20260921_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'seeds': list(SEEDS), 'compatibility_keys': list(COMPAT_KEYS), 'donor_keys': list(DONOR_KEYS), 'datasets': []}
    permutations = {}
    for disclosure in DISCLOSURES:
        original = MECH/'data'/f'{disclosure}_causal_family'
        rows = list(load_from_disk(str(original/'train'))['train'])
        for seed in SEEDS:
            shuffled, donors, audit = shuffled_rows(rows, seed)
            if seed in permutations:
                assert donors == permutations[seed], 'disclosure permutations differ'
            permutations[seed] = donors
            path = args.output/f'{disclosure}_s{seed}'
            path.mkdir()
            DatasetDict({'train': Dataset.from_list(shuffled)}).save_to_disk(str(path/'train'))
            # Online eval remains unchanged and true-target, matching the historical recipe.
            # Fresh final evaluation is separate and is never used for optimization/selection.
            (path/'heldout').symlink_to(original/'heldout', target_is_directory=True)
            donor_map = [{'recipient': json.loads(rows[i]['reference'])['task_id'],
                          'donor': json.loads(rows[j]['reference'])['task_id']} for i,j in enumerate(donors)]
            (path/'donor_map.json').write_text(json.dumps(donor_map, indent=2)+'\n')
            files = {str(p.relative_to(path)): sha(p) for p in sorted((path/'train').rglob('*')) if p.is_file()}
            record = {'disclosure': disclosure, 'seed':seed, 'path':str(path), 'source':str(original),
                      'audit':audit, 'files_sha256':files, 'donor_map_sha256':sha(path/'donor_map.json')}
            manifest['datasets'].append(record)
            print(disclosure, seed, audit, flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

if __name__ == '__main__': main()
