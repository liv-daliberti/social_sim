#!/usr/bin/env python3
"""Extend the shuffled-target control to seeds 47-48 without touching the frozen data."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import build as base

EXT_SEEDS = (47, 48)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=base.ROOT / 'data_ext')
    args = parser.parse_args()
    from datasets import Dataset, DatasetDict, load_from_disk

    frozen = json.loads((base.ROOT / 'data/manifest.json').read_text())
    assert frozen['seeds'] == [42, 43, 44, 45, 46], 'unexpected frozen seed roster'
    assert not set(EXT_SEEDS) & set(frozen['seeds']), 'extension seed collides with frozen roster'
    if (args.output / 'manifest.json').exists():
        raise SystemExit('Extension data already exists; refuse overwrite')
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {'protocol': 'mechanism_shuffled_target_20260921_v1_ext4748',
                'extends': 'mechanism_shuffled_target_20260921_v1',
                'frozen_manifest_sha256': base.sha(base.ROOT / 'data/manifest.json'),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'seeds': list(EXT_SEEDS), 'compatibility_keys': list(base.COMPAT_KEYS),
                'donor_keys': list(base.DONOR_KEYS), 'datasets': []}
    permutations = {}
    for disclosure in base.DISCLOSURES:
        original = base.MECH / 'data' / f'{disclosure}_causal_family'
        rows = list(load_from_disk(str(original / 'train'))['train'])
        for seed in EXT_SEEDS:
            shuffled, donors, audit = base.shuffled_rows(rows, seed)
            if seed in permutations:
                assert donors == permutations[seed], 'disclosure permutations differ'
            permutations[seed] = donors
            path = args.output / f'{disclosure}_s{seed}'
            path.mkdir()
            DatasetDict({'train': Dataset.from_list(shuffled)}).save_to_disk(str(path / 'train'))
            (path / 'heldout').symlink_to(original / 'heldout', target_is_directory=True)
            donor_map = [{'recipient': json.loads(rows[i]['reference'])['task_id'],
                          'donor': json.loads(rows[j]['reference'])['task_id']}
                         for i, j in enumerate(donors)]
            (path / 'donor_map.json').write_text(json.dumps(donor_map, indent=2) + '\n')
            files = {str(p.relative_to(path)): base.sha(p)
                     for p in sorted((path / 'train').rglob('*')) if p.is_file()}
            manifest['datasets'].append({'disclosure': disclosure, 'seed': seed, 'path': str(path),
                                         'source': str(original), 'audit': audit, 'files_sha256': files,
                                         'donor_map_sha256': base.sha(path / 'donor_map.json')})
            print(disclosure, seed, audit, flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('wrote', args.output / 'manifest.json')


if __name__ == '__main__':
    main()
