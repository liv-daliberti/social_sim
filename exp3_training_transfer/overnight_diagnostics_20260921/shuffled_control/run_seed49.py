#!/usr/bin/env python3
"""Seed 49 runs: same hardware and memory allocation as the ext4748 roster.

Mirrors run_extension.py. See freeze_seed49.json -- this seed was added after
the seven-seed result was known and is recorded as an outcome-informed
amendment, not a pre-registered extension.
"""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    amend = json.loads((ROOT / 'freeze_seed49.json').read_text())
    assert sha(ROOT / 'freeze.json') == amend['parent_freeze_sha256'], 'parent freeze changed'
    assert sha(ROOT / 'freeze_extension.json') == amend['ext4748_freeze_sha256'], 'ext4748 freeze changed'
    assert sha(ROOT / 'data_ext49/manifest.json') == amend['seed49_data_manifest_sha256'], 'seed49 data changed'
    assert sha(ROOT / 'roster_seed49.json') == amend['source_sha256']['roster_seed49.json'], 'roster changed'
    manifest = json.loads((ROOT / 'data_ext49/manifest.json').read_text())
    for record in manifest['datasets']:
        path = Path(record['path'])
        for rel, expected in record['files_sha256'].items():
            assert sha(path / rel) == expected, f'seed49 dataset changed: {path/rel}'
        assert sha(path / 'donor_map.json') == record['donor_map_sha256'], f'donor map changed: {path}'

    row = json.loads((ROOT / 'roster_seed49.json').read_text())[args.index]
    env = os.environ.copy(); env.update(row['environment']); env.pop('VLLM_SLEEP', None)
    assert env['VLLM_RATIO'] == '0.38' and env['COLLOCATE'] == '0' and env['GPUS'] == '2'
    env['VLLM_RATIO'] = '0.76'   # the same transformation the completed A5000 runs applied
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                       '--format=csv,noheader'], text=True)
    print(json.dumps({'array_index': args.index, 'record': row, 'device_info': devices,
                      'seed49_freeze_sha256': sha(ROOT / 'freeze_seed49.json'),
                      'run_wrapper_sha256': sha(__file__),
                      'decision_is_outcome_informed': amend['decision_is_outcome_informed']},
                     indent=2, sort_keys=True), flush=True)
    subprocess.run(['bash', str(ROOT / 'train.sh')], env=env, cwd=REPO, check=True)


if __name__ == '__main__':
    main()
