#!/usr/bin/env python3
"""Seed 47-48 extension runs: same hardware and memory allocation as the frozen roster."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    amend = json.loads((ROOT / 'freeze_extension.json').read_text())
    assert sha(ROOT / 'freeze.json') == amend['parent_freeze_sha256'], 'parent freeze changed'
    assert sha(ROOT / 'data_ext/manifest.json') == amend['extension_data_manifest_sha256'], 'extension data changed'
    manifest = json.loads((ROOT / 'data_ext/manifest.json').read_text())
    for record in manifest['datasets']:
        path = Path(record['path'])
        for rel, expected in record['files_sha256'].items():
            assert sha(path / rel) == expected, f'extension dataset changed: {path/rel}'
        assert sha(path / 'donor_map.json') == record['donor_map_sha256'], f'donor map changed: {path}'

    row = json.loads((ROOT / 'extension_roster.json').read_text())[args.index]
    env = os.environ.copy(); env.update(row['environment']); env.pop('VLLM_SLEEP', None)
    # Same transformation the completed A5000 runs applied.
    assert env['VLLM_RATIO'] == '0.38' and env['COLLOCATE'] == '0' and env['GPUS'] == '2'
    env['VLLM_RATIO'] = '0.76'
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                       '--format=csv,noheader'], text=True)
    print(json.dumps({'array_index': args.index, 'record': row, 'device_info': devices,
                      'extension_freeze_sha256': sha(ROOT / 'freeze_extension.json'),
                      'run_wrapper_sha256': sha(__file__)}, indent=2, sort_keys=True), flush=True)
    subprocess.run(['bash', str(ROOT / 'train.sh')], env=env, cwd=REPO, check=True)


if __name__ == '__main__':
    main()
