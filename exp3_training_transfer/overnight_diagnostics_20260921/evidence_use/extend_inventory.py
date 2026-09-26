#!/usr/bin/env python3
"""Expand the evidence-use checkpoint roster to the completed A5000 matched/shuffled runs.

The frozen gain-pair and held-out data are reused byte-for-byte; only the checkpoint
roster grows, which PROTOCOL.md anticipates ("datasets are shared by the shuffled
control and matched checkpoint evaluation"). Seeds 42-44 were listed as missing at
freeze time because no final adapter existed; they exist now.
"""
from __future__ import annotations
import hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT.parent / 'shuffled_control'
SEEDS = (42, 43, 44, 45, 46)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

PATTERNS = {'causal_family': ('matched_recovery', 'matched_hardware'),
            'shuffled_target': ('shuffled_target',)}


def find_adapter(disclosure, arm, seed):
    hits = []
    for infix in PATTERNS[arm]:
        hits += sorted(CONTROL.glob(
            f'reports/{disclosure}_{infix}_qwen3_4b_s{seed}_*/debug_*/saved_models/step_00301/adapter_model.safetensors'))
    hits = [h for h in hits if h.is_file()]
    if len(hits) != 1:
        raise SystemExit(f'{disclosure}/{arm}/s{seed}: expected one adapter, found {len(hits)}')
    return hits[0]


def main():
    src, out = ROOT / 'data', ROOT / 'data_ext'
    parent = json.loads((src / 'frozen_manifest.json').read_text())
    if out.exists():
        raise SystemExit('data_ext already exists; refuse overwrite')
    out.mkdir(parents=True)

    # Reuse the frozen data byte-for-byte.
    for name, art in parent['artifacts'].items():
        for key in ('jsonl', 'dataset'):
            s = Path(art[key])
            d = out / s.name
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
        assert sha(out / Path(art['jsonl']).name) == art['sha256'], f'{name}: data hash drift'

    old = json.loads((src / 'checkpoint_inventory.json').read_text())
    available = [c for c in old['available'] if c['arm'] == 'base']
    for disclosure in ('disclosed', 'undisclosed'):
        for arm in ('causal_family', 'shuffled_target'):
            for seed in SEEDS:
                w = find_adapter(disclosure, arm, seed)
                available.append({'id': f'{disclosure}_{arm}_s{seed}', 'disclosure': disclosure,
                                  'arm': arm, 'seed': seed, 'adapter': str(w.parent.resolve()),
                                  'adapter_sha256': sha(w),
                                  'adapter_config_sha256': sha(w.parent / 'adapter_config.json')})
    inventory = {'model': old['model'], 'available': available, 'missing': [],
                 'provenance': 'All adapters from the completed A5000 shuffled-control campaign; '
                               'identical hardware and actor-memory allocation across arms.'}
    inv_path = out / 'checkpoint_inventory.json'
    inv_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')

    manifest = dict(parent)
    manifest.update({
        'protocol': parent['protocol'] + '_ext_full_roster',
        'extends': parent['protocol'],
        'parent_manifest_sha256': sha(src / 'frozen_manifest.json'),
        'parent_checkpoint_inventory_sha256': parent['checkpoint_inventory_sha256'],
        'checkpoint_inventory_sha256': sha(inv_path),
        'amended_at': datetime.now(timezone.utc).isoformat(),
        'amendment': 'Checkpoint roster only. Gain-pair and held-out data are reused byte-for-byte '
                     '(hashes unchanged). Seeds 42-44 were missing at freeze time and now exist; '
                     'the shuffled-target arm replaces population_prior as the distribution-matched '
                     'comparator, which PROTOCOL.md already contemplated.',
    })
    (out / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    n = len(available)
    print(json.dumps({'checkpoints': n, 'per_disclosure': n // 2,
                      'arms': sorted({c['arm'] for c in available}),
                      'data_hashes_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
