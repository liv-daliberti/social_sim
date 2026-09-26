#!/usr/bin/env python3
"""Native-cell SFT training sets: is the structure task learnable *within* each cell?

The parent protocol trains only on Coin City with semantic labels and holds out
Coin Harbor and arbitrary labels. Supervised training reaches 97.9% in the trained
cell and 39.6-50% in the held-out cells, with reference identification perfect
everywhere. That is consistent with two different things: a semantic shortcut that
arbitrary symbols break, or an inability to bind an arbitrary symbol to a numeric
pattern at all. Training natively in each cell separates them.

Episodes use a seed band disjoint from the parent's training (61M) and evaluation
(71M+) bands, so no training episode can coincide with an evaluation episode.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SEL = REPO / 'exp3_training_transfer/coin_city_structure_selection'
sys.path.insert(0, str(SEL))
import make_dataset as M

# (cell name, domain, label_kind, disjoint seed base)
CELLS = (('coin_city_arbitrary',   'coin_city',   'arbitrary', 62_000_000),
         ('coin_harbor_semantic',  'coin_harbor', 'semantic',  63_000_000),
         ('coin_harbor_arbitrary', 'coin_harbor', 'arbitrary', 64_000_000))


def sha_text(t): return hashlib.sha256(t.encode()).hexdigest()


def main():
    assert M.TRAIN_SEED_BASE == 61_000_000 and M.EVAL_SEED_BASE == 71_000_000, 'parent seed bands moved'
    assert M.TRAIN_ROWS == 4800
    manifest = {'protocol': 'native_cell_supervised_learnability_v1',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'parent': 'coin_city_structure_selection_v1',
                'generator_sha256': hashlib.sha256((SEL / 'make_dataset.py').read_bytes()).hexdigest(),
                'rows_per_cell': M.TRAIN_ROWS, 'cue': 'correct', 'cells': []}
    for name, domain_name, label_kind, base in CELLS:
        assert not (61_000_000 <= base < 61_000_000 + M.TRAIN_ROWS), 'collides with parent train band'
        assert base + M.TRAIN_ROWS <= 71_000_000, 'collides with parent eval band'
        domain = M.DOMAINS[domain_name]
        rows, seen = [], set()
        for index in range(M.TRAIN_ROWS):
            seed = base + index
            k = M.K_VALUES[index % len(M.K_VALUES)]
            structure = M.STRUCTURES[(index // len(M.K_VALUES)) % len(M.STRUCTURES)]
            ep = M._episode(domain, seed, k, structure, 'correct', label_kind)
            pair = f'native:{name}:{index}:k{k}'
            row = M._row(ep, 'causal', pair, pair)
            ref = json.loads(row['reference'])
            assert ref['domain'] == domain_name and ref['label_kind'] == label_kind
            assert ref['cue'] == 'correct'
            forecasts = json.loads(ref['gold'])['forecasts']
            assert len(forecasts) == 10
            assert max(abs(a - b) for a, b in zip(forecasts, ref['truth_targets'])) <= 0.000501
            assert ref['task_id'] not in seen
            seen.add(ref['task_id'])
            rows.append({'task_id': ref['task_id'], 'input': row['input'], 'completion': ref['gold']})
        text = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
        out = ROOT / f'train_{name}.jsonl'
        if out.exists(): raise SystemExit(f'{out} exists; refuse overwrite')
        out.write_text(text)
        manifest['cells'].append({'cell': name, 'domain': domain_name, 'label_kind': label_kind,
                                  'seed_base': base, 'rows': len(rows), 'path': str(out),
                                  'data_sha256': sha_text(text)})
        print(f'{name}: {len(rows)} rows  sha={sha_text(text)[:16]}', flush=True)
    (ROOT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('wrote', ROOT / 'data_manifest.json')


if __name__ == '__main__':
    main()
