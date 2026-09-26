#!/usr/bin/env python3
"""Design property: the within-group shuffle preserves episode level, not response shape.

This reads only the frozen training targets, never a model prediction, so it is a
property of the manipulation rather than a result. It is the reason a level-error
benefit is not predicted for matched training.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_from_disk

import build as b

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44, 45, 46)


def main():
    out = {'measured_at': datetime.now(timezone.utc).isoformat(),
           'definition': {'level': 'abs(mean(recipient) - mean(donor)) of response_targets',
                          'shape': 'mean abs((recipient - mean) - (donor - mean)) of response_targets'},
           'note': 'Computed from frozen training targets only; no predictions are read.',
           'cells': []}
    for disclosure in b.DISCLOSURES:
        rows = list(load_from_disk(str(b.MECH / 'data' / f'{disclosure}_causal_family' / 'train'))['train'])
        refs = [json.loads(r['reference']) for r in rows]
        within = []
        groups = {}
        for i, ref in enumerate(refs):
            groups.setdefault(b.group_key(ref), []).append(i)
        for idx in groups.values():
            within.append(np.std([np.asarray(refs[i]['response_targets'], float).mean() for i in idx]))
        for seed in SEEDS:
            _, donors, _ = b.shuffled_rows(rows, seed)
            lv, sh = [], []
            for i, j in enumerate(donors):
                a = np.asarray(refs[i]['response_targets'], float).ravel()
                c = np.asarray(refs[j]['response_targets'], float).ravel()
                lv.append(abs(a.mean() - c.mean()))
                sh.append(np.abs((a - a.mean()) - (c - c.mean())).mean())
            out['cells'].append({'disclosure': disclosure, 'seed': seed, 'episodes': len(donors),
                                 'level_delta': float(np.mean(lv)), 'shape_delta': float(np.mean(sh)),
                                 'ratio': float(np.mean(sh) / max(np.mean(lv), 1e-12)),
                                 'within_group_level_sd': float(np.mean(within))})
    lv = [c['level_delta'] for c in out['cells']]
    sh = [c['shape_delta'] for c in out['cells']]
    out['summary'] = {'level_delta_min': min(lv), 'level_delta_max': max(lv),
                      'shape_delta_min': min(sh), 'shape_delta_max': max(sh),
                      'ratio_min': min(c['ratio'] for c in out['cells']),
                      'ratio_max': max(c['ratio'] for c in out['cells']),
                      'within_group_level_sd': out['cells'][0]['within_group_level_sd']}
    (ROOT / 'level_shape_asymmetry.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out['summary'], indent=2))


if __name__ == '__main__':
    main()
