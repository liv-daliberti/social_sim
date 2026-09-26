#!/usr/bin/env python3
"""Per-layer decoding of the City C binding, for choosing a causal patch window.

The relational protocol selects the activation-patching window at the argmax of
development decoding CV. That is a poor criterion, because decoding accuracy is
not peaked: it rises while the binding is being computed and then sits on a long,
nearly flat plateau while the binding is merely carried. Argmax over a flat
plateau is noise-dominated, and the deeper the model the longer the plateau, so
the rule degrades with scale.

This sweep prints the whole profile instead, so the window can be placed at the
rise or its peak rather than wherever the argmax happens to fall. Development
episodes only; it computes no sealed-test metric. Hidden states for every layer
are already saved by the extraction step, so this costs no GPU time.

Run:
    python layer_decoding_sweep.py --run-dir <study directory>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ridge_fit(X, y, alpha):
    """Dual-form ridge, cheap when features greatly outnumber rows."""
    Xc = X - X.mean(0, keepdims=True)
    K = Xc @ Xc.T
    a = np.linalg.solve(K + alpha * np.eye(len(K)), y - y.mean())
    return Xc.T @ a, X.mean(0), y.mean()


def roc_auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float('nan')
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def sweep(run_dir: Path, alpha: float = 0.1):
    tasks = {}
    for line in (run_dir / 'tasks.jsonl').read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            tasks[r['sample_id']] = r
    z = np.load(run_dir / 'label_token_states.npz')
    states, sample_ids, symbols = z['label_token_states'], z['sample_ids'], z['symbols']

    keep, y, fold = [], [], []
    for i, sid in enumerate(sample_ids):
        t = tasks.get(str(sid))
        if t is None or t['split'] != 'dev':
            continue
        match = int(symbols[i, 2] != symbols[i, 0])       # which reference C binds
        strong = 0 if t['strong_reference_city'] == 'A' else 1
        keep.append(i)
        y.append(int(match == strong))                    # does C inherit the strong regime?
        fold.append(t['cv_fold'])
    keep, y, fold = np.array(keep), np.array(y), np.array(fold)

    out = []
    for layer in range(states.shape[1]):
        c = states[keep, layer, 2].astype(np.float32)
        m = np.stack([states[k, layer, int(symbols[k, 2] != symbols[k, 0])] for k in keep]).astype(np.float32)
        X = (c - m).mean(axis=1)
        scores = np.zeros(len(y))
        for f in sorted(set(fold.tolist())):
            tr, te = fold != f, fold == f
            if len(set(y[tr].tolist())) < 2:
                continue
            w, mu, b = ridge_fit(X[tr], y[tr].astype(np.float64), alpha)
            scores[te] = (X[te] - mu) @ w + b
        out.append({'layer': layer, 'dev_cv_auc': float(roc_auc(y, scores))})
    return {'run_dir': str(run_dir), 'alpha': alpha, 'development_rows': int(len(keep)),
            'scope': 'development episodes only; no sealed-test metric', 'layers': out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    report = sweep(a.run_dir, a.alpha)
    peak = max(report['layers'], key=lambda r: r['dev_cv_auc'])
    print(f"development rows: {report['development_rows']}")
    for r in report['layers']:
        bar = '#' * int(round(max(0.0, r['dev_cv_auc'] - 0.5) * 60))
        print(f"{r['layer']:5d} {r['dev_cv_auc']:7.3f}   {bar}")
    print(f"\npeak: layer {peak['layer']} at AUC {peak['dev_cv_auc']:.3f}")
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + '\n')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
