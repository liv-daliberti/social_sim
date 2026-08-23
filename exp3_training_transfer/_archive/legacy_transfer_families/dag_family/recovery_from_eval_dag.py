#!/usr/bin/env python3
"""Read OAT held-out eval dumps and report the Experiment-3 TRANSFER result on the DAG family.

OAT writes <save_path>/.../eval_results/<step>.json at every eval: one record per held-out prompt
with the model `output` and our `reference` JSON. The held-out set is the 4 TEST structures, four
shocks per (structure, city, prefix k) -- so we group by (structure, seed, k) and per group compute

    forecast MAE = mean_shock |pred - target|         (poll pts; the transfer headline)
    recovery ghat = slope_shock(pred)                  (where the structure has a one-step gain)

then aggregate per structure. Point it at the step-0 dump (base model = BEFORE training) and the
final dump (AFTER training) to read the transfer: does training on the 16 OTHER structures improve
forecasting on the 4 held-out ones?

Usage:
    python recovery_from_eval_dag.py <eval_results_dir>
    python recovery_from_eval_dag.py --before <0.json> --after <NN.json>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

_PRED_RE = re.compile(r'"?predicted_poll"?\s*:\s*"?(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def _parse_pred(text):
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    m = None
    for m in _PRED_RE.finditer(text or ""):
        pass
    return max(0.0, min(100.0, float(m.group(1)))) if m else None


def read_dump(path: Path) -> dict:
    records = json.loads(Path(path).read_text())
    groups = defaultdict(lambda: {"shock": [], "pred": [], "target": [],
                                  "true_gain": None, "recover": False})
    n_unparsed = n_total = 0
    for rec in records:
        ref = json.loads(rec["reference"])
        key = (ref["structure"], ref["seed"], ref["k"])
        gd = groups[key]
        gd["recover"] = ref.get("recover", False); gd["true_gain"] = ref.get("true_gain")
        n_total += 1
        pred = _parse_pred(rec["output"])
        if pred is None:
            n_unparsed += 1
            continue
        gd["shock"].append(float(ref["test_news"])); gd["pred"].append(pred)
        gd["target"].append(float(ref["target"]))

    per = defaultdict(lambda: {"fmae": [], "rmae": [], "n": 0})
    for (struct, _seed, _k), gd in groups.items():
        per[struct]["n"] += 1
        if gd["pred"]:
            per[struct]["fmae"].append(np.mean([abs(p - t) for p, t in zip(gd["pred"], gd["target"])]))
        if gd["recover"] and gd["true_gain"] is not None and len(gd["pred"]) >= 2:
            x = np.asarray(gd["shock"]); y = np.asarray(gd["pred"])
            if np.var(x) > 0:
                ghat = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
                per[struct]["rmae"].append(abs(ghat - gd["true_gain"]))
    out = {st: {"fmae": round(float(np.mean(d["fmae"])), 2) if d["fmae"] else None,
                "rmae": round(float(np.mean(d["rmae"])), 2) if d["rmae"] else None,
                "n": d["n"]} for st, d in per.items()}
    out["_miss"] = round(n_unparsed / max(n_total, 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", help="eval_results dir (uses first & last step dumps)")
    ap.add_argument("--before"); ap.add_argument("--after")
    args = ap.parse_args()
    if args.dir:
        dumps = sorted(Path(args.dir).glob("*.json"),
                       key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
        if len(dumps) < 1:
            print("no eval dumps in", args.dir); return
        before, after = dumps[0], dumps[-1]
    else:
        before, after = Path(args.before), Path(args.after)

    b = read_dump(before); a = read_dump(after)
    print(f"BEFORE {before.name}  (miss {b['_miss']})   AFTER {after.name}  (miss {a['_miss']})\n")
    print(f"{'held-out structure':20s} {'forecast MAE':>22s} {'recovery MAE':>22s}")
    print(f"{'':20s} {'base  ->  trained':>22s} {'base  ->  trained':>22s}")

    def arrow(x, y):
        sx = f"{x:.2f}" if isinstance(x, float) else "--"
        sy = f"{y:.2f}" if isinstance(y, float) else "--"
        return f"{sx:>7s} -> {sy:<7s}"

    fbs, fas, rbs, ras = [], [], [], []
    for st in sorted(k for k in b if not k.startswith("_")):
        print(f"{st:20s} {arrow(b[st]['fmae'], a[st]['fmae']):>22s} "
              f"{arrow(b[st]['rmae'], a[st]['rmae']):>22s}")
        for L, v in ((fbs, b[st]['fmae']), (fas, a[st]['fmae']), (rbs, b[st]['rmae']), (ras, a[st]['rmae'])):
            if isinstance(v, float):
                L.append(v)
    mean = lambda L: (round(float(np.mean(L)), 2) if L else None)
    print("-" * 66)
    print(f"{'TEST mean':20s} {arrow(mean(fbs), mean(fas)):>22s} {arrow(mean(rbs), mean(ras)):>22s}")
    print("\n(reference bracket @k~3: oracle forecast ~1.7 / recovery ~0.14; "
          "naive ~3.85 / 0.37; persistence ~4.6)")


if __name__ == "__main__":
    main()
