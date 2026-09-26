#!/usr/bin/env python3
"""Hosted vs. open-weight group contrast for Experiment 1 selectivity.

The manuscript's scale claim compares three hosted API systems against six
scale-valid open-weight models, but reports only per-model intervals. This script
supplies the group-level test, using the same market-clustered resampling scheme
as the published intervals: 20,000 bootstrap replicates over the 100-market core,
resampling whole markets and retaining every packet and repeated run belonging to
a sampled market.

Five statistics, each as a hosted-minus-open-weight difference of macro-averaged
model means, so that a model with more stored runs does not dominate its group:

  AUC          within-market P(|dp| directional > |dp| orthogonal)
  Difference   mean directional minus mean orthogonal absolute revision, in pp
  Sensitivity  ratio of those means
  Abstention   share of orthogonal runs whose revision is exactly zero
  Sens|move    sensitivity recomputed among runs that moved at all

AUC and abstention are the statistics to read. The published ratio has an
unstable denominator and, as audit_orthogonal_abstention.py shows, mixes
abstention with magnitude; the raw difference is in units that vary with each
model's overall movement scale.

Records come from the frozen June 17 report, which enumerates every scored run.

Outputs:
  data/results/group_contrast.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from audit_selectivity_robustness import _auc, _load_frozen_runs, FROZEN, LABELS
from audit_orthogonal_abstention import ZERO_TOL_PP

ROOT = _HERE.parent
OUT_JSON = ROOT / "data" / "results" / "group_contrast.json"

HOSTED = ("claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro")
LOCAL = ("qwen2.5:72b", "llama3.3:70b", "llama3.1:70b",
         "qwen2.5:32b", "qwen2.5:14b", "llama3.1:8b")
N_BOOT = 20_000
SEED = 20260617
KEYS = ("auc", "diff", "sens", "abstention", "sens_move")


def main() -> None:
    report = json.loads(FROZEN.read_text())
    rows, _ = _load_frozen_runs(report)

    by_market: dict[tuple, dict] = defaultdict(lambda: {"dir": [], "orth": []})
    for r in rows:
        arm = "orth" if r["direction"] == "orthogonal" else "dir"
        by_market[(r["model"], r["task_id"])][arm].append(r["abs_delta_pp"])

    models = HOSTED + LOCAL
    markets = sorted({m for (_, m) in by_market})

    # Precompute each model's per-market cell so resampling is a lookup.
    cell: dict[tuple, dict] = {}
    for model in models:
        for market in markets:
            d = by_market.get((model, market), {"dir": [], "orth": []})
            if d["dir"] and d["orth"]:
                cell[(model, market)] = {
                    "auc": _auc(d["dir"], d["orth"]),
                    "dir_sum": sum(d["dir"]), "dir_n": len(d["dir"]),
                    "orth_sum": sum(d["orth"]), "orth_n": len(d["orth"]),
                    "dir_zero": sum(1 for v in d["dir"] if v <= ZERO_TOL_PP),
                    "orth_zero": sum(1 for v in d["orth"] if v <= ZERO_TOL_PP),
                }

    def stats_for(model, market_idx):
        aucs, ds, dn, os_, on, dz, oz = [], 0.0, 0, 0.0, 0, 0, 0
        for i in market_idx:
            c = cell.get((model, markets[i]))
            if c is None:
                continue
            aucs.append(c["auc"])
            ds += c["dir_sum"]; dn += c["dir_n"]; dz += c["dir_zero"]
            os_ += c["orth_sum"]; on += c["orth_n"]; oz += c["orth_zero"]
        if not aucs or not dn or not on:
            return None
        dmean, omean = ds / dn, os_ / on
        dmoved, omoved = dn - dz, on - oz
        return {
            "auc": st.mean(aucs),
            "diff": dmean - omean,
            "sens": dmean / omean if omean > 0 else None,
            "abstention": oz / on,
            "sens_move": ((ds / dmoved) / (os_ / omoved)
                          if dmoved and omoved and os_ > 0 else None),
        }

    full = np.arange(len(markets))
    point = {m: stats_for(m, full) for m in models}

    def group_diff(per_model, key):
        h = [per_model[m][key] for m in HOSTED if per_model.get(m) and per_model[m][key] is not None]
        l = [per_model[m][key] for m in LOCAL if per_model.get(m) and per_model[m][key] is not None]
        if not h or not l:
            return None
        return st.mean(h) - st.mean(l)

    observed = {k: group_diff(point, k) for k in KEYS}

    rng = np.random.RandomState(SEED)
    draws = {k: [] for k in KEYS}
    n = len(markets)
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        per_model = {m: stats_for(m, idx) for m in models}
        for k in draws:
            v = group_diff(per_model, k)
            if v is not None:
                draws[k].append(v)

    result = {
        "analysis": "exp1_hosted_vs_open_weight_group_contrast",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_source": FROZEN.name,
        "resamples": N_BOOT,
        "seed": SEED,
        "clustering": "whole markets, retaining all packets and repeated runs",
        "hosted": list(HOSTED),
        "open_weight": list(LOCAL),
        "per_model": {m: {"label": LABELS[m], **{k: round(v, 4) for k, v in point[m].items()
                                                 if v is not None}} for m in models},
        "group_means": {
            k: {"hosted": round(st.mean([point[m][k] for m in HOSTED]), 4),
                "open_weight": round(st.mean([point[m][k] for m in LOCAL]), 4)}
            for k in KEYS
        },
        "contrasts": {},
    }
    for k in KEYS:
        arr = np.array(draws[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        result["contrasts"][k] = {
            "hosted_minus_open_weight": round(observed[k], 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "share_of_resamples_favouring_hosted": round(float((arr > 0).mean()), 4),
        }

    # Overlap check: does any open-weight model beat the weakest hosted one?
    for k in KEYS:
        worst_hosted = min(point[m][k] for m in HOSTED)
        best_local = max(point[m][k] for m in LOCAL)
        result["contrasts"][k]["groups_fully_separated"] = bool(best_local < worst_hosted)
        result["contrasts"][k]["open_weight_models_above_weakest_hosted"] = [
            LABELS[m] for m in LOCAL if point[m][k] >= worst_hosted]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'model':18s} {'AUC':>7s} {'diff pp':>8s} {'sens':>7s} "
          f"{'abstain':>8s} {'sens|mv':>8s}")
    for m in models:
        tag = "H" if m in HOSTED else "L"
        print(f"{LABELS[m]:18s} {point[m]['auc']:7.3f} {point[m]['diff']:8.2f} "
              f"{point[m]['sens']:7.2f} {point[m]['abstention']:8.3f} "
              f"{point[m]['sens_move']:8.2f}  [{tag}]")
    print()
    for k, name in (("auc", "AUC"), ("diff", "Difference (pp)"),
                    ("sens", "Sensitivity ratio"),
                    ("abstention", "Orthogonal abstention"),
                    ("sens_move", "Sensitivity | moved")):
        c = result["contrasts"][k]
        g = result["group_means"][k]
        print(f"{name:20s} hosted {g['hosted']:8.3f}  open {g['open_weight']:8.3f}  "
              f"diff {c['hosted_minus_open_weight']:+.3f} "
              f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
              f"excludes 0: {c['excludes_zero']}  separated: {c['groups_fully_separated']}")
        if c["open_weight_models_above_weakest_hosted"]:
            print(f"{'':20s} open-weight at or above weakest hosted: "
                  f"{', '.join(c['open_weight_models_above_weakest_hosted'])}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
