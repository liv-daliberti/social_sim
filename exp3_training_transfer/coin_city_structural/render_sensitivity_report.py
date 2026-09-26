#!/usr/bin/env python3
"""Sensitivity and measurement-validity report for the Coin City endpoints.

Two weaknesses in the reported measures, documented here rather than left for a
reader to find.

1. Classifier brittleness. `structure_correct` is the argmax of cosine
   similarity against two response templates. In the mediated_b cells those
   templates are close enough that a perfectly direct_a-shaped answer still
   scores ~0.83 cosine against mediated truth, so the label flips on a small
   margin. The headline 94% -> 0% collapse it reports is therefore a knife-edge
   relabelling, not a capability measurement, and trained cosine-to-truth in
   those cells is actually *better* than base. This script prints the template
   separation and cosine alongside every accuracy so the reader can see which
   is doing the work.

2. Parse-penalty leverage. Unparsed draws take the full observable-range
   penalty -- 280 points in Coin Harbor, ~50x a typical error. Base parse rate
   is 99.83%, trained is 100%, so every base-versus-trained contrast is
   partially a parse-rate comparison. One unparsed draw moves the base joint
   MAE from 5.26 to 5.72. This script reports every base contrast twice, once
   under the registered fail-closed policy and once on parsed draws only.

Reads registered endpoints only. Writes nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CELLS = (("ID", "coin_city", "direct_a"), ("Domain", "coin_harbor", "direct_a"),
         ("Structure", "coin_city", "mediated_b"), ("Joint", "coin_harbor", "mediated_b"))
TCRIT = {2: 4.302653, 4: 2.776445, 7: 2.364624}


def load(pattern: str) -> list[dict] | None:
    dirs = [d for d in glob.glob(str(REPORTS / pattern)) if "failed_attempts" not in d]
    if len(dirs) != 1:
        return None
    path = Path(dirs[0]) / "stochastic_n5.scores.jsonl"
    if not path.is_file():
        return None
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def cell_rows(rows: list[dict], domain: str, structure: str) -> list[dict]:
    return [r for r in rows if r["domain"] == domain
            and r["target_structure"] == structure and r["cue"] == "correct"]


def episode_mae(rows: list[dict], parsed_only: bool) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if parsed_only and not r.get("parsed"):
            continue
        acc[r["task_id"]].append(float(r["response_mae"]))
    return {t: float(np.mean(v)) for t, v in acc.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48, 49])
    args = parser.parse_args()
    model, seeds = args.model, tuple(args.seeds)

    base = load(f"base_{model}_j*")
    trained = {arm: {s: load(f"{arm}_{model}_s{s}_*") for s in seeds}
               for arm in ("causal", "population_prior")}
    if base is None or any(v is None for a in trained.values() for v in a.values()):
        raise SystemExit(f"incomplete registered roster for {model} at seeds {list(seeds)}")

    print(f"SENSITIVITY REPORT  {model}  G={len(seeds)}  seeds {list(seeds)}\n")

    print("=" * 78)
    print("1. IS THE STRUCTURE CLASSIFIER MEASURING ANYTHING?")
    print("=" * 78)
    print("template separation = cosine between the two candidate response shapes.")
    print("Near 1.0 means structure_correct is deciding on a sliver.\n")
    print(f"{'cell':<10}{'template cos':>13}{'base acc':>10}{'trained acc':>12}"
          f"{'base cos':>10}{'trained cos':>12}")
    for name, domain, structure in CELLS:
        brows = cell_rows(base, domain, structure)
        # Cosine between the mediated truth and the direct-shaped answer (same
        # horizon-1 response, zero at horizon 3). Only defined where the target
        # is mediated: for a direct_a target the direct shape *is* the truth,
        # and the score files do not carry the mediated template, so there is
        # no separation to report.
        sep = []
        if structure == "mediated_b":
            for r in brows:
                d = np.asarray(r.get("truth_response", []), dtype=float)
                if d.size:
                    alt = d.copy()
                    alt[4:] = 0.0
                    if np.linalg.norm(alt) > 0 and np.linalg.norm(d) > 0:
                        sep.append(float(np.dot(alt, d)
                                         / (np.linalg.norm(alt) * np.linalg.norm(d))))
        trows = [r for s in seeds for arm in trained for r in
                 cell_rows(trained[arm][s], domain, structure)]
        bc = [r["response_cosine"] for r in brows if r.get("response_cosine") is not None]
        tc = [r["response_cosine"] for r in trows if r.get("response_cosine") is not None]
        septxt = f"{np.mean(sep):.3f}" if sep else "n/a"
        print(f"{name:<10}{septxt:>13}"
              f"{np.mean([bool(r['structure_correct']) for r in brows]):>10.3f}"
              f"{np.mean([bool(r['structure_correct']) for r in trows]):>12.3f}"
              f"{np.mean(bc):>10.3f}{np.mean(tc):>12.3f}")
    print("\nWhere template cosine is high and trained cosine exceeds base while")
    print("trained accuracy collapses, the accuracy figure is an artefact of the")
    print("argmax, not a drop in response quality.")

    print("\n" + "=" * 78)
    print("2. PARSE-PENALTY LEVERAGE ON EVERY BASE CONTRAST")
    print("=" * 78)
    bp = 100 * np.mean([bool(r.get("parsed")) for r in base])
    tp = 100 * np.mean([bool(r.get("parsed")) for s in seeds for arm in trained
                        for r in trained[arm][s]])
    print(f"parse rate: base {bp:.2f}%   trained {tp:.2f}%\n")
    print(f"{'cell':<10}{'fail-closed':>26}{'parsed-only':>26}{'shift':>9}")
    for name, domain, structure in CELLS:
        out = []
        for parsed_only in (False, True):
            b = episode_mae(cell_rows(base, domain, structure), parsed_only)
            per_seed = []
            for s in seeds:
                c = episode_mae(cell_rows(trained["causal"][s], domain, structure), parsed_only)
                p = episode_mae(cell_rows(trained["population_prior"][s], domain, structure),
                                parsed_only)
                common = sorted(set(b) & set(c) & set(p))
                per_seed.append(float(np.mean([b[t] - (c[t] + p[t]) / 2 for t in common])))
            v = np.asarray(per_seed)
            m, sd = float(v.mean()), float(v.std(ddof=1))
            h = TCRIT[len(seeds) - 1] * sd / math.sqrt(len(seeds))
            out.append((m, m - h, m + h))
        (m0, l0, h0), (m1, l1, h1) = out
        print(f"{name:<10}{f'{m0:+.3f} [{l0:+.3f},{h0:+.3f}]':>26}"
              f"{f'{m1:+.3f} [{l1:+.3f},{h1:+.3f}]':>26}{m1 - m0:>+9.3f}")
    print("\nA large shift means that contrast is substantially a parse-rate")
    print("comparison rather than a forecast-quality one, and both numbers belong")
    print("in the manuscript.")

    print("\n" + "=" * 78)
    print("3. CONTINUOUS ALTERNATIVE TO THE CLASSIFIER")
    print("=" * 78)
    print("implied persistence = mean|predicted h3| / mean|predicted h1|, as a")
    print("ratio of means so it is stable when individual h1 values are near zero.")
    print("truth: 0.00 for direct_a targets, ~0.65 for mediated_b.\n")
    print(f"{'cell':<10}{'truth':>8}{'base':>9}{'matched':>10}{'prior':>9}")
    for name, domain, structure in CELLS:
        def ratio(rows):
            h1 = [np.mean(np.abs(np.asarray(r["predicted_response"])[:4]))
                  for r in rows if r.get("predicted_response")]
            h3 = [np.mean(np.abs(np.asarray(r["predicted_response"])[4:]))
                  for r in rows if r.get("predicted_response")]
            return float(np.mean(h3) / max(np.mean(h1), 1e-9)) if h1 else float("nan")
        brows = cell_rows(base, domain, structure)
        truth_h1 = np.mean([np.mean(np.abs(np.asarray(r["truth_response"])[:4]))
                            for r in brows if r.get("truth_response")])
        truth_h3 = np.mean([np.mean(np.abs(np.asarray(r["truth_response"])[4:]))
                            for r in brows if r.get("truth_response")])
        c = [r for s in seeds for r in cell_rows(trained["causal"][s], domain, structure)]
        p = [r for s in seeds for r in
             cell_rows(trained["population_prior"][s], domain, structure)]
        print(f"{name:<10}{truth_h3 / max(truth_h1, 1e-9):>8.3f}{ratio(brows):>9.3f}"
              f"{ratio(c):>10.3f}{ratio(p):>9.3f}")


if __name__ == "__main__":
    main()
