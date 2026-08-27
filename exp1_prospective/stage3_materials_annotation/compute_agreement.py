#!/usr/bin/env python3
"""Agreement statistics for the Stage 3 synthetic-materials check.

Computes, for the completed quality-filtered reviewer panel:

  * human-human   -- Fleiss' kappa over the registered four response options,
                     plus mean pairwise Cohen's kappa;
  * human-key     -- per-reviewer and majority-vote Cohen's kappa against the
                     registered conditional direction;
  * human-model   -- per-model Cohen's kappa against the registered key and
                     against the human majority label, on the same 18 packets.

Model direction labels come from the mean signed revision for that packet,
thresholded at the same 3 percentage points used for EHC/HFC. A model is scored
only on packets for which it has at least one parseable update carrying a
delta; per-model item coverage is reported so partial panels are visible.

Usage:  python compute_agreement.py [--emit-tex PATH]
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
import glob
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GENERATED = HERE / "generated_v6"
RESPONSES = HERE / "data/exports/registered_20260827T204456Z.csv"
UPDATES = ROOT / "data/updated_forecasts"

INCLUDED = [
    "annotator_01",
    "annotator_03",
    "annotator_04",
    "annotator_05",
    "annotator_06",
    "annotator_07",
    "annotator_08",
    "annotator_09",
]
LABEL = {reviewer_id: f"R{int(reviewer_id[-2:])}" for reviewer_id in INCLUDED}
CATS = ["more_likely", "less_likely", "no_material_effect", "ambiguous"]
THRESHOLD = 0.03
MODEL_LABEL = {
    "gpt-5.4": "GPT-5.4",
    "claude-opus-4-8": "Claude Opus~4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro",
    "qwen2.5:7b": "Qwen~2.5-7B",
    "qwen2.5:14b": "Qwen~2.5-14B",
    "qwen2.5:32b": "Qwen~2.5-32B",
    "qwen2.5:72b": "Qwen~2.5-72B",
    "llama3.1:8b": "Llama~3.1-8B",
    "llama3.1:70b": "Llama~3.1-70B",
    "llama3.3:70b": "Llama~3.3-70B",
}


def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in CATS)
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po


def fleiss(per_item_counts, cats):
    n = sum(per_item_counts[0].values())
    N = len(per_item_counts)
    pbar = sum((sum(v * v for v in m.values()) - n) / (n * (n - 1))
               for m in per_item_counts) / N
    totals = collections.Counter()
    for m in per_item_counts:
        totals.update(m)
    pe = sum((totals[c] / (N * n)) ** 2 for c in cats)
    return (pbar - pe) / (1 - pe)


def direction_label(delta):
    if delta > THRESHOLD:
        return "more_likely"
    if delta < -THRESHOLD:
        return "less_likely"
    return "no_material_effect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-tex", type=Path)
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    key = {r["item_id"]: r for r in
           (json.loads(l) for l in (GENERATED / "private_key.jsonl").read_text().splitlines() if l.strip())}
    items = sorted(key)
    gold = {i: key[i]["expected_conditional_direction"] for i in items}
    klass = {i: key[i]["registered_direction"] for i in items}

    human = collections.defaultdict(dict)
    with RESPONSES.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            human[row["reviewer_id"]][row["item_id"]] = row["conditional_direction"]

    # ---- human-human -----------------------------------------------------
    counts = [collections.Counter(human[r][i] for r in INCLUDED) for i in items]
    fleiss_4 = fleiss(counts, CATS)
    folded = [collections.Counter(
        ("no_material_effect" if human[r][i] == "ambiguous" else human[r][i])
        for r in INCLUDED) for i in items]
    fleiss_3 = fleiss(folded, CATS[:3])
    pairwise = [cohen([human[a][i] for i in items], [human[b][i] for i in items])[0]
                for a, b in itertools.combinations(INCLUDED, 2)]

    # ---- human-key -------------------------------------------------------
    def majority(i):
        c = collections.Counter(human[r][i] for r in INCLUDED).most_common()
        return "ambiguous" if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]
    maj = {i: majority(i) for i in items}
    per_reviewer = {}
    for r in INCLUDED:
        k, po = cohen([human[r][i] for i in items], [gold[i] for i in items])
        per_reviewer[LABEL[r]] = {"kappa": k, "exact": po,
                                  "correct": sum(human[r][i] == gold[i] for i in items)}
    maj_k, maj_po = cohen([maj[i] for i in items], [gold[i] for i in items])

    by_class = {}
    for c in ("pro_H1", "anti_H1", "orthogonal"):
        sub = [i for i in items if klass[i] == c]
        by_class[c] = {
            "reviewer_correct": sum(human[r][i] == gold[i] for r in INCLUDED for i in sub),
            "reviewer_denom": len(sub) * len(INCLUDED),
            "majority_correct": sum(maj[i] == gold[i] for i in sub),
            "majority_denom": len(sub),
        }

    # ---- registered item-retention gate ----------------------------------
    # Registered rule: at most one dissenting reviewer on direction, median
    # confidence and clarity >= 4/5, median plausibility >= 3/5, and at most one
    # reviewer marking the premise unusable. The packet-level gate additionally
    # requires >= 15/18 items to pass and Fleiss' kappa >= .70.
    raw = collections.defaultdict(dict)
    with RESPONSES.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            raw[row["item_id"]][row["reviewer_id"]] = row
    gate_items = []
    for i in items:
        rs = [raw[i][a] for a in INCLUDED]
        rec = {
            "item_id": i,
            "direction_match": sum(r["conditional_direction"] == gold[i] for r in rs),
            "median_confidence": statistics.median(int(r["direction_confidence"]) for r in rs),
            "median_clarity": statistics.median(int(r["clarity"]) for r in rs),
            "median_plausibility": statistics.median(int(r["plausibility"]) for r in rs),
            "premise_usable": sum(r["usable_premise"] == "yes" for r in rs),
        }
        rec["passes"] = (rec["direction_match"] >= len(INCLUDED) - 1
                         and rec["median_confidence"] >= 4 and rec["median_clarity"] >= 4
                         and rec["median_plausibility"] >= 3
                         and rec["premise_usable"] >= len(INCLUDED) - 1)
        gate_items.append(rec)
    gate = {
        "items_passing": sum(r["passes"] for r in gate_items),
        "items_total": len(items),
        "registered_items_required": 15,
        "registered_fleiss_kappa_required": 0.70,
        "observed_fleiss_kappa": fleiss_4,
        "direction_only_at_most_one_dissent": sum(r["direction_match"] >= len(INCLUDED) - 1
                                                  for r in gate_items),
        "direction_only_unanimous": sum(r["direction_match"] == len(INCLUDED) for r in gate_items),
        "packet_gate_met": False,
        "per_item": gate_items,
    }
    gate["packet_gate_met"] = (gate["items_passing"] >= gate["registered_items_required"]
                               and fleiss_4 >= gate["registered_fleiss_kappa_required"])

    # ---- models on the same packets --------------------------------------
    cf2item = {key[i]["cf_id"]: i for i in items}
    deltas = collections.defaultdict(lambda: collections.defaultdict(list))
    unparseable = 0
    for path in sorted(glob.glob(str(UPDATES / "*.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                unparseable += 1
                continue
            if rec.get("cf_id") in cf2item and rec.get("delta_yes_prob") is not None:
                deltas[rec["forecast_model"]][cf2item[rec["cf_id"]]].append(
                    float(rec["delta_yes_prob"]))

    models = {}
    for m in sorted(deltas):
        covered = [i for i in items if deltas[m][i]]
        labels = [direction_label(sum(deltas[m][i]) / len(deltas[m][i])) for i in covered]
        k_key, exact = cohen(labels, [gold[i] for i in covered])
        k_maj, _ = cohen(labels, [maj[i] for i in covered])
        per_human = [cohen(labels, [human[r][i] for i in covered])[0] for r in INCLUDED]
        models[m] = {
            "items_covered": len(covered), "runs": sum(len(deltas[m][i]) for i in covered),
            "kappa_vs_key": k_key, "exact_vs_key": exact, "kappa_vs_human_majority": k_maj,
            "mean_kappa_vs_reviewers": sum(per_human) / len(per_human),
        }

    full = {m: v for m, v in models.items() if v["items_covered"] == len(items)}
    summary = {
        "n_items": len(items), "n_reviewers": len(INCLUDED),
        "reviewer_ids": [LABEL[r] for r in INCLUDED],
        "threshold_pp": THRESHOLD * 100,
        "unparseable_update_lines_skipped": unparseable,
        "human_human": {
            "fleiss_kappa_4cat": fleiss_4, "fleiss_kappa_3cat_ambiguous_folded": fleiss_3,
            "mean_pairwise_cohen_kappa": sum(pairwise) / len(pairwise),
            "pairwise_min": min(pairwise), "pairwise_max": max(pairwise),
        },
        "human_key": {
            "per_reviewer": per_reviewer,
            "pooled_correct": sum(v["correct"] for v in per_reviewer.values()),
            "pooled_denom": len(items) * len(INCLUDED),
            "majority_kappa": maj_k, "majority_exact": maj_po,
            "by_packet_class": by_class,
        },
        "registered_gate": gate,
        "models": models,
        "models_full_coverage": {
            "n": len(full),
            "kappa_vs_key_min": min(v["kappa_vs_key"] for v in full.values()),
            "kappa_vs_key_max": max(v["kappa_vs_key"] for v in full.values()),
            "kappa_vs_majority_min": min(v["kappa_vs_human_majority"] for v in full.values()),
            "kappa_vs_majority_max": max(v["kappa_vs_human_majority"] for v in full.values()),
        },
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}")

    if args.emit_tex:
        rows = []
        for m, v in sorted(models.items(), key=lambda kv: -kv[1]["kappa_vs_key"]):
            note = "" if v["items_covered"] == len(items) else f" ({v['items_covered']}/18)"
            display = MODEL_LABEL.get(m, m.replace("_", "-"))
            rows.append(
                f"{display}{note} & {v['kappa_vs_key']:.2f} & "
                f"{v['exact_vs_key']*100:.1f}\\% & "
                f"{v['kappa_vs_human_majority']:.2f} \\\\")
        args.emit_tex.write_text(
            "% Generated by stage3_materials_annotation/compute_agreement.py -- do not hand-edit.\n"
            "\\begin{tabular}{lccc}\n\\toprule\n"
            "Model & $\\kappa$ vs.\\ key & Exact vs.\\ key & "
            "$\\kappa$ vs.\\ human majority \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        print(f"\nwrote {args.emit_tex}")


if __name__ == "__main__":
    main()
