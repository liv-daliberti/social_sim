#!/usr/bin/env python3
"""Robustness checks for the Experiment 1 selectivity claim.

Source of record: the frozen June 17 consistency report. That report enumerates
every scored run --- model, market, packet, run index, initial and updated
probability, delta, and the EHC/HFC/ICS flags --- so these checks run on exactly
the data behind the manuscript's numbers rather than on the mutable collection
directory. The enumeration reproduces the report's own per-arm aggregate counts
exactly for eight of ten models; Claude Opus 4.8 and GPT-5.4 each carry a handful
of runs that the aggregate counted in markets absent from the per-market join
(18 runs in total, 0.05% of 38,730). That residual is reported, not assumed away.

Each check answers an alternative explanation for the directional/orthogonal
movement gap.

  COVERAGE   Per-model per-arm market coverage. Claude Opus 4.8 does not have all
             three arms in all 100 core markets, so its pooled sensitivity ratio
             mixes slightly different market sets across numerator and
             denominator. We recompute sensitivity restricted to markets covered
             on all three arms, and compare every other model's sensitivity on
             the markets Claude covers against the ones it misses, to test
             whether the missingness is selective.

  SCALE-FREE Sensitivity is a ratio of means whose denominator can be under one
             percentage point, so it is unstable by construction. We report the
             difference of means in percentage points, the common-language effect
             size AUC = P(|dp| directional > |dp| orthogonal) computed within
             market and averaged, and the number of markets whose mean
             directional movement exceeds its mean orthogonal movement.

  SURFACE    Orthogonal packets are shorter and carry fewer digits than
             directional ones by construction of the slot types. We report the
             asymmetry, the within-directional-arm rank correlation between
             surface features and movement, and sensitivity recomputed on the
             word-count band the two arms share. These three checks need packet
             text, so they join runs to the released June 10 packet set; runs
             from the superseded June 7 pilot generation cannot join and are
             counted separately.

Outputs:
  data/results/selectivity_robustness.json
  paper/tables/exp1_selectivity_robustness.tex
"""

from __future__ import annotations

import bisect
import json
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluate_consistency import _spearman

ROOT = _HERE.parent
REPO_ROOT = ROOT.parent
CF_FILE = ROOT / "data" / "counterfactuals" / "counterfactuals_2026-06-10.jsonl"
FROZEN = ROOT / "data" / "results" / "consistency_report_2026-06-17.json"
OUT_JSON = ROOT / "data" / "results" / "selectivity_robustness.json"
OUT_TEX = REPO_ROOT / "paper" / "tables" / "exp1_selectivity_robustness.tex"

# Qwen2.5-7B is excluded throughout: its archived updates mix 0-1 and 0-100
# scales, so absolute magnitudes are not comparable.
MODEL_ORDER = (
    "claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro",
    "qwen2.5:72b", "llama3.3:70b", "llama3.1:70b",
    "qwen2.5:32b", "qwen2.5:14b", "llama3.1:8b",
)
LABELS = {
    "claude-opus-4-8": "Claude Opus~4.8", "gpt-5.4": "GPT-5.4",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro", "qwen2.5:72b": "Qwen2.5-72B",
    "llama3.3:70b": "Llama-3.3-70B", "llama3.1:70b": "Llama-3.1-70B",
    "qwen2.5:32b": "Qwen2.5-32B", "qwen2.5:14b": "Qwen2.5-14B",
    "llama3.1:8b": "Llama-3.1-8B",
}
ARMS = ("pro_H1", "anti_H1", "orthogonal")


def _half_up(value: float, places: int = 1) -> float:
    """Round half away from zero, matching the rounding used in the frozen tables."""
    return float(Decimal(repr(value)).quantize(Decimal(1).scaleb(-places),
                                               rounding=ROUND_HALF_UP))


def _auc(xs: list[float], ys: list[float]) -> float:
    """P(X > Y) + .5 P(X = Y) for the two samples."""
    ys = sorted(ys)
    total = 0.0
    for x in xs:
        lo, hi = bisect.bisect_left(ys, x), bisect.bisect_right(ys, x)
        total += lo + 0.5 * (hi - lo)
    return total / (len(xs) * len(ys))


def _load_frozen_runs(report: dict) -> tuple[list, dict]:
    """Flatten the frozen report into one row per scored run."""
    rows = []
    for model, model_result in report["per_model"].items():
        for market in model_result["per_market"]:
            for packet in market.get("cf_results", []):
                direction = packet.get("direction")
                if direction not in ARMS:
                    continue
                for run in packet.get("runs", []):
                    delta = run.get("delta_yes_prob")
                    if delta is None:
                        continue
                    rows.append({
                        "model": model,
                        "task_id": market.get("task_id"),
                        "cf_id": packet.get("cf_id"),
                        "direction": direction,
                        "abs_delta_pp": abs(delta) * 100,
                    })

    # Reconcile the enumeration against the report's own aggregate.
    enumerated = defaultdict(lambda: defaultdict(int))
    for r in rows:
        enumerated[r["model"]][r["direction"]] += 1
    reconciliation = {}
    for model in MODEL_ORDER:
        anchoring = report["per_model"][model]["anchoring"]
        aggregate = {a: anchoring[a]["n"] for a in ARMS}
        present = {a: enumerated[model][a] for a in ARMS}
        reconciliation[model] = {
            "aggregate_records_by_arm": aggregate,
            "enumerated_records_by_arm": present,
            "exact_match": aggregate == present,
            "runs_in_aggregate_not_enumerated":
                sum(aggregate.values()) - sum(present.values()),
        }
    return rows, reconciliation


def write_table(per_model: dict) -> None:
    lines = []
    for model in MODEL_ORDER:
        r = per_model[model]
        lines.append(
            f"{r['label']:16s} & ${_half_up(r['frozen_sensitivity_ratio']):.1f}\\times$ & "
            f"{_half_up(r['difference_pp']):.1f}\\,pp & {r['auc_within_market']:.3f} & "
            f"{r['markets_directional_exceeds_orthogonal']}/{r['markets_paired']} & "
            f"${_half_up(r['sensitivity_length_matched']):.1f}\\times$ \\\\"
        )
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(
        "% Generated by exp1_prospective/agent/audit_selectivity_robustness.py\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "\\textbf{Model} & \\textbf{Sens.} & \\textbf{Diff.} & \\textbf{AUC} "
        "& \\textbf{Markets} & \\textbf{Length-restr.} \\\\\n\\midrule\n"
        + "\n".join(lines[:3]) + "\n\\midrule\n" + "\n".join(lines[3:])
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def main() -> None:
    report = json.loads(FROZEN.read_text())
    rows, reconciliation = _load_frozen_runs(report)
    packets = {json.loads(l)["cf_id"]: json.loads(l)
               for l in CF_FILE.read_text().splitlines() if l.strip()}

    by_market: dict[tuple, dict] = defaultdict(lambda: {"dir": [], "orth": []})
    surface: dict[str, dict] = defaultdict(lambda: {"words": [], "digits": [], "move": []})
    unjoinable: dict[str, int] = defaultdict(int)
    band_moves: dict[str, dict] = defaultdict(lambda: {"dir": [], "orth": []})

    words_by_arm = defaultdict(list)
    for p in packets.values():
        arm = "orth" if p["direction"] == "orthogonal" else "dir"
        words_by_arm[arm].append(len((p.get("evidence_text") or "").split()))
    lo = max(min(words_by_arm["orth"]), min(words_by_arm["dir"]))
    hi = min(max(words_by_arm["orth"]), max(words_by_arm["dir"]))

    for r in rows:
        model, arm = r["model"], ("orth" if r["direction"] == "orthogonal" else "dir")
        by_market[(model, r["task_id"])][arm].append(r["abs_delta_pp"])
        packet = packets.get(r["cf_id"])
        if packet is None:
            unjoinable[model] += 1
            continue
        text = packet.get("evidence_text") or ""
        words = len(text.split())
        if arm == "dir":
            surface[model]["words"].append(words)
            surface[model]["digits"].append(len(re.findall(r"\d", text)))
            surface[model]["move"].append(r["abs_delta_pp"])
        if lo <= words <= hi:
            band_moves[model][arm].append(r["abs_delta_pp"])

    def scored(entry, arm):
        block = (entry.get("by_direction") or {}).get(arm) or {}
        return (block.get("n_ICS") or 0) > 0

    coverage = {}
    for model in MODEL_ORDER:
        per_market = report["per_model"][model]["per_market"]
        coverage[model] = {
            "markets_in_report": len(per_market),
            "markets_all_three_arms":
                sum(1 for e in per_market if all(scored(e, a) for a in ARMS)),
            "markets_with_orthogonal":
                sum(1 for e in per_market if scored(e, "orthogonal")),
            "markets_with_directional":
                sum(1 for e in per_market if scored(e, "pro_H1") or scored(e, "anti_H1")),
            "frozen_sensitivity_ratio":
                report["per_model"][model]["anchoring"]["sensitivity_ratio"],
            "_covered": {e["task_id"] for e in per_market
                         if all(scored(e, a) for a in ARMS)},
        }

    def sensitivity(model, markets):
        d = [v for m in markets for v in by_market[(model, m)]["dir"]]
        o = [v for m in markets for v in by_market[(model, m)]["orth"]]
        if not d or not o or st.mean(o) == 0:
            return None
        return round(st.mean(d) / st.mean(o), 2)

    claude_covered = coverage["claude-opus-4-8"]["_covered"]
    per_model = {}
    for model in MODEL_ORDER:
        markets = {m for (mm, m) in by_market if mm == model}
        paired = [m for m in markets
                  if by_market[(model, m)]["dir"] and by_market[(model, m)]["orth"]]
        d = [v for m in markets for v in by_market[(model, m)]["dir"]]
        o = [v for m in markets for v in by_market[(model, m)]["orth"]]
        band = band_moves[model]
        per_model[model] = {
            "label": LABELS[model],
            "frozen_sensitivity_ratio": coverage[model]["frozen_sensitivity_ratio"],
            "reproduced_sensitivity_ratio": sensitivity(model, markets),
            "difference_pp": round(st.mean(d) - st.mean(o), 2),
            "auc_within_market": round(
                st.mean([_auc(by_market[(model, m)]["dir"], by_market[(model, m)]["orth"])
                         for m in paired]), 3),
            "markets_directional_exceeds_orthogonal": sum(
                1 for m in paired
                if st.mean(by_market[(model, m)]["dir"])
                > st.mean(by_market[(model, m)]["orth"])),
            "markets_paired": len(paired),
            "sensitivity_all_three_arm_markets":
                sensitivity(model, coverage[model]["_covered"] & markets),
            "sensitivity_on_markets_claude_covers":
                sensitivity(model, claude_covered & markets),
            "sensitivity_on_markets_claude_misses":
                sensitivity(model, markets - claude_covered),
            "sensitivity_length_matched":
                round(st.mean(band["dir"]) / st.mean(band["orth"]), 2)
                if band["dir"] and band["orth"] else None,
            "spearman_words_vs_movement":
                _spearman(surface[model]["words"], surface[model]["move"]),
            "spearman_digits_vs_movement":
                _spearman(surface[model]["digits"], surface[model]["move"]),
            "runs_without_june10_packet": unjoinable.get(model, 0),
        }

    arm_surface = {}
    for arm in ARMS:
        texts = [p.get("evidence_text") or "" for p in packets.values()
                 if p["direction"] == arm]
        arm_surface[arm] = {
            "mean_words": round(st.mean([len(t.split()) for t in texts]), 1),
            "mean_digits": round(st.mean([len(re.findall(r"\d", t)) for t in texts]), 1),
            "n": len(texts),
        }

    result = {
        "analysis": "exp1_selectivity_robustness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_source": FROZEN.name,
        "frozen_enumeration_reconciliation": reconciliation,
        "runs_without_june10_packet_total": sum(unjoinable.values()),
        "excluded_models": ["qwen2.5:7b"],
        "frozen_arm_coverage": {
            m: {k: v for k, v in c.items() if not k.startswith("_")}
            for m, c in coverage.items()
        },
        "shared_word_count_band": [lo, hi],
        "packet_surface_by_arm": arm_surface,
        "per_model": per_model,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    write_table(per_model)

    exact = [m for m in MODEL_ORDER if reconciliation[m]["exact_match"]]
    print(f"frozen enumeration vs frozen aggregate: {len(exact)}/{len(MODEL_ORDER)} "
          f"models match exactly")
    for model in MODEL_ORDER:
        rec = reconciliation[model]
        if not rec["exact_match"]:
            print(f"  {LABELS[model]}: {rec['runs_in_aggregate_not_enumerated']} runs "
                  f"counted by the aggregate outside the per-market join")
    print(f"runs not joinable to a June 10 packet: {result['runs_without_june10_packet_total']}")
    print()
    print(f"{'model':18s} {'frozen':>7s} {'repro':>7s} {'diff':>7s} {'AUC':>6s} "
          f"{'markets':>9s} {'len-m':>6s} {'3-arm':>6s}")
    for model in MODEL_ORDER:
        r = per_model[model]
        print(f"{r['label']:18s} {r['frozen_sensitivity_ratio']:7.1f} "
              f"{r['reproduced_sensitivity_ratio']:7.1f} {r['difference_pp']:7.1f} "
              f"{r['auc_within_market']:6.3f} "
              f"{r['markets_directional_exceeds_orthogonal']:4d}/{r['markets_paired']:<4d} "
              f"{r['sensitivity_length_matched']:6.1f} "
              f"{r['sensitivity_all_three_arm_markets']:6.1f}")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
