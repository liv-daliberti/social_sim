#!/usr/bin/env python3
"""Regenerate the published Experiment 1 numbers from the frozen report and check
them against the manuscript.

The June 17 aggregate is the manuscript's numerical authority. It stores per-model
summary statistics and enumerates every scored run, so most published Experiment 1
values can be re-derived from that one file and compared against what the paper
actually prints. This script does that comparison and fails if any published value
has drifted from the frozen data.

What is checked, all sourced from the frozen report alone:

  Table \\ref{tab:model-summary}      valid records and represented markets per model
  Table \\ref{tab:anchoring}          mean absolute revision by arm, sensitivity, n
  Figure \\ref{fig:exp1-table}        sensitivity and directional/orthogonal means
  Table \\ref{tab:exp1-arm-coverage}  per-arm market coverage

What is deliberately not checked here, and why:

  Figure \\ref{fig:exp1-table}, EHC column
      the figure's EHC is the 3-percentage-point threshold statistic, which comes
      from threshold_robustness.json rather than from the frozen aggregate's own
      EHC field; the two differ by up to .015. It is checked here against that
      artifact instead, and flagged as record-derived.
  Table \\ref{tab:exp1-threshold-robustness}
      evaluate_threshold_robustness.py re-reads the row-level update directory
      and applies its own scale-normalisation and validity filters, so its
      denominators (for example 2,119 valid directional Claude records) do not
      correspond to the frozen aggregate's arm counts. It is reproducible from
      the record directory, not from the freeze.
  Stage 3 human materials review
      a separate frozen response export with its own generator.
  Post-freeze settlement audit
      a dated market snapshot, descriptive and outside the frozen aggregate.

Exit status is non-zero if any checked value disagrees.

Outputs:
  data/results/freeze_reproduction.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
FROZEN = ROOT / "data" / "results" / "consistency_report_2026-06-17.json"
THRESHOLD = ROOT / "data" / "results" / "threshold_robustness.json"
APPENDIX = REPO_ROOT / "paper" / "experiment1_appendix.tex"
SECTION = REPO_ROOT / "paper" / "experiment1_section.tex"
OUT_JSON = ROOT / "data" / "results" / "freeze_reproduction.json"

EXPECTED_SHA256 = "b041e7e54f7b75f476e7dabe87823d162b0af72c07ca7cccadc4b2fc005ebfb5"

ARMS = ("pro_H1", "anti_H1", "orthogonal")
LABEL_TO_KEY = {
    "Claude Opus~4.8": "claude-opus-4-8",
    "GPT-5.4": "gpt-5.4",
    "DeepSeek V4-Pro": "DeepSeek-V4-Pro",
    "Qwen~2.5-72B": "qwen2.5:72b",
    "Llama~3.3-70B": "llama3.3:70b",
    "Llama~3.1-70B": "llama3.1:70b",
    "Qwen~2.5-32B": "qwen2.5:32b",
    "Qwen~2.5-14B": "qwen2.5:14b",
    "Llama~3.1-8B": "llama3.1:8b",
    "Qwen~2.5-7B": "qwen2.5:7b",
}
# Figure 5(b) abbreviates the model names.
FIG_LABEL_TO_KEY = {
    "Opus~4.8": "claude-opus-4-8", "GPT-5.4": "gpt-5.4", "V4-Pro": "DeepSeek-V4-Pro",
    "2.5-72B": "qwen2.5:72b", "3.3-70B": "llama3.3:70b", "3.1-70B": "llama3.1:70b",
    "2.5-32B": "qwen2.5:32b", "2.5-14B": "qwen2.5:14b", "3.1-8B": "llama3.1:8b",
}


def _num(text: str) -> float:
    return float(text.replace("{,}", "").replace(",", "").strip())


def _half_up(value: float, places: int) -> float:
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(value)).quantize(Decimal(1).scaleb(-places),
                                               rounding=ROUND_HALF_UP))


def derive(report: dict) -> dict:
    """Every quantity the manuscript prints, recomputed from the frozen report."""
    out = {}
    for key, block in report["per_model"].items():
        anchoring = block["anchoring"]
        n_by_arm = {a: anchoring[a]["n"] for a in ARMS}
        means_pp = {a: (anchoring[a]["mean_abs_delta"] or 0) * 100 for a in ARMS}
        directional_n = n_by_arm["pro_H1"] + n_by_arm["anti_H1"]
        directional_mean = (
            (anchoring["pro_H1"]["mean_abs_delta"] * n_by_arm["pro_H1"]
             + anchoring["anti_H1"]["mean_abs_delta"] * n_by_arm["anti_H1"])
            / directional_n * 100) if directional_n else None

        def scored(entry, arm):
            b = (entry.get("by_direction") or {}).get(arm) or {}
            return (b.get("n_ICS") or 0) > 0

        per_market = block["per_market"]

        out[key] = {
            "records_total": sum(n_by_arm.values()),
            "records_by_arm": n_by_arm,
            # Markets with at least one scored update, which is what the
            # manuscript's market column reports. block["n_markets"] counts the
            # raw per-market list, which for three Qwen deployments carries 20
            # extra markets that never produced a scored update.
            "markets": sum(1 for e in per_market
                           if any(scored(e, a) for a in ARMS)),
            "mean_abs_delta_pp": means_pp,
            "directional_mean_pp": directional_mean,
            "sensitivity_ratio": anchoring["sensitivity_ratio"],
            "ehc_rate": block["consistency"]["EHC_rate"],
            "coverage": {
                "markets_in_report": len(per_market),
                "all_three_arms": sum(1 for e in per_market
                                      if all(scored(e, a) for a in ARMS)),
                "orthogonal": sum(1 for e in per_market if scored(e, "orthogonal")),
                "directional": sum(1 for e in per_market
                                   if scored(e, "pro_H1") or scored(e, "anti_H1")),
            },
        }
    return out


def parse_model_summary(tex: str) -> dict:
    body = tex.split(r"\label{tab:model-summary}", 1)[1].split(r"\end{tabular}", 1)[0]
    rows = {}
    for line in body.splitlines():
        m = re.match(r"\s*([A-Za-z0-9~\.\- ]+?)\s*&\s*(?:Hosted|Local[^&]*)&\s*"
                     r"([0-9{},]+)\s*&\s*([0-9]+)\s*\\\\", line)
        if m and m.group(1).strip() in LABEL_TO_KEY:
            rows[LABEL_TO_KEY[m.group(1).strip()]] = {
                "records": int(_num(m.group(2))), "markets": int(_num(m.group(3)))}
    return rows


def parse_anchoring(tex: str) -> dict:
    body = tex.split(r"\label{tab:anchoring}", 1)[1].split(r"\end{tabular}", 1)[0]
    rows = {}
    for line in body.splitlines():
        m = re.match(r"\s*([A-Za-z0-9~\.\- ]+?)\s*&\s*(?:Hosted|Local[^&]*)&\s*"
                     r"([0-9\.]+)\\,pp\s*&\s*([0-9\.]+)\\,pp\s*&\s*([0-9\.]+)\\,pp\s*&\s*"
                     r"\$([0-9\.]+)\\times\$\s*&\s*([0-9{},]+)\s*\\\\", line)
        if m and m.group(1).strip() in LABEL_TO_KEY:
            rows[LABEL_TO_KEY[m.group(1).strip()]] = {
                "pro_pp": float(m.group(2)), "anti_pp": float(m.group(3)),
                "orth_pp": float(m.group(4)), "sensitivity": float(m.group(5)),
                "n": int(_num(m.group(6)))}
    return rows


def parse_figure_table(tex: str) -> dict:
    rows = {}
    for line in tex.splitlines():
        m = re.search(r"\\micon\{[^}]+\}~([A-Za-z0-9~\.\-]+)\s*&[^&]*&[^$]*\$([0-9\.]+)"
                      r"\\times\$\s*&[^$]*\$([0-9\.]+)/([0-9\.]+)\$\s*&[^0-9]*\.([0-9]+)",
                      line)
        if m and m.group(1) in FIG_LABEL_TO_KEY:
            rows[FIG_LABEL_TO_KEY[m.group(1)]] = {
                "sensitivity": float(m.group(2)), "directional_pp": float(m.group(3)),
                "orthogonal_pp": float(m.group(4)), "ehc": float("0." + m.group(5))}
    return rows


def parse_arm_coverage(tex: str) -> dict:
    body = tex.split(r"\label{tab:exp1-arm-coverage}", 1)[1].split(r"\end{tabular}", 1)[0]
    rows = {}
    for line in body.splitlines():
        m = re.match(r"\s*([A-Za-z0-9~\.\- ]+?)\s*&\s*([0-9]+)\s*&\s*([0-9]+)\s*&\s*"
                     r"([0-9]+)\s*&\s*([0-9]+)\s*\\\\", line)
        if m and m.group(1).strip() in LABEL_TO_KEY:
            rows[LABEL_TO_KEY[m.group(1).strip()]] = {
                "markets_in_report": int(m.group(2)), "directional": int(m.group(3)),
                "orthogonal": int(m.group(4)), "all_three_arms": int(m.group(5))}
    return rows


def main() -> None:
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    report = json.loads(FROZEN.read_text())
    appendix, section = APPENDIX.read_text(), SECTION.read_text()
    threshold = json.loads(THRESHOLD.read_text())
    derived = derive(report)

    checks: list[dict] = []

    def check(table, model, field, published, expected, places=None):
        """Counts must match exactly; displayed decimals must agree to within half
        a display unit, so a rounding convention is not mistaken for data drift."""
        if places is None:
            ok, got = published == expected, expected
        else:
            got = _half_up(expected, places)
            ok = abs(published - expected) <= 0.5 * 10 ** (-places) + 1e-9
        checks.append({"table": table, "model": model, "field": field,
                       "published": published, "recomputed": round(expected, 4),
                       "rounded": got, "pass": ok})

    for key, row in parse_model_summary(appendix).items():
        check("tab:model-summary", key, "records", row["records"], derived[key]["records_total"])
        check("tab:model-summary", key, "markets", row["markets"], derived[key]["markets"])

    for key, row in parse_anchoring(appendix).items():
        d = derived[key]
        if d["records_total"] == 0 or d["sensitivity_ratio"] is None:
            continue
        check("tab:anchoring", key, "pro pp", row["pro_pp"], d["mean_abs_delta_pp"]["pro_H1"], 1)
        check("tab:anchoring", key, "anti pp", row["anti_pp"], d["mean_abs_delta_pp"]["anti_H1"], 1)
        check("tab:anchoring", key, "orth pp", row["orth_pp"], d["mean_abs_delta_pp"]["orthogonal"], 1)
        check("tab:anchoring", key, "sensitivity", row["sensitivity"], d["sensitivity_ratio"], 1)
        check("tab:anchoring", key, "n", row["n"], d["records_total"])

    for key, row in parse_figure_table(section).items():
        d = derived[key]
        check("fig:exp1-table", key, "sensitivity", row["sensitivity"], d["sensitivity_ratio"], 1)
        check("fig:exp1-table", key, "directional pp", row["directional_pp"], d["directional_mean_pp"], 1)
        check("fig:exp1-table", key, "orthogonal pp", row["orthogonal_pp"], d["mean_abs_delta_pp"]["orthogonal"], 1)
        # Record-derived, not freeze-derived: see the module docstring.
        t3 = (threshold["per_model"][key]["EHC"]["thresholds"]["3"]
              ["market_macro_directional_correctness"])
        check("fig:exp1-table (record-derived)", key, "EHC at 3pp", row["ehc"], t3, 3)

    for key, row in parse_arm_coverage(appendix).items():
        c = derived[key]["coverage"]
        for field, name in (("markets_in_report", "markets"), ("directional", "directional"),
                            ("orthogonal", "orthogonal"), ("all_three_arms", "all three arms")):
            check("tab:exp1-arm-coverage", key, name, row[field], c[field])

    failures = [c for c in checks if not c["pass"]]
    result = {
        "analysis": "exp1_freeze_reproduction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_report": FROZEN.name,
        "frozen_report_sha256": digest,
        "checksum_matches_pinned": digest == EXPECTED_SHA256,
        "checks_run": len(checks),
        "checks_failed": len(failures),
        "failures": failures,
        "not_reproducible_from_freeze": {
            "tab:exp1-threshold-robustness":
                "evaluate_threshold_robustness.py re-reads the row-level update "
                "directory and applies its own scale normalisation and validity "
                "filters; its denominators do not correspond to the frozen "
                "aggregate's arm counts.",
            "stage3 human materials review":
                "separate frozen response export with its own generator.",
            "post-freeze settlement audit":
                "dated market snapshot, descriptive and outside the frozen aggregate.",
        },
        "checks": checks,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    print(f"frozen report sha256 {digest}")
    print(f"  matches pinned value: {digest == EXPECTED_SHA256}")
    by_table: dict[str, list] = {}
    for c in checks:
        by_table.setdefault(c["table"], []).append(c)
    for table, cs in by_table.items():
        bad = [c for c in cs if not c["pass"]]
        print(f"  {table:28s} {len(cs) - len(bad):3d}/{len(cs):3d} values reproduce"
              + ("" if not bad else "   <-- MISMATCH"))
        for c in bad:
            print(f"      {c['model']:18s} {c['field']:16s} "
                  f"paper={c['published']} recomputed={c['recomputed']} "
                  f"(rounds to {c['rounded']})")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} published values reproduce "
          f"from {FROZEN.name}")
    print(f"wrote {OUT_JSON}")
    sys.exit(1 if failures or digest != EXPECTED_SHA256 else 0)


if __name__ == "__main__":
    main()
