#!/usr/bin/env python3
"""Standalone consistency evaluation script (Step 5).

Reads all updated forecasts, counterfactuals, and initial forecasts, then:
  1. Computes EHC / HFC / ICS per market per model (replicating viewer logic).
  2. Computes anchoring baseline (Step 6a): correlation between CF direction and |Δyes_prob|.
  3. Computes market-price divergence baseline (Step 6b): |agent_yes_prob - market_price|.
  4. Writes:
       data/results/consistency_report_{date}.json  — full machine-readable data
       data/results/summary_{date}.md               — human-readable Markdown

Usage (from exp1_prospective/):
    python agent/evaluate_consistency.py
    python agent/evaluate_consistency.py --date 2026-06-07
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent.parent
_IF_DIR  = _ROOT / "data" / "initial_forecasts"
_CF_DIR  = _ROOT / "data" / "counterfactuals"
_UF_DIR  = _ROOT / "data" / "updated_forecasts"
_OUT_DIR = _ROOT / "data" / "results"


# ── helpers (mirrors viewer/app.py) ───────────────────────────────────────────

def _mean_se(vals: list) -> tuple:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var / len(vals))


def _load_initial_forecasts() -> dict[str, dict]:
    """Returns {model_name: {task_id: record}}."""
    by_model: dict[str, dict] = {}
    model_order: list[str] = []

    for path in sorted(_IF_DIR.glob("forecasts_*.jsonl"), reverse=True):
        mani = path.with_suffix("").with_suffix(".manifest.json")
        model_name = None
        if mani.exists():
            try:
                m = json.loads(mani.read_text())
                model_name = m.get("model") or None
            except Exception:
                pass
        if not model_name:
            fm = re.match(r"forecasts_(.+)_\d{4}-\d{2}-\d{2}$", path.stem)
            if fm:
                model_name = fm.group(1)
            else:
                continue

        if model_name not in by_model:
            by_model[model_name] = {}
            model_order.append(model_name)

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    tid = rec.get("task_id")
                    if tid and tid not in by_model[model_name]:
                        by_model[model_name][tid] = rec
                except json.JSONDecodeError:
                    pass

    return by_model, model_order


def _load_counterfactuals(task_id: str) -> list[dict]:
    packets: list[dict] = []
    seen: set[str] = set()
    _order = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}

    for path in sorted(_CF_DIR.glob("counterfactuals_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("task_id") != task_id:
                        continue
                    if rec.get("cf_index") is None:
                        continue
                    cf_id = rec.get("cf_id", "")
                    if cf_id and cf_id not in seen:
                        seen.add(cf_id)
                        packets.append(rec)
                except json.JSONDecodeError:
                    pass

    packets.sort(key=lambda r: _order.get(r.get("direction", ""), 99))
    return packets


def _load_updated_forecasts(task_id: str, model: str | None = None) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()

    for path in sorted(_UF_DIR.glob("updated_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("task_id") != task_id:
                        continue
                    if model and rec.get("forecast_model") != model:
                        continue
                    uid = rec.get("update_id", "")
                    if uid and uid not in seen:
                        seen.add(uid)
                        records.append(rec)
                except json.JSONDecodeError:
                    pass

    return records


def _compute_consistency(uf_records: list[dict], cf_lookup: dict) -> dict:
    """Exact replica of viewer's _compute_consistency."""

    by_cf: dict[str, list] = {}
    for r in uf_records:
        by_cf.setdefault(r.get("cf_id", ""), []).append(r)

    dir_order = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}

    cf_results = []
    for cf_id in sorted(by_cf, key=lambda k: (
        dir_order.get((by_cf[k][0].get("direction") or ""), 9),
        by_cf[k][0].get("cf_index") or 0,
    )):
        runs_raw  = sorted(by_cf[cf_id], key=lambda r: r.get("initial_run_id") or 0)
        cf_info   = cf_lookup.get(cf_id, {})
        direction = (runs_raw[0].get("direction") or "")

        run_data = []
        for r in runs_raw:
            initial_yp = r.get("initial_yes_prob")
            updated_yp = r.get("updated_yes_prob")
            delta_yp   = r.get("delta_yes_prob")

            usf    = r.get("updated_structured_forecast") or {}
            hyps   = usf.get("hypotheses", [])
            h1_upd = next((h for h in hyps if h.get("id") == "H1"), {})
            h1_post = h1_upd.get("posterior_probability")

            ehc = None
            if direction in ("pro_H1", "anti_H1") and h1_post is not None and initial_yp is not None:
                h1_delta = h1_post - initial_yp
                if abs(h1_delta) >= 0.03:
                    ehc = 1 if (
                        (direction == "pro_H1" and h1_delta > 0) or
                        (direction == "anti_H1" and h1_delta < 0)
                    ) else 0

            hfc = None
            if direction in ("pro_H1", "anti_H1") and delta_yp is not None:
                if abs(delta_yp) >= 0.03:
                    hfc = 1 if (
                        (direction == "pro_H1" and delta_yp > 0) or
                        (direction == "anti_H1" and delta_yp < 0)
                    ) else 0

            ics, ics_dev = None, None
            if updated_yp is not None and h1_post is not None:
                ics_dev = abs(updated_yp - h1_post)
                ics = 1 if ics_dev < 0.02 else 0

            run_data.append({
                "run_id":               r.get("initial_run_id"),
                "initial_yes_prob":     initial_yp,
                "updated_yes_prob":     updated_yp,
                "delta_yes_prob":       delta_yp,
                "h1_posterior_updated": h1_post,
                "EHC": ehc,
                "HFC": hfc,
                "ICS": ics,
                "ics_deviation": round(ics_dev, 4) if ics_dev is not None else None,
                "parse_error": r.get("parse_error"),
            })

        ehc_vals   = [r["EHC"] for r in run_data if r["EHC"] is not None]
        hfc_vals   = [r["HFC"] for r in run_data if r["HFC"] is not None]
        ics_vals   = [r["ICS"] for r in run_data if r["ICS"] is not None]
        delta_vals = [r["delta_yes_prob"] for r in run_data if r["delta_yes_prob"] is not None]

        ehc_rate, _ = _mean_se(ehc_vals)
        hfc_rate, _ = _mean_se(hfc_vals)
        ics_rate, _ = _mean_se(ics_vals)
        mean_delta  = sum(delta_vals) / len(delta_vals) if delta_vals else None

        slot_raw   = cf_info.get("slot_type") or ""
        slot_label = slot_raw.split("—")[0].strip() if "—" in slot_raw else slot_raw.split(" ")[0].strip()

        cf_results.append({
            "cf_id":              cf_id,
            "direction":          direction,
            "cf_index":           cf_info.get("cf_index"),
            "evidence_headline":  cf_info.get("evidence_headline", ""),
            "mechanism_targeted": cf_info.get("mechanism_targeted", ""),
            "slot_type":          slot_label,
            "runs":               run_data,
            "EHC_rate":   round(ehc_rate, 3) if ehc_rate is not None else None,
            "HFC_rate":   round(hfc_rate, 3) if hfc_rate is not None else None,
            "ICS_rate":   round(ics_rate, 3) if ics_rate is not None else None,
            "mean_delta": round(mean_delta, 4) if mean_delta is not None else None,
            "n_runs":     len(run_data),
        })

    all_ehc = [r["EHC"] for cf in cf_results for r in cf["runs"] if r["EHC"] is not None]
    all_hfc = [r["HFC"] for cf in cf_results for r in cf["runs"] if r["HFC"] is not None]
    all_ics = [r["ICS"] for cf in cf_results for r in cf["runs"] if r["ICS"] is not None]

    by_dir = {}
    for d in ("pro_H1", "anti_H1", "orthogonal"):
        d_cfs = [cf for cf in cf_results if cf["direction"] == d]
        d_ehc = [r["EHC"] for cf in d_cfs for r in cf["runs"] if r["EHC"] is not None]
        d_hfc = [r["HFC"] for cf in d_cfs for r in cf["runs"] if r["HFC"] is not None]
        d_ics = [r["ICS"] for cf in d_cfs for r in cf["runs"] if r["ICS"] is not None]
        er, es = _mean_se(d_ehc)
        hr, hs = _mean_se(d_hfc)
        ir, isr = _mean_se(d_ics)
        by_dir[d] = {
            "EHC_rate": round(er,  3) if er  is not None else None,
            "EHC_se":   round(es,  3) if es  is not None else None,
            "HFC_rate": round(hr,  3) if hr  is not None else None,
            "HFC_se":   round(hs,  3) if hs  is not None else None,
            "ICS_rate": round(ir,  3) if ir  is not None else None,
            "ICS_se":   round(isr, 3) if isr is not None else None,
            "n_EHC": len(d_ehc),
            "n_HFC": len(d_hfc),
            "n_ICS": len(d_ics),
        }

    gr, gs   = _mean_se(all_ehc)
    hr2, hs2 = _mean_se(all_hfc)
    ir2, is2 = _mean_se(all_ics)

    return {
        "task_id":   (uf_records[0].get("task_id") if uf_records else None),
        "n_updates": len(uf_records),
        "cf_results": cf_results,
        "summary": {
            "EHC_rate": round(gr,  3) if gr  is not None else None,
            "EHC_se":   round(gs,  3) if gs  is not None else None,
            "HFC_rate": round(hr2, 3) if hr2 is not None else None,
            "HFC_se":   round(hs2, 3) if hs2 is not None else None,
            "ICS_rate": round(ir2, 3) if ir2 is not None else None,
            "ICS_se":   round(is2, 3) if is2 is not None else None,
            "n_EHC": len(all_ehc),
            "n_HFC": len(all_hfc),
            "n_ICS": len(all_ics),
            "by_direction": by_dir,
        },
    }


# ── Step 6a: anchoring check ───────────────────────────────────────────────────

def _anchoring_check(all_uf_records: list[dict]) -> dict:
    """Anchoring baseline: does CF direction predict |Δyes_prob|?

    A well-calibrated agent should show larger |Δyes_prob| for pro_H1/anti_H1
    CFs than for orthogonal ones.  Low sensitivity (|Δ| near zero regardless of
    direction) = anchoring to initial estimate.

    Returns:
      mean |Δyes_prob| by direction, and a simple sensitivity ratio:
        sensitivity = mean_|Δ|_{pro+anti} / mean_|Δ|_{orthogonal}   (>1 = good)
    """
    by_dir: dict[str, list[float]] = {"pro_H1": [], "anti_H1": [], "orthogonal": []}

    for r in all_uf_records:
        d   = r.get("direction", "")
        dyp = r.get("delta_yes_prob")
        if d in by_dir and dyp is not None:
            by_dir[d].append(abs(dyp))

    result: dict = {}
    all_directed = by_dir["pro_H1"] + by_dir["anti_H1"]
    ortho        = by_dir["orthogonal"]

    for d, vals in by_dir.items():
        m, se = _mean_se(vals)
        result[d] = {
            "mean_abs_delta": round(m, 4) if m is not None else None,
            "se":             round(se, 4) if se is not None else None,
            "n": len(vals),
        }

    m_dir, _  = _mean_se(all_directed)
    m_ort, _  = _mean_se(ortho)
    sensitivity = (round(m_dir / m_ort, 3)
                   if (m_dir is not None and m_ort is not None and m_ort > 0)
                   else None)

    result["sensitivity_ratio"] = sensitivity
    result["interpretation"] = (
        "sensitivity_ratio > 1 means the agent moves more for directional CFs "
        "than orthogonal ones (good). <1 means it moves equally regardless (anchoring)."
    )
    return result


# ── Step 6b: market-price divergence ──────────────────────────────────────────

def _market_divergence(initial_recs: dict[str, dict]) -> dict:
    """Per-market: |agent_yes_prob - market_price|."""
    divs = []
    per_market = []
    for tid, rec in initial_recs.items():
        yp = rec.get("yes_prob")
        mp = rec.get("yes_price_market")
        if yp is not None and mp is not None:
            d = abs(yp - mp)
            divs.append(d)
            per_market.append({
                "task_id":         tid,
                "question":        rec.get("question", "")[:80],
                "yes_prob":        round(yp, 3),
                "market_price":    round(mp, 3),
                "abs_divergence":  round(d, 3),
                "direction":       "above_market" if yp > mp else "below_market",
            })
    per_market.sort(key=lambda r: -r["abs_divergence"])
    mean_div, se_div = _mean_se(divs)
    return {
        "mean_abs_divergence": round(mean_div, 4) if mean_div is not None else None,
        "se":                  round(se_div,  4) if se_div  is not None else None,
        "n": len(divs),
        "per_market": per_market,
    }


# ── report assembly ────────────────────────────────────────────────────────────

def _fmt(val, digits=3) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


def _pct(val) -> str:
    if val is None:
        return "—"
    return f"{val*100:.1f}%"


def build_report(date_str: str) -> tuple[dict, str]:
    if_by_model, model_order = _load_initial_forecasts()

    if not model_order:
        print("ERROR: no initial forecast files found in data/initial_forecasts/", file=sys.stderr)
        sys.exit(1)

    # discover all task_ids across models
    all_task_ids: set[str] = set()
    for recs in if_by_model.values():
        all_task_ids.update(recs.keys())

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date":         date_str,
        "models":       model_order,
        "n_markets":    len(all_task_ids),
        "per_model":    {},
    }

    all_uf_by_model: dict[str, list[dict]] = {}

    for model in model_order:
        model_ifs = if_by_model.get(model, {})
        task_ids  = sorted(model_ifs.keys())

        per_market_rows = []
        all_uf_for_model: list[dict] = []

        for tid in task_ids:
            rec        = model_ifs[tid]
            uf_records = _load_updated_forecasts(tid, model)
            cf_packets = _load_counterfactuals(tid)
            cf_lookup  = {p["cf_id"]: p for p in cf_packets}

            all_uf_for_model.extend(uf_records)

            cons_result = None
            if uf_records:
                cons_result = _compute_consistency(uf_records, cf_lookup)

            summary = (cons_result or {}).get("summary") or {}
            yp  = rec.get("yes_prob")
            mp  = rec.get("yes_price_market")
            per_market_rows.append({
                "task_id":            tid,
                "question":           rec.get("question", "")[:100],
                "category":           rec.get("category", ""),
                "yes_prob":           round(yp, 3) if yp is not None else None,
                "market_price":       round(mp, 3) if mp is not None else None,
                "abs_divergence":     round(abs(yp - mp), 3) if (yp is not None and mp is not None) else None,
                "n_updates":          len(uf_records),
                "n_cfs":              len(cf_packets),
                "EHC_rate":           summary.get("EHC_rate"),
                "HFC_rate":           summary.get("HFC_rate"),
                "ICS_rate":           summary.get("ICS_rate"),
                "n_EHC":              summary.get("n_EHC"),
                "n_HFC":              summary.get("n_HFC"),
                "n_ICS":              summary.get("n_ICS"),
                "by_direction":       summary.get("by_direction", {}),
                "cf_results":         (cons_result or {}).get("cf_results", []),
            })

        all_uf_by_model[model] = all_uf_for_model

        # aggregate
        ehc_r, ehc_se = _mean_se([m["EHC_rate"] for m in per_market_rows])
        hfc_r, hfc_se = _mean_se([m["HFC_rate"] for m in per_market_rows])
        ics_r, ics_se = _mean_se([m["ICS_rate"] for m in per_market_rows])
        div_r, div_se = _mean_se([m["abs_divergence"] for m in per_market_rows])

        anchoring = _anchoring_check(all_uf_for_model)
        divergence = _market_divergence(model_ifs)

        report["per_model"][model] = {
            "n_markets":   len(task_ids),
            "n_with_updates": sum(1 for m in per_market_rows if m["n_updates"] > 0),
            "consistency": {
                "EHC_rate": round(ehc_r,  3) if ehc_r  is not None else None,
                "EHC_se":   round(ehc_se, 3) if ehc_se is not None else None,
                "HFC_rate": round(hfc_r,  3) if hfc_r  is not None else None,
                "HFC_se":   round(hfc_se, 3) if hfc_se is not None else None,
                "ICS_rate": round(ics_r,  3) if ics_r  is not None else None,
                "ICS_se":   round(ics_se, 3) if ics_se is not None else None,
            },
            "market_divergence": {
                "mean_abs_divergence": round(div_r,  4) if div_r  is not None else None,
                "se":                  round(div_se, 4) if div_se is not None else None,
            },
            "anchoring":   anchoring,
            "divergence":  divergence,
            "per_market":  per_market_rows,
        }

    # ── Markdown summary ───────────────────────────────────────────────────────
    lines = [
        f"# Consistency Report — {date_str}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"Markets: {len(all_task_ids)}  |  Models: {', '.join(model_order)}",
        "",
    ]

    for model in model_order:
        md = report["per_model"][model]
        c  = md["consistency"]
        a  = md["anchoring"]
        dv = md["market_divergence"]

        lines += [
            f"## {model}",
            "",
            "### Consistency Metrics (averaged across markets)",
            "",
            f"| Metric | Rate | SE |",
            f"|--------|------|----|",
            f"| EHC (Evidence-Hypothesis) | {_pct(c['EHC_rate'])} | ±{_fmt(c['EHC_se'])} |",
            f"| HFC (Hypothesis-Forecast)  | {_pct(c['HFC_rate'])} | ±{_fmt(c['HFC_se'])} |",
            f"| ICS (Internal Coherence)   | {_pct(c['ICS_rate'])} | ±{_fmt(c['ICS_se'])} |",
            "",
            "### Per-Market Breakdown",
            "",
            "| Market | EHC | HFC | ICS | |Agent−Mkt| |",
            "|--------|-----|-----|-----|-----------:|",
        ]

        for pm in md["per_market"]:
            q_short = pm["question"][:55] + ("…" if len(pm["question"]) > 55 else "")
            lines.append(
                f"| {q_short} | {_pct(pm['EHC_rate'])} | {_pct(pm['HFC_rate'])} "
                f"| {_pct(pm['ICS_rate'])} | {_fmt(pm['abs_divergence'])} |"
            )

        lines += [
            "",
            "### Baseline Checks",
            "",
            "**Market-price divergence** (mean |agent − market|):  ",
            f"`{_fmt(dv['mean_abs_divergence'], 4)}`",
            "",
            "**Anchoring check** — mean |Δyes_prob| by CF direction:",
            "",
            f"| Direction | mean |Δ| | SE | n |",
            f"|-----------|----------|----|---|",
        ]
        for d in ("pro_H1", "anti_H1", "orthogonal"):
            row = a.get(d, {})
            lines.append(
                f"| {d} | {_fmt(row.get('mean_abs_delta'), 4)} "
                f"| ±{_fmt(row.get('se'), 4)} | {row.get('n', 0)} |"
            )

        sens = a.get("sensitivity_ratio")
        lines += [
            "",
            f"**Sensitivity ratio** (directed/orthogonal): **{_fmt(sens)}**  ",
            ("> 1 = responds more to directional CFs than orthogonal (healthy signal)"
             if (sens is not None and sens > 1)
             else "> ≤ 1 suggests possible anchoring to initial estimate"),
            "",
        ]

    md_text = "\n".join(lines)
    return report, md_text


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate consistency + baselines (Step 5/6)")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    help="Date label for output files (default: today)")
    ap.add_argument("--out-dir", default=str(_OUT_DIR),
                    help="Output directory (default: data/results/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building consistency report for {args.date} …")
    report, md_text = build_report(args.date)

    json_path = out_dir / f"consistency_report_{args.date}.json"
    md_path   = out_dir / f"summary_{args.date}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(md_text)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    # Quick summary to stdout
    print()
    for model in report["models"]:
        c = report["per_model"][model]["consistency"]
        a = report["per_model"][model]["anchoring"]
        sens = a.get("sensitivity_ratio")
        print(f"  {model}:")
        print(f"    EHC={_pct(c['EHC_rate'])}  HFC={_pct(c['HFC_rate'])}  ICS={_pct(c['ICS_rate'])}")
        print(f"    Sensitivity ratio: {_fmt(sens)}")


if __name__ == "__main__":
    main()
