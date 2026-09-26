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

_DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")

def _norm_tid(tid: str | None) -> str | None:
    """Strip date suffix from task_ids: pm_12345_2026-06-11 → pm_12345."""
    if tid is None:
        return None
    return _DATE_SUFFIX_RE.sub("", tid)


def _mean_se(vals: list) -> tuple:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var / len(vals))


def _norm01(v):
    """Normalise a yes-probability to [0,1].  Some models (e.g. Qwen-7B) emit
    probabilities on a 0–100 scale; divide those by 100 so all forecasts are
    comparable to the Polymarket mid-price."""
    if v is None:
        return None
    return v / 100.0 if v > 1.5 else v


def _spearman(xs: list, ys: list):
    """Spearman rank correlation between two equal-length lists (ties → average
    ranks).  Returns None if fewer than 3 paired points or zero variance."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None

    def _rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    xr = _rank([p[0] for p in pairs])
    yr = _rank([p[1] for p in pairs])
    mx = sum(xr) / n
    my = sum(yr) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    den = (sum((a - mx) ** 2 for a in xr) * sum((b - my) ** 2 for b in yr)) ** 0.5
    return num / den if den else None


def _load_initial_forecasts() -> dict[str, dict]:
    """Returns {model_name: {task_id: record}}."""
    by_model: dict[str, dict] = {}
    model_order: list[str] = []

    for path in sorted(_IF_DIR.glob("forecasts_*.jsonl"), reverse=True):
        mani = path.with_name(path.stem + ".manifest.json")
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
                    tid = _norm_tid(rec.get("task_id"))
                    if tid and tid not in by_model[model_name]:
                        by_model[model_name][tid] = rec
                except json.JSONDecodeError:
                    pass

    return by_model, model_order


def _load_all_counterfactuals() -> dict[str, list[dict]]:
    """Load all counterfactuals once; returns {task_id: [packets sorted by direction]}.
    task_ids are normalized (date suffix stripped) so they match updated-forecast keys.
    """
    _order = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}
    by_task: dict[str, dict] = {}  # norm_task_id -> {cf_id: rec}

    for path in sorted(_CF_DIR.glob("counterfactuals_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec   = json.loads(line)
                    tid   = _norm_tid(rec.get("task_id"))
                    cf_id = rec.get("cf_id", "")
                    if not tid or rec.get("cf_index") is None or not cf_id:
                        continue
                    if tid not in by_task:
                        by_task[tid] = {}
                    if cf_id not in by_task[tid]:
                        by_task[tid][cf_id] = rec
                except json.JSONDecodeError:
                    pass

    result: dict[str, list[dict]] = {}
    for tid, cf_map in by_task.items():
        packets = list(cf_map.values())
        packets.sort(key=lambda r: _order.get(r.get("direction", ""), 99))
        result[tid] = packets
    return result


def _load_all_updated_forecasts() -> dict[tuple, list[dict]]:
    """Load all updated forecasts once; returns {(task_id, model): [records]}.
    task_ids are normalized (date suffix stripped) so they match initial-forecast keys.
    """
    by_key: dict[tuple, dict] = {}  # (norm_task_id, model) -> {update_id: rec}

    for path in sorted(_UF_DIR.glob("updated_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec   = json.loads(line)
                    tid   = _norm_tid(rec.get("task_id"))
                    model = rec.get("forecast_model", "")
                    uid   = rec.get("update_id", "")
                    if not tid or not model or not uid:
                        continue
                    key = (tid, model)
                    if key not in by_key:
                        by_key[key] = {}
                    existing = by_key[key].get(uid)
                    if existing is None:
                        by_key[key][uid] = rec
                    elif (existing.get("delta_yes_prob") is None
                          and rec.get("delta_yes_prob") is not None):
                        # prefer a valid record over an error/parse-error record
                        # (a failed run with a later date should not mask a good earlier run)
                        by_key[key][uid] = rec
                except json.JSONDecodeError:
                    pass

    return {k: list(v.values()) for k, v in by_key.items()}


def _load_counterfactuals(task_id: str) -> list[dict]:
    """Legacy single-task loader (kept for compatibility)."""
    return _load_all_counterfactuals().get(task_id, [])


def _load_updated_forecasts(task_id: str, model: str | None = None) -> list[dict]:
    """Legacy single-task loader (kept for compatibility)."""
    all_uf = _load_all_updated_forecasts()
    if model:
        return all_uf.get((task_id, model), [])
    return [r for (tid, _), recs in all_uf.items() if tid == task_id for r in recs]


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
            hyps   = [h for h in usf.get("hypotheses", []) if isinstance(h, dict)]
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
    """Agent-vs-market relationship, framed as a market-imitation control (not
    calibration: markets are unresolved, so the mid-price is a proxy, not ground
    truth).  Reports two threshold-free quantities over markets:
      * spearman_rho — rank correlation between the agent's yes-probability and
        the Polymarket mid-price (association / "does it track the crowd").
      * mean_abs_divergence — mean |agent - market| in [0,1] (magnitude of
        divergence / "does it copy the price").
    yes_prob is normalised to [0,1] first so models on a 0–100 scale are handled.
    """
    divs = []
    per_market = []
    agent_ps, market_ps = [], []
    for tid, rec in initial_recs.items():
        yp = _norm01(rec.get("yes_prob"))
        mp = rec.get("yes_price_market")
        if yp is not None and mp is not None:
            d = abs(yp - mp)
            divs.append(d)
            agent_ps.append(yp)
            market_ps.append(mp)
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
    rho = _spearman(agent_ps, market_ps)
    return {
        "spearman_rho":        round(rho, 3) if rho is not None else None,
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

    # Load all updated forecasts and counterfactuals once (avoids re-scanning files per market)
    print("  Loading updated forecasts index …")
    uf_index  = _load_all_updated_forecasts()   # {(task_id, model): [recs]}
    print("  Loading counterfactuals index …")
    cf_index  = _load_all_counterfactuals()      # {task_id: [packets]}

    all_uf_by_model: dict[str, list[dict]] = {}

    for model in model_order:
        model_ifs = if_by_model.get(model, {})
        task_ids  = sorted(model_ifs.keys())

        per_market_rows = []
        all_uf_for_model: list[dict] = []

        for tid in task_ids:
            rec        = model_ifs[tid]
            uf_records = uf_index.get((tid, model), [])
            cf_packets = cf_index.get(tid, [])
            cf_lookup  = {p["cf_id"]: p for p in cf_packets}

            all_uf_for_model.extend(uf_records)

            cons_result = None
            if uf_records:
                cons_result = _compute_consistency(uf_records, cf_lookup)

            summary = (cons_result or {}).get("summary") or {}
            yp  = _norm01(rec.get("yes_prob"))
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
                "spearman_rho":        divergence.get("spearman_rho"),
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
            "| Metric | Rate | SE |",
            "|--------|------|----|",
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
            "**Market association** (Spearman ρ, agent yes-prob vs. mid-price):  ",
            f"`{_fmt(dv.get('spearman_rho'))}`",
            "",
            "**Market-price divergence** (mean |agent − market|, normalised):  ",
            f"`{_fmt(dv['mean_abs_divergence'], 4)}`",
            "",
            "**Anchoring check** — mean |Δyes_prob| by CF direction:",
            "",
            "| Direction | mean |Δ| | SE | n |",
            "|-----------|----------|----|---|",
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
        dv = report["per_model"][model]["market_divergence"]
        sens = a.get("sensitivity_ratio")
        print(f"  {model}:")
        print(f"    EHC={_pct(c['EHC_rate'])}  HFC={_pct(c['HFC_rate'])}  ICS={_pct(c['ICS_rate'])}")
        print(f"    Sensitivity ratio: {_fmt(sens)}")
        print(f"    Market: rho={_fmt(dv.get('spearman_rho'))}  "
              f"|A-M|={_fmt(dv.get('mean_abs_divergence'), 4)}")


if __name__ == "__main__":
    main()
