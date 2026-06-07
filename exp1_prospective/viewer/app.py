#!/usr/bin/env python3
"""Flask viewer for Exp 1 forecast conversations.

Usage (from exp1_prospective/):
    python viewer/app.py
    python viewer/app.py --port 5050
"""

import argparse
import json
import math
import re
from pathlib import Path

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

_ROOT           = Path(__file__).resolve().parent.parent
_FORECAST_DIR   = _ROOT / "data" / "initial_forecasts"
_MARKETS_DIR    = _ROOT / "data" / "selected_markets"
_CF_DIR         = _ROOT / "data" / "counterfactuals"
_UF_DIR         = _ROOT / "data" / "updated_forecasts"


# ── market price history (loaded once at startup) ──────────────────────────────

def _load_market_histories() -> dict[str, dict]:
    """Return {market_id: {price_history: [...], volume_history: [...]}}."""
    result: dict[str, dict] = {}
    files = sorted(_MARKETS_DIR.glob("diverse_*.jsonl"), reverse=True)
    if not files:
        return result
    with open(files[0]) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                m = json.loads(line)
                mid = m.get("market_id", "")
                if mid:
                    result[mid] = {
                        "price_history":  m.get("price_history", []),
                        "volume_history": m.get("volume_history", []),
                        "open_time":      m.get("open_time"),
                        "end_time":       m.get("end_time"),
                    }
            except json.JSONDecodeError:
                pass
    return result


_market_histories: dict[str, dict] = _load_market_histories()


# ── forecast loading ────────────────────────────────────────────────────────────

def _load_all_forecasts():
    """Load all forecast records grouped by model.

    Returns:
        models_data: {model_name: [records sorted by yes_prob desc]}
        model_order: [model names in discovery order, newest-run first]
    """
    by_model = {}       # model_name -> {task_id -> record}
    model_order = []

    for path in sorted(_FORECAST_DIR.glob("forecasts_*.jsonl"), reverse=True):
        mani = path.with_suffix("").with_suffix(".manifest.json")
        # Try manifest first; fall back to filename slug for in-progress runs
        model_name = None
        if mani.exists():
            try:
                m = json.loads(mani.read_text())
                model_name = m.get("model") or None
            except Exception:
                pass
        if not model_name:
            # forecasts_{slug}_{YYYY-MM-DD}.jsonl  →  slug is the model name
            fm = re.match(r"forecasts_(.+)_\d{4}-\d{2}-\d{2}$", path.stem)
            if fm:
                model_name = fm.group(1)
            else:
                # Plain-date filename (forecasts_YYYY-MM-DD.jsonl) with no manifest
                # yet — skip until the manifest is written at run completion.
                continue

        if model_name not in by_model:
            by_model[model_name] = {}
        if model_name not in model_order:
            model_order.append(model_name)

        latest = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    tid = rec.get("task_id", "")
                    if tid:
                        latest[tid] = rec
                except json.JSONDecodeError:
                    pass

        for tid, rec in latest.items():
            if tid not in by_model[model_name]:
                rec["model"] = model_name
                by_model[model_name][tid] = rec

    models_data = {}
    for model_name, tid_map in by_model.items():
        recs = list(tid_map.values())
        recs.sort(key=lambda r: (-(r.get("yes_price_market") or 0), r.get("task_id", "")))
        models_data[model_name] = recs

    return models_data, model_order


def _attach_price_history(rec: dict) -> dict:
    """Attach market price history to a forecast record (mutates a copy)."""
    mid = rec.get("market_id", "")
    h = _market_histories.get(mid, {})
    rec = dict(rec)
    rec["price_history"]  = h.get("price_history", [])
    rec["volume_history"] = h.get("volume_history", [])
    return rec


# ── routes ──────────────────────────────────────────────────────────────────────

def _build_market_list(models_data, model_order):
    """Return ordered list of markets with per-model probability snapshots.

    Order: union of task_ids across all models, primary-model-first (most
    markets wins), stable across page reloads.
    """
    primary_order = sorted(model_order, key=lambda m: -len(models_data.get(m, [])))
    seen = {}
    for model_name in primary_order:
        for rec in models_data.get(model_name, []):
            tid = rec.get("task_id", "")
            if not tid:
                continue
            if tid not in seen:
                seen[tid] = {
                    "task_id":            tid,
                    "question":           rec.get("question", ""),
                    "category":           rec.get("category"),
                    "days_to_resolution": rec.get("days_to_resolution"),
                    "yes_price_market":   rec.get("yes_price_market"),
                    "per_model":          {},
                }
            seen[tid]["per_model"][model_name] = {
                "yes_prob": rec.get("yes_prob"),
                "k_done":   rec.get("k_done", len(rec.get("k_runs", []))),
                "k":        rec.get("k"),
            }
    # Only keep markets covered by every known model
    all_models = set(model_order)
    matched = [m for m in seen.values() if set(m["per_model"].keys()) >= all_models]
    # Stable shared order: highest market probability first, then by task_id
    matched.sort(key=lambda m: (-(m.get("yes_price_market") or 0), m["task_id"]))
    return matched


@app.route("/")
def index():
    models_data, model_order = _load_all_forecasts()
    market_list = _build_market_list(models_data, model_order)
    # display order: most-markets first so the primary model's chips appear first
    display_order = sorted(model_order, key=lambda m: -len(models_data.get(m, [])))
    return render_template("index.html",
                           models_data=models_data,
                           model_order=display_order,
                           market_list=market_list)


@app.route("/api/forecast/<task_id>")
def get_forecast(task_id: str):
    model_filter = request.args.get("model")
    models_data, _ = _load_all_forecasts()
    search_models = ([model_filter] if model_filter and model_filter in models_data
                     else list(models_data.keys()))
    for m in search_models:
        for rec in models_data[m]:
            if rec.get("task_id") == task_id:
                return jsonify(_attach_price_history(rec))
    return jsonify({"error": "not found"}), 404


def _load_counterfactuals(task_id: str) -> list[dict]:
    """Return all CF packets for a task_id, newest file first, deduplicated by cf_id."""
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
                    # skip old single-packet format (no cf_index field)
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


@app.route("/api/counterfactuals/<task_id>")
def get_counterfactuals(task_id: str):
    return jsonify(_load_counterfactuals(task_id))


def _load_updated_forecasts(task_id, model=None):
    """Return updated-forecast records for a task_id, optionally filtered by forecast_model."""
    records = []
    seen = set()
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


@app.route("/api/updated_forecasts/<task_id>")
def get_updated_forecasts(task_id: str):
    model = request.args.get("model")
    return jsonify(_load_updated_forecasts(task_id, model))


# ── shared helper ───────────────────────────────────────────────────────────────

def _mean_se(vals):
    """Return (mean, SE) for a list of floats; either may be None."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var / len(vals))


# ── aggregate report ─────────────────────────────────────────────────────────────

def _load_aggregate():
    """Aggregate stats across all markets for every model.

    Returns a dict: { model_order: [...], models: { model_name: {...} } }
    """
    models_data, model_order = _load_all_forecasts()
    result = {}

    for model_name in model_order:
        recs = models_data[model_name]

        yp_vals  = [r["yes_prob"]        for r in recs if r.get("yes_prob") is not None]
        std_vals = [r["yes_prob_std"]     for r in recs if r.get("yes_prob_std") is not None]
        mp_vals  = [r["yes_price_market"] for r in recs if r.get("yes_price_market") is not None]
        div_vals = [
            abs(r["yes_prob"] - r["yes_price_market"])
            for r in recs
            if r.get("yes_prob") is not None and r.get("yes_price_market") is not None
        ]

        per_market = []
        for rec in recs:
            tid        = rec["task_id"]
            uf_records = _load_updated_forecasts(tid, model_name)
            cf_packets = _load_counterfactuals(tid)
            cf_lookup  = {p["cf_id"]: p for p in cf_packets}

            entry = {
                "task_id":          tid,
                "question":         rec.get("question", ""),
                "category":         rec.get("category", ""),
                "yes_prob":         rec.get("yes_prob"),
                "yes_prob_std":     rec.get("yes_prob_std"),
                "yes_price_market": rec.get("yes_price_market"),
                "n_updates":        len(uf_records),
                "EHC_rate":         None,
                "HFC_rate":         None,
                "ICS_rate":         None,
                "by_direction":     {},
            }

            if uf_records:
                cons = _compute_consistency(uf_records, cf_lookup)
                if cons and cons.get("summary"):
                    s = cons["summary"]
                    entry["EHC_rate"]     = s.get("EHC_rate")
                    entry["HFC_rate"]     = s.get("HFC_rate")
                    entry["ICS_rate"]     = s.get("ICS_rate")
                    entry["by_direction"] = s.get("by_direction", {})

            per_market.append(entry)

        # global consistency aggregated across markets
        ehc_r, ehc_se = _mean_se([m["EHC_rate"] for m in per_market])
        hfc_r, hfc_se = _mean_se([m["HFC_rate"] for m in per_market])
        ics_r, ics_se = _mean_se([m["ICS_rate"] for m in per_market])
        yp_r,  _      = _mean_se(yp_vals)
        std_r, std_se = _mean_se(std_vals)
        div_r, div_se = _mean_se(div_vals)

        # direction-level aggregate (average market-level rates per direction)
        dir_agg = {}
        for d in ("pro_H1", "anti_H1", "orthogonal"):
            d_ehc = [m["by_direction"].get(d, {}).get("EHC_rate") for m in per_market]
            d_hfc = [m["by_direction"].get(d, {}).get("HFC_rate") for m in per_market]
            d_ics = [m["by_direction"].get(d, {}).get("ICS_rate") for m in per_market]
            er, es = _mean_se(d_ehc)
            hr, hs = _mean_se(d_hfc)
            ir, isr = _mean_se(d_ics)
            dir_agg[d] = {
                "EHC_rate": round(er, 3) if er is not None else None,
                "EHC_se":   round(es, 3) if es is not None else None,
                "HFC_rate": round(hr, 3) if hr is not None else None,
                "HFC_se":   round(hs, 3) if hs is not None else None,
                "ICS_rate": round(ir, 3) if ir is not None else None,
                "ICS_se":   round(isr, 3) if isr is not None else None,
            }

        # category-level aggregate
        cat_agg = {}
        cats = sorted({m["category"] for m in per_market if m["category"]})
        for cat in cats:
            cm = [m for m in per_market if m["category"] == cat]
            er, es = _mean_se([m["EHC_rate"] for m in cm])
            hr, hs = _mean_se([m["HFC_rate"] for m in cm])
            ir, isr = _mean_se([m["ICS_rate"] for m in cm])
            cat_agg[cat] = {
                "EHC_rate": round(er, 3) if er is not None else None,
                "EHC_se":   round(es, 3) if es is not None else None,
                "HFC_rate": round(hr, 3) if hr is not None else None,
                "HFC_se":   round(hs, 3) if hs is not None else None,
                "ICS_rate": round(ir, 3) if ir is not None else None,
                "ICS_se":   round(isr, 3) if isr is not None else None,
                "n":        len(cm),
            }

        result[model_name] = {
            "n_markets":  len(recs),
            "n_with_uf":  sum(1 for m in per_market if m["n_updates"] > 0),
            "forecast": {
                "mean_yes_prob":      round(yp_r,  3) if yp_r  is not None else None,
                "mean_run_std":       round(std_r, 3) if std_r is not None else None,
                "mean_run_std_se":    round(std_se,3) if std_se is not None else None,
                "mean_divergence":    round(div_r, 3) if div_r is not None else None,
                "mean_divergence_se": round(div_se,3) if div_se is not None else None,
            },
            "consistency": {
                "EHC_rate": round(ehc_r, 3) if ehc_r is not None else None,
                "EHC_se":   round(ehc_se,3) if ehc_se is not None else None,
                "HFC_rate": round(hfc_r, 3) if hfc_r is not None else None,
                "HFC_se":   round(hfc_se,3) if hfc_se is not None else None,
                "ICS_rate": round(ics_r, 3) if ics_r is not None else None,
                "ICS_se":   round(ics_se,3) if ics_se is not None else None,
                "n_markets_with_data": sum(1 for m in per_market if m["EHC_rate"] is not None),
            },
            "by_direction": dir_agg,
            "by_category":  cat_agg,
            "per_market":   per_market,
        }

    return {"models": result, "model_order": model_order}


@app.route("/api/aggregate")
def get_aggregate():
    return jsonify(_load_aggregate())


# ── consistency metrics ─────────────────────────────────────────────────────────

def _compute_consistency(uf_records, cf_lookup):
    """Compute EHC, HFC, ICS per (cf_id, run) and aggregate.

    EHC — Evidence-Hypothesis Consistency:
        Did H1.posterior_probability in the updated structured forecast move
        in the expected direction implied by the counterfactual evidence?
        Only counted when |H1_delta| >= 0.03 (a meaningful update occurred).

    HFC — Hypothesis-Forecast Consistency:
        Did the final yes_prob field move in the expected direction?
        This tests end-to-end propagation from evidence → final number.
        Only counted when |yes_prob_delta| >= 0.03.

    ICS — Internal Coherence Score:
        After updating, is yes_prob ≈ H1.posterior_probability?
        (The model's own invariant: yes_prob must equal P(H1).)
        ICS = 1 if |yes_prob - H1.posterior| < 0.02, else 0.
    """

    _rate_se = _mean_se  # use module-level helper

    # group by cf_id
    by_cf = {}
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

            # extract H1.posterior from updated structured forecast
            usf    = r.get("updated_structured_forecast") or {}
            hyps   = usf.get("hypotheses", [])
            h1_upd = next((h for h in hyps if h.get("id") == "H1"), {})
            h1_post = h1_upd.get("posterior_probability")

            # EHC
            ehc = None
            if direction in ("pro_H1", "anti_H1") and h1_post is not None and initial_yp is not None:
                h1_delta = h1_post - initial_yp
                if abs(h1_delta) >= 0.03:
                    ehc = 1 if (
                        (direction == "pro_H1" and h1_delta > 0) or
                        (direction == "anti_H1" and h1_delta < 0)
                    ) else 0

            # HFC
            hfc = None
            if direction in ("pro_H1", "anti_H1") and delta_yp is not None:
                if abs(delta_yp) >= 0.03:
                    hfc = 1 if (
                        (direction == "pro_H1" and delta_yp > 0) or
                        (direction == "anti_H1" and delta_yp < 0)
                    ) else 0

            # ICS
            ics, ics_dev = None, None
            if updated_yp is not None and h1_post is not None:
                ics_dev = abs(updated_yp - h1_post)
                ics = 1 if ics_dev < 0.02 else 0

            run_data.append({
                "run_id":              r.get("initial_run_id"),
                "initial_yes_prob":    initial_yp,
                "updated_yes_prob":    updated_yp,
                "delta_yes_prob":      delta_yp,
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

        ehc_rate, _ = _rate_se(ehc_vals)
        hfc_rate, _ = _rate_se(hfc_vals)
        ics_rate, _ = _rate_se(ics_vals)
        mean_delta  = sum(delta_vals) / len(delta_vals) if delta_vals else None

        slot_raw   = cf_info.get("slot_type") or ""
        slot_label = slot_raw.split("—")[0].strip() if "—" in slot_raw else slot_raw.split(" ")[0].strip()

        cf_results.append({
            "cf_id":             cf_id,
            "direction":         direction,
            "cf_index":          cf_info.get("cf_index"),
            "evidence_headline": cf_info.get("evidence_headline", ""),
            "mechanism_targeted":cf_info.get("mechanism_targeted", ""),
            "slot_type":         slot_label,
            "runs":              run_data,
            "EHC_rate":  round(ehc_rate, 3) if ehc_rate is not None else None,
            "HFC_rate":  round(hfc_rate, 3) if hfc_rate is not None else None,
            "ICS_rate":  round(ics_rate, 3) if ics_rate is not None else None,
            "mean_delta":round(mean_delta, 4) if mean_delta is not None else None,
            "n_runs":    len(run_data),
        })

    # ── global summary ──
    all_ehc = [r["EHC"] for cf in cf_results for r in cf["runs"] if r["EHC"] is not None]
    all_hfc = [r["HFC"] for cf in cf_results for r in cf["runs"] if r["HFC"] is not None]
    all_ics = [r["ICS"] for cf in cf_results for r in cf["runs"] if r["ICS"] is not None]

    by_dir = {}
    for d in ("pro_H1", "anti_H1", "orthogonal"):
        d_cfs = [cf for cf in cf_results if cf["direction"] == d]
        d_ehc = [r["EHC"] for cf in d_cfs for r in cf["runs"] if r["EHC"] is not None]
        d_hfc = [r["HFC"] for cf in d_cfs for r in cf["runs"] if r["HFC"] is not None]
        d_ics = [r["ICS"] for cf in d_cfs for r in cf["runs"] if r["ICS"] is not None]
        er, es = _rate_se(d_ehc)
        hr, hs = _rate_se(d_hfc)
        ir, is_ = _rate_se(d_ics)
        by_dir[d] = {
            "EHC_rate": round(er, 3) if er is not None else None,
            "EHC_se":   round(es, 3) if es is not None else None,
            "HFC_rate": round(hr, 3) if hr is not None else None,
            "HFC_se":   round(hs, 3) if hs is not None else None,
            "ICS_rate": round(ir, 3) if ir is not None else None,
            "ICS_se":   round(is_, 3) if is_ is not None else None,
            "n_EHC": len(d_ehc),
            "n_HFC": len(d_hfc),
            "n_ICS": len(d_ics),
        }

    gr, gs = _rate_se(all_ehc)
    hr2, hs2 = _rate_se(all_hfc)
    ir2, is2 = _rate_se(all_ics)

    return {
        "task_id":   (uf_records[0].get("task_id") if uf_records else None),
        "n_updates": len(uf_records),
        "cf_results": cf_results,
        "summary": {
            "EHC_rate": round(gr,   3) if gr   is not None else None,
            "EHC_se":   round(gs,   3) if gs   is not None else None,
            "HFC_rate": round(hr2,  3) if hr2  is not None else None,
            "HFC_se":   round(hs2,  3) if hs2  is not None else None,
            "ICS_rate": round(ir2,  3) if ir2  is not None else None,
            "ICS_se":   round(is2,  3) if is2  is not None else None,
            "n_EHC": len(all_ehc),
            "n_HFC": len(all_hfc),
            "n_ICS": len(all_ics),
            "by_direction": by_dir,
        },
    }


@app.route("/api/consistency/<task_id>")
def get_consistency(task_id: str):
    model      = request.args.get("model")
    uf_records = _load_updated_forecasts(task_id, model)
    cf_packets = _load_counterfactuals(task_id)
    cf_lookup  = {p["cf_id"]: p for p in cf_packets}
    if not uf_records:
        return jsonify({"task_id": task_id, "n_updates": 0, "cf_results": [], "summary": None})
    return jsonify(_compute_consistency(uf_records, cf_lookup))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"Starting viewer at http://{args.host}:{args.port}")
    print(f"Data directory: {_FORECAST_DIR}")
    app.run(host=args.host, port=args.port, debug=True)
