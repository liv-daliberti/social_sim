#!/usr/bin/env python3
"""Flask viewer for Exp 1 forecast conversations.

Usage (from exp1_prospective/):
    python viewer/app.py
    python viewer/app.py --port 5050
"""

import argparse
import json
import math
import os
import re
import statistics
import threading
from pathlib import Path

try:
    import orjson as _json_fast
    _fast_loads = _json_fast.loads
except ImportError:
    _fast_loads = json.loads

import numpy as np

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

_ROOT           = Path(__file__).resolve().parent.parent
_FORECAST_DIR   = _ROOT / "data" / "initial_forecasts"
_MARKETS_DIR    = _ROOT / "data" / "selected_markets"
_CF_DIR         = _ROOT / "data" / "counterfactuals"
_UF_DIR         = _ROOT / "data" / "updated_forecasts"
_CAT_MAP_PATH   = _ROOT / "configs" / "category_map.json"


def _load_category_map() -> dict[str, str]:
    try:
        raw = json.loads(_CAT_MAP_PATH.read_text())
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:
        return {}

_CATEGORY_MAP: dict[str, str] = _load_category_map()


def _canon_cat(raw: str) -> str:
    """Normalize a raw Polymarket category label to its canonical form."""
    return _CATEGORY_MAP.get(raw, raw) if raw else raw


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


def _norm_prob(v):
    """Normalize yes_prob to [0, 1].  Handles 0.6, 60, 60.0, and '60%'."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.endswith("%"):
            try:
                v = float(v[:-1]) / 100.0
            except ValueError:
                return None
        else:
            try:
                v = float(v)
            except ValueError:
                return None
    else:
        v = float(v)
    if v > 1.0:
        v = v / 100.0
    return round(v, 6)


# ── in-memory cache ─────────────────────────────────────────────────────────────
# Rebuilt automatically whenever any data file's mtime changes.
# _lean_cache holds forecast records with turns/k_runs stripped (small).
# _detail_cache holds full records (with conversations) loaded on-demand per market.

_cache_lock   = threading.Lock()
_lean_cache: dict = {}    # {mtimes, models_data, model_order, cf_index, uf_index}
_detail_cache: dict = {}  # {(task_id, model_name): full_rec}


def _data_mtimes() -> dict[str, float]:
    """Check individual file mtimes so cache invalidates when files are appended to."""
    out: dict[str, float] = {}
    for d in (_FORECAST_DIR, _UF_DIR, _CF_DIR):
        for f in d.glob("*.jsonl"):
            try:
                out[str(f)] = f.stat().st_mtime
            except OSError:
                pass
    return out


def _build_cf_index() -> dict[str, list]:
    """Load all counterfactual packets, indexed by task_id."""
    index: dict[str, list] = {}
    seen: set[str] = set()
    _order = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}
    for path in sorted(_CF_DIR.glob("counterfactuals_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = _fast_loads(line)
                    if rec.get("cf_index") is None:
                        continue
                    cf_id = rec.get("cf_id", "")
                    if cf_id and cf_id not in seen:
                        seen.add(cf_id)
                        norm_tid = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", rec.get("task_id", ""))
                        index.setdefault(norm_tid, []).append(rec)
                except json.JSONDecodeError:
                    pass
    for lst in index.values():
        lst.sort(key=lambda r: _order.get(r.get("direction", ""), 99))
    return index


def _build_uf_index() -> dict[str, list]:
    """Load all updated-forecast records, indexed by task_id.

    Error records (those with an 'error' key or missing 'direction') are skipped
    entirely — they are also excluded from the seen-set so that an older successful
    record for the same update_id can still be indexed.
    """
    index: dict[str, list] = {}
    seen: set[str] = set()
    for path in sorted(_UF_DIR.glob("updated_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = _fast_loads(line)
                    # Skip failed runs — don't add to seen so a later (older) file
                    # can contribute a successful record for the same update_id.
                    if rec.get("error") or not rec.get("direction"):
                        continue
                    uid = rec.get("update_id", "")
                    model_key = rec.get("forecast_model", "").replace(":", "-")
                    dedup_key = (model_key, uid)
                    if uid and dedup_key not in seen:
                        seen.add(dedup_key)
                        rec["initial_yes_prob"] = _norm_prob(rec.get("initial_yes_prob"))
                        rec["updated_yes_prob"] = _norm_prob(rec.get("updated_yes_prob"))
                        dy = rec.get("delta_yes_prob")
                        if dy is not None:
                            dy = float(dy)
                            if abs(dy) > 1.0:
                                dy = dy / 100.0
                            rec["delta_yes_prob"] = round(dy, 6)
                        # Normalize Ollama model names (qwen2.5:7b → qwen2.5-7b)
                        if "forecast_model" in rec:
                            rec["forecast_model"] = rec["forecast_model"].replace(":", "-")
                        norm_tid = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", rec.get("task_id", ""))
                        index.setdefault(norm_tid, []).append(rec)
                except (json.JSONDecodeError, Exception):
                    pass
    return index


_RAW_CACHE_TTL = 60  # seconds between raw-index rebuilds


def _filter_to_core_markets(models_data: dict):
    """Filter every model to the intersection of markets shared by all models
    with a full run (≥ 100 markets), then return (filtered_data, core_set).

    This removes the extra 20 markets that qwen2.5-14b/32b/72b have from an
    earlier, wider market selection not shared by the other models.
    """
    market_sets = {m: {r["task_id"] for r in recs} for m, recs in models_data.items()}
    full_sets   = [s for s in market_sets.values() if len(s) >= 100]
    if not full_sets:
        return models_data, set()
    core = set.intersection(*full_sets)
    filtered = {
        m: [r for r in recs if r["task_id"] in core]
        for m, recs in models_data.items()
    }
    return filtered, core


def _ensure_cache(force: bool = False) -> dict:
    """Return the lean app cache, rebuilding at most once per _RAW_CACHE_TTL seconds.

    Uses a time-based TTL rather than per-file mtime checks, because mtime
    checks on this cluster frequently show changes (ongoing experiments writing
    data) and trigger expensive JSONL re-parses on every page load.
    Call with force=True (e.g. from /api/refresh) to rebuild immediately.
    """
    import time as _time
    now = _time.monotonic()
    last_built = _lean_cache.get("__built_at", 0)
    if not force and _lean_cache.get("mtimes") is not None and (now - last_built) < _RAW_CACHE_TTL:
        return _lean_cache
    with _cache_lock:
        last_built = _lean_cache.get("__built_at", 0)
        if not force and _lean_cache.get("mtimes") is not None and (now - last_built) < _RAW_CACHE_TTL:
            return _lean_cache
        mtimes = _data_mtimes()
        models_data, model_order = _load_all_forecasts(lean=True)
        models_data, core_markets = _filter_to_core_markets(models_data)
        uf_index = _build_uf_index()
        if core_markets:
            uf_index = {tid: recs for tid, recs in uf_index.items() if tid in core_markets}
        _lean_cache.update({
            "mtimes":      mtimes,
            "models_data": models_data,
            "model_order": model_order,
            "cf_index":    _build_cf_index(),
            "uf_index":    uf_index,
            "__built_at":  _time.monotonic(),
        })
        _detail_cache.clear()
    return _lean_cache


_derived_locks: dict = {}
_derived_locks_lock = threading.Lock()
_DERIVED_TTL = 60  # seconds; re-compute derived results at most this often


def _ensure_derived(key: str, builder):
    """Lazily compute and cache a derived result for up to _DERIVED_TTL seconds.

    Per-key locks allow aggregate and regression to compute in parallel on
    cold start. TTL avoids expensive recomputes when data files change often.
    """
    import time as _time
    now = _time.monotonic()
    stored    = _lean_cache.get(key)
    stored_at = _lean_cache.get(f"__{key}_at", 0)
    if stored is not None and (now - stored_at) < _DERIVED_TTL:
        return stored
    with _derived_locks_lock:
        if key not in _derived_locks:
            _derived_locks[key] = threading.Lock()
    lock = _derived_locks[key]
    with lock:
        stored    = _lean_cache.get(key)
        stored_at = _lean_cache.get(f"__{key}_at", 0)
        if stored is not None and (now - stored_at) < _DERIVED_TTL:
            return stored
        _ensure_cache()  # refresh raw index if TTL expired
        result = builder()
        _lean_cache[key]           = result
        _lean_cache[f"__{key}_at"] = _time.monotonic()
        return result


# ── forecast loading ────────────────────────────────────────────────────────────

# Fields kept in the lean (index) record; conversations stripped to save memory.
_LEAN_DROP = {"turns", "k_runs"}


def _load_all_forecasts(lean: bool = False):
    """Load all forecast records grouped by model.

    When lean=True, strips conversation fields (turns, k_runs) and adds
    _source_file so detail records can be fetched on demand.

    Returns:
        models_data: {model_name: [records sorted by yes_prob desc]}
        model_order: [model names in discovery order, newest-run first]
    """
    by_model = {}       # model_name -> {task_id -> record}
    model_order = []

    for path in sorted(_FORECAST_DIR.glob("forecasts_*.jsonl"), reverse=True):
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
        model_name = model_name.replace(":", "-")

        if model_name not in by_model:
            by_model[model_name] = {}
        if model_name not in model_order:
            model_order.append(model_name)

        latest = {}
        with open(path, "rb") as fb:
            while True:
                offset = fb.tell()
                raw = fb.readline()
                if not raw:
                    break
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = _fast_loads(stripped)
                    tid = rec.get("task_id", "")
                    if tid:
                        rec["_source_offset"] = offset
                        # Normalize task_id: strip trailing date suffix so
                        # pm_601825_2026-06-09 and pm_601825_2026-06-11
                        # resolve to the same market across different run dates.
                        norm_tid = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", tid)
                        rec["task_id"] = norm_tid
                        latest[norm_tid] = rec
                except (json.JSONDecodeError, ValueError):
                    pass

        for tid, rec in latest.items():
            if tid not in by_model[model_name]:
                rec["model"] = model_name
                rec["_source_file"] = str(path)
                raw_runs = rec.get("yes_prob_runs") or []
                normed_runs = [_norm_prob(v) for v in raw_runs]
                normed_runs = [v for v in normed_runs if v is not None]
                if normed_runs:
                    rec["yes_prob_runs"] = normed_runs
                    rec["yes_prob"] = round(statistics.median(normed_runs), 4)
                    rec["yes_prob_std"] = (round(statistics.stdev(normed_runs), 4)
                                          if len(normed_runs) > 1 else 0.0)
                else:
                    rec["yes_prob"] = _norm_prob(rec.get("yes_prob"))
                if lean:
                    for field in _LEAN_DROP:
                        rec.pop(field, None)
                by_model[model_name][tid] = rec

    models_data = {}
    for model_name, tid_map in by_model.items():
        recs = list(tid_map.values())
        recs.sort(key=lambda r: (-(r.get("yes_price_market") or 0), r.get("task_id", "")))
        models_data[model_name] = recs

    def _size_key(mn):
        m = re.search(r'[:\-](\d+)b$', mn.lower())
        return (0, int(m.group(1))) if m else (1, 0)

    model_order.sort(key=_size_key)
    return models_data, model_order


def _load_one_forecast(task_id: str, model_name: str):
    """Return the full forecast record (with turns/k_runs) for one market.

    Uses the lean cache to locate the source file, then reads only that file.
    Result is cached in _detail_cache so repeated clicks are instant.
    """
    key = (task_id, model_name)
    if key in _detail_cache:
        return _detail_cache[key]

    cache = _ensure_cache()
    lean_recs = cache["models_data"].get(model_name, [])
    lean_rec = next((r for r in lean_recs if r.get("task_id") == task_id), None)
    if lean_rec is None:
        return None
    source_file   = lean_rec.get("_source_file")
    source_offset = lean_rec.get("_source_offset")
    if not source_file:
        return None

    try:
        with open(source_file, "rb") as fb:
            if source_offset is not None:
                fb.seek(source_offset)
                raw = fb.readline()
            else:
                raw = b""
                for raw in fb:
                    if raw.strip():
                        try:
                            raw_tid = json.loads(raw).get("task_id", "")
                            if re.sub(r"_\d{4}-\d{2}-\d{2}$", "", raw_tid) == task_id:
                                break
                        except (json.JSONDecodeError, ValueError):
                            pass
            rec = json.loads(raw)
    except Exception:
        return None

    raw_tid = rec.get("task_id", "")
    if re.sub(r"_\d{4}-\d{2}-\d{2}$", "", raw_tid) != task_id:
        return None

    rec["task_id"] = task_id  # store normalized form
    rec["model"] = model_name
    rec["_source_file"] = source_file
    raw_runs = rec.get("yes_prob_runs") or []
    normed = [_norm_prob(v) for v in raw_runs]
    normed = [v for v in normed if v is not None]
    if normed:
        rec["yes_prob_runs"] = normed
        rec["yes_prob"] = round(statistics.median(normed), 4)
        rec["yes_prob_std"] = (round(statistics.stdev(normed), 4)
                              if len(normed) > 1 else 0.0)
    else:
        rec["yes_prob"] = _norm_prob(rec.get("yes_prob"))
    _detail_cache[key] = rec
    return rec


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
    """Return ordered list of markets with per-model probability snapshots."""
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
                    "category":           _canon_cat(rec.get("category", "")),
                    "days_to_resolution": rec.get("days_to_resolution"),
                    "yes_price_market":   rec.get("yes_price_market"),
                    "per_model":          {},
                }
            seen[tid]["per_model"][model_name] = {
                "yes_prob": rec.get("yes_prob"),
                "k_done":   rec.get("k_done", len(rec.get("k_runs", []))),
                "k":        rec.get("k"),
            }
    matched = [m for m in seen.values() if m["per_model"]]
    matched.sort(key=lambda m: (-(m.get("yes_price_market") or 0), m["task_id"]))
    return matched


@app.route("/")
def index():
    cache = _ensure_cache()
    models_data  = cache["models_data"]
    model_order  = cache["model_order"]
    market_list  = _build_market_list(models_data, model_order)
    return render_template("index.html",
                           models_data=models_data,
                           model_order=model_order,
                           market_list=market_list)


@app.route("/api/forecast/<task_id>")
def get_forecast(task_id: str):
    model_filter = request.args.get("model")
    cache = _ensure_cache()
    search_models = ([model_filter] if model_filter and model_filter in cache["models_data"]
                     else list(cache["models_data"].keys()))
    for m in search_models:
        rec = _load_one_forecast(task_id, m)
        if rec:
            return jsonify(_attach_price_history(rec))
    return jsonify({"error": "not found"}), 404


def _load_counterfactuals(task_id: str, _cache=None) -> list[dict]:
    """Return all CF packets for a task_id (from cache index)."""
    return (_cache or _ensure_cache())["cf_index"].get(task_id, [])


@app.route("/api/counterfactuals/<task_id>")
def get_counterfactuals(task_id: str):
    return jsonify(_load_counterfactuals(task_id))


def _load_updated_forecasts(task_id, model=None, _cache=None):
    """Return updated-forecast records for a task_id (from cache index)."""
    recs = (_cache or _ensure_cache())["uf_index"].get(task_id, [])
    if model:
        recs = [r for r in recs if r.get("forecast_model") == model]
    return recs


@app.route("/api/updated_forecasts/<task_id>")
def get_updated_forecasts(task_id: str):
    model = request.args.get("model")
    return jsonify(_load_updated_forecasts(task_id, model))


# ── shared helper ───────────────────────────────────────────────────────────────

def _cohen_kappa(labels1, labels2):
    """Compute Cohen's kappa for two lists of binary labels (0/1)."""
    n = len(labels1)
    if n < 5:
        return None
    agree  = sum(a == b for a, b in zip(labels1, labels2))
    p_o    = agree / n
    p1_yes = sum(labels1) / n
    p2_yes = sum(labels2) / n
    p_e    = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    if abs(1 - p_e) < 1e-9:
        return 1.0 if abs(p_o - 1.0) < 1e-9 else None
    return (p_o - p_e) / (1 - p_e)


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
    """Aggregate stats across all markets for every model."""
    cache = _ensure_cache()
    models_data = cache["models_data"]
    model_order = cache["model_order"]
    result = {}

    for model_name in model_order:
        recs = models_data[model_name]

        yp_vals  = [r["yes_prob"]        for r in recs if r.get("yes_prob") is not None]
        std_vals = [r["yes_prob_std"]     for r in recs if r.get("yes_prob_std") is not None]
        div_vals = [
            abs(r["yes_prob"] - r["yes_price_market"])
            for r in recs
            if r.get("yes_prob") is not None and r.get("yes_price_market") is not None
        ]

        per_market = []
        all_uf_records = []
        for rec in recs:
            tid        = rec["task_id"]
            uf_records = _load_updated_forecasts(tid, model_name, _cache=cache)
            all_uf_records.extend(uf_records)
            cf_packets = _load_counterfactuals(tid, _cache=cache)
            cf_lookup  = {p["cf_id"]: p for p in cf_packets}

            entry = {
                "task_id":          tid,
                "question":         rec.get("question", ""),
                "category":         _canon_cat(rec.get("category", "")),
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

        ehc_r, ehc_se = _mean_se([m["EHC_rate"] for m in per_market])
        hfc_r, hfc_se = _mean_se([m["HFC_rate"] for m in per_market])
        ics_r, ics_se = _mean_se([m["ICS_rate"] for m in per_market])
        yp_r,  _      = _mean_se(yp_vals)
        std_r, std_se = _mean_se(std_vals)
        div_r, div_se = _mean_se(div_vals)

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

        _anc_dir: dict[str, list[float]] = {"pro_H1": [], "anti_H1": [], "orthogonal": []}
        for ufr in all_uf_records:
            d   = ufr.get("direction", "")
            dyp = ufr.get("delta_yes_prob")
            if d in _anc_dir and dyp is not None:
                _anc_dir[d].append(abs(dyp))

        anc_agg: dict = {}
        for d, vals in _anc_dir.items():
            m_v, se_v = _mean_se(vals)
            anc_agg[d] = {
                "mean_abs_delta": round(m_v,  4) if m_v  is not None else None,
                "se":             round(se_v, 4) if se_v is not None else None,
                "n": len(vals),
            }
        _directed = _anc_dir["pro_H1"] + _anc_dir["anti_H1"]
        m_dir, _ = _mean_se(_directed)
        m_ort, _ = _mean_se(_anc_dir["orthogonal"])
        sensitivity = (round(m_dir / m_ort, 2)
                       if (m_dir is not None and m_ort is not None and m_ort > 0)
                       else None)
        anc_agg["sensitivity_ratio"] = sensitivity

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
            "anchoring":    anc_agg,
        }

    return {"models": result, "model_order": model_order}


@app.route("/api/aggregate")
def get_aggregate():
    return jsonify(_ensure_derived("aggregate", _load_aggregate))


@app.route("/api/delta_scatter")
def get_delta_scatter():
    """Raw delta_yes_prob values per model per direction, for scatter plotting."""
    cache = _ensure_cache()
    model_order = cache["model_order"]
    uf_index    = cache["uf_index"]

    by_model: dict[str, dict] = {}
    for task_recs in uf_index.values():
        for r in task_recs:
            mn = r.get("forecast_model", "")
            if not mn:
                continue
            if mn not in by_model:
                by_model[mn] = {"pro_H1": [], "anti_H1": [], "orthogonal": []}
            d  = r.get("direction", "")
            dy = r.get("delta_yes_prob")
            if d in by_model[mn] and dy is not None:
                try:
                    by_model[mn][d].append(round(float(dy), 4))
                except (TypeError, ValueError):
                    pass

    return jsonify({"models": by_model, "model_order": model_order})


def _fit_ols(records):
    """Fit delta_yes_prob ~ pro_H1 + anti_H1 + initial_yes_prob via OLS.

    Direction dummies: orthogonal = reference (intercept).
    Returns dict with coefficients, SE, t-stats, p-values, R², n,
    and predicted lines for plotting (initial_yes_prob 0→1 grid, per direction).
    Returns None if fewer than 10 usable records.
    """
    rows = []
    for r in records:
        d   = r.get("direction", "")
        ip  = r.get("initial_yes_prob")   # already _norm_prob'd in _build_uf_index
        up  = r.get("updated_yes_prob")   # already _norm_prob'd
        if ip is None or up is None or d not in ("pro_H1", "anti_H1", "orthogonal"):
            continue
        try:
            ip_f = float(ip); up_f = float(up)
            # Recompute delta from normalised probs so Qwen's raw-scale
            # delta_yes_prob values (which can be ±30) don't contaminate the fit.
            dy = up_f - ip_f
            rows.append((dy, ip_f, d))
        except (TypeError, ValueError):
            pass
    if len(rows) < 10:
        return None

    Y  = np.array([r[0] for r in rows])
    ip = np.array([r[1] for r in rows])
    is_pro  = np.array([1.0 if r[2] == "pro_H1"  else 0.0 for r in rows])
    is_anti = np.array([1.0 if r[2] == "anti_H1" else 0.0 for r in rows])

    X = np.column_stack([np.ones(len(Y)), is_pro, is_anti, ip])

    try:
        beta, res, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    Y_hat   = X @ beta
    ss_res  = float(np.sum((Y - Y_hat) ** 2))
    ss_tot  = float(np.sum((Y - Y.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n, k    = len(Y), X.shape[1]
    sigma2  = ss_res / max(n - k, 1)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se  = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, float("nan"))

    t_stats = beta / se
    # Two-tailed p-values via regularized incomplete beta (t-distribution CDF)
    import math as _math
    def _t_pval(t_val, df):
        # Wilson-Hilferty approximation for t → p
        if not _math.isfinite(t_val) or df <= 0:
            return float("nan")
        x = df / (df + t_val ** 2)
        # regularized incomplete beta I_x(df/2, 1/2) via lgamma
        a, b = df / 2.0, 0.5
        # Use normal approximation for large df (accurate to <0.001)
        if df > 30:
            z = abs(t_val) * (1 - 1 / (4 * df)) / _math.sqrt(1 + t_val**2 / (2 * df))
            p_one = 0.5 * _math.erfc(z / _math.sqrt(2))
        else:
            # Abramowitz & Stegun 26.7.8 iterative
            p_one = 0.5 * x ** a * (1 - x) ** b
            for _ in range(50):
                p_one = max(0.0, min(0.5, p_one))
            # Fallback to normal approx
            z = abs(t_val)
            p_one = 0.5 * _math.erfc(z / _math.sqrt(2))
        return min(1.0, 2 * p_one)
    p_vals = np.array([_t_pval(float(t), n - k) for t in t_stats])

    names = ["intercept", "pro_H1", "anti_H1", "initial_yes_prob"]

    # Predicted lines: for each direction, predicted delta across ip grid 0→1
    grid = np.linspace(0, 1, 41)
    lines = {}
    for dir_label, is_p, is_a in [("pro_H1", 1, 0), ("anti_H1", 0, 1), ("orthogonal", 0, 0)]:
        X_grid = np.column_stack([np.ones(41), np.full(41, is_p), np.full(41, is_a), grid])
        lines[dir_label] = [round(float(v), 4) for v in (X_grid @ beta)]

    # Per-direction scatter: (x=ip, y=delta) for regression plot
    # and (x=ip, y=updated) for before/after plot
    scatter     = {"pro_H1": [], "anti_H1": [], "orthogonal": []}
    before_after = {"pro_H1": [], "anti_H1": [], "orthogonal": []}
    for dy, ip_val, d in rows:
        scatter[d].append([round(ip_val, 4), round(dy, 4)])
        before_after[d].append([round(ip_val, 4), round(ip_val + dy, 4)])

    # Wald test: H₀: β_pro + β_anti = 0  (symmetric sensitivity)
    # L = [0, 1, 1, 0]  →  L'β = β_pro + β_anti
    wald = None
    try:
        L      = np.array([0., 1., 1., 0.])
        Lbeta  = float(L @ beta)
        LcovL  = float(L @ cov @ L)
        if LcovL > 0:
            W      = Lbeta ** 2 / LcovL          # chi-squared(1) under H₀
            # p from chi²(1): erfc(sqrt(W/2))
            wald_p = math.erfc(math.sqrt(W / 2))
            wald = {
                "statistic": round(W, 4),
                "p_value":   round(wald_p, 6),
                "lbeta":     round(Lbeta, 5),   # β_pro + β_anti (should be ~0 if symmetric)
                "se_lbeta":  round(math.sqrt(LcovL), 5),
            }
    except Exception:
        pass

    def _fmt(arr):
        return {n: round(float(v), 5) for n, v in zip(names, arr)}

    return {
        "n":            n,
        "r_squared":    round(r2, 4),
        "rmse":         round(math.sqrt(sigma2), 4),
        "coefficients": _fmt(beta),
        "se":           _fmt(se),
        "t_stats":      _fmt(t_stats),
        "p_values":     _fmt(p_vals),
        "lines":        lines,
        "grid":         [round(float(v), 3) for v in grid],
        "scatter":      scatter,
        "before_after": before_after,   # {direction: [[p0, p_updated], ...]}
        "wald_symmetry": wald,          # H₀: |β_pro| = |β_anti|
    }


def _compute_regression() -> dict:
    cache    = _ensure_cache()
    uf_index = cache["uf_index"]

    all_recs: list[dict] = []
    by_model: dict[str, list] = {}
    for task_recs in uf_index.values():
        for r in task_recs:
            all_recs.append(r)
            mn = r.get("forecast_model", "")
            if mn:
                by_model.setdefault(mn, []).append(r)

    return {
        "overall":     _fit_ols(all_recs),
        "by_model":    {mn: _fit_ols(recs) for mn, recs in by_model.items()},
        "model_order": cache["model_order"],
    }


@app.route("/api/regression")
def get_regression():
    """OLS regression: delta_yes_prob ~ direction + initial_yes_prob, overall + per model."""
    return jsonify(_ensure_derived("regression", _compute_regression))


@app.route("/api/update_agreement")
def get_update_agreement():
    """Cross-model sign agreement and CF packet consistency.

    Groups UF records by (market_id, direction, cf_index) and checks whether
    all models that have data for that (market, CF snippet) agree on the sign
    of Δp̂.  Also computes within-market variance across the 3 CF snippets per
    direction (CF packet consistency).
    """
    cache    = _ensure_cache()
    uf_index = cache["uf_index"]
    model_order = cache["model_order"]

    DIRS = ("pro_H1", "anti_H1", "orthogonal")

    # ── collect mean Δp̂ per (market, direction, cf_index, model) ────────────
    # key = (market_id, direction, cf_index)  →  {model: [deltas]}
    bucket: dict = {}
    for task_recs in uf_index.values():
        for r in task_recs:
            mn  = r.get("forecast_model", "")
            mid = r.get("market_id",      "")
            d   = r.get("direction",      "")
            ci  = r.get("cf_index")
            ip  = r.get("initial_yes_prob")
            up  = r.get("updated_yes_prob")
            if not mn or not mid or d not in DIRS or ci is None or ip is None or up is None:
                continue
            try:
                dy = float(up) - float(ip)
            except (TypeError, ValueError):
                continue
            key = (mid, d, int(ci))
            bucket.setdefault(key, {}).setdefault(mn, []).append(dy)

    # Reduce to mean per (market, dir, cf_index, model)
    means: dict = {}   # key → {model: mean_delta}
    for key, model_deltas in bucket.items():
        means[key] = {mn: sum(v) / len(v) for mn, v in model_deltas.items() if v}

    # ── cross-model sign agreement ────────────────────────────────────────────
    # Only consider keys where ≥2 models have data
    def _sign(v): return 1 if v > 1e-6 else (-1 if v < -1e-6 else 0)

    agree_overall: list[bool] = []
    agree_by_dir:  dict = {d: [] for d in DIRS}
    pairwise_agree: dict = {}   # "m1||m2" → [agree_bools]

    for key, model_means in means.items():
        mid, d, ci = key
        models_here = [mn for mn, v in model_means.items() if v is not None]
        if len(models_here) < 2:
            continue
        signs = {mn: _sign(model_means[mn]) for mn in models_here}
        # unanimous non-zero sign?
        non_zero = [s for s in signs.values() if s != 0]
        unanimous = len(non_zero) == len(models_here) and len(set(non_zero)) == 1
        agree_overall.append(unanimous)
        agree_by_dir[d].append(unanimous)
        # pairwise
        for i, m1 in enumerate(models_here):
            for m2 in models_here[i+1:]:
                pk = "||".join(sorted([m1, m2]))
                same = (signs[m1] == signs[m2]) and (signs[m1] != 0)
                pairwise_agree.setdefault(pk, []).append(same)

    def _rate(lst): return round(sum(lst) / len(lst), 4) if lst else None

    pairwise_summary = {pk: {"rate": _rate(v), "n": len(v)}
                        for pk, v in pairwise_agree.items()}

    # ── CF packet consistency ────────────────────────────────────────────────
    # For each (market, model, direction): std of mean_delta across cf_index 0,1,2
    # key2 = (market_id, model, direction) → {cf_index: mean_delta}
    consistency_bucket: dict = {}
    for (mid, d, ci), model_means in means.items():
        for mn, v in model_means.items():
            consistency_bucket.setdefault((mid, mn, d), {})[ci] = v

    # Compute std across cf indices; only use entries with all 3 present
    consist_vals: dict = {}      # model → list of within-market stds
    consist_by_dir: dict = {d: [] for d in DIRS}
    consist_per_market: dict = {}  # (mid, mn, d) → std

    for (mid, mn, d), ci_map in consistency_bucket.items():
        if len(ci_map) < 2:
            continue
        vals = list(ci_map.values())
        mu   = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        consist_vals.setdefault(mn, []).append(std)
        consist_by_dir[d].append(std)
        consist_per_market[(mid, mn, d)] = std

    def _mean(lst): return round(sum(lst) / len(lst), 5) if lst else None

    consist_by_model = {mn: {"mean_std": _mean(v), "n": len(v)}
                        for mn, v in consist_vals.items()}
    consist_by_direction = {d: {"mean_std": _mean(v), "n": len(v)}
                             for d, v in consist_by_dir.items()}

    # ── per-market summary ───────────────────────────────────────────────────
    # Collect market-level: agreement rate + consistency per direction
    mkt_summary: dict = {}
    for (mid, d, ci), model_means in means.items():
        ms = mkt_summary.setdefault(mid, {"directions": {}})
        ds = ms["directions"].setdefault(d, {"cf_deltas": {}, "model_means": {}})
        for mn, v in model_means.items():
            ds["cf_deltas"].setdefault(mn, {})[ci] = v
            # running mean per model across all cf indices
            ds["model_means"].setdefault(mn, []).append(v)

    # Reduce model_means to simple mean
    per_market_out = []
    # Get question text from models_data
    q_lookup = {}
    for task_recs in cache["models_data"].values():
        for r in task_recs:
            if r.get("market_id") and r.get("question"):
                q_lookup[r["market_id"]] = r["question"]
                break

    for mid, ms in mkt_summary.items():
        dirs_out = {}
        for d, ds in ms["directions"].items():
            model_dir_means = {mn: round(sum(v)/len(v), 4)
                               for mn, v in ds["model_means"].items() if v}
            signs_here = [_sign(v) for v in model_dir_means.values()]
            non_zero   = [s for s in signs_here if s != 0]
            agree = (len(non_zero) == len(signs_here) == len(model_dir_means)
                     and len(set(non_zero)) == 1) if model_dir_means else False
            dirs_out[d] = {
                "model_means": model_dir_means,
                "agree":       agree,
            }
        per_market_out.append({
            "market_id": mid,
            "question":  q_lookup.get(mid, "")[:80],
            "n_models":  max((len(ds["model_means"]) for ds in ms["directions"].values()), default=0),
            "directions": dirs_out,
        })

    per_market_out.sort(key=lambda x: -x["n_models"])

    return jsonify({
        "sign_agreement": {
            "overall_rate":    _rate(agree_overall),
            "n_cf_instances":  len(agree_overall),
            "by_direction":    {d: {"rate": _rate(v), "n": len(v)}
                                for d, v in agree_by_dir.items()},
            "pairwise":        pairwise_summary,
        },
        "cf_consistency": {
            "by_model":     consist_by_model,
            "by_direction": consist_by_direction,
        },
        "per_market":   per_market_out,
        "model_order":  model_order,
    })


def _load_cross_model():
    """Cross-model agreement: pairwise Cohen's kappa + metric comparison table."""
    cache       = _ensure_cache()
    models_data = cache["models_data"]
    model_order = cache["model_order"]
    THRESHOLD   = 0.5

    # {task_id: {model_name: yes_prob}} + meta
    task_probs: dict[str, dict[str, float]] = {}
    task_meta:  dict[str, dict] = {}
    for model_name in model_order:
        for rec in models_data[model_name]:
            tid = rec["task_id"]
            yp  = rec.get("yes_prob")
            if yp is not None:
                task_probs.setdefault(tid, {})[model_name] = yp
            if tid not in task_meta:
                task_meta[tid] = {
                    "question":         rec.get("question", ""),
                    "category":         _canon_cat(rec.get("category", "")),
                    "yes_price_market": rec.get("yes_price_market"),
                }

    # pairwise Cohen's kappa
    kappa_pairs = []
    for i, m1 in enumerate(model_order):
        for j, m2 in enumerate(model_order):
            if i >= j:
                continue
            shared  = [tid for tid in task_probs
                       if m1 in task_probs[tid] and m2 in task_probs[tid]]
            n_shared = len(shared)
            l1 = [1 if task_probs[tid][m1] > THRESHOLD else 0 for tid in shared]
            l2 = [1 if task_probs[tid][m2] > THRESHOLD else 0 for tid in shared]
            kappa = _cohen_kappa(l1, l2)
            kappa_pairs.append({
                "m1": m1, "m2": m2,
                "n_shared": n_shared,
                "kappa": round(kappa, 3) if kappa is not None else None,
            })

    # per-model kappa vs market price
    market_kappas = []
    for mn in model_order:
        shared = [tid for tid in task_probs
                  if mn in task_probs[tid]
                  and task_meta.get(tid, {}).get("yes_price_market") is not None]
        n_shared = len(shared)
        l_model  = [1 if task_probs[tid][mn] > THRESHOLD else 0 for tid in shared]
        l_market = [1 if task_meta[tid]["yes_price_market"] > THRESHOLD else 0 for tid in shared]
        kappa    = _cohen_kappa(l_model, l_market)
        market_kappas.append({
            "model":    mn,
            "n_shared": n_shared,
            "kappa":    round(kappa, 3) if kappa is not None else None,
        })

    # kappa by category
    def _kappa_pairs_for(tids):
        pairs = []
        for i, m1 in enumerate(model_order):
            for j, m2 in enumerate(model_order):
                if i >= j:
                    continue
                shared = [t for t in tids if m1 in task_probs.get(t, {}) and m2 in task_probs.get(t, {})]
                l1 = [1 if task_probs[t][m1] > THRESHOLD else 0 for t in shared]
                l2 = [1 if task_probs[t][m2] > THRESHOLD else 0 for t in shared]
                k  = _cohen_kappa(l1, l2)
                pairs.append({"m1": m1, "m2": m2, "n_shared": len(shared),
                               "kappa": round(k, 3) if k is not None else None})
        return pairs

    def _market_kappas_for(tids):
        out = []
        for mn in model_order:
            shared = [t for t in tids
                      if mn in task_probs.get(t, {})
                      and task_meta.get(t, {}).get("yes_price_market") is not None]
            lm = [1 if task_probs[t][mn] > THRESHOLD else 0 for t in shared]
            lp = [1 if task_meta[t]["yes_price_market"] > THRESHOLD else 0 for t in shared]
            k  = _cohen_kappa(lm, lp)
            out.append({"model": mn, "n_shared": len(shared),
                        "kappa": round(k, 3) if k is not None else None})
        return out

    cats = sorted({task_meta[t]["category"] for t in task_meta if task_meta[t].get("category")})
    kappa_by_category: dict[str, dict] = {}
    for cat in cats:
        cat_tids = [t for t in task_probs if task_meta.get(t, {}).get("category") == cat]
        kappa_by_category[cat] = {
            "n":             len(cat_tids),
            "pairs":         _kappa_pairs_for(cat_tids),
            "market_kappas": _market_kappas_for(cat_tids),
        }

    # per-model summary from aggregate
    agg = _load_aggregate()
    model_summary = []
    for mn in model_order:
        m    = agg["models"].get(mn, {})
        f    = m.get("forecast",    {})
        cons = m.get("consistency", {})
        anc  = m.get("anchoring",   {})
        model_summary.append({
            "model":           mn,
            "n_markets":       m.get("n_markets"),
            "mean_yes_prob":   f.get("mean_yes_prob"),
            "mean_divergence": f.get("mean_divergence"),
            "mean_run_std":    f.get("mean_run_std"),
            "EHC_rate":        cons.get("EHC_rate"),
            "HFC_rate":        cons.get("HFC_rate"),
            "ICS_rate":        cons.get("ICS_rate"),
            "sensitivity":     anc.get("sensitivity_ratio"),
        })

    def _cmean(key):
        vals = [s[key] for s in model_summary if s.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    cross_avg = {
        "mean_yes_prob":   _cmean("mean_yes_prob"),
        "mean_divergence": _cmean("mean_divergence"),
        "mean_run_std":    _cmean("mean_run_std"),
        "EHC_rate":        _cmean("EHC_rate"),
        "HFC_rate":        _cmean("HFC_rate"),
        "ICS_rate":        _cmean("ICS_rate"),
        "sensitivity":     _cmean("sensitivity"),
    }

    # per-market consensus
    per_market = []
    for tid in sorted(task_probs, key=lambda t: -(task_meta[t].get("yes_price_market") or 0)):
        probs = task_probs[tid]
        if len(probs) < 2:
            continue
        vals      = list(probs.values())
        mean_p    = sum(vals) / len(vals)
        std_p     = (sum((v - mean_p) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        labels    = [1 if v > THRESHOLD else 0 for v in vals]
        unanimous = all(lv == labels[0] for lv in labels)
        meta      = task_meta.get(tid, {})
        per_market.append({
            "task_id":          tid,
            "question":         meta.get("question", ""),
            "category":         meta.get("category", ""),
            "yes_price_market": meta.get("yes_price_market"),
            "model_probs":      probs,
            "mean_prob":        round(mean_p, 3),
            "std_prob":         round(std_p, 3),
            "n_models":         len(probs),
            "unanimous":        unanimous,
        })

    return {
        "model_order":       model_order,
        "kappa_pairs":       kappa_pairs,
        "market_kappas":     market_kappas,
        "kappa_by_category": kappa_by_category,
        "model_summary":     model_summary,
        "cross_avg":         cross_avg,
        "per_market":        per_market,
        "threshold":         THRESHOLD,
    }


@app.route("/api/cross_model")
def get_cross_model():
    return jsonify(_load_cross_model())


@app.route("/api/refresh")
def refresh_cache():
    """Force a cache reload (useful after new data arrives without server restart)."""
    _lean_cache.clear()
    _detail_cache.clear()
    _ensure_cache(force=True)
    return jsonify({"status": "ok"})


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

    _rate_se = _mean_se

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

            usf    = r.get("updated_structured_forecast") or {}
            hyps   = usf.get("hypotheses", [])
            if isinstance(hyps, dict):
                h1_upd = hyps.get("H1", {})
            else:
                h1_upd = next((h for h in hyps if isinstance(h, dict) and h.get("id") == "H1"), {})
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


# ── startup cache warmup ────────────────────────────────────────────────────────
# Load the lean index once at startup so the first request is fast.
print("Warming cache...", flush=True)
_ensure_cache()
print("Cache warm.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"Starting viewer at http://{args.host}:{args.port}")
    print(f"Data directory: {_FORECAST_DIR}")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=args.host, port=args.port, debug=debug, use_reloader=False)
