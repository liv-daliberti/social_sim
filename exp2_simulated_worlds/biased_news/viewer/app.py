#!/usr/bin/env python3
"""Exp 2 Biased News — single viewer app (Simulator + Results).

The HTML has two screens (top-nav): a live Episode Simulator (Bayes oracle +
naive baseline + any matching batch LM predictions) and Batch Results
(learning curves and per-episode detail read from data/batch/results_*.jsonl).

Usage (from biased_news/):
    python viewer/app.py
    python viewer/app.py --port 5052
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.temporal_dag import (
    generate_episode, compute_bayes_forecast, compute_blind_forecast,
    VARIANTS, BASE_EFFECT, BIASED_GAIN, SIGMA_OPINION, SIGMA_SURVEY,
)

app = Flask(__name__)

_ROOT    = Path(__file__).resolve().parent.parent
_BATCHDIR = _ROOT / "data" / "batch"

# ── in-memory cache ─────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_cache: dict = {}   # {mtimes, records, index}


def _mtimes() -> dict[str, float]:
    out: dict[str, float] = {}
    for f in _BATCHDIR.glob("results_*.jsonl"):
        try:
            out[str(f)] = f.stat().st_mtime
        except OSError:
            pass
    return out


def _load_records() -> list[dict]:
    latest: dict[str, dict] = {}
    for path in sorted(_BATCHDIR.glob("results_*.jsonl"), reverse=True):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["episode_id"], rec["variant"], rec["model"])
                    if key not in latest:
                        latest[key] = rec
                except Exception:
                    pass
    return list(latest.values())


def _ensure_cache() -> dict:
    mt = _mtimes()
    if _cache.get("mtimes") == mt:
        return _cache
    with _lock:
        if _cache.get("mtimes") == mt:
            return _cache
        records = _load_records()
        _cache.update({"mtimes": mt, "records": records})
    return _cache


# ── aggregate helpers ──────────────────────────────────────────────────────────

def _mae(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return round(statistics.mean(v), 3) if v else None


def _aggregate(records: list[dict], models: list[str], variants: list[str], biases: list[str]):
    """Return per-T_obs MAE for each predictor, filtered."""
    by_t: dict[int, dict[str, list]] = defaultdict(lambda: {
        "lm": [], "bayes": [], "blind": [], "naive": []
    })

    for rec in records:
        if rec["model"] not in models:
            continue
        if rec["variant"] not in variants:
            continue
        if rec["bias"] not in biases:
            continue
        for step in rec["steps"]:
            t = step["T_obs"]
            by_t[t]["lm"].append(step["lm_abs_err"])
            by_t[t]["bayes"].append(step["bayes_abs_err"])
            by_t[t]["blind"].append(step.get("blind_abs_err"))   # None for pre-blind records
            by_t[t]["naive"].append(step["naive_abs_err"])

    t_vals = sorted(by_t.keys())
    return {
        "T_obs":     t_vals,
        "lm_mae":    [_mae(by_t[t]["lm"])    for t in t_vals],
        "bayes_mae": [_mae(by_t[t]["bayes"]) for t in t_vals],
        "blind_mae": [_mae(by_t[t]["blind"]) for t in t_vals],
        "naive_mae": [_mae(by_t[t]["naive"]) for t in t_vals],
        "n":         max((len(by_t[t]["bayes"]) for t in t_vals), default=0),
    }


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    """Available models, variants, bias conditions in loaded data."""
    c = _ensure_cache()
    recs = c.get("records", [])
    models   = sorted({r["model"]   for r in recs})
    variants = sorted({r["variant"] for r in recs})
    biases   = sorted({r["bias"]    for r in recs})
    # max T_obs across all records
    max_t = max((max((s["T_obs"] for s in r["steps"]), default=0) for r in recs), default=0)
    return jsonify({
        "models":   models,
        "variants": variants,
        "biases":   biases,
        "max_T":    max_t,
        "n_records": len(recs),
    })


@app.route("/api/aggregate")
def api_aggregate():
    """MAE vs T_obs, split neutral vs biased, for selected model+variant."""
    c        = _ensure_cache()
    recs     = c.get("records", [])
    models   = request.args.getlist("model")   or sorted({r["model"]   for r in recs})
    variants = request.args.getlist("variant") or sorted({r["variant"] for r in recs})

    neutral = _aggregate(recs, models, variants, ["neutral"])
    biased  = _aggregate(recs, models, variants, ["biased"])
    return jsonify({"neutral": neutral, "biased": biased})


@app.route("/api/episodes")
def api_episodes():
    """Paginated episode list with summary stats."""
    c        = _ensure_cache()
    recs     = c.get("records", [])
    models   = set(request.args.getlist("model")   or [r["model"]   for r in recs])
    variants = set(request.args.getlist("variant") or [r["variant"] for r in recs])
    biases   = set(request.args.getlist("bias")    or [r["bias"]    for r in recs])
    sort_by  = request.args.get("sort", "lm_err_desc")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    filtered = [r for r in recs
                if r["model"]   in models
                and r["variant"] in variants
                and r["bias"]    in biases]

    # Summarise each record
    summaries = []
    for r in filtered:
        steps     = r["steps"]
        lm_errs   = [s["lm_abs_err"]    for s in steps if s["lm_abs_err"]   is not None]
        bayes_errs = [s["bayes_abs_err"] for s in steps]
        naive_errs = [s["naive_abs_err"] for s in steps]
        lm_wins = sum(
            1 for s in steps
            if s["lm_abs_err"] is not None and s["lm_abs_err"] < s["bayes_abs_err"]
        )
        summaries.append({
            "episode_id":  r["episode_id"],
            "bias":        r["bias"],
            "variant":     r["variant"],
            "model":       r["model"],
            "T_max":       r["T_max"],
            "mean_lm_err":    round(statistics.mean(lm_errs),    2) if lm_errs    else None,
            "mean_bayes_err": round(statistics.mean(bayes_errs), 2) if bayes_errs else None,
            "mean_naive_err": round(statistics.mean(naive_errs), 2) if naive_errs else None,
            "lm_wins":     lm_wins,
            "lm_err_by_t": [s["lm_abs_err"]    for s in steps],
            "bayes_err_by_t": [s["bayes_abs_err"] for s in steps],
        })

    # Sort
    rev = "desc" in sort_by
    key_map = {
        "lm_err":    lambda s: s["mean_lm_err"]    or 999,
        "bayes_err": lambda s: s["mean_bayes_err"] or 999,
        "lm_wins":   lambda s: s["lm_wins"],
        "episode":   lambda s: s["episode_id"],
    }
    for k, fn in key_map.items():
        if sort_by.startswith(k):
            summaries.sort(key=fn, reverse=rev)
            break

    total  = len(summaries)
    start  = (page - 1) * per_page
    return jsonify({
        "episodes": summaries[start: start + per_page],
        "total":    total,
        "page":     page,
        "pages":    math.ceil(total / per_page),
    })


@app.route("/api/episode/<episode_id>")
def api_episode(episode_id: str):
    """Full detail for one episode (all steps + episode trajectory)."""
    c        = _ensure_cache()
    recs     = c.get("records", [])
    model    = request.args.get("model")
    variant  = request.args.get("variant")

    matches = [
        r for r in recs
        if r["episode_id"] == episode_id
        and (model   is None or r["model"]   == model)
        and (variant is None or r["variant"] == variant)
    ]
    if not matches:
        return jsonify({"error": "not found"}), 404

    rec = matches[0]
    return jsonify({
        "episode_id": rec["episode_id"],
        "bias":       rec["bias"],
        "variant":    rec["variant"],
        "model":      rec["model"],
        "T_max":      rec["T_max"],
        "ep_seed":    rec.get("ep_seed"),
        "steps":      rec["steps"],
        "episode":    rec.get("episode", {}),
    })


# ── simulator ──────────────────────────────────────────────────────────────────

@app.route("/api/simulate")
def api_simulate():
    """Generate one episode + Bayes trajectory; look up any matching batch LM data."""
    seed   = int(request.args.get("seed", 42))
    bias   = request.args.get("bias", "random")
    T      = int(request.args.get("T", 12))
    model  = request.args.get("model")   # optional: look up LM preds
    variant = request.args.get("variant", "V2")

    # World parameters (default = canonical Regime B in the engine constants)
    world = dict(
        base_effect   = float(request.args.get("base_effect",   BASE_EFFECT)),
        biased_gain   = float(request.args.get("biased_gain",   BIASED_GAIN)),
        sigma_opinion = float(request.args.get("sigma_opinion", SIGMA_OPINION)),
        sigma_survey  = float(request.args.get("sigma_survey",  SIGMA_SURVEY)),
    )

    import random as _rnd
    if bias == "random":
        bias = _rnd.choice(["neutral", "biased"])

    ep = generate_episode(bias=bias, seed=seed, T=T, **world)

    # Forecasters at every step
    steps = []
    alpha = 0.35
    for T_obs in range(1, T + 1):
        obs_evts  = ep["events"][:T_obs]
        obs_survs = ep["surveys"][:T_obs]
        nxt       = ep["events"][T_obs]
        gold_surv = ep["surveys"][T_obs]
        gold_opin = ep["opinion_traj"][T_obs]

        bayes = compute_bayes_forecast(
            events=obs_evts, surveys=obs_survs,
            next_event_polarity=nxt["polarity"], **world,
        )
        blind = compute_blind_forecast(
            events=obs_evts, surveys=obs_survs,
            next_event_polarity=nxt["polarity"], **world,
        )

        ema = float(obs_survs[0])
        for s in obs_survs[1:]:
            ema = alpha * s + (1 - alpha) * ema
        naive_pred = round(ema)

        steps.append({
            "T_obs":         T_obs,
            "obs_event":     obs_evts[-1],
            "obs_survey":    int(obs_survs[-1]),
            "obs_opinion":   round(ep["opinion_traj"][T_obs - 1], 1),
            "next_event":    nxt,
            "gold_survey":   int(gold_surv),
            "gold_opinion":  round(gold_opin, 1),
            "bayes_pred":    round(bayes["predicted_survey"], 1),
            "bayes_p_biased": round(bayes["p_biased"], 3),
            "blind_pred":    round(blind["predicted_survey"], 1),
            "naive_pred":    naive_pred,
        })

    # Look up LM predictions from batch data if requested
    lm_by_t: dict = {}
    if model:
        c    = _ensure_cache()
        recs = c.get("records", [])
        ep_id = f"{'neutral' if bias=='neutral' else 'biased'}_{seed:04d}"
        match = next(
            (r for r in recs
             if r["episode_id"] == ep_id and r["model"] == model and r["variant"] == variant),
            None,
        )
        if match:
            for s in match["steps"]:
                lm_by_t[s["T_obs"]] = s.get("lm_pred")

    return jsonify({
        "bias":         bias,
        "gain":         ep.get("gain"),
        "seed":         seed,
        "T":            T,
        "opinion_init": round(ep["opinion_init"], 1),
        "steps":        steps,
        "lm_by_t":      lm_by_t,
        "variant_text": VARIANTS.get(variant, ""),
        "world":        world,
    })


@app.route("/api/variants")
def api_variants():
    return jsonify({k: v for k, v in VARIANTS.items()})


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5052)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"\n  Exp 2 Batch Viewer  →  http://{args.host}:{args.port}")
    print(f"  Data dir: {_BATCHDIR}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
