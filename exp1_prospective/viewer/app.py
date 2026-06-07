#!/usr/bin/env python3
"""Flask viewer for Exp 1 forecast conversations.

Usage (from exp1_prospective/):
    python viewer/app.py
    python viewer/app.py --port 5050
"""

import argparse
import json
from pathlib import Path

from flask import Flask, render_template, jsonify

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

def _load_forecasts() -> tuple[list[dict], str]:
    """Load all forecast records from all JSONL files, newest file first.
    Returns (records, model_name)."""
    records: list[dict] = []
    seen: set[str] = set()
    model_name = "gpt-5.4"

    files = sorted(_FORECAST_DIR.glob("forecasts_*.jsonl"), reverse=True)
    for path in files:
        mani = path.with_suffix("").with_suffix(".manifest.json")
        if mani.exists():
            try:
                m = json.loads(mani.read_text())
                if m.get("model"):
                    model_name = m["model"]
            except Exception:
                pass

        latest: dict[str, dict] = {}
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
            if tid not in seen:
                seen.add(tid)
                records.append(rec)

    records.sort(key=lambda r: (r.get("yes_prob") is None, -(r.get("yes_prob") or 0)))
    return records, model_name


def _attach_price_history(rec: dict) -> dict:
    """Attach market price history to a forecast record (mutates a copy)."""
    mid = rec.get("market_id", "")
    h = _market_histories.get(mid, {})
    rec = dict(rec)
    rec["price_history"]  = h.get("price_history", [])
    rec["volume_history"] = h.get("volume_history", [])
    return rec


# ── routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    records, model_name = _load_forecasts()
    return render_template("index.html", records=records, model_name=model_name)


@app.route("/api/forecast/<task_id>")
def get_forecast(task_id: str):
    records, _ = _load_forecasts()
    for rec in records:
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


def _load_updated_forecasts(task_id: str) -> list[dict]:
    """Return all updated-forecast records for a task_id, deduplicated by update_id."""
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
                    uid = rec.get("update_id", "")
                    if uid and uid not in seen:
                        seen.add(uid)
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    return records


@app.route("/api/updated_forecasts/<task_id>")
def get_updated_forecasts(task_id: str):
    return jsonify(_load_updated_forecasts(task_id))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"Starting viewer at http://{args.host}:{args.port}")
    print(f"Data directory: {_FORECAST_DIR}")
    app.run(host=args.host, port=args.port, debug=True)
