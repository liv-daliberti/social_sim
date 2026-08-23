#!/usr/bin/env python3
"""Render the interim v12 dashboard and live structure curve."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v12_confirmatory as analysis
from analysis.live_monitor_three_city_c2_v11 import _read_jsonl_tolerant
from engine.three_city_c2_v12 import PREFIX_LADDER, PROMPT_ARMS

_RUN = _ROOT / "data" / "three_city_c2_v12" / "full_k1_5_noisier_structural_choice"
_DESIGN = _RUN / "design"
_RESPONSES = _RUN / "responses"
_OUTDIR = _RUN / "live"
_EXPECTED_CELL = 120 * len(PREFIX_LADDER)


def _latest():
    latest = {}
    for path in sorted(_RESPONSES.glob("responses_*.jsonl")):
        for record in _read_jsonl_tolerant(path):
            try:
                key = (record["model"], record["arm"], record["task_id"])
            except KeyError:
                continue
            priority = 2 if record.get("predicted_poll") is not None else 1
            old = latest.get(key)
            old_priority = -1 if old is None else (2 if old.get("predicted_poll") is not None else 1)
            if priority >= old_priority:
                latest[key] = record
    return latest


def render(draws: int, iteration: int) -> None:
    latest = _latest()
    keys = {
        row["task_id"]: row for row in _read_jsonl_tolerant(
            _DESIGN / "answer_key_c2_v12.jsonl"
        )
    }
    rows = analysis._score_rows(latest, keys)
    curves = {}
    for model_index, model in enumerate(analysis._MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = analysis._statistics(
                analysis._paired(rows, model=model, k=k),
                draws=draws, seed=20260805 + 1000 * model_index + k,
            )
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    analysis._make_structure_figure(
        _OUTDIR / "live_structure.png", curves=curves, keys=keys,
        models=analysis._MODEL_ORDER, interim=True,
    )
    updated = datetime.now(timezone.utc).isoformat()
    progress = {
        "experiment": "three_city_c2_v12_noisier_structural_choice",
        "status": "interim_incomplete_not_confirmatory",
        "updated_at": updated,
        "iteration": iteration,
        "expected_total_calls": _EXPECTED_CELL * len(PROMPT_ARMS) * len(analysis._MODEL_ORDER),
        "received_total": len(latest),
        "parsed_total": len(rows),
    }
    (_OUTDIR / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(f"{updated} received={len(latest)}/5400 parsed={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--draws", type=int, default=300)
    args = parser.parse_args()
    iteration = 0
    while True:
        iteration += 1
        render(args.draws, iteration)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
