#!/usr/bin/env python3
"""Render the interim dashboard for the C2 v10 paper rerun."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v10_confirmatory as analysis
from engine.three_city_c2_v10 import PREFIX_LADDER, PROMPT_ARMS
from eval.build_three_city_c2_v10_tasks import STRUCTURAL_CHOICE_CLUE

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v10"
    / "full_k1_5_structural_choice"
)
_DESIGN = _RUN / "design"
_RESPONSES = _RUN / "responses"
_OUTDIR = _RUN / "live"
_EXPECTED_PER_CELL = 120 * len(PREFIX_LADDER)


def _read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _latest() -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(_RESPONSES.glob("responses_*.jsonl")):
        for record in _read_jsonl_tolerant(path):
            try:
                key = (record["model"], record["arm"], record["task_id"])
            except KeyError:
                continue
            priority = 2 if record.get("predicted_poll") is not None else 1
            old = latest.get(key)
            old_priority = -1 if old is None else (
                2 if old.get("predicted_poll") is not None else 1
            )
            if priority >= old_priority:
                latest[key] = record
    return latest


def _render(*, draws: int, iteration: int) -> None:
    latest = _latest()
    keys = {
        row["task_id"]: row
        for row in _read_jsonl_tolerant(_DESIGN / "answer_key_c2_v10.jsonl")
    }
    rows = analysis._score_rows(latest, keys)
    curves = {}
    cells = []
    for model_index, model in enumerate(analysis._MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = analysis._statistics(
                analysis._paired(rows, model=model, k=k),
                draws=draws,
                seed=20260805 + 1000 * model_index + k,
            )
        for arm in PROMPT_ARMS:
            records = [
                record
                for (record_model, record_arm, _), record in latest.items()
                if record_model == model and record_arm == arm
            ]
            received = len(records)
            parsed = sum(
                record.get("predicted_poll") is not None
                for record in records
            )
            cells.append(
                {
                    "model": model,
                    "model_label": analysis._DISPLAY[model],
                    "arm": arm,
                    "arm_label": analysis._ARM_LABEL[arm],
                    "received": received,
                    "parsed": parsed,
                    "expected": _EXPECTED_PER_CELL,
                    "parse_rate": parsed / received if received else None,
                }
            )
    updated = datetime.now(timezone.utc).isoformat()
    received_total = sum(cell["received"] for cell in cells)
    parsed_total = sum(cell["parsed"] for cell in cells)
    expected_total = _EXPECTED_PER_CELL * len(PROMPT_ARMS) * len(analysis._MODEL_ORDER)
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    analysis._make_structure_figure(
        _OUTDIR / "live_structure.png",
        curves=curves,
        keys=keys,
        models=analysis._MODEL_ORDER,
        interim=True,
    )
    progress = {
        "experiment": "three_city_c2_v10_structural_choice_clue",
        "status": "interim_incomplete_not_confirmatory",
        "updated_at": updated,
        "iteration": iteration,
        "expected_total_calls": expected_total,
        "received_total": received_total,
        "parsed_total": parsed_total,
        "cells": cells,
    }
    (_OUTDIR / "progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n"
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(cell['model_label'])}</td>"
        f"<td>{html.escape(cell['arm_label'])}</td>"
        f"<td>{cell['received']}/{cell['expected']}</td>"
        f"<td>{cell['parsed']}</td>"
        "</tr>"
        for cell in cells
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>C2 v10 live</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1200px}}img{{max-width:100%}}
table{{border-collapse:collapse}}td,th{{padding:.35rem .7rem;border:1px solid #ccc}}
.warn{{color:#a61b1b;font-weight:700}}</style></head>
<body><p class="warn">INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY</p>
<h1>Structural-choice clue paper rerun</h1>
<p>{received_total}/{expected_total} received; {parsed_total} parsed. Updated {html.escape(updated)}.</p>
<img src="live_structure.png?v={iteration}" alt="Live structure curve">
<h2>Exact third-arm clue</h2><blockquote>{html.escape(STRUCTURAL_CHOICE_CLUE)}</blockquote>
<h2>Progress</h2><table><tr><th>Model</th><th>Arm</th><th>Received</th><th>Parsed</th></tr>{table_rows}</table>
</body></html>"""
    (_OUTDIR / "index.html").write_text(page)
    print(
        f"{updated} received={received_total}/{expected_total} parsed={parsed_total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--draws", type=int, default=300)
    args = parser.parse_args()
    iteration = 0
    while True:
        iteration += 1
        _render(draws=args.draws, iteration=iteration)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
