#!/usr/bin/env python3
"""Continuously render the exact prompt structure above live v11 curves."""

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

from analysis import analyze_three_city_c2_v11_confirmatory as analysis
from analysis import live_monitor_three_city_c2_v11 as collector_live
from analysis.render_three_city_c2_v11_paper_prompt_figure import render_exact
from engine.three_city_c2_v11 import PREFIX_LADDER, PROMPT_ARMS

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v11"
    / "full_k1_5_identifiable_structural_choice"
)
_DESIGN = _RUN / "design"
_OUTDIR = _RUN / "live"
_EXPECTED = 120 * len(PREFIX_LADDER) * len(PROMPT_ARMS) * len(analysis._MODEL_ORDER)


def _render(*, draws: int, iteration: int) -> None:
    latest = collector_live._latest()
    keys = {
        row["task_id"]: row
        for row in collector_live._read_jsonl_tolerant(
            _DESIGN / "answer_key_c2_v11.jsonl"
        )
    }
    rows = analysis._score_rows(latest, keys)
    curves = {}
    for model_index, model in enumerate(analysis._MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = analysis._statistics(
                analysis._paired(rows, model=model, k=k),
                draws=draws,
                seed=20260805 + 1000 * model_index + k,
            )
    updated = datetime.now(timezone.utc).isoformat()
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    output = _OUTDIR / "live_structure_annotated.png"
    render_exact(
        output,
        curves=curves,
        models=analysis._MODEL_ORDER,
        interim=len(rows) < _EXPECTED,
        progress_label=(
            f"{len(rows):,}/{_EXPECTED:,} parsed · updated {updated} · "
            "incomplete curves are descriptive only"
        ),
    )
    progress = {
        "experiment": "three_city_c2_v11_identifiable_structural_choice",
        "status": "interim" if len(rows) < _EXPECTED else "complete",
        "iteration": iteration,
        "updated_at": updated,
        "expected": _EXPECTED,
        "received": len(latest),
        "parsed": len(rows),
        "figure": output.name,
    }
    (_OUTDIR / "annotated_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{updated} annotated received={len(latest)}/{_EXPECTED} "
        f"parsed={len(rows)}",
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
