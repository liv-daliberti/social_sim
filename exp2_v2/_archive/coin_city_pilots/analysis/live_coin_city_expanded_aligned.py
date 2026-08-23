#!/usr/bin/env python3
"""Live/final graph for the fresh 100-episode always-aligned expansion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis import analyze_coin_city_llm_pilot as base
from analysis import render_coin_city_llm_pilot_by_condition as view
from engine.coin_city_expanded_aligned import (
    EXPERIMENT,
    PLANNED_CALLS,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
LIVE = RUN / "live"


def draw(output: Path) -> dict:
    latest = base._latest_responses(RESPONSES)
    keys = {
        row["task_id"]: row
        for row in base._read_jsonl(DESIGN / "answer_key.jsonl")
    }
    prompt_sections = json.loads(
        (DESIGN / "example_prompt_sections.json").read_text()
    )
    curves = view._globally_matched(latest, keys)
    received = len(latest)
    parsed = sum(row.get("predicted_poll") is not None for row in latest.values())
    if received == parsed == PLANNED_CALLS:
        status = "complete"
    elif received == PLANNED_CALLS:
        status = "complete_with_parse_failures"
    else:
        status = "interim_incomplete"
    view._render(
        output,
        curves=curves,
        prompt_sections=prompt_sections,
        received=received,
        expected_calls=PLANNED_CALLS,
        title="Expanded coin-to-city test: context always points correctly",
        subtitle_detail=(
            "100 fresh episodes · 100/100 aligned clues · "
            "all plotted curves use one common model × condition subset"
        ),
    )
    result = {
        "experiment": EXPERIMENT,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "expected": PLANNED_CALLS,
        "received": received,
        "parsed": parsed,
        "parse_failures": received - parsed,
        "view": "condition_panels_with_two_regression_references",
        "context_alignment": "100_percent_correct_direction",
        "curves": curves,
        "figure": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{result['updated_at']} {status}: received={received}/{PLANNED_CALLS} "
        f"parsed={parsed}; wrote {output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument(
        "--output", type=Path, default=LIVE / "live_structure.png"
    )
    args = parser.parse_args()
    while True:
        result = draw(args.output)
        if not args.watch or result["received"] >= result["expected"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
