#!/usr/bin/env python3
"""Render the Claude-only n=250 stable-response design and live curves."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from matplotlib.figure import Figure


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import render_coin_city_variable_regression as base
from engine.coin_city_stable_relationship_claude_n250 import (
    C_CASE_LEVELS,
    EXPERIMENT,
    FUTURE_CALLS,
    MODELS,
    PROMPT_ARMS,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
LIVE = RUN / "live"

# Reuse the approved three-panel live-structure layout while redirecting every
# data dependency to the separate v2 design. The v1 experiment remains intact.
base.C_CASE_LEVELS = C_CASE_LEVELS
base.EXPERIMENT = EXPERIMENT
base.MODELS = MODELS
base.PLANNED_CALLS = FUTURE_CALLS
base.PROMPT_ARMS = PROMPT_ARMS
base.RUN = RUN
base.DESIGN = DESIGN
base.RESPONSES = RESPONSES
base.LIVE = LIVE


def render(output: Path) -> dict:
    original_suptitle = Figure.suptitle
    original_text = Figure.text

    def stable_title(self, text, *args, **kwargs):
        if text == "Variable-predictor coin-to-city experiment":
            text = "Claude Opus 4.8 · fair n=250 stable-response design"
        return original_suptitle(self, text, *args, **kwargs)

    def stable_text(self, x, y, text, *args, **kwargs):
        if isinstance(text, str):
            text = text.replace(f" · {FUTURE_CALLS:,}-call ceiling", "")
            text = text.replace("City C cases 0–6", "City C cases 0–4")
            text = text.replace("C=0–6", "C=0–4")
            text = text.replace(
                "fixed 3 A cases + 3 B cases", "fixed 4 A cases + 4 B cases"
            )
            if text.startswith("Both dashed regressions are reconstructed"):
                text = (
                    "Yellow boxes are analyst benchmarks reconstructed only from "
                    "displayed rows; they are not included in any prompt. City C "
                    "regression is undefined at C=0."
                )
        return original_text(self, x, y, text, *args, **kwargs)

    Figure.suptitle = stable_title
    Figure.text = stable_text
    try:
        result = base.render(output)
    finally:
        Figure.suptitle = original_suptitle
        Figure.text = original_text

    result["analysis_revision"] = "stable_relationship_claude_n250_v4"
    result["llm_calls_made"] = result["received"]
    result["future_calls_if_run"] = FUTURE_CALLS
    (output.parent / "progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=LIVE / "live_structure.png"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh the live figure until interrupted or all calls are parsed.",
    )
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    while True:
        result = render(args.output)
        if not args.watch or result["parsed"] >= FUTURE_CALLS:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
