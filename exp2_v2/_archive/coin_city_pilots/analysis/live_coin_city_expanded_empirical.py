#!/usr/bin/env python3
"""Prompt-only empirical live/final analysis for the expanded run."""

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
from analysis.empirical_coin_city_estimators import empirical_analysis_keys


RUN = ROOT / "data" / "coin_city_expanded_aligned_v1"
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
LIVE = RUN / "live"
MANIFEST = json.loads((DESIGN / "manifest.json").read_text())
EXPERIMENT = str(MANIFEST["experiment"])
PLANNED_CALLS = int(MANIFEST["planned_calls"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def draw(output: Path) -> dict:
    latest = base._latest_responses(RESPONSES)
    original_keys = {
        row["task_id"]: row
        for row in base._read_jsonl(DESIGN / "answer_key.jsonl")
    }
    keys, empirical_records = empirical_analysis_keys(DESIGN, original_keys)
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
        title="Expanded coin-to-city test: empirical regressions only",
        subtitle_detail=(
            "100 fresh episodes · 100/100 aligned clues · regressions use "
            "displayed rows only · curves use one common completed subset"
        ),
    )
    result = {
        "experiment": EXPERIMENT,
        "analysis_revision": "prompt_only_empirical_v1",
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "expected": PLANNED_CALLS,
        "received": received,
        "parsed": parsed,
        "parse_failures": received - parsed,
        "view": "condition_panels_with_two_prompt_only_empirical_regressions",
        "context_alignment": "100_percent_correct_direction",
        "estimator_definitions": {
            "city_c_regression": "mean of visible City C rows",
            "abc_no_context_regression": (
                "empirical random-intercept partial pooling; reference grand mean, "
                "within variance, and between variance estimated from visible A/B rows"
            ),
            "forbidden_inputs": [
                "simulation variances",
                "latent response type",
                "cue reliability",
                "unseen future City C rows",
                "target truth during estimation",
            ],
        },
        "scoring_only_field": "gold_expected_poll",
        "discarded_before_scoring": [
            "original simulator baselines",
            "target_high",
            "high_reference_city",
            "cue_high",
            "cue_correct",
        ],
        "curves": curves,
        "figure": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output.parent / "empirical_progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_jsonl(output.parent / "empirical_benchmarks.jsonl", empirical_records)
    print(
        f"{result['updated_at']} {status}: empirical-only "
        f"received={received}/{PLANNED_CALLS} parsed={parsed}; wrote {output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--wait-for-controller", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=LIVE / "live_structure.png"
    )
    args = parser.parse_args()
    while True:
        result = draw(args.output)
        if result["received"] >= result["expected"]:
            if not args.wait_for_controller:
                break
            status_path = RUN / "controller_exit_status.txt"
            if status_path.exists():
                # The frozen launcher performs one legacy render immediately
                # before writing its status; redraw empirically after that.
                time.sleep(3.0)
                draw(args.output)
                break
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
