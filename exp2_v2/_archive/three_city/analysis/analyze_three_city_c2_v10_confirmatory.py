#!/usr/bin/env python3
"""Analyze the C2 v10 structural-choice-clue paper rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v9_confirmatory as base
from engine.three_city_c2_v10 import CASES_BY_PREFIX, PREFIX_LADDER, PROMPT_ARMS

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v10"
    / "full_k1_5_structural_choice"
)
_RESPONSES = _RUN / "responses"
_ANSWER_KEY = _RUN / "design" / "answer_key_c2_v10.jsonl"
_OUTDIR = _RUN / "analysis"
_EXPECTED_PER_CELL = 120 * len(PREFIX_LADDER)

base.PROMPT_ARMS = PROMPT_ARMS
base._ARM_LABEL = {
    "c_only": "City C evidence only",
    "abc": "A/B/C information",
    "abc_structural_clue": "A/B/C + structural-choice clue",
}
base._ARM_COLOR = {
    "c_only": "#E68613",
    "abc": "#168A72",
    "abc_structural_clue": "#267CB5",
}

_DISPLAY = base._DISPLAY
_MODEL_ORDER = base._MODEL_ORDER
_MODEL_STYLE = base._MODEL_STYLE
_ARM_LABEL = base._ARM_LABEL
_ARM_COLOR = base._ARM_COLOR
_TARGET = base._TARGET
_ABC = base._ABC
_GRID = base._GRID
plt = base.plt
Line2D = base.Line2D

_read_jsonl = base._read_jsonl
_latest_responses = base._latest_responses
_score_rows = base._score_rows
_paired = base._paired
_statistics_v9 = base._statistics
_baseline_mae = base._baseline_mae
_make_structure_figure = base._make_structure_figure


def _statistics(paired, *, draws: int, seed: int):
    """Adapt the audited v9 bootstrap to the renamed v10 third arm."""
    translated = []
    for pair in paired:
        value = dict(pair)
        value["abc_relevance"] = value["abc_structural_clue"]
        translated.append(value)
    original_arms = base.PROMPT_ARMS
    try:
        base.PROMPT_ARMS = ("c_only", "abc", "abc_relevance")
        result = _statistics_v9(translated, draws=draws, seed=seed)
    finally:
        base.PROMPT_ARMS = original_arms
    result["arms"]["abc_structural_clue"] = result["arms"].pop("abc_relevance")
    if "relevance_minus_abc" in result["contrasts"]:
        result["contrasts"]["structural_clue_minus_abc"] = result["contrasts"].pop("relevance_minus_abc")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=_RESPONSES)
    parser.add_argument("--answer-key", type=Path, default=_ANSWER_KEY)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    latest = _latest_responses(sorted(args.responses.glob("responses_*.jsonl")))
    keys = {row["task_id"]: row for row in _read_jsonl(args.answer_key)}
    rows = _score_rows(latest, keys)
    if not args.allow_incomplete:
        expected = _EXPECTED_PER_CELL * len(PROMPT_ARMS) * len(_MODEL_ORDER)
        if len(rows) != expected:
            raise SystemExit(f"collection incomplete: {len(rows)}/{expected} parsed")
    curves = {}
    for model_index, model in enumerate(_MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = _statistics(
                _paired(rows, model=model, k=k),
                draws=args.draws,
                seed=20260805 + 1000 * model_index + k,
            )
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure_path = args.outdir / "three_city_c2_v10_structure.png"
    _make_structure_figure(
        figure_path,
        curves=curves,
        keys=keys,
        models=_MODEL_ORDER,
        interim=args.allow_incomplete,
    )
    result = {
        "experiment": "three_city_c2_v10_structural_choice_clue",
        "status": "interim" if args.allow_incomplete else "complete",
        "parsed_rows": len(rows),
        "cases_by_round": CASES_BY_PREFIX,
        "curves": curves,
        "baseline_mae": {
            method: {
                str(k): _baseline_mae(keys, method=method, k=k)
                for k in PREFIX_LADDER
            }
            for method in ("target_only", "abc_shrinkage")
        },
    }
    (args.outdir / "three_city_c2_v10_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {figure_path} and {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
