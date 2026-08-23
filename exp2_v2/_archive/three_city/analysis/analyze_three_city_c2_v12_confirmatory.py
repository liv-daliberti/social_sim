#!/usr/bin/env python3
"""Analyze the C2 v12 noisier structural-choice rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v11_confirmatory as base
from engine.three_city_c2_v12 import CASES_BY_PREFIX, PREFIX_LADDER, PROMPT_ARMS

_RUN = _ROOT / "data" / "three_city_c2_v12" / "full_k1_5_noisier_structural_choice"
_RESPONSES = _RUN / "responses"
_ANSWER_KEY = _RUN / "design" / "answer_key_c2_v12.jsonl"
_OUTDIR = _RUN / "analysis"
_EXPECTED = 120 * len(PREFIX_LADDER)

for name in (
    "_DISPLAY", "_MODEL_ORDER", "_MODEL_STYLE", "_ARM_LABEL", "_ARM_COLOR",
    "_TARGET", "_ABC", "_GRID", "plt", "Line2D", "_read_jsonl",
    "_latest_responses", "_score_rows", "_paired", "_statistics",
    "_baseline_mae", "_make_structure_figure",
):
    globals()[name] = getattr(base, name)


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
        expected = _EXPECTED * len(PROMPT_ARMS) * len(_MODEL_ORDER)
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
    figure = args.outdir / "three_city_c2_v12_structure.png"
    _make_structure_figure(
        figure, curves=curves, keys=keys, models=_MODEL_ORDER,
        interim=args.allow_incomplete,
    )
    result = {
        "experiment": "three_city_c2_v12_noisier_structural_choice",
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
    (args.outdir / "three_city_c2_v12_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {figure} and {figure.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
