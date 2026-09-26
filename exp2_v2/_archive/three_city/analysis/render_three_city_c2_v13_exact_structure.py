#!/usr/bin/env python3
"""Render the approved exact-text layout for frozen v13 tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v13_confirmatory as analysis
from analysis import render_three_city_c2_v12_exact_structure as layout
from engine.three_city_c2_v13 import PREFIX_LADDER, PROMPT_ARMS
from eval.build_three_city_c2_v13_tasks import STRUCTURAL_CHOICE_CLUE

_RUN = _ROOT / "data" / "three_city_c2_v13" / "full_k1_5_two_regime_structural_choice"
_DESIGN = _RUN / "design"
_OUTDIR = _RUN / "presentation"
_REPRESENTATIVE_ID = "c2v13_0000_k2"

layout._RUN = _RUN
layout._OUTDIR = _OUTDIR
layout.PREFIX_LADDER = PREFIX_LADDER
layout.PROMPT_ARMS = PROMPT_ARMS
layout.analysis = analysis
layout.base._RUN = _RUN
layout.base._DESIGN = _DESIGN
layout.base._OUTDIR = _OUTDIR
layout.base._REPRESENTATIVE_ID = _REPRESENTATIVE_ID
layout.base.analysis = analysis
layout.base.PREFIX_LADDER = PREFIX_LADDER
layout.base.PROMPT_ARMS = PROMPT_ARMS
layout.base.STRUCTURAL_CHOICE_CLUE = STRUCTURAL_CHOICE_CLUE


def _read_task(arm: str) -> dict:
    path = _DESIGN / f"tasks_c2_v13_{arm}.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return record
    raise ValueError(f"missing representative task: {arm}")


layout._read_task = _read_task
layout.base._read_task = _read_task
_original_read_jsonl = analysis._read_jsonl


def _read_jsonl(path: Path):
    if path.name in ("answer_key_c2_v11.jsonl", "answer_key_c2_v12.jsonl"):
        path = _DESIGN / "answer_key_c2_v13.jsonl"
    return _original_read_jsonl(path)


analysis._read_jsonl = _read_jsonl
_original_baseline_mae = analysis._baseline_mae


def _baseline_mae(keys, *, method: str, k: int):
    if method == "abc_shrinkage":
        method = "abc_no_structure"
    return _original_baseline_mae(keys, method=method, k=k)


analysis._baseline_mae = _baseline_mae

_FORMULA_BY_ARM = {
    "c_only": (
        r"City C regression:  $\hat\beta_C="
        r"\frac{\sum_i x_{Ci}(y_{Ci}-50)}{\sum_i x_{Ci}^2}$"
        "\n"
        r"Prediction:  $\hat y_C(+8)=50+8\hat\beta_C$"
    ),
    "abc": (
        r"A/B/C pooling:  $\hat\beta_{AB}="
        r"(\hat\beta_A+\hat\beta_B)/2$"
        "\n"
        r"$\hat\beta_{pool}=w_k\hat\beta_{AB}+(1-w_k)\hat\beta_C$,  "
        r"$w_k=2/(2+n_C(k))$"
    ),
    "abc_structural_clue": (
        r"Structural choice:  $j^*=\arg\min_{j\in\{A,B\}}|p_C-p_j|$"
        "\n"
        r"$\hat\beta_{struct}=w_k\hat\beta_{j^*}+(1-w_k)\hat\beta_C$,  "
        r"$w_k=2/(2+n_C(k))$"
    ),
}


def render(
    path: Path,
    *,
    curves=None,
    models=(),
    interim: bool = False,
    progress_label: str | None = None,
) -> None:
    layout.render(
        path,
        curves=curves,
        models=models,
        interim=interim,
        progress_label=progress_label,
        structural_benchmark_method="abc_structural",
        matched_benchmark_label="A/B-average estimator",
        structural_benchmark_label="Structural-choice estimator",
        formula_by_arm=_FORMULA_BY_ARM,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=_OUTDIR / "prompt_structure_paper.png",
    )
    args = parser.parse_args()
    render(args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
