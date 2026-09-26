#!/usr/bin/env python3
"""Presentation-only live figure attaching exact v9 prompt changes to curves.

This companion renderer is intentionally outside the frozen task manifest. It
reads the frozen prompts and responses but never changes experimental inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v9_confirmatory as analysis
from analysis import live_monitor_three_city_c2_v9 as live
from engine.three_city_c2_v9 import PREFIX_LADDER, PROMPT_ARMS
from eval.build_three_city_c2_v9_tasks import RELEVANCE_HINT_BLOCK

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v9"
    / "full_k1_5_clean_information"
)
_DESIGN = _RUN / "design"
_OUTDIR = _RUN / "live"
_REPRESENTATIVE_ID = "c2v9_0000_k2"
_EXPECTED_TOTAL = 120 * len(PREFIX_LADDER) * len(PROMPT_ARMS) * 3


def _representative_prompt(arm: str) -> str:
    path = _DESIGN / f"tasks_c2_v9_{arm}.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return str(record["prompt"])
    raise ValueError(f"missing representative task: {arm}/{_REPRESENTATIVE_ID}")


def _prompt_material() -> tuple[str, str]:
    c_only = _representative_prompt("c_only")
    abc = _representative_prompt("abc")
    marker = "Below are records from two earlier cities in the same region."
    reference_block = marker + abc.split(marker, 1)[1].split("\nCITY C\n", 1)[0]
    return c_only, reference_block.strip()


def _live_statistics(draws: int) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    latest = live._latest()
    keys = {
        row["task_id"]: row
        for row in live._read_jsonl_tolerant(_DESIGN / "answer_key_c2_v9.jsonl")
    }
    rows = analysis._score_rows(latest, keys)
    curves: dict[str, Any] = {}
    for model_index, model in enumerate(analysis._MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = analysis._statistics(
                analysis._paired(rows, model=model, k=k),
                draws=draws,
                seed=20260804 + 1000 * model_index + k,
            )
    received = len(latest)
    parsed = sum(record.get("predicted_poll") is not None for record in latest.values())
    return curves, keys, received, parsed


def _wrapped_prompt(prompt: str, width: int = 148) -> str:
    lines = []
    for line in prompt.splitlines():
        if not line or line.startswith("|") or len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width))
    return "\n".join(lines)


def _draw(path: Path, *, draws: int) -> tuple[int, int]:
    curves, keys, received, parsed = _live_statistics(draws)
    base_prompt, reference_block = _prompt_material()
    analysis.plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure = analysis.plt.figure(figsize=(13.6, 12.4))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(3.0, 2.8, 6.4),
        hspace=0.16,
        wspace=0.08,
        left=0.06,
        right=0.985,
        top=0.87,
        bottom=0.035,
    )
    difference_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    plot_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    base_axis = figure.add_subplot(grid[2, :])
    for axis in difference_axes + [base_axis]:
        axis.set_axis_off()

    difference_axes[0].text(
        0.0,
        1.0,
        "INFORMATION OMITTED",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=analysis._ARM_COLOR["c_only"],
        transform=difference_axes[0].transAxes,
    )
    difference_axes[0].text(
        0.0,
        0.86,
        (
            "City A and City B records are not shown.\n\n"
            "The complete City C material and forecasting question remain "
            "identical and are printed in the shared block below."
        ),
        ha="left",
        va="top",
        fontsize=8.4,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#FFF5E8",
            "edgecolor": analysis._ARM_COLOR["c_only"],
            "linewidth": 1.0,
        },
        transform=difference_axes[0].transAxes,
    )

    difference_axes[1].text(
        0.0,
        1.0,
        "INFORMATION ADDED (EXACT REPRESENTATIVE BLOCK)",
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=analysis._ARM_COLOR["abc"],
        transform=difference_axes[1].transAxes,
    )
    difference_axes[1].text(
        0.0,
        0.93,
        reference_block,
        ha="left",
        va="top",
        fontsize=4.45,
        family="monospace",
        linespacing=1.05,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F0FAF7",
            "edgecolor": analysis._ARM_COLOR["abc"],
            "linewidth": 0.9,
        },
        transform=difference_axes[1].transAxes,
    )

    difference_axes[2].text(
        0.0,
        1.0,
        "ONLY B→C PROMPT CHANGE (EXACT TEXT)",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=analysis._ARM_COLOR["abc_relevance"],
        transform=difference_axes[2].transAxes,
    )
    difference_axes[2].text(
        0.0,
        0.86,
        (
            "Same City A/B records as panel B, plus:\n\n"
            + RELEVANCE_HINT_BLOCK
        ),
        ha="left",
        va="top",
        fontsize=8.4,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#EDF5FC",
            "edgecolor": analysis._ARM_COLOR["abc_relevance"],
            "linewidth": 1.0,
        },
        transform=difference_axes[2].transAxes,
    )

    ks = list(PREFIX_LADDER)
    target = [analysis._baseline_mae(keys, method="target_only", k=k) for k in ks]
    abc = [analysis._baseline_mae(keys, method="abc_shrinkage", k=k) for k in ks]
    upper = max(target + abc) * 1.14
    for axis, arm, panel in zip(plot_axes, PROMPT_ARMS, ("A", "B", "C")):
        for model in analysis._MODEL_ORDER:
            color, marker = analysis._MODEL_STYLE[model]
            estimates = [curves[model][str(k)]["arms"][arm]["mae"] for k in ks]
            lows = [curves[model][str(k)]["arms"][arm]["mae_ci"][0] for k in ks]
            highs = [curves[model][str(k)]["arms"][arm]["mae_ci"][1] for k in ks]
            finite = [value for value in estimates + highs if math.isfinite(value)]
            if finite:
                upper = max(upper, max(finite) * 1.08)
            axis.plot(
                ks,
                estimates,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.1,
                linewidth=1.7,
                markersize=4.5,
                zorder=4,
            )
            axis.fill_between(ks, lows, highs, color=color, alpha=0.10, linewidth=0)
        axis.plot(ks, target, color=analysis._TARGET, linestyle=(0, (3, 2)), linewidth=1.35)
        axis.plot(ks, abc, color=analysis._ABC, linestyle=(0, (3, 2)), linewidth=1.35)
        axis.set_title(
            f"({panel}) {analysis._ARM_LABEL[arm]}",
            fontsize=9.2,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=5,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.5)
    plot_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8)
    for index, axis in enumerate(plot_axes):
        axis.set_ylim(0, upper)
        if index:
            axis.tick_params(labelleft=False)

    handles = [
        analysis.Line2D(
            [0],
            [0],
            color=analysis._MODEL_STYLE[model][0],
            marker=analysis._MODEL_STYLE[model][1],
            linewidth=1.7,
            markersize=4.5,
            label=analysis._DISPLAY[model],
        )
        for model in analysis._MODEL_ORDER
    ]
    handles.extend(
        [
            analysis.Line2D(
                [0], [0], color=analysis._TARGET, linestyle=(0, (3, 2)),
                linewidth=1.35, label="City C estimator"
            ),
            analysis.Line2D(
                [0], [0], color=analysis._ABC, linestyle=(0, (3, 2)),
                linewidth=1.35, label="Matched A/B/C estimator"
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(handles),
        frameon=False,
        fontsize=7.1,
        columnspacing=0.9,
    )
    figure.text(
        0.5,
        0.995,
        "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#A61B1B",
    )
    figure.text(
        0.5,
        0.968,
        (
            f"Exact frozen prompt manipulation · representative task {_REPRESENTATIVE_ID} "
            f"(round 2, two City C cases) · live progress {parsed}/{_EXPECTED_TOTAL} parsed"
        ),
        ha="center",
        fontsize=7.8,
        color="#555555",
    )
    base_axis.text(
        0.0,
        0.985,
        "FULL SHARED CITY C PROMPT (EXACT REPRESENTATIVE TASK)",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#333333",
        transform=base_axis.transAxes,
    )
    base_axis.text(
        1.0,
        0.985,
        "Panel B inserts the A/B block before CITY C; panel C also inserts the blue sentence before ‘Think carefully.’",
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
        transform=base_axis.transAxes,
    )
    base_axis.text(
        0.0,
        0.935,
        _wrapped_prompt(base_prompt),
        ha="left",
        va="top",
        fontsize=5.25,
        family="monospace",
        linespacing=1.12,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#FAFAFA",
            "edgecolor": "#777777",
            "linewidth": 0.8,
        },
        transform=base_axis.transAxes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    analysis.plt.close(figure)
    return received, parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTDIR / "live_structure_prompt_annotated.png",
    )
    args = parser.parse_args()
    while True:
        received, parsed = _draw(args.output, draws=args.draws)
        print(
            f"prompt_figure received={received}/{_EXPECTED_TOTAL} parsed={parsed}",
            flush=True,
        )
        if not args.watch or received >= _EXPECTED_TOTAL:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
