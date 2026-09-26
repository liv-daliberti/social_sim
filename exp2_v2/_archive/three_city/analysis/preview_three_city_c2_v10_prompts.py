#!/usr/bin/env python3
"""Render the exact pre-collection v10 prompt manipulation with benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

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
_OUTDIR = _RUN / "preview"
_REPRESENTATIVE_ID = "c2v10_0000_k2"


def _prompt(arm: str) -> str:
    path = _DESIGN / f"tasks_c2_v10_{arm}.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return str(record["prompt"])
    raise ValueError(f"missing representative task: {arm}/{_REPRESENTATIVE_ID}")


def _wrapped(prompt: str, width: int = 148) -> str:
    lines = []
    for line in prompt.splitlines():
        if not line or line.startswith("|") or len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width))
    return "\n".join(lines)


def render(path: Path) -> None:
    c_only = _prompt("c_only")
    abc = _prompt("abc")
    clue = _prompt("abc_structural_clue")
    marker = "Below are records from two earlier cities in the same region."
    reference_block = marker + abc.split(marker, 1)[1].split("\nCITY C\n", 1)[0]
    assert clue.replace(STRUCTURAL_CHOICE_CLUE + "\n\n", "") == abc
    keys = {
        row["task_id"]: row
        for row in analysis._read_jsonl(_DESIGN / "answer_key_c2_v10.jsonl")
    }
    ks = list(PREFIX_LADDER)
    target = [
        analysis._baseline_mae(keys, method="target_only", k=k)
        for k in ks
    ]
    matched = [
        analysis._baseline_mae(keys, method="abc_shrinkage", k=k)
        for k in ks
    ]

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
            "The complete City C evidence and forecasting question remain "
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
        reference_block.strip(),
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
        color=analysis._ARM_COLOR["abc_structural_clue"],
        transform=difference_axes[2].transAxes,
    )
    difference_axes[2].text(
        0.0,
        0.86,
        "Same City A/B records as panel B, plus:\n\n" + STRUCTURAL_CHOICE_CLUE,
        ha="left",
        va="top",
        fontsize=8.4,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#EDF5FC",
            "edgecolor": analysis._ARM_COLOR["abc_structural_clue"],
            "linewidth": 1.0,
        },
        transform=difference_axes[2].transAxes,
    )

    for axis, arm, panel in zip(plot_axes, PROMPT_ARMS, ("A", "B", "C")):
        axis.plot(
            ks,
            target,
            color=analysis._TARGET,
            marker="o",
            markerfacecolor="white",
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            label="City C estimator",
        )
        axis.plot(
            ks,
            matched,
            color=analysis._ABC,
            marker="s",
            markerfacecolor="white",
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            label="Matched A/B/C estimator",
        )
        axis.set_title(
            f"({panel}) {analysis._ARM_LABEL[arm]}",
            fontsize=9.2,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=5,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8)
        axis.set_ylim(0, 3.8)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.5)
    plot_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8)
    for axis in plot_axes[1:]:
        axis.tick_params(labelleft=False)

    figure.legend(
        handles=plot_axes[0].get_legend_handles_labels()[0],
        labels=plot_axes[0].get_legend_handles_labels()[1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=7.4,
    )
    figure.text(
        0.5,
        0.995,
        "PRE-COLLECTION PROMPT PREVIEW — NO MODEL RESULTS",
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
            f"Exact frozen manipulation · representative task {_REPRESENTATIVE_ID} "
            "(round 2, two City C cases) · curves are preregistered estimator benchmarks"
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
        (
            "Panel B inserts the A/B block before CITY C; panel C also inserts "
            "the blue clue before ‘Think carefully.’"
        ),
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
        transform=base_axis.transAxes,
    )
    base_axis.text(
        0.0,
        0.935,
        _wrapped(c_only),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTDIR / "prompt_structure_preview.png",
    )
    args = parser.parse_args()
    render(args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
