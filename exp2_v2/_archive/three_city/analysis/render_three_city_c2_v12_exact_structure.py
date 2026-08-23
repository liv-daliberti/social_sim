#!/usr/bin/env python3
"""Render the approved exact-text structure layout for frozen v12 tasks."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v12_confirmatory as analysis
from analysis import render_three_city_c2_v11_paper_prompt_figure as base
from engine.three_city_c2_v12 import PREFIX_LADDER, PROMPT_ARMS

_RUN = _ROOT / "data" / "three_city_c2_v12" / "full_k1_5_noisier_structural_choice"
_OUTDIR = _RUN / "presentation"

base._RUN = _RUN
base._DESIGN = _RUN / "design"
base._OUTDIR = _OUTDIR
base._REPRESENTATIVE_ID = "c2v12_0000_k2"
base.analysis = analysis
base.PREFIX_LADDER = PREFIX_LADDER
base.PROMPT_ARMS = PROMPT_ARMS


def _read_task(arm: str) -> dict:
    path = base._DESIGN / f"tasks_c2_v12_{arm}.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] == base._REPRESENTATIVE_ID:
            return record
    raise ValueError(f"missing representative task: {arm}")


base._read_task = _read_task
_original_read_jsonl = analysis._read_jsonl


def _read_jsonl(path: Path):
    if path.name == "answer_key_c2_v11.jsonl":
        path = path.with_name("answer_key_c2_v12.jsonl")
    return _original_read_jsonl(path)


analysis._read_jsonl = _read_jsonl


def render(
    path: Path,
    *,
    curves=None,
    models=(),
    interim: bool = False,
    progress_label: str | None = None,
    structural_benchmark_method: str | None = None,
    matched_benchmark_label: str = "Matched A/B/C estimator",
    structural_benchmark_label: str = "Structural-choice estimator",
    formula_by_arm: dict[str, str] | None = None,
) -> None:
    tasks = {arm: _read_task(arm) for arm in PROMPT_ARMS}
    shared_prompt = str(tasks["c_only"]["prompt"])
    abc_prompt = str(tasks["abc"]["prompt"])
    clue_prompt = str(tasks["abc_structural_clue"]["prompt"])
    clue = base.STRUCTURAL_CHOICE_CLUE
    assert clue_prompt.replace(clue + "\n\n", "") == abc_prompt
    marker = "Below are records from two earlier cities in the same region."
    reference_block = (
        marker + abc_prompt.split(marker, 1)[1].split("\nCITY C\n", 1)[0]
    ).strip()

    keys = {
        row["task_id"]: row
        for row in analysis._read_jsonl(
            base._DESIGN / "answer_key_c2_v12.jsonl"
        )
    }
    ks = list(PREFIX_LADDER)
    target = [
        analysis._baseline_mae(keys, method="target_only", k=k) for k in ks
    ]
    matched = [
        analysis._baseline_mae(keys, method="abc_shrinkage", k=k) for k in ks
    ]
    structural = (
        [
            analysis._baseline_mae(
                keys, method=structural_benchmark_method, k=k
            )
            for k in ks
        ]
        if structural_benchmark_method is not None
        else None
    )

    analysis.plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure = analysis.plt.figure(
        figsize=(16.0, 11.2 if formula_by_arm is not None else 10.0)
    )
    grid = figure.add_gridspec(
        4 if formula_by_arm is not None else 3,
        3,
        height_ratios=(2.2, 2.75, 3.2, 0.8)
        if formula_by_arm is not None
        else (2.2, 3.1, 3.4),
        hspace=0.09 if formula_by_arm is not None else 0.075,
        wspace=0.10,
        left=0.045,
        right=0.985,
        top=0.92,
        bottom=0.085,
    )
    shared_axis = figure.add_subplot(grid[0, :])
    addition_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    plot_axes = [figure.add_subplot(grid[2, 0])]
    plot_axes.extend(
        figure.add_subplot(grid[2, index], sharey=plot_axes[0])
        for index in (1, 2)
    )
    formula_axes = (
        [figure.add_subplot(grid[3, index]) for index in range(3)]
        if formula_by_arm is not None
        else []
    )
    shared_axis.set_axis_off()
    for axis in addition_axes:
        axis.set_axis_off()

    shared_axis.text(
        0.5,
        1.0,
        "SHARED PROMPT IN ALL THREE ARMS  ·  EXACT FROZEN TEXT",
        ha="center",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color="#333333",
        transform=shared_axis.transAxes,
    )
    shared_axis.text(
        0.5,
        0.91,
        base._wrap_prompt(shared_prompt, width=118),
        ha="center",
        va="top",
        fontsize=6.15,
        family="monospace",
        linespacing=1.04,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FAFAFA",
            "edgecolor": "#777777",
            "linewidth": 0.9,
        },
        transform=shared_axis.transAxes,
    )

    headings = (
        "A  ·  ADDITION TO SHARED PROMPT",
        "B  ·  EXACT ADDITION TO SHARED PROMPT",
        "C  ·  EXACT ADDITIONS TO SHARED PROMPT",
    )
    for axis, arm, heading in zip(addition_axes, PROMPT_ARMS, headings):
        axis.text(
            0.5,
            1.0,
            heading,
            ha="center",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            transform=axis.transAxes,
        )

    addition_axes[0].text(
        0.5,
        0.52,
        "NO ADDITION\n\nArm A is exactly the shared prompt above.",
        ha="center",
        va="center",
        fontsize=11.0,
        linespacing=1.35,
        color="#6F4100",
        bbox={
            "boxstyle": "round,pad=1.2",
            "facecolor": "#FFF5E8",
            "edgecolor": analysis._ARM_COLOR["c_only"],
            "linewidth": 1.1,
        },
        transform=addition_axes[0].transAxes,
    )
    addition_axes[1].text(
        0.5,
        0.91,
        reference_block,
        ha="center",
        va="top",
        fontsize=4.5,
        family="monospace",
        linespacing=1.02,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#F0FAF7",
            "edgecolor": analysis._ARM_COLOR["abc"],
            "linewidth": 0.9,
        },
        transform=addition_axes[1].transAxes,
    )
    addition_axes[2].text(
        0.5,
        0.91,
        reference_block,
        ha="center",
        va="top",
        fontsize=4.5,
        family="monospace",
        linespacing=1.02,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#F3F9FD",
            "edgecolor": analysis._ARM_COLOR["abc_structural_clue"],
            "linewidth": 0.9,
        },
        transform=addition_axes[2].transAxes,
    )
    addition_axes[2].text(
        0.5,
        0.13,
        textwrap.fill(
            clue,
            width=72,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        ha="center",
        va="bottom",
        fontsize=5.8,
        linespacing=1.06,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#E8F3FC",
            "edgecolor": analysis._ARM_COLOR["abc_structural_clue"],
            "linewidth": 1.0,
        },
        transform=addition_axes[2].transAxes,
    )
    addition_axes[1].text(
        0.5,
        0.055,
        "No clue text is added in Arm B.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=analysis._ARM_COLOR["abc"],
        transform=addition_axes[1].transAxes,
    )

    plot_titles = {
        "c_only": "(A) City C evidence only",
        "abc": "(B) A/B/C information",
        "abc_structural_clue": "(C) A/B/C + structural clue",
    }
    for axis, arm in zip(plot_axes, PROMPT_ARMS):
        if curves is not None:
            for model in models:
                color, point = analysis._MODEL_STYLE[model]
                estimates = [
                    curves[model][str(k)]["arms"][arm]["mae"] for k in ks
                ]
                lows = [
                    curves[model][str(k)]["arms"][arm]["mae_ci"][0]
                    for k in ks
                ]
                highs = [
                    curves[model][str(k)]["arms"][arm]["mae_ci"][1]
                    for k in ks
                ]
                axis.plot(
                    ks,
                    estimates,
                    color=color,
                    marker=point,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    linewidth=1.6,
                    markersize=4.3,
                    zorder=4,
                )
                axis.fill_between(
                    ks, lows, highs, color=color, alpha=0.10, linewidth=0
                )
        axis.plot(
            ks, target, color=analysis._TARGET,
            linestyle=(0, (3, 2)), linewidth=1.35,
        )
        axis.plot(
            ks, matched, color=analysis._ABC,
            linestyle=(0, (3, 2)), linewidth=1.35,
        )
        if structural is not None:
            axis.plot(
                ks, structural, color="#2A6FBB",
                linestyle=(0, (3, 2)), linewidth=1.35,
            )
        axis.set_title(
            plot_titles[arm],
            fontsize=9.5,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=6,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8.4)
        axis.set_ylim(bottom=0.0)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.8)
    plot_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8.4)
    for axis in plot_axes[1:]:
        axis.tick_params(labelleft=False)

    for axis, arm in zip(formula_axes, PROMPT_ARMS):
        axis.set_axis_off()
        axis.text(
            0.5,
            0.88,
            formula_by_arm[arm],
            ha="center",
            va="top",
            fontsize=7.2,
            linespacing=1.2,
            color="#2B2500",
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": "#FFF2A8",
                "edgecolor": "#C9A400",
                "linewidth": 1.0,
            },
            transform=axis.transAxes,
        )

    handles = []
    if curves is not None:
        handles.extend(
            analysis.Line2D(
                [0], [0], color=analysis._MODEL_STYLE[model][0],
                marker=analysis._MODEL_STYLE[model][1], linewidth=1.6,
                markersize=4.3, label=analysis._DISPLAY[model],
            )
            for model in models
        )
    handles.extend(
        [
            analysis.Line2D(
                [0], [0], color=analysis._TARGET,
                linestyle=(0, (3, 2)), label="City C estimator",
            ),
            analysis.Line2D(
                [0], [0], color=analysis._ABC,
                linestyle=(0, (3, 2)), label=matched_benchmark_label,
            ),
        ]
    )
    if structural is not None:
        handles.append(
            analysis.Line2D(
                [0], [0], color="#2A6FBB",
                linestyle=(0, (3, 2)), label=structural_benchmark_label,
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=False,
        fontsize=7.3,
    )
    figure.suptitle(
        "Prompt structure and live performance"
        if curves is not None
        else "Prompt structure and preregistered estimator benchmarks",
        fontsize=13.0,
        fontweight="bold",
        y=0.988,
    )
    subtitle = progress_label or (
        f"Frozen task {base._REPRESENTATIVE_ID} · round 2 · "
        "benchmark curves are not model results"
    )
    figure.text(0.5, 0.962, subtitle, ha="center", fontsize=8.2, color="#555555")
    if interim:
        figure.text(
            0.012, 0.987, "INTERIM · NOT CONFIRMATORY",
            ha="left", va="top", fontsize=8.0,
            fontweight="bold", color="#A61B1B",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    analysis.plt.close(figure)


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
