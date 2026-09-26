#!/usr/bin/env python3
"""Render a concise paper-facing view of the frozen v11 prompt structure.

This is deliberately separate from the hashed pre-collection renderer.  It
changes presentation only: all displayed values and prompt text are read from
the frozen task files and answer key.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v11_confirmatory as analysis
from engine.three_city_c2_v11 import PREFIX_LADDER, PROMPT_ARMS
from eval.build_three_city_c2_v11_tasks import STRUCTURAL_CHOICE_CLUE

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v11"
    / "full_k1_5_identifiable_structural_choice"
)
_DESIGN = _RUN / "design"
_OUTDIR = _RUN / "presentation"
_REPRESENTATIVE_ID = "c2v11_0000_k2"


def _read_task(arm: str) -> dict:
    path = _DESIGN / f"tasks_c2_v11_{arm}.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return record
    raise ValueError(f"missing representative task: {arm}/{_REPRESENTATIVE_ID}")


def _read_key() -> dict:
    path = _DESIGN / "answer_key_c2_v11.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return record
    raise ValueError(f"missing representative key: {_REPRESENTATIVE_ID}")


def _wrap_prompt(prompt: str, width: int = 112) -> str:
    wrapped = []
    for line in prompt.splitlines():
        if not line or line.startswith("|") or len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(wrapped)


def _compact_city(name: str, city: dict) -> str:
    endings = [float(value) for value in city["ending_polls"]]
    starts = {float(value) for value in [city["starting_poll"]]}
    news = {float(value) for value in city["news"]}
    if len(starts) != 1 or len(news) != 1:
        raise ValueError("compact reference display requires shared inputs")
    return "\n".join(
        [
            f"CITY {name}  ·  profile {float(city['profile_index']):.1f}/100",
            f"All cases: start {float(city['starting_poll']):.1f}, news {float(city['news'][0]):+g}",
            "Cases 1–4 end: " + " · ".join(f"{value:.1f}" for value in endings[:4]),
            "Cases 5–8 end: " + " · ".join(f"{value:.1f}" for value in endings[4:]),
        ]
    )


def _card(axis, *, title: str, body: str, color: str, face: str) -> None:
    axis.set_axis_off()
    axis.text(
        0.0,
        1.0,
        title,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=color,
        transform=axis.transAxes,
    )
    axis.text(
        0.0,
        0.88,
        body,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.32,
        bbox={
            "boxstyle": "round,pad=0.75",
            "facecolor": face,
            "edgecolor": color,
            "linewidth": 1.1,
        },
        transform=axis.transAxes,
    )


def render(path: Path) -> None:
    tasks = {arm: _read_task(arm) for arm in PROMPT_ARMS}
    key = _read_key()
    c_only = str(tasks["c_only"]["prompt"])
    abc = str(tasks["abc"]["prompt"])
    structural = str(tasks["abc_structural_clue"]["prompt"])
    assert structural.replace(STRUCTURAL_CHOICE_CLUE + "\n\n", "") == abc

    all_keys = {
        row["task_id"]: row
        for row in analysis._read_jsonl(_DESIGN / "answer_key_c2_v11.jsonl")
    }
    ks = list(PREFIX_LADDER)
    target = [
        analysis._baseline_mae(all_keys, method="target_only", k=k)
        for k in ks
    ]
    matched = [
        analysis._baseline_mae(all_keys, method="abc_shrinkage", k=k)
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
    figure = analysis.plt.figure(figsize=(13.8, 12.0))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(3.1, 5.1, 3.2),
        hspace=0.17,
        wspace=0.10,
        left=0.055,
        right=0.985,
        top=0.925,
        bottom=0.055,
    )
    card_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    shared_axis = figure.add_subplot(grid[1, :])
    plot_axes = [figure.add_subplot(grid[2, index]) for index in range(3)]

    _card(
        card_axes[0],
        title="A  ·  CITY C EVIDENCE ONLY",
        body=(
            "City A and City B records are omitted.\n\n"
            "The shared City C prompt below is unchanged."
        ),
        color=analysis._ARM_COLOR["c_only"],
        face="#FFF5E8",
    )
    reference_a = key["public"]["reference_a"]
    reference_b = key["public"]["reference_b"]
    _card(
        card_axes[1],
        title="B  ·  A/B/C INFORMATION",
        body=(
            "Added reference records (exact values, compact layout):\n\n"
            + _compact_city("A", reference_a)
            + "\n\n"
            + _compact_city("B", reference_b)
        ),
        color=analysis._ARM_COLOR["abc"],
        face="#F0FAF7",
    )
    _card(
        card_axes[2],
        title="C  ·  STRUCTURAL-CHOICE CLUE",
        body=(
            "Same records as B, plus this exact text:\n\n"
            + textwrap.fill(
                STRUCTURAL_CHOICE_CLUE,
                width=53,
                break_long_words=False,
                break_on_hyphens=False,
            )
        ),
        color=analysis._ARM_COLOR["abc_structural_clue"],
        face="#EDF5FC",
    )

    shared_axis.set_axis_off()
    shared_axis.text(
        0.0,
        1.0,
        "SHARED CITY C PROMPT  ·  EXACT REPRESENTATIVE TASK",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#333333",
        transform=shared_axis.transAxes,
    )
    shared_axis.text(
        1.0,
        1.0,
        "Identical in A, B, and C",
        ha="right",
        va="top",
        fontsize=8.4,
        color="#555555",
        transform=shared_axis.transAxes,
    )
    shared_axis.text(
        0.0,
        0.94,
        _wrap_prompt(c_only),
        ha="left",
        va="top",
        fontsize=6.45,
        family="monospace",
        linespacing=1.13,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.75",
            "facecolor": "#FAFAFA",
            "edgecolor": "#777777",
            "linewidth": 0.9,
        },
        transform=shared_axis.transAxes,
    )

    short_titles = {
        "c_only": "(A) City C evidence only",
        "abc": "(B) A/B/C information",
        "abc_structural_clue": "(C) A/B/C + structural clue",
    }
    for axis, arm in zip(plot_axes, PROMPT_ARMS):
        axis.plot(
            ks,
            target,
            color=analysis._TARGET,
            marker="o",
            markerfacecolor="white",
            linestyle=(0, (3, 2)),
            linewidth=1.6,
            label="City C estimator",
        )
        axis.plot(
            ks,
            matched,
            color=analysis._ABC,
            marker="s",
            markerfacecolor="white",
            linestyle=(0, (3, 2)),
            linewidth=1.6,
            label="Matched A/B/C estimator",
        )
        axis.set_title(
            short_titles[arm],
            fontsize=9.7,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=6,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8.7)
        axis.set_ylim(0, 3.8)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8.0)
    plot_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8.7)
    for axis in plot_axes[1:]:
        axis.tick_params(labelleft=False)

    figure.legend(
        handles=plot_axes[0].get_legend_handles_labels()[0],
        labels=plot_axes[0].get_legend_handles_labels()[1],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    figure.suptitle(
        "Prompt structure and preregistered estimator benchmarks",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.958,
        (
            f"Frozen task {_REPRESENTATIVE_ID} · round 2 · two City C cases · "
            "benchmark curves are not model results"
        ),
        ha="center",
        fontsize=8.3,
        color="#555555",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    analysis.plt.close(figure)


def render_exact(
    path: Path,
    *,
    curves=None,
    models=(),
    interim: bool = False,
    progress_label: str | None = None,
) -> None:
    """Render exact frozen prompt material above static or live curves."""
    tasks = {arm: _read_task(arm) for arm in PROMPT_ARMS}
    c_only = str(tasks["c_only"]["prompt"])
    abc = str(tasks["abc"]["prompt"])
    structural = str(tasks["abc_structural_clue"]["prompt"])
    assert structural.replace(STRUCTURAL_CHOICE_CLUE + "\n\n", "") == abc

    marker = "Below are records from two earlier cities in the same region."
    reference_block = (
        marker + abc.split(marker, 1)[1].split("\nCITY C\n", 1)[0]
    ).strip()
    after_marker = reference_block.split(marker, 1)[1].strip()
    city_a_text, city_b_tail = after_marker.split("\n\nCITY B\n", 1)
    city_a_text = city_a_text.strip()
    city_b_text = "CITY B\n" + city_b_tail.strip()

    keys = {
        row["task_id"]: row
        for row in analysis._read_jsonl(_DESIGN / "answer_key_c2_v11.jsonl")
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
    figure = analysis.plt.figure(figsize=(13.8, 9.8))
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(2.5, 1.9, 0.75, 3.2),
        hspace=0.035,
        wspace=0.10,
        left=0.055,
        right=0.985,
        top=0.935,
        bottom=0.09,
    )
    shared_axis = figure.add_subplot(grid[0, :])
    reference_axis = figure.add_subplot(grid[1, :])
    clue_axis = figure.add_subplot(grid[2, :])
    plot_axes = [figure.add_subplot(grid[3, index]) for index in range(3)]
    for axis in (shared_axis, reference_axis, clue_axis):
        axis.set_axis_off()

    shared_axis.text(
        0.5,
        1.0,
        "SHARED CITY C PROMPT  ·  EXACT FROZEN TEXT",
        ha="center",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color="#333333",
        transform=shared_axis.transAxes,
    )
    shared_axis.text(
        0.5,
        0.94,
        _wrap_prompt(c_only, width=108),
        ha="center",
        va="top",
        fontsize=6.25,
        family="monospace",
        linespacing=1.10,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#FAFAFA",
            "edgecolor": "#777777",
            "linewidth": 0.9,
        },
        transform=shared_axis.transAxes,
    )

    reference_axis.text(
        0.0,
        1.0,
        "A  ·  NO TEXT ADDED",
        ha="left",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        color=analysis._ARM_COLOR["c_only"],
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#FFF5E8",
            "edgecolor": analysis._ARM_COLOR["c_only"],
            "linewidth": 0.9,
        },
        transform=reference_axis.transAxes,
    )
    reference_axis.text(
        1.0,
        1.0,
        "B AND C  ·  EXACT BLOCK ADDED BEFORE ‘CITY C’",
        ha="right",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        color=analysis._ARM_COLOR["abc"],
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#F0FAF7",
            "edgecolor": analysis._ARM_COLOR["abc"],
            "linewidth": 0.9,
        },
        transform=reference_axis.transAxes,
    )
    reference_axis.text(
        0.5,
        0.90,
        marker,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#333333",
        transform=reference_axis.transAxes,
    )
    for x, city_text in ((0.015, city_a_text), (0.515, city_b_text)):
        reference_axis.text(
            x,
            0.74,
            city_text,
            ha="left",
            va="top",
            fontsize=5.45,
            family="monospace",
            linespacing=1.05,
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#F7FCFA",
                "edgecolor": analysis._ARM_COLOR["abc"],
                "linewidth": 0.85,
            },
            transform=reference_axis.transAxes,
        )

    clue_axis.text(
        0.5,
        1.0,
        "C ONLY  ·  EXACT TEXT ADDED BEFORE ‘THINK CAREFULLY’",
        ha="center",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        color=analysis._ARM_COLOR["abc_structural_clue"],
        transform=clue_axis.transAxes,
    )
    clue_axis.text(
        0.5,
        0.82,
        textwrap.fill(
            STRUCTURAL_CHOICE_CLUE,
            width=132,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        ha="center",
        va="top",
        fontsize=8.0,
        linespacing=1.18,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#EDF5FC",
            "edgecolor": analysis._ARM_COLOR["abc_structural_clue"],
            "linewidth": 1.0,
        },
        transform=clue_axis.transAxes,
    )

    short_titles = {
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
                    markeredgewidth=1.1,
                    linewidth=1.7,
                    markersize=4.5,
                    zorder=4,
                )
                axis.fill_between(
                    ks,
                    lows,
                    highs,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
        axis.plot(
            ks,
            target,
            color=analysis._TARGET,
            linestyle=(0, (3, 2)),
            linewidth=1.35,
        )
        axis.plot(
            ks,
            matched,
            color=analysis._ABC,
            linestyle=(0, (3, 2)),
            linewidth=1.35,
        )
        axis.set_title(
            short_titles[arm],
            fontsize=9.7,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=6,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8.7)
        axis.set_ylim(bottom=0.0)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8.0)
    plot_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8.7)
    for axis in plot_axes[1:]:
        axis.tick_params(labelleft=False)

    handles = []
    if curves is not None:
        handles.extend(
            analysis.Line2D(
                [0],
                [0],
                color=analysis._MODEL_STYLE[model][0],
                marker=analysis._MODEL_STYLE[model][1],
                linewidth=1.7,
                markersize=4.5,
                label=analysis._DISPLAY[model],
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
                linestyle=(0, (3, 2)), label="Matched A/B/C estimator",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=False,
        fontsize=7.5,
    )
    figure.suptitle(
        "Prompt structure and live performance"
        if curves is not None
        else "Prompt structure and preregistered estimator benchmarks",
        fontsize=13.0,
        fontweight="bold",
        y=0.989,
    )
    subtitle = (
        progress_label
        if progress_label
        else (
            f"Frozen task {_REPRESENTATIVE_ID} · round 2 · two City C cases · "
            "benchmark curves are not model results"
        )
    )
    figure.text(
        0.5,
        0.964,
        subtitle,
        ha="center",
        fontsize=8.3,
        color="#555555",
    )
    if interim:
        figure.text(
            0.012,
            0.988,
            "INTERIM · NOT CONFIRMATORY",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#A61B1B",
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
        default=_OUTDIR / "prompt_structure_paper.png",
    )
    args = parser.parse_args()
    render_exact(args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
