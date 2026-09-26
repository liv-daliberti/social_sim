#!/usr/bin/env python3
"""Post-collection v13 figures with City-C-only repeated in every panel.

This renderer is intentionally separate from the source files hashed into the
frozen v13 design manifest.  It changes presentation only.
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
sys.path.insert(0, str(_ROOT / "eval"))

from analysis import analyze_three_city_c2_v13_confirmatory as analysis
from engine.three_city_c2_v13 import PREFIX_LADDER, PROMPT_ARMS

_RUN = (
    _ROOT / "data" / "three_city_c2_v13"
    / "full_k1_5_three_tier_structural_pooling"
)
_RESULTS = _RUN / "analysis" / "three_city_c2_v13_results.json"
_DESIGN = _RUN / "design"
_ANALYSIS_OUT = _RUN / "analysis"
_PRESENTATION_OUT = _RUN / "presentation"

_C_ONLY_STYLE = {
    "linestyle": (0, (1.2, 2.0)),
    "linewidth": 1.2,
    "alpha": 0.55,
    "zorder": 2,
}

_REPRESENTATIVE_ID = "c2v13_0000_k2"
_DEFAULT_RULE = (
    "Default modeling rule: Unless additional structural guidance is "
    "provided, treat Cities A and B as equally relevant reference sources "
    "for City C."
)
_STRUCTURAL_CLUE = (
    "Structural clue: One of Cities A and B is a more relevant analogue for "
    "City C than the other. The displayed regional-profile indices are "
    "informative about which reference is more relevant, although the "
    "resemblance is imperfect and City C may still respond differently."
)
_PANEL_LABEL = {
    "c_only": "City C regression only",
    "abc": "A/B/C equal pooling",
    "abc_structural_clue": "A/B/C + structural-choice clue",
}


def _read_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _answer_keys() -> dict:
    return {
        row["task_id"]: row
        for row in analysis._read_jsonl(
            _DESIGN / "answer_key_c2_v13.jsonl"
        )
    }


def _read_task(arm: str) -> dict:
    path = _DESIGN / f"tasks_c2_v13_{arm}.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return record
    raise ValueError(f"missing representative task for {arm}")


def _wrap(value: str, width: int = 67) -> str:
    return textwrap.fill(
        value, width=width, break_long_words=False, break_on_hyphens=False
    )


def _addition_text(reference_block: str, *, include_clue: bool) -> str:
    pieces = [reference_block, _wrap(_DEFAULT_RULE)]
    if include_clue:
        pieces.append(_wrap(_STRUCTURAL_CLUE))
    return "\n\n".join(pieces)


def _plot_panel(axis, *, arm: str, curves, models, keys) -> None:
    ks = list(PREFIX_LADDER)
    methods = {
        "c_only": "target_only",
        "abc": "symmetric_pooling",
        "abc_structural_clue": "structural_pooling",
    }

    # Repeat each model's City-C-only performance behind the focal arm.
    if arm != "c_only":
        for model in models:
            color, _ = analysis._MODEL_STYLE[model]
            city_c = [
                curves[model][str(k)]["arms"]["c_only"]["mae"]
                for k in ks
            ]
            axis.plot(ks, city_c, color=color, **_C_ONLY_STYLE)

    for model in models:
        color, marker = analysis._MODEL_STYLE[model]
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
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.75,
            markersize=4.8,
            zorder=4,
        )
        axis.fill_between(
            ks, lows, highs, color=color, alpha=0.09, linewidth=0, zorder=1
        )

    focal_benchmark = [
        analysis._baseline_mae(keys, method=methods[arm], k=k) for k in ks
    ]
    axis.plot(
        ks,
        focal_benchmark,
        color=analysis._ARM_COLOR[arm],
        linestyle=(0, (3, 2)),
        linewidth=1.65,
        zorder=3,
    )
    if arm != "c_only":
        city_c_benchmark = [
            analysis._baseline_mae(keys, method="target_only", k=k)
            for k in ks
        ]
        axis.plot(
            ks,
            city_c_benchmark,
            color=analysis._ARM_COLOR["c_only"],
            linestyle="dashdot",
            linewidth=1.35,
            alpha=0.85,
            zorder=3,
        )


def _legend_handles(models):
    handles = [
        analysis.Line2D(
            [0],
            [0],
            color=analysis._MODEL_STYLE[model][0],
            marker=analysis._MODEL_STYLE[model][1],
            markerfacecolor="white",
            linewidth=1.75,
            markersize=4.8,
            label=analysis._DISPLAY[model],
        )
        for model in models
    ]
    handles.extend(
        [
            analysis.Line2D(
                [0], [0], color="#666666", linewidth=1.75,
                label="Solid + markers: focal prompt arm",
            ),
            analysis.Line2D(
                [0], [0], color="#666666", **_C_ONLY_STYLE,
                label="Dotted: same model, City C regression only",
            ),
            analysis.Line2D(
                [0], [0], color="#555555", linestyle=(0, (3, 2)),
                linewidth=1.65, label="Dashed: estimator for focal arm",
            ),
            analysis.Line2D(
                [0], [0], color=analysis._ARM_COLOR["c_only"],
                linestyle="dashdot", linewidth=1.35,
                label="Orange dash-dot: City C-only estimator",
            ),
        ]
    )
    return handles


def _format_plot_axes(axes) -> None:
    for axis, arm, panel in zip(axes, PROMPT_ARMS, ("A", "B", "C")):
        axis.set_title(
            f"({panel}) {_PANEL_LABEL[arm]}",
            fontsize=9.7,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=6,
        )
        axis.set_xticks(list(PREFIX_LADDER))
        axis.set_xlabel("Evidence round $k$", fontsize=8.7)
        axis.set_ylim(bottom=0.0)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8.1)
    axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8.7)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)


def render_results(path: Path, *, result: dict) -> None:
    curves = result["curves"]
    models = tuple(analysis._MODEL_ORDER)
    keys = _answer_keys()
    analysis.plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure, axes = analysis.plt.subplots(
        1, 3, figsize=(11.6, 3.75), sharex=True, sharey=True
    )
    for axis, arm in zip(axes, PROMPT_ARMS):
        _plot_panel(
            axis, arm=arm, curves=curves, models=models, keys=keys
        )
    _format_plot_axes(axes)
    figure.legend(
        handles=_legend_handles(models),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=7.15,
    )
    figure.suptitle(
        "City C regression-only performance shown in every panel",
        fontsize=11.6,
        fontweight="bold",
        y=1.105,
    )
    figure.text(
        0.5,
        -0.015,
        "At k=1, the City C slope is defined by one point: "
        "$\\hat{\\beta}_C=(\\mathrm{ending\\ poll}-50)/8$; "
        "rounds k=1,...,5 use 1, 2, 4, 8, and 16 City C cases.",
        ha="center",
        fontsize=8.0,
        color="#444444",
    )
    if result.get("status") != "complete":
        figure.text(
            0.01,
            1.09,
            f"INTERIM · {result.get('parsed_rows', 0):,}/5,400 parsed",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color="#A61B1B",
        )
    figure.tight_layout(rect=(0, 0.055, 1, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    analysis.plt.close(figure)


def render_prompt_and_results(path: Path, *, result: dict) -> None:
    curves = result["curves"]
    models = tuple(analysis._MODEL_ORDER)
    tasks = {arm: _read_task(arm) for arm in PROMPT_ARMS}
    shared_prompt = str(tasks["c_only"]["prompt"])
    abc_prompt = str(tasks["abc"]["prompt"])
    reference_marker = "Below are records from two earlier cities in the same region."
    reference_block = (
        reference_marker
        + abc_prompt.split(reference_marker, 1)[1].split("\nCITY C\n", 1)[0]
    ).strip()
    additions = {
        "c_only": "NO ADDITION\n\nArm A is exactly the shared prompt above.",
        "abc": _addition_text(reference_block, include_clue=False),
        "abc_structural_clue": _addition_text(
            reference_block, include_clue=True
        ),
    }
    keys = _answer_keys()

    figure = analysis.plt.figure(figsize=(16.0, 11.1))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(2.45, 4.35, 3.2),
        hspace=0.055,
        wspace=0.09,
        left=0.045,
        right=0.985,
        top=0.925,
        bottom=0.09,
    )
    shared_axis = figure.add_subplot(grid[0, :])
    addition_axes = [figure.add_subplot(grid[1, i]) for i in range(3)]
    plot_axes = [figure.add_subplot(grid[2, 0])]
    plot_axes.extend(
        figure.add_subplot(grid[2, i], sharey=plot_axes[0])
        for i in (1, 2)
    )
    shared_axis.set_axis_off()
    for axis in addition_axes:
        axis.set_axis_off()

    shared_axis.text(
        0.5,
        1.0,
        "SHARED PROMPT IN ALL THREE ARMS · EXACT FROZEN TEXT",
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color="#333333",
        transform=shared_axis.transAxes,
    )
    shared_axis.text(
        0.5,
        0.91,
        shared_prompt,
        ha="center",
        va="top",
        fontsize=6.45,
        family="monospace",
        linespacing=1.06,
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
        "A · EXACT ADDITION TO SHARED PROMPT",
        "B · EXACT ADDITION TO SHARED PROMPT",
        "C · EXACT ADDITIONS TO SHARED PROMPT",
    )
    backgrounds = {
        "c_only": "#FFF5E8",
        "abc": "#F0FAF7",
        "abc_structural_clue": "#EDF6FC",
    }
    for axis, arm, heading in zip(addition_axes, PROMPT_ARMS, headings):
        axis.text(
            0.5,
            1.0,
            heading,
            ha="center",
            va="top",
            fontsize=10.0,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            transform=axis.transAxes,
        )
        is_a = arm == "c_only"
        axis.text(
            0.5,
            0.89,
            additions[arm],
            ha="center",
            va="top",
            fontsize=5.05 if not is_a else 11.2,
            family="monospace" if not is_a else "serif",
            linespacing=1.04 if not is_a else 1.28,
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.58" if not is_a else "round,pad=1.0",
                "facecolor": backgrounds[arm],
                "edgecolor": analysis._ARM_COLOR[arm],
                "linewidth": 1.0,
            },
            transform=axis.transAxes,
        )

    for axis, arm in zip(plot_axes, PROMPT_ARMS):
        _plot_panel(
            axis, arm=arm, curves=curves, models=models, keys=keys
        )
    _format_plot_axes(plot_axes)
    figure.legend(
        handles=_legend_handles(models),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=7.6,
    )
    figure.suptitle(
        "Prompt structure and performance · City C-only repeated in every panel",
        fontsize=13.4,
        fontweight="bold",
        y=0.988,
    )
    figure.text(
        0.5,
        0.956,
        f"INTERIM · {result.get('parsed_rows', 0):,}/5,400 parsed · "
        "at k=1, $\\hat{\\beta}_C=(\\mathrm{ending\\ poll}-50)/8$",
        ha="center",
        fontsize=8.5,
        color="#A61B1B",
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    analysis.plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=_RESULTS)
    parser.add_argument(
        "--results-output",
        type=Path,
        default=_ANALYSIS_OUT / "three_city_c2_v13_with_c_only_reference.png",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=_PRESENTATION_OUT / "prompt_structure_with_c_only_reference.png",
    )
    args = parser.parse_args()
    result = _read_results(args.results)
    render_results(args.results_output, result=result)
    render_prompt_and_results(args.prompt_output, result=result)
    print(f"Wrote {args.results_output} and {args.results_output.with_suffix('.pdf')}")
    print(f"Wrote {args.prompt_output} and {args.prompt_output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
