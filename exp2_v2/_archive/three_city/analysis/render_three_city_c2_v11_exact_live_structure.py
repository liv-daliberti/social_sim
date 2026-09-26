#!/usr/bin/env python3
"""Render exact frozen prompt additions above benchmark or live v11 curves."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v11_confirmatory as analysis
from analysis import live_monitor_three_city_c2_v11 as monitor
from engine.three_city_c2_v11 import PREFIX_LADDER, PROMPT_ARMS
from eval.build_three_city_c2_v11_tasks import STRUCTURAL_CHOICE_CLUE

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v11"
    / "full_k1_5_identifiable_structural_choice"
)
_DESIGN = _RUN / "design"
_LIVE = _RUN / "live"
_PRESENTATION = _RUN / "presentation"
_REPRESENTATIVE_ID = "c2v11_0000_k2"


def _read_prompt(arm: str) -> str:
    path = _DESIGN / f"tasks_c2_v11_{arm}.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] == _REPRESENTATIVE_ID:
            return str(record["prompt"])
    raise ValueError(f"missing representative prompt: {arm}")


def _keys() -> dict:
    return {
        row["task_id"]: row
        for row in analysis._read_jsonl(_DESIGN / "answer_key_c2_v11.jsonl")
    }


def _wrap_prompt(prompt: str, width: int = 108) -> str:
    lines = []
    for line in prompt.splitlines():
        if not line or line.startswith("|") or len(line) <= width:
            lines.append(line)
        else:
            lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(lines)


def _exact_blocks() -> tuple[str, str, str, str]:
    c_only = _read_prompt("c_only")
    abc = _read_prompt("abc")
    structural = _read_prompt("abc_structural_clue")
    if structural.replace(STRUCTURAL_CHOICE_CLUE + "\n\n", "") != abc:
        raise ValueError("frozen structural prompt does not differ only by clue")
    marker = "Below are records from two earlier cities in the same region."
    reference = (
        marker + abc.split(marker, 1)[1].split("\nCITY C\n", 1)[0]
    ).strip()
    after_marker = reference.split(marker, 1)[1].strip()
    city_a, city_b_tail = after_marker.split("\n\nCITY B\n", 1)
    return c_only, marker, city_a.strip(), "CITY B\n" + city_b_tail.strip()


def _live_curves(*, rows: list, draws: int) -> dict:
    curves = {}
    for model_index, model in enumerate(analysis._MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = analysis._statistics(
                analysis._paired(rows, model=model, k=k),
                draws=draws,
                seed=20260805 + 1000 * model_index + k,
            )
    return curves


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def render(
    path: Path,
    *,
    keys: dict,
    curves: dict | None,
    parsed: int,
    interim: bool,
) -> None:
    c_only, marker, city_a, city_b = _exact_blocks()
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
    figure = analysis.plt.figure(figsize=(13.8, 14.25))
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(4.25, 5.0, 1.25, 3.35),
        hspace=0.12,
        wspace=0.10,
        left=0.055,
        right=0.985,
        top=0.915,
        bottom=0.055,
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
        _wrap_prompt(c_only),
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
    for x, text_value in ((0.015, city_a), (0.515, city_b)):
        reference_axis.text(
            x,
            0.83,
            text_value,
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
        0.78,
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

    y_values = list(target) + list(matched)
    titles = {
        "c_only": "(A) City C evidence only",
        "abc": "(B) A/B/C information",
        "abc_structural_clue": "(C) A/B/C + structural clue",
    }
    for axis, arm in zip(plot_axes, PROMPT_ARMS):
        if curves is not None:
            for model in analysis._MODEL_ORDER:
                color, point = analysis._MODEL_STYLE[model]
                estimates = [
                    curves[model][str(k)]["arms"][arm]["mae"] for k in ks
                ]
                lows = [
                    curves[model][str(k)]["arms"][arm]["mae_ci"][0] for k in ks
                ]
                highs = [
                    curves[model][str(k)]["arms"][arm]["mae_ci"][1] for k in ks
                ]
                if any(_finite(value) for value in estimates):
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
                    y_values.extend(value for value in highs if _finite(value))
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
            titles[arm],
            fontsize=9.7,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            pad=6,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8.7)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8.0)
    upper = max(3.8, max(float(value) for value in y_values if _finite(value)) * 1.08)
    for axis in plot_axes:
        axis.set_ylim(0.0, upper)
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
            for model in analysis._MODEL_ORDER
        )
    handles.extend(
        [
            analysis.Line2D(
                [0], [0], color=analysis._TARGET,
                linestyle=(0, (3, 2)), label="City C estimator"
            ),
            analysis.Line2D(
                [0], [0], color=analysis._ABC,
                linestyle=(0, (3, 2)), label="Matched A/B/C estimator"
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=len(handles),
        frameon=False,
        fontsize=7.5,
    )
    title = (
        "Live prompt structure and forecast accuracy"
        if curves is not None
        else "Prompt structure and preregistered estimator benchmarks"
    )
    figure.suptitle(title, fontsize=13.0, fontweight="bold", y=0.987)
    subtitle = (
        f"INTERIM — INCOMPLETE COLLECTION — {parsed}/5,400 parsed — NOT CONFIRMATORY"
        if interim
        else "Complete collection"
    )
    if curves is None:
        subtitle = (
            f"Frozen task {_REPRESENTATIVE_ID} · round 2 · two City C cases · "
            "benchmark curves are not model results"
        )
    figure.text(
        0.5,
        0.962,
        subtitle,
        ha="center",
        fontsize=8.5,
        fontweight="bold" if interim else "normal",
        color="#A61B1B" if interim else "#555555",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    analysis.plt.close(figure)


def _write_dashboard(path: Path, *, parsed: int, iteration: int) -> None:
    updated = datetime.now(timezone.utc).isoformat()
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>C2 v11 annotated live structure</title>
<style>body{{font-family:system-ui;margin:1.2rem;max-width:1500px}}img{{max-width:100%}}
.warn{{color:#a61b1b;font-weight:700}}</style></head>
<body><p class="warn">INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY</p>
<p>{parsed}/5,400 parsed. Updated {html.escape(updated)}.</p>
<img src="live_structure_annotated.png?v={iteration}" alt="Annotated live structure">
</body></html>"""
    path.write_text(page)


def render_once(*, draws: int, iteration: int, presentation: bool) -> None:
    keys = _keys()
    latest = monitor._latest()
    rows = analysis._score_rows(latest, keys)
    curves = _live_curves(rows=rows, draws=draws)
    render(
        _LIVE / "live_structure_annotated.png",
        keys=keys,
        curves=curves,
        parsed=len(rows),
        interim=len(rows) < 5_400,
    )
    _write_dashboard(
        _LIVE / "annotated_index.html",
        parsed=len(rows),
        iteration=iteration,
    )
    if presentation:
        render(
            _PRESENTATION / "prompt_structure_paper.png",
            keys=keys,
            curves=None,
            parsed=0,
            interim=False,
        )
    print(f"annotated live structure: {len(rows)}/5400 parsed", flush=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--render-presentation", action="store_true")
    args = parser.parse_args()
    iteration = 0
    while True:
        iteration += 1
        parsed = render_once(
            draws=args.draws,
            iteration=iteration,
            presentation=args.render_presentation and iteration == 1,
        )
        if not args.watch or parsed >= 5_400:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
