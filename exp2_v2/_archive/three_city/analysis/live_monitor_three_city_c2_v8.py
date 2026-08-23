#!/usr/bin/env python3
"""Continuously render clearly labelled interim C2 v8 dashboards."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from analysis import analyze_three_city_c2_v8_confirmatory as analysis
from engine.three_city_c2_v8 import PREFIX_LADDER, PROMPT_ARMS
from eval.build_three_city_c2_v8_tasks import HINT_BLOCK, STRONG_HINT_BLOCK

_DATA = _ROOT / "data" / "three_city_c2_v8"
_RUN = _DATA / "full_k1_5_similarity"
_DESIGN = _RUN / "design"
_RESPONSES = _RUN / "responses"
_OUTDIR = _RUN / "live"
_EXPECTED_PER_CELL = 120 * 4 * len(PREFIX_LADDER)


def _read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read a file that another process may currently be appending."""
    records = []
    if not path.exists():
        return records
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _latest_responses(
    paths: Iterable[Path],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        for record in _read_jsonl_tolerant(path):
            try:
                key = (record["model"], record["arm"], record["task_id"])
            except KeyError:
                continue
            priority = (
                2
                if record.get("predicted_poll") is not None
                else (1 if record.get("response_received") is True else 0)
            )
            old = latest.get(key)
            old_priority = (
                -1
                if old is None
                else (
                    2
                    if old.get("predicted_poll") is not None
                    else (1 if old.get("response_received") is True else 0)
                )
            )
            if priority >= old_priority:
                latest[key] = record
    return latest


def _progress(
    latest: Mapping[tuple[str, str, str], Mapping[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    cells = []
    paired_k2: dict[str, dict[str, int]] = {}
    for model in analysis._MODEL_ORDER:
        paired_k2[model] = {}
        pairs = analysis._paired_structure(rows, model=model, k=2)
        paired_n = len(pairs)
        for arm in PROMPT_ARMS:
            selected = [
                record
                for (record_model, record_arm, _), record in latest.items()
                if record_model == model and record_arm == arm
            ]
            received = sum(
                record.get("response_received") is True
                or record.get("predicted_poll") is not None
                for record in selected
            )
            parsed = sum(
                record.get("predicted_poll") is not None
                for record in selected
            )
            cells.append(
                {
                    "model": model,
                    "model_label": analysis._DISPLAY[model],
                    "arm": arm,
                    "arm_label": analysis._ARM_LABEL[arm],
                    "received": received,
                    "parsed": parsed,
                    "expected": _EXPECTED_PER_CELL,
                    "completion_fraction": received / _EXPECTED_PER_CELL,
                    "parse_rate": parsed / received if received else None,
                    "paired_no_context_k2": paired_n,
                }
            )
            paired_k2[model][arm] = paired_n
    return cells, paired_k2


def _statistics(
    rows: list[dict[str, Any]],
    *,
    draws: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    curves = {}
    context = {}
    for model_index, model in enumerate(analysis._MODEL_ORDER):
        curves[model] = {}
        context[model] = {}
        for k_index, k in enumerate(PREFIX_LADDER):
            curves[model][str(k)] = analysis._structure_statistics(
                analysis._paired_structure(rows, model=model, k=k),
                draws=draws,
                seed=20260729 + 1_000 * model_index + 10 * k_index,
                include_beta=False,
            )
            context[model][str(k)] = analysis._context_statistics(
                analysis._paired_context(rows, model=model, k=k),
                draws=draws,
                seed=20261729 + 1_000 * model_index + 10 * k_index,
            )
    return curves, context


def _make_prompt_annotated_structure_figure(
    path: Path,
    *,
    curves: Mapping[str, Any],
    keys: Mapping[str, Mapping[str, Any]],
    base_prompt: str,
    base_task_id: str,
) -> None:
    """Render the live structure curve with exact arm differences attached."""
    analysis.plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure = analysis.plt.figure(figsize=(13.4, 11.4))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(1.25, 3.0, 5.0),
        hspace=0.15,
        wspace=0.08,
        left=0.065,
        right=0.985,
        top=0.86,
        bottom=0.035,
    )
    prompt_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    plot_axes = []
    for index in range(3):
        plot_axes.append(
            figure.add_subplot(
                grid[1, index],
                sharex=plot_axes[0] if plot_axes else None,
                sharey=plot_axes[0] if plot_axes else None,
            )
        )
    base_axis = figure.add_subplot(grid[2, :])
    base_axis.set_axis_off()

    additions = {
        "blind": "No added structural instruction.",
        "hint": HINT_BLOCK,
        "strong_hint": STRONG_HINT_BLOCK,
    }
    fills = {
        "blind": "#F0F0F0",
        "hint": "#E7F4EF",
        "strong_hint": "#E8F2F9",
    }
    panels = ("A", "B", "C")
    for axis, arm, panel in zip(prompt_axes, PROMPT_ARMS, panels):
        axis.set_axis_off()
        axis.text(
            0.5,
            1.02,
            f"({panel}) {analysis._ARM_LABEL[arm]}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            transform=axis.transAxes,
        )
        axis.text(
            0.02,
            0.82,
            "ONLY ARM-SPECIFIC PROMPT TEXT",
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color=analysis._ARM_COLOR[arm],
            transform=axis.transAxes,
        )
        wrapped = textwrap.fill(
            additions[arm],
            width=56 if arm == "strong_hint" else 50,
        )
        axis.text(
            0.02,
            0.68,
            wrapped,
            ha="left",
            va="top",
            fontsize=7.1 if arm == "strong_hint" else 7.7,
            linespacing=1.12,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.52",
                "facecolor": fills[arm],
                "edgecolor": analysis._ARM_COLOR[arm],
                "linewidth": 1.2,
            },
            transform=axis.transAxes,
        )

    ks = list(PREFIX_LADDER)
    target = [
        analysis._baseline_mae(keys, method="target_only", k=k) for k in ks
    ]
    abc = [
        analysis._baseline_mae(keys, method="abc_shrinkage", k=k) for k in ks
    ]
    common_upper = 0.0
    for axis, arm in zip(plot_axes, PROMPT_ARMS):
        for model in analysis._MODEL_ORDER:
            color, marker = analysis._MODEL_STYLE[model]
            estimates = [
                curves[model][str(k)]["arms"][arm]["mae"] for k in ks
            ]
            lows = [
                curves[model][str(k)]["arms"][arm]["mae_ci"][0] for k in ks
            ]
            highs = [
                curves[model][str(k)]["arms"][arm]["mae_ci"][1] for k in ks
            ]
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
            axis.fill_between(
                ks,
                lows,
                highs,
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )
        axis.plot(
            ks,
            target,
            color=analysis._TARGET,
            linestyle=(0, (3, 2)),
            linewidth=1.35,
            zorder=3,
        )
        axis.plot(
            ks,
            abc,
            color=analysis._ABC,
            linestyle=(0, (3, 2)),
            linewidth=1.35,
            zorder=3,
        )
        axis.axvline(2, color="#BBBBBB", linewidth=0.7, zorder=1)
        axis.axvline(analysis._CONVERGENCE_K, color="#DDDDDD", linewidth=0.7, zorder=1)
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8)
        axis.grid(color=analysis._GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.5)
        common_upper = max(common_upper, axis.get_ylim()[1])
    plot_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8)
    for index, axis in enumerate(plot_axes):
        axis.set_ylim(0, common_upper)
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
                [0],
                [0],
                color=analysis._TARGET,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                label="Use City C only",
            ),
            analysis.Line2D(
                [0],
                [0],
                color=analysis._ABC,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                label="Combine A/B/C (matched)",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=len(handles),
        frameon=False,
        fontsize=7.1,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.4,
    )
    figure.text(
        0.5,
        0.985,
        "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#A61B1B",
    )
    figure.text(
        0.5,
        0.947,
        (
            "All task data and common instructions are identical across "
            "panels; the shaded block is the complete prompt difference. "
            "Evidence rounds 1–5 contain 1, 2, 4, 8, and 16 completed City C cases."
        ),
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    base_axis.text(
        0.0,
        0.965,
        "FULL SHARED BASE PROMPT",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#333333",
        transform=base_axis.transAxes,
    )
    base_axis.text(
        1.0,
        0.965,
        (
            f"Exact frozen task {base_task_id} · round 2 · 2 City C cases · no City C background · "
            "B/C insert the highlighted block immediately before "
            "“Think carefully”"
        ),
        ha="right",
        va="top",
        fontsize=6.5,
        color="#555555",
        transform=base_axis.transAxes,
    )
    prompt_lines = []
    for line in base_prompt.splitlines():
        if (
            not line
            or line.startswith("|")
            or line.startswith("{")
            or len(line) <= 145
        ):
            prompt_lines.append(line)
        else:
            prompt_lines.extend(textwrap.wrap(line, width=145))
    base_axis.text(
        0.0,
        0.915,
        "\n".join(prompt_lines),
        ha="left",
        va="top",
        fontsize=5.25,
        family="monospace",
        linespacing=1.12,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.6",
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


def _write_html(
    path: Path,
    *,
    created_at: str,
    cells: list[dict[str, Any]],
    iteration: int,
    base_prompt: str,
    base_task_id: str,
) -> None:
    rows = []
    for cell in cells:
        parse_rate = cell["parse_rate"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(cell['model_label'])}</td>"
            f"<td>{html.escape(cell['arm_label'])}</td>"
            f"<td>{cell['received']:,} / {cell['expected']:,}</td>"
            f"<td>{100.0 * cell['completion_fraction']:.1f}%</td>"
            f"<td>{'—' if parse_rate is None else f'{100.0 * parse_rate:.1f}%'}</td>"
            f"<td>{cell['paired_no_context_k2']}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>Three-city C2 v8 live monitor</title>
<style>
body {{ margin: 24px auto; max-width: 1500px; font-family: system-ui, sans-serif;
       color: #222; background: #fafafa; }}
h1 {{ margin-bottom: 4px; }}
.warning {{ color: #a61b1b; font-weight: 700; }}
.meta {{ color: #555; }}
img {{ width: 100%; height: auto; background: white; border: 1px solid #ddd;
       margin: 12px 0 28px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: white;
       border: 1px solid #ccc; border-left: 5px solid #4a4a4a;
       padding: 18px; font: 14px/1.35 ui-monospace, SFMono-Regular, Menlo,
       Consolas, monospace; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
  text-align: left;
}}
</style>
</head>
<body>
<h1>Three-city C2 v8 live monitor</h1>
<p class="warning">INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY</p>
<p class="meta">Updated {html.escape(created_at)} · refresh #{iteration}.
This page reloads every 15 seconds; figures are recomputed every monitor interval.
All plotted model estimates use complete parsed episode cells required by the
corresponding contrast.</p>
<h2>Main C2a structure curve</h2>
<img src="live_structure.png?v={iteration}" alt="Live structure curve">
<h2>Full shared base prompt</h2>
<p class="meta">Exact frozen structure-blind prompt for representative main-curve
task <code>{html.escape(base_task_id)}</code> (round 2, 2 City C cases, no City C background).
The city records vary across tasks. Panels B and C insert only their highlighted
arm-specific block immediately before “Think carefully”; panel A inserts
nothing.</p>
<pre>{html.escape(base_prompt)}</pre>
<p><a href="full_base_prompt.txt">Open the base prompt as plain text</a></p>
<h2>Separate C2b context panel</h2>
<img src="live_context.png?v={iteration}" alt="Live context panel">
<h2>Collection progress</h2>
<table>
<thead><tr><th>Model</th><th>Prompt arm</th><th>Received</th>
<th>Complete</th><th>Parse rate</th><th>Paired n at round 2</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""
    temporary = path.with_suffix(".tmp.html")
    temporary.write_text(document)
    os.replace(temporary, path)


def render_once(
    *,
    answer_key: Path,
    responses_dir: Path,
    outdir: Path,
    draws: int,
    iteration: int,
) -> dict[str, Any]:
    keys = {
        row["task_id"]: row
        for row in _read_jsonl_tolerant(answer_key)
        if "task_id" in row
    }
    latest = _latest_responses(
        sorted(responses_dir.glob("responses_*.jsonl"))
    )
    base_task_id = "c2v8_0000_v0_k2"
    base_records = _read_jsonl_tolerant(
        _DESIGN / "tasks_c2_v8_blind.jsonl"
    )
    base_record = next(
        (
            record
            for record in base_records
            if record.get("task_id") == base_task_id
        ),
        base_records[0],
    )
    base_prompt = str(base_record["prompt"])
    base_task_id = str(base_record["task_id"])
    rows = analysis._score_rows(latest, keys)
    curves, context = _statistics(rows, draws=draws)
    cells, paired_k2 = _progress(latest, rows)
    timestamp = datetime.now(timezone.utc)

    outdir.mkdir(parents=True, exist_ok=True)
    stage = outdir / ".stage"
    stage.mkdir(parents=True, exist_ok=True)
    structure_stage = stage / "live_structure.png"
    context_stage = stage / "live_context.png"
    _make_prompt_annotated_structure_figure(
        structure_stage,
        curves=curves,
        keys=keys,
        base_prompt=base_prompt,
        base_task_id=base_task_id,
    )
    analysis._make_context_figure(
        context_stage,
        context=context,
        oracle=analysis._oracle_context_curves(keys),
        models=analysis._MODEL_ORDER,
        interim=True,
    )
    for source, destination in (
        (structure_stage, outdir / "live_structure.png"),
        (structure_stage.with_suffix(".pdf"), outdir / "live_structure.pdf"),
        (context_stage, outdir / "live_context.png"),
        (context_stage.with_suffix(".pdf"), outdir / "live_context.pdf"),
    ):
        os.replace(source, destination)

    progress = {
        "experiment": "three_city_c2_v8_full_k1_5_similarity_matched",
        "status": "interim_incomplete_not_confirmatory",
        "updated_at": timestamp.isoformat(),
        "iteration": iteration,
        "expected_total_calls": _EXPECTED_PER_CELL * len(cells),
        "received_total": sum(cell["received"] for cell in cells),
        "parsed_total": sum(cell["parsed"] for cell in cells),
        "cells": cells,
        "paired_no_context_k2": paired_k2,
        "response_files": sorted(
            path.name for path in responses_dir.glob("responses_*.jsonl")
        ),
    }
    progress_stage = outdir / "progress.tmp.json"
    progress_stage.write_text(json.dumps(progress, indent=2) + "\n")
    os.replace(progress_stage, outdir / "progress.json")
    prompt_stage = outdir / "full_base_prompt.tmp.txt"
    prompt_stage.write_text(base_prompt + "\n")
    os.replace(prompt_stage, outdir / "full_base_prompt.txt")
    _write_html(
        outdir / "index.html",
        created_at=timestamp.isoformat(),
        cells=cells,
        iteration=iteration,
        base_prompt=base_prompt,
        base_task_id=base_task_id,
    )
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_DESIGN / "answer_key_c2_v8.jsonl",
    )
    parser.add_argument("--responses-dir", type=Path, default=_RESPONSES)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    iteration = 0
    while True:
        iteration += 1
        try:
            progress = render_once(
                answer_key=args.answer_key,
                responses_dir=args.responses_dir,
                outdir=args.outdir,
                draws=args.draws,
                iteration=iteration,
            )
            print(
                f"{progress['updated_at']} received="
                f"{progress['received_total']}/{progress['expected_total_calls']} parsed="
                f"{progress['parsed_total']}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"live monitor render failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not args.watch:
                raise
        if not args.watch:
            break
        time.sleep(max(10.0, args.interval))


if __name__ == "__main__":
    main()
