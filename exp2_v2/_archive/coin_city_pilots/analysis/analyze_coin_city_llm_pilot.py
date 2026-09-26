#!/usr/bin/env python3
"""Matched-subset analysis and prompt-aware figure for the coin-city pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_llm_pilot import (
    CASES_BY_ROUND,
    EPISODES,
    EXPERIMENT,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    SHARED_PROMPT_CLOSING,
    SHARED_PROMPT_INTRO,
)


RUN = ROOT / "data" / EXPERIMENT
RESPONSES = RUN / "responses"
DESIGN = RUN / "design"
OUTDIR = RUN / "analysis"

ARM_LABEL = {
    "baseline": "City C evidence only",
    "abc_no_context": "A/B/C, no City C context",
    "abc_context": "A/B/C + City C context",
}
ARM_COLOR = {
    "baseline": "#E68613",
    "abc_no_context": "#168A72",
    "abc_context": "#267CB5",
}
MODEL_LABEL = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}


def _read_jsonl(path: Path, *, tolerant: bool = False) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not tolerant:
                raise
    return rows


def _latest_responses(response_dir: Path) -> dict[tuple[str, str, str], dict]:
    latest: dict[tuple[str, str, str], dict] = {}
    for path in sorted(response_dir.glob("responses_*.jsonl")):
        for row in _read_jsonl(path, tolerant=True):
            try:
                key = (row["model"], row["arm"], row["task_id"])
            except KeyError:
                continue
            old = latest.get(key)
            priority = 2 if row.get("predicted_poll") is not None else 1
            old_priority = -1 if old is None else (
                2 if old.get("predicted_poll") is not None else 1
            )
            if priority >= old_priority:
                latest[key] = row
    return latest


def _mae(predictions: list[float], truths: list[float]) -> float | None:
    if not predictions:
        return None
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(truths))))


def _curves(
    latest: dict[tuple[str, str, str], dict], keys: dict[str, dict]
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        result[model] = {}
        for k in ROUNDS:
            ids_by_arm = {
                arm: {
                    task_id
                    for (observed_model, observed_arm, task_id), row in latest.items()
                    if observed_model == model
                    and observed_arm == arm
                    and row.get("predicted_poll") is not None
                    and task_id in keys
                    and keys[task_id]["k"] == k
                }
                for arm in PROMPT_ARMS
            }
            matched = set.intersection(*(ids_by_arm[arm] for arm in PROMPT_ARMS))
            ordered = sorted(matched)
            llm_mae: dict[str, float | None] = {}
            benchmark_mae: dict[str, float | None] = {}
            for arm in PROMPT_ARMS:
                truths = [keys[task_id]["gold_expected_poll"] for task_id in ordered]
                predictions = [latest[(model, arm, task_id)]["predicted_poll"] for task_id in ordered]
                benchmarks = [keys[task_id]["baselines"][arm] for task_id in ordered]
                llm_mae[arm] = _mae(predictions, truths)
                benchmark_mae[arm] = _mae(benchmarks, truths)
            paired_improvement = {
                "abc_no_context_over_baseline": (
                    None
                    if not ordered
                    else llm_mae["baseline"] - llm_mae["abc_no_context"]
                ),
                "context_over_no_context": (
                    None
                    if not ordered
                    else llm_mae["abc_no_context"] - llm_mae["abc_context"]
                ),
            }
            result[model][str(k)] = {
                "matched_n": len(ordered),
                "llm_mae": llm_mae,
                "matched_benchmark_mae": benchmark_mae,
                "mae_improvement": paired_improvement,
            }
    return result


def _wrap_display(value: str, width: int) -> str:
    lines = []
    for line in value.splitlines():
        if not line or line.startswith("|"):
            lines.append(line)
        else:
            lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [""]
            )
    return "\n".join(lines)


def _render(
    output: Path,
    *,
    curves: dict[str, dict[str, dict[str, Any]]],
    prompt_sections: dict[str, str],
    received: int,
    parsed: int,
    status: str,
) -> None:
    fig = plt.figure(figsize=(16.0, 13.2), facecolor="white")
    fig.suptitle(
        "Coin-to-city LLM development pilot",
        y=0.988,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.952,
        f"{received:,}/{PLANNED_CALLS:,} responses received · "
        f"{parsed:,} parsed · {status.replace('_', ' ')} · "
        "curves use only episodes completed in all three conditions",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    axes = [
        fig.add_axes([0.055 + 0.315 * index, 0.665, 0.285, 0.235])
        for index in range(3)
    ]
    x = np.asarray(ROUNDS)
    all_values = []
    for model in MODELS:
        for k in ROUNDS:
            cell = curves[model][str(k)]
            for kind in ("llm_mae", "matched_benchmark_mae"):
                all_values.extend(
                    value for value in cell[kind].values() if value is not None
                )
    ymax = max(4.5, (max(all_values) * 1.18 if all_values else 4.5))
    for ax, model in zip(axes, MODELS):
        for arm in PROMPT_ARMS:
            llm = [
                curves[model][str(k)]["llm_mae"][arm]
                if curves[model][str(k)]["llm_mae"][arm] is not None
                else np.nan
                for k in ROUNDS
            ]
            benchmark = [
                curves[model][str(k)]["matched_benchmark_mae"][arm]
                if curves[model][str(k)]["matched_benchmark_mae"][arm] is not None
                else np.nan
                for k in ROUNDS
            ]
            ax.plot(
                x,
                benchmark,
                color=ARM_COLOR[arm],
                linewidth=1.4,
                linestyle="--",
                alpha=0.65,
            )
            ax.plot(
                x,
                llm,
                color=ARM_COLOR[arm],
                linewidth=2.5,
                marker="o",
                markersize=5.5,
            )
        matched_counts = [curves[model][str(k)]["matched_n"] for k in ROUNDS]
        ax.set_title(MODEL_LABEL[model], fontsize=13, fontweight="bold")
        ax.set_xticks(
            x,
            [f"k={k}\n{CASES_BY_ROUND[k]} C" for k in ROUNDS],
            fontsize=8.5,
        )
        ax.set_ylim(0, ymax)
        ax.set_ylabel("MAE (poll points)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.5,
            -0.25,
            "matched n: " + ", ".join(map(str, matched_counts)),
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#555555",
        )
    arm_handles = [
        Line2D([0], [0], color=ARM_COLOR[arm], lw=2.6, marker="o", label=ARM_LABEL[arm])
        for arm in PROMPT_ARMS
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", lw=2.4, label="LLM"),
        Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="matched statistical benchmark"),
    ]
    fig.legend(
        handles=arm_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.635),
        ncol=5,
        frameon=False,
        fontsize=9.2,
    )

    fig.text(
        0.5,
        0.585,
        "What the LLM sees — frozen episode 0 at k=2",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.545,
        _wrap_display(SHARED_PROMPT_INTRO, 150),
        ha="center",
        va="top",
        family="monospace",
        fontsize=8.1,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F2F2F2",
            "edgecolor": "#AAAAAA",
        },
    )
    lefts = (0.025, 0.345, 0.665)
    panel_titles = {
        "baseline": "A. City C evidence only",
        "abc_no_context": "B. A/B/C, no City C context",
        "abc_context": "C. A/B/C + City C context",
    }
    for left, arm in zip(lefts, PROMPT_ARMS):
        ax = fig.add_axes([left, 0.145, 0.31, 0.34])
        ax.set_axis_off()
        ax.set_title(
            panel_titles[arm],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=ARM_COLOR[arm],
        )
        ax.text(
            0.01,
            0.985,
            _wrap_display(prompt_sections[arm], 62),
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=6.35,
            linespacing=1.08,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#FCFCFC",
                "edgecolor": ARM_COLOR[arm],
                "linewidth": 1.3,
            },
        )
    fig.text(
        0.82,
        0.125,
        "B → C changes only the City C background sentence.",
        ha="center",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#FFF4B8",
            "edgecolor": "#D4B547",
        },
    )
    fig.text(
        0.5,
        0.085,
        _wrap_display(SHARED_PROMPT_CLOSING, 150),
        ha="center",
        va="top",
        family="monospace",
        fontsize=7.7,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F2F2F2",
            "edgecolor": "#AAAAAA",
        },
    )
    fig.text(
        0.5,
        0.012,
        "Dashed benchmarks are recomputed on each model's same currently matched episodes; display wrapping does not alter prompts.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(
    *,
    response_dir: Path = RESPONSES,
    design_dir: Path = DESIGN,
    outdir: Path = OUTDIR,
    output_name: str = "coin_city_llm_pilot_results.png",
) -> dict:
    latest = _latest_responses(response_dir)
    keys = {
        row["task_id"]: row
        for row in _read_jsonl(design_dir / "answer_key.jsonl")
    }
    prompt_sections = json.loads(
        (design_dir / "example_prompt_sections.json").read_text()
    )
    received = len(latest)
    parsed = sum(row.get("predicted_poll") is not None for row in latest.values())
    if received == PLANNED_CALLS and parsed == PLANNED_CALLS:
        status = "complete"
    elif received == PLANNED_CALLS:
        status = "complete_with_parse_failures"
    else:
        status = "interim_incomplete"
    curves = _curves(latest, keys)
    output = outdir / output_name
    _render(
        output,
        curves=curves,
        prompt_sections=prompt_sections,
        received=received,
        parsed=parsed,
        status=status,
    )
    result = {
        "experiment": EXPERIMENT,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "expected": PLANNED_CALLS,
        "received": received,
        "parsed": parsed,
        "parse_failures": received - parsed,
        "curves": curves,
        "figure": str(output),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=RESPONSES)
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--output-name", default="coin_city_llm_pilot_results.png")
    args = parser.parse_args()
    result = analyze(
        response_dir=args.responses,
        design_dir=args.design,
        outdir=args.outdir,
        output_name=args.output_name,
    )
    print(
        f"{result['status']}: {result['received']}/{result['expected']} received; "
        f"wrote {result['figure']}"
    )


if __name__ == "__main__":
    main()
