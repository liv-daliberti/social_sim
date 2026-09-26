#!/usr/bin/env python3
"""Render the completed pilot by condition, with models inside each panel."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis import analyze_coin_city_llm_pilot as base
from engine.coin_city_llm_pilot import (
    CASES_BY_ROUND,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    SHARED_PROMPT_CLOSING,
    SHARED_PROMPT_INTRO,
)


MODEL_COLOR = {
    "claude-opus-4-8": "#7A5195",
    "DeepSeek-V4-Pro": "#D95F02",
    "gpt-5.4": "#2C3E50",
}
MODEL_MARKER = {
    "claude-opus-4-8": "o",
    "DeepSeek-V4-Pro": "s",
    "gpt-5.4": "^",
}


def _mae(predictions: list[float], truths: list[float]) -> float | None:
    if not predictions:
        return None
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(truths))))


def _globally_matched(latest: dict, keys: dict[str, dict]) -> dict:
    """Use one common episode subset across all nine cells at each k."""
    curves = {}
    for k in ROUNDS:
        ids_by_cell = []
        for model in MODELS:
            for arm in PROMPT_ARMS:
                ids_by_cell.append(
                    {
                        task_id
                        for (observed_model, observed_arm, task_id), row in latest.items()
                        if observed_model == model
                        and observed_arm == arm
                        and row.get("predicted_poll") is not None
                        and task_id in keys
                        and keys[task_id]["k"] == k
                    }
                )
        matched = sorted(set.intersection(*ids_by_cell))
        truths = [keys[task_id]["gold_expected_poll"] for task_id in matched]
        cell = {"common_n": len(matched), "arms": {}}
        for arm in PROMPT_ARMS:
            benchmark = [keys[task_id]["baselines"][arm] for task_id in matched]
            model_mae = {}
            for model in MODELS:
                predictions = [
                    latest[(model, arm, task_id)]["predicted_poll"]
                    for task_id in matched
                ]
                model_mae[model] = _mae(predictions, truths)
            cell["arms"][arm] = {
                "matched_statistical_benchmark_mae": _mae(benchmark, truths),
                "llm_mae": model_mae,
            }
        curves[str(k)] = cell
    return curves


def _render(
    output: Path,
    *,
    curves: dict,
    prompt_sections: dict,
    received: int,
    expected_calls: int = PLANNED_CALLS,
    title: str = "Coin-to-city LLM development pilot",
    subtitle_detail: str = (
        "each panel is one prompt condition · all curves use the same 40 episodes"
    ),
) -> None:
    fig = plt.figure(figsize=(16.0, 13.2), facecolor="white")
    fig.suptitle(
        title,
        y=0.988,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.952,
        f"{received:,}/{expected_calls:,} responses received and parsed · "
        + subtitle_detail,
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
    for k in ROUNDS:
        for arm in PROMPT_ARMS:
            arm_cell = curves[str(k)]["arms"][arm]
            all_values.append(arm_cell["matched_statistical_benchmark_mae"])
            all_values.extend(arm_cell["llm_mae"].values())
    observed_values = [value for value in all_values if value is not None]
    ymax = max(
        4.5,
        max(observed_values) * 1.18 if observed_values else 4.5,
    )
    panel_titles = {
        "baseline": "A. City C evidence only",
        "abc_no_context": "B. A/B/C, no City C context",
        "abc_context": "C. A/B/C + City C context",
    }
    for ax, arm in zip(axes, PROMPT_ARMS):
        city_c_regression = [
            curves[str(k)]["arms"]["baseline"][
                "matched_statistical_benchmark_mae"
            ]
            for k in ROUNDS
        ]
        abc_no_context_regression = [
            curves[str(k)]["arms"]["abc_no_context"][
                "matched_statistical_benchmark_mae"
            ]
            for k in ROUNDS
        ]
        ax.plot(
            x,
            city_c_regression,
            color=base.ARM_COLOR["baseline"],
            linewidth=2.4,
            linestyle="--",
            marker="D",
            markersize=4.7,
            label="City C regression",
        )
        ax.plot(
            x,
            abc_no_context_regression,
            color=base.ARM_COLOR["abc_no_context"],
            linewidth=2.4,
            linestyle="--",
            marker="X",
            markersize=5.2,
            label="A/B/C no-context regression",
        )
        for model in MODELS:
            values = [
                curves[str(k)]["arms"][arm]["llm_mae"][model]
                for k in ROUNDS
            ]
            ax.plot(
                x,
                values,
                color=MODEL_COLOR[model],
                linewidth=2.4,
                marker=MODEL_MARKER[model],
                markersize=5.5,
                label=base.MODEL_LABEL[model],
            )
        ax.set_title(
            panel_titles[arm],
            loc="left",
            fontsize=12.5,
            fontweight="bold",
            color=base.ARM_COLOR[arm],
        )
        ax.set_xticks(
            x,
            [f"k={k}\n{CASES_BY_ROUND[k]} C" for k in ROUNDS],
            fontsize=8.5,
        )
        ax.set_ylim(0, ymax)
        ax.set_ylabel("MAE (poll points)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        counts = [curves[str(k)]["common_n"] for k in ROUNDS]
        ax.text(
            0.5,
            -0.25,
            "common matched n: " + ", ".join(map(str, counts)),
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#555555",
        )
    legend_handles = [
        Line2D(
            [0], [0], color=MODEL_COLOR[model], linewidth=2.5,
            marker=MODEL_MARKER[model], label=base.MODEL_LABEL[model],
        )
        for model in MODELS
    ]
    legend_handles.extend(
        [
        Line2D(
                [0], [0], color=base.ARM_COLOR["baseline"],
                linewidth=2.4, linestyle="--", marker="D",
                label="City C regression",
            ),
            Line2D(
                [0], [0], color=base.ARM_COLOR["abc_no_context"],
                linewidth=2.4, linestyle="--", marker="X",
                label="A/B/C no-context regression",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.635),
        ncol=5,
        frameon=False,
        fontsize=9.5,
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
        base._wrap_display(SHARED_PROMPT_INTRO, 150),
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
    for left, arm in zip(lefts, PROMPT_ARMS):
        ax = fig.add_axes([left, 0.145, 0.31, 0.34])
        ax.set_axis_off()
        ax.set_title(
            panel_titles[arm],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=base.ARM_COLOR[arm],
        )
        ax.text(
            0.01,
            0.985,
            base._wrap_display(prompt_sections[arm], 62),
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=6.35,
            linespacing=1.08,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#FCFCFC",
                "edgecolor": base.ARM_COLOR[arm],
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
        base._wrap_display(SHARED_PROMPT_CLOSING, 150),
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
        "Every panel repeats only the City C and A/B/C no-context regression references; all lines use one common matched set.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    latest = base._latest_responses(base.RESPONSES)
    keys = {
        row["task_id"]: row
        for row in base._read_jsonl(base.DESIGN / "answer_key.jsonl")
    }
    prompt_sections = json.loads(
        (base.DESIGN / "example_prompt_sections.json").read_text()
    )
    curves = _globally_matched(latest, keys)
    output = base.RUN / "live" / "live_structure.png"
    _render(
        output,
        curves=curves,
        prompt_sections=prompt_sections,
        received=len(latest),
    )
    result = {
        "experiment": "coin_city_llm_pilot_v1",
        "view": "condition_panels_with_two_regression_references",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "received": len(latest),
        "expected": PLANNED_CALLS,
        "curves": curves,
        "figure": str(output),
    }
    (base.RUN / "live" / "results_by_condition.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote condition-separated figure to {output}")


if __name__ == "__main__":
    main()
