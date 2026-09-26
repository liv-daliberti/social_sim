#!/usr/bin/env python3
"""Live/final three-panel view using always-aligned context in panel C."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    ROUNDS,
    SHARED_PROMPT_CLOSING,
    SHARED_PROMPT_INTRO,
)


PARENT = ROOT / "data" / "coin_city_llm_pilot_v1"
RERUN = PARENT / "context100_rerun"
PARENT_LIVE_FIGURE = PARENT / "live" / "live_structure.png"
RERUN_LIVE_FIGURE = RERUN / "live" / "context100_structure.png"
ARM_CONTEXT100 = "abc_context_100pct"
EXPECTED_NEW = 600
PRIOR_CALLS = 1800
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
PANELS = (
    {
        "response_arm": "baseline",
        "benchmark": "baseline",
        "title": "A. City C evidence only",
        "benchmark_label": "City C evidence-only regression",
        "prompt_key": "baseline",
    },
    {
        "response_arm": "abc_no_context",
        "benchmark": "abc_no_context",
        "title": "B. A/B/C, no City C context",
        "benchmark_label": "A/B/C regression (no context)",
        "prompt_key": "abc_no_context",
    },
    {
        "response_arm": ARM_CONTEXT100,
        "benchmark": "abc_no_context",
        "title": "C. A/B/C + always-useful City C context",
        "benchmark_label": "A/B/C regression (no context)",
        "prompt_key": ARM_CONTEXT100,
    },
)


def _latest() -> dict[tuple[str, str, str], dict]:
    latest = base._latest_responses(PARENT / "responses")
    latest.update(base._latest_responses(RERUN / "responses"))
    return latest


def _mae(predictions: list[float], truths: list[float]) -> float | None:
    if not predictions:
        return None
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(truths))))


def _curves(latest: dict, keys: dict[str, dict]) -> dict:
    curves = {}
    for k in ROUNDS:
        completed_sets = []
        for model in MODELS:
            for panel in PANELS:
                arm = panel["response_arm"]
                completed_sets.append(
                    {
                        task_id
                        for (seen_model, seen_arm, task_id), row in latest.items()
                        if seen_model == model
                        and seen_arm == arm
                        and row.get("predicted_poll") is not None
                        and task_id in keys
                        and keys[task_id]["k"] == k
                    }
                )
        matched = sorted(set.intersection(*completed_sets))
        truths = [keys[task_id]["gold_expected_poll"] for task_id in matched]
        panel_results = []
        for panel in PANELS:
            benchmark_values = [
                keys[task_id]["baselines"][panel["benchmark"]]
                for task_id in matched
            ]
            llm = {}
            for model in MODELS:
                llm[model] = _mae(
                    [
                        latest[(model, panel["response_arm"], task_id)]["predicted_poll"]
                        for task_id in matched
                    ],
                    truths,
                )
            panel_results.append(
                {
                    "response_arm": panel["response_arm"],
                    "benchmark": panel["benchmark"],
                    "benchmark_label": panel["benchmark_label"],
                    "regression_mae": _mae(benchmark_values, truths),
                    "llm_mae": llm,
                }
            )
        curves[str(k)] = {"common_n": len(matched), "panels": panel_results}
    return curves


def _render(output: Path, *, result: dict, prompt_sections: dict) -> None:
    curves = result["curves"]
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
        f"Always-useful context rerun: {result['new_received']:,}/{EXPECTED_NEW:,} responses · "
        f"cumulative test calls {PRIOR_CALLS + result['new_received']:,}/5,000 · "
        "all plotted curves use one common episode subset",
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
        for panel in curves[str(k)]["panels"]:
            if panel["regression_mae"] is not None:
                all_values.append(panel["regression_mae"])
            all_values.extend(value for value in panel["llm_mae"].values() if value is not None)
    ymax = max(4.5, max(all_values) * 1.18 if all_values else 4.5)
    common_counts = [curves[str(k)]["common_n"] for k in ROUNDS]
    for panel_index, (ax, panel_spec) in enumerate(zip(axes, PANELS)):
        regression = [
            curves[str(k)]["panels"][panel_index]["regression_mae"]
            if curves[str(k)]["panels"][panel_index]["regression_mae"] is not None
            else np.nan
            for k in ROUNDS
        ]
        benchmark_line, = ax.plot(
            x,
            regression,
            color="#111111",
            linewidth=2.7,
            linestyle="--",
            marker="D",
            markersize=5,
            label=panel_spec["benchmark_label"],
        )
        for model in MODELS:
            values = [
                curves[str(k)]["panels"][panel_index]["llm_mae"][model]
                if curves[str(k)]["panels"][panel_index]["llm_mae"][model] is not None
                else np.nan
                for k in ROUNDS
            ]
            ax.plot(
                x,
                values,
                color=MODEL_COLOR[model],
                linewidth=2.4,
                marker=MODEL_MARKER[model],
                markersize=5.5,
            )
        color = base.ARM_COLOR[
            "abc_context" if panel_index == 2 else panel_spec["response_arm"]
        ]
        ax.set_title(
            panel_spec["title"], loc="left", fontsize=12.2,
            fontweight="bold", color=color,
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
        ax.legend(handles=[benchmark_line], loc="upper right", frameon=False, fontsize=8.2)
        ax.text(
            0.5,
            -0.25,
            "common matched n: " + ", ".join(map(str, common_counts)),
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#555555",
        )
    model_handles = [
        Line2D(
            [0], [0], color=MODEL_COLOR[model], linewidth=2.5,
            marker=MODEL_MARKER[model], label=base.MODEL_LABEL[model],
        )
        for model in MODELS
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.635),
        ncol=3,
        frameon=False,
        fontsize=9.7,
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
            "boxstyle": "round,pad=0.6", "facecolor": "#F2F2F2",
            "edgecolor": "#AAAAAA",
        },
    )
    for panel_index, (left, panel_spec) in enumerate(zip((0.025, 0.345, 0.665), PANELS)):
        ax = fig.add_axes([left, 0.145, 0.31, 0.34])
        ax.set_axis_off()
        color = base.ARM_COLOR[
            "abc_context" if panel_index == 2 else panel_spec["response_arm"]
        ]
        ax.set_title(
            panel_spec["title"], loc="left", fontsize=11.3,
            fontweight="bold", color=color,
        )
        ax.text(
            0.01,
            0.985,
            base._wrap_display(prompt_sections[panel_spec["prompt_key"]], 62),
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=6.35,
            linespacing=1.08,
            bbox={
                "boxstyle": "round,pad=0.55", "facecolor": "#FCFCFC",
                "edgecolor": color, "linewidth": 1.3,
            },
        )
    fig.text(
        0.82,
        0.125,
        "B → C adds only a City C clue that is useful in every episode.",
        ha="center",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.4", "facecolor": "#FFF4B8",
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
            "boxstyle": "round,pad=0.55", "facecolor": "#F2F2F2",
            "edgecolor": "#AAAAAA",
        },
    )
    fig.text(
        0.5,
        0.012,
        "Panels B and C use the same no-context A/B/C regression; only the LLM in C receives the aligned language clue.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render(output: Path) -> dict:
    latest = _latest()
    keys = {
        row["task_id"]: row
        for row in base._read_jsonl(PARENT / "design" / "answer_key.jsonl")
    }
    old_sections = json.loads(
        (PARENT / "design" / "example_prompt_sections.json").read_text()
    )
    context_sections = json.loads(
        (RERUN / "design" / "example_prompt_sections.json").read_text()
    )
    prompt_sections = {**old_sections, **context_sections}
    new_latest = base._latest_responses(RERUN / "responses")
    new_received = len(new_latest)
    new_parsed = sum(row.get("predicted_poll") is not None for row in new_latest.values())
    result = {
        "status": (
            "complete" if new_received == new_parsed == EXPECTED_NEW
            else "interim_incomplete"
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "new_expected": EXPECTED_NEW,
        "new_received": new_received,
        "new_parsed": new_parsed,
        "cumulative_test_calls": PRIOR_CALLS + new_received,
        "hard_call_cap": 5000,
        "curves": _curves(latest, keys),
        "figure": str(output),
    }
    _render(output, result=result, prompt_sections=prompt_sections)
    (output.parent / "context100_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{result['status']}: aligned context {new_received}/{EXPECTED_NEW}; wrote {output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=RERUN_LIVE_FIGURE
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    output = args.output
    if output.resolve() == PARENT_LIVE_FIGURE.resolve():
        print(
            f"Refusing to overwrite completed pilot figure; redirecting to "
            f"{RERUN_LIVE_FIGURE}",
            flush=True,
        )
        output = RERUN_LIVE_FIGURE
    while True:
        result = render(output)
        if not args.watch or result["new_received"] >= EXPECTED_NEW:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
