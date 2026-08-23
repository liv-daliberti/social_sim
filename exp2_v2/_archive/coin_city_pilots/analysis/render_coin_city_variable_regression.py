#!/usr/bin/env python3
"""Render the variable-predictor design and, when present, live LLM curves."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
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

from engine.coin_city_variable_regression import (
    C_CASE_LEVELS,
    EXPERIMENT,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
LIVE = RUN / "live"

ARM_COLOR = {
    "baseline": "#E69F00",
    "abc_no_context": "#1B9E77",
    "abc_context": "#2A7FA3",
}
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
MODEL_LABEL = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}
PANEL_TITLE = {
    "baseline": "A. City C evidence only",
    "abc_no_context": "B. A/B/C, no City C context",
    "abc_context": "C. A/B/C + news-source context",
}


def _read_jsonl(path: Path, *, tolerant: bool = False) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerant and index == len(lines) - 1:
                continue
            raise
    return rows


def _latest_responses() -> dict[tuple[str, str, str], dict]:
    latest = {}
    for path in sorted(RESPONSES.glob("responses_*.jsonl")):
        for row in _read_jsonl(path, tolerant=True):
            try:
                key = (row["model"], row["arm"], row["task_id"])
            except KeyError:
                continue
            old = latest.get(key)
            new_priority = 2 if row.get("predicted_poll") is not None else 1
            old_priority = -1 if old is None else (
                2 if old.get("predicted_poll") is not None else 1
            )
            if new_priority >= old_priority:
                latest[key] = row
    return latest


def _mae(predictions: list[float], truths: list[float]) -> float | None:
    if not predictions:
        return None
    return float(
        np.mean(np.abs(np.asarray(predictions, dtype=float) - np.asarray(truths)))
    )


def _curves(keys: dict[str, dict], latest: dict) -> tuple[dict, str]:
    preview = not latest
    result = {}
    for c_cases in C_CASE_LEVELS:
        all_ids = {
            task_id
            for task_id, key in keys.items()
            if int(key["c_cases"]) == int(c_cases)
        }
        if preview:
            matched = sorted(all_ids)
        else:
            cells = []
            for model in MODELS:
                for arm in PROMPT_ARMS:
                    cells.append(
                        {
                            task_id
                            for task_id in all_ids
                            if latest.get((model, arm, task_id), {}).get(
                                "predicted_poll"
                            )
                            is not None
                        }
                    )
            matched = sorted(set.intersection(*cells))
        truths = [keys[task_id]["gold_expected_poll"] for task_id in matched]
        city_c = [keys[task_id]["baselines"]["baseline"] for task_id in matched]
        abc = [
            keys[task_id]["baselines"]["abc_no_context"] for task_id in matched
        ]
        arms = {}
        for arm in PROMPT_ARMS:
            llm = {}
            for model in MODELS:
                if preview:
                    llm[model] = None
                else:
                    llm[model] = _mae(
                        [
                            latest[(model, arm, task_id)]["predicted_poll"]
                            for task_id in matched
                        ],
                        truths,
                    )
            arms[arm] = {"llm_mae": llm}
        result[str(c_cases)] = {
            "common_n": len(matched),
            "city_c_regression_mae": (
                None if c_cases == 0 else _mae(city_c, truths)
            ),
            "abc_no_context_regression_mae": _mae(abc, truths),
            "arms": arms,
        }
    return result, ("design_preview" if preview else "matched_live")


def _wrap_prompt(text: str, width: int = 54) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("Background:") or line.startswith("No additional"):
            lines.extend(textwrap.wrap(line, width=width))
        else:
            lines.append(line)
    return "\n".join(lines)


def _wrap_shared(text: str, width: int = 145) -> str:
    return "\n".join(textwrap.wrap(text.replace("\n", " "), width=width))


def render(output: Path) -> dict:
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    validation_path = DESIGN / "validation.json"
    validation_status = (
        json.loads(validation_path.read_text()).get("status")
        if validation_path.exists()
        else "not_run"
    )
    keys = {
        row["task_id"]: row for row in _read_jsonl(DESIGN / "scoring_key.jsonl")
    }
    components = json.loads(
        (DESIGN / "example_prompt_components.json").read_text()
    )
    latest = _latest_responses()
    curves, curve_mode = _curves(keys, latest)
    received = len(latest)
    parsed = sum(row.get("predicted_poll") is not None for row in latest.values())

    fig = plt.figure(figsize=(16.0, 14.8), facecolor="white")
    fig.suptitle(
        "Variable-predictor coin-to-city experiment",
        y=0.987,
        fontsize=21,
        fontweight="bold",
    )
    subtitle = (
        (
            "C=0–6 live-design preview · PRE-FLIGHT WARNING"
            if validation_status == "failed"
            else f"Frozen {manifest['episodes']}-episode live-design preview"
        )
        + " · no LLM calls in this version"
        if curve_mode == "design_preview"
        else f"{parsed:,}/{PLANNED_CALLS:,} parsed responses · common matched subsets"
    )
    fig.text(
        0.5,
        0.961,
        subtitle
        + " · data noise "
        + f"ε ~ N(0, {manifest['data_generating_process']['case_noise_sd']:.1f}²) (DGP only)"
        + " · fixed 3 A cases + 3 B cases · City C cases 0–6"
        + f" · {PLANNED_CALLS:,}-call ceiling",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.text(
        0.5,
        0.916,
        _wrap_shared(components["shared_intro"]),
        ha="center",
        va="center",
        fontsize=9.4,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F5F5F5",
            "edgecolor": "#AAAAAA",
            "linewidth": 0.9,
        },
    )

    axes = [
        fig.add_axes([0.055 + 0.315 * index, 0.625, 0.285, 0.225])
        for index in range(3)
    ]
    x = np.asarray(C_CASE_LEVELS, dtype=float)
    values = []
    for c_cases in C_CASE_LEVELS:
        cell = curves[str(c_cases)]
        values.extend(
            [
                cell["city_c_regression_mae"],
                cell["abc_no_context_regression_mae"],
            ]
        )
        for arm in PROMPT_ARMS:
            values.extend(cell["arms"][arm]["llm_mae"].values())
    observed = [value for value in values if value is not None]
    ymax = max(4.5, max(observed) * 1.15 if observed else 4.5)
    for ax, arm in zip(axes, PROMPT_ARMS):
        c_values = np.asarray(
            [
                curves[str(c_cases)]["city_c_regression_mae"]
                for c_cases in C_CASE_LEVELS
            ],
            dtype=float,
        )
        abc_values = [
            curves[str(c_cases)]["abc_no_context_regression_mae"]
            for c_cases in C_CASE_LEVELS
        ]
        ax.plot(
            x,
            c_values,
            color=ARM_COLOR["baseline"],
            linestyle="--",
            linewidth=2.6,
            marker="D",
            markersize=5,
        )
        ax.plot(
            x,
            abc_values,
            color=ARM_COLOR["abc_no_context"],
            linestyle="--",
            linewidth=2.6,
            marker="X",
            markersize=5.5,
        )
        for model in MODELS:
            model_values = [
                curves[str(c_cases)]["arms"][arm]["llm_mae"][model]
                for c_cases in C_CASE_LEVELS
            ]
            if any(value is not None for value in model_values):
                ax.plot(
                    x,
                    model_values,
                    color=MODEL_COLOR[model],
                    marker=MODEL_MARKER[model],
                    linewidth=2.3,
                    markersize=5.5,
                )
        ax.set_title(
            PANEL_TITLE[arm],
            loc="left",
            fontsize=12.4,
            fontweight="bold",
            color=ARM_COLOR[arm],
        )
        ax.set_xticks(x, [str(c_cases) for c_cases in C_CASE_LEVELS], fontsize=8.7)
        ax.set_xlabel("Number of displayed City C cases", labelpad=7)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("MAE (poll points)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.574,
        f"Design n = {manifest['episodes']}",
        ha="center",
        fontsize=9.0,
        color="#555555",
    )

    legend = [
        Line2D(
            [0],
            [0],
            color=ARM_COLOR["baseline"],
            linestyle="--",
            marker="D",
            linewidth=2.5,
            label="City C regression",
        ),
        Line2D(
            [0],
            [0],
            color=ARM_COLOR["abc_no_context"],
            linestyle="--",
            marker="X",
            linewidth=2.5,
            label="A/B/C regression",
        ),
    ]
    if curve_mode != "design_preview":
        legend = [
            Line2D(
                [0],
                [0],
                color=MODEL_COLOR[model],
                marker=MODEL_MARKER[model],
                linewidth=2.3,
                label=MODEL_LABEL[model],
            )
            for model in MODELS
        ] + legend
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.858),
        ncol=len(legend),
        frameon=False,
        fontsize=9.2,
    )

    formula_text = {
        "baseline": (
            r"City C regression:  $\hat\beta_C=\frac{\sum_{i\in C}X_i(Y_i-S_i)}"
            r"{\sum_{i\in C}X_i^2}$   ·   "
            r"$\hat Y_* = S_*+X_*\hat\beta_C$"
            "\nUndefined at C=0 (left blank)."
        ),
        "abc_no_context": (
            r"A/B/C regression:  $\hat\beta_{ABC}=\frac{\sum_{i\in A,B,C}"
            r"X_i(Y_i-S_i)}{\sum_{i\in A,B,C}X_i^2}$"
            "\n"
            r"$\hat Y_* = S_*+X_*\hat\beta_{ABC}$"
        ),
        "abc_context": (
            r"Same A/B/C regression:  $\hat\beta_{ABC}=\frac{\sum_{i\in A,B,C}"
            r"X_i(Y_i-S_i)}{\sum_{i\in A,B,C}X_i^2}$"
            "\nLocal-vs-national news sentence added only to the LLM prompt."
        ),
    }
    for index, arm in enumerate(PROMPT_ARMS):
        fig.text(
            0.1975 + 0.315 * index,
            0.535,
            formula_text[arm],
            ha="center",
            va="center",
            fontsize=9.0 if arm != "abc_no_context" else 8.5,
            color="#3C3520",
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": "#FFF1A8",
                "edgecolor": "#C9A227",
                "linewidth": 1.0,
            },
        )

    fig.text(
        0.5,
        0.476,
        "What the LLM sees — frozen episode "
        f"{components['episode']} with {components['c_cases']} City C cases",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    sections = components["sections"]
    prompt_y = 0.443
    for index, arm in enumerate(PROMPT_ARMS):
        x_center = 0.1975 + 0.315 * index
        fig.text(
            0.055 + 0.315 * index,
            prompt_y,
            PANEL_TITLE[arm],
            ha="left",
            va="top",
            fontsize=11.0,
            fontweight="bold",
            color=ARM_COLOR[arm],
        )
        fig.text(
            x_center,
            prompt_y - 0.025,
            _wrap_prompt(sections[arm]),
            ha="center",
            va="top",
            fontsize=7.15,
            linespacing=1.12,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": "#FCFCFC",
                "edgecolor": ARM_COLOR[arm],
                "linewidth": 1.25,
            },
        )

    fig.text(
        0.5,
        0.105,
        _wrap_shared(components["shared_closing"], width=110),
        ha="center",
        va="center",
        fontsize=8.7,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F5F5F5",
            "edgecolor": "#AAAAAA",
            "linewidth": 0.9,
        },
    )
    fig.text(
        0.5,
        0.058,
        "Both dashed regressions are reconstructed only from displayed rows. "
        "City C regression is undefined at C=0; truth is used only afterward to calculate MAE.",
        ha="center",
        fontsize=8.2,
        color="#666666",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    status = (
        (
            "design_preview_preflight_failed_no_model_calls"
            if validation_status == "failed"
            else "frozen_design_preview_no_model_calls"
        )
        if curve_mode == "design_preview"
        else (
            "complete" if parsed == PLANNED_CALLS else "interim_incomplete"
        )
    )
    result = {
        "experiment": EXPERIMENT,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "curve_mode": curve_mode,
        "expected": PLANNED_CALLS,
        "received": received,
        "parsed": parsed,
        "analysis_revision": "variable_predictor_prompt_only_ols_v1",
        "design_validation_status": validation_status,
        "curves": curves,
        "figure": str(output),
        "regression_inputs": "displayed rows and displayed query only",
        "epsilon": {
            "case_noise_sd": manifest["data_generating_process"]["case_noise_sd"],
            "shown_to_llm": False,
        },
    }
    (output.parent / "progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{result['updated_at']} {status}: received={received}/{PLANNED_CALLS}; "
        f"wrote {output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=LIVE / "live_structure.png"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh the live figure until interrupted or all calls are parsed.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=20.0,
        help="Seconds between live refreshes in watch mode.",
    )
    args = parser.parse_args()
    while True:
        result = render(args.output)
        if not args.watch or result["parsed"] >= PLANNED_CALLS:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
