#!/usr/bin/env python3
"""Score and plot the small, paired three-city C2 pilot.

The model runner cannot read the answer key. This post-hoc analysis is the only
place where saved model forecasts are joined to hidden conditions and gold
values.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DATA = _ROOT / "data" / "three_city_c2"
_PILOT = _DATA / "pilot_v1"
_ANSWER_KEY = _DATA / "answer_key_c2_v1.jsonl"
_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}
_COLORS = {
    "claude-opus-4-8": "#7b4ab5",
    "DeepSeek-V4-Pro": "#168a86",
    "gpt-5.4": "#d65f45",
}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.mean(clean) if clean else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_latest_responses(paths: Iterable[Path]):
    latest: Dict[tuple[str, str], Dict[str, Any]] = {}
    failures: Dict[tuple[str, str], Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for record in _read_jsonl(path):
            key = (record["model"], record["task_id"])
            if record.get("predicted_poll") is not None:
                latest[key] = record
            elif key not in latest:
                failures[key] = record
    return latest, failures


def _current_poll(key: Mapping[str, Any]) -> float:
    k = int(key["k"])
    target = key["public"]["target"]
    if k:
        return float(target["polls"][k - 1])
    return float(target["initial_poll"])


def _behavioral_gain(key: Mapping[str, Any], predicted_poll: float) -> float:
    return (
        float(predicted_poll) - _current_poll(key)
    ) / float(key["public"]["test_news"])


def _type_score(gain: float, true_gain: float) -> float:
    false_gain = 0.25 if true_gain == 1.0 else 1.0
    true_distance = abs(gain - true_gain)
    false_distance = abs(gain - false_gain)
    if abs(true_distance - false_distance) < 1e-12:
        return 0.5
    return float(true_distance < false_distance)


def _context_role(key: Mapping[str, Any]) -> str:
    condition = key["condition"]
    if condition in ("none", "orthogonal"):
        return condition
    target_high = key["gold"]["target_gain"] == 1.0
    aligned = (
        (condition == "cue_high" and target_high)
        or (condition == "cue_low" and not target_high)
    )
    return "aligned" if aligned else "misleading"


def _score_prediction(
    key: Mapping[str, Any],
    predicted_poll: Optional[float],
) -> Dict[str, Any]:
    if predicted_poll is None:
        return {
            "predicted_poll": None,
            "mae": None,
            "gain": None,
            "gain_error": None,
            "type_accuracy": None,
        }
    predicted_poll = float(predicted_poll)
    gain = _behavioral_gain(key, predicted_poll)
    true_gain = float(key["gold"]["target_gain"])
    return {
        "predicted_poll": predicted_poll,
        "mae": abs(predicted_poll - float(key["gold"]["expected_poll"])),
        "gain": gain,
        "gain_error": abs(gain - true_gain),
        "type_accuracy": _type_score(gain, true_gain),
    }


def _model_rows(
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows = []
    for (model, task_id), response in responses.items():
        if task_id not in keys:
            continue
        key = keys[task_id]
        rows.append(
            {
                "source": model,
                "episode_id": key["episode_id"],
                "task_id": task_id,
                "k": int(key["k"]),
                "condition": key["condition"],
                "role": _context_role(key),
                **_score_prediction(key, response.get("predicted_poll")),
            }
        )
    return rows


def _baseline_rows(
    keys: Mapping[str, Mapping[str, Any]],
    selected_task_ids: set[str],
) -> list[Dict[str, Any]]:
    rows = []
    methods = (
        "naive",
        "frequentist",
        "two_prototype_exemplar",
        "context_oracle",
        "clairvoyant",
    )
    for task_id in selected_task_ids:
        key = keys[task_id]
        for method in methods:
            predicted = key["baselines"][method].get("predicted_poll")
            rows.append(
                {
                    "source": method,
                    "episode_id": key["episode_id"],
                    "task_id": task_id,
                    "k": int(key["k"]),
                    "condition": key["condition"],
                    "role": _context_role(key),
                    **_score_prediction(key, predicted),
                }
            )
    return rows


def _group_mean(rows, source, k, field, *, role=None):
    return _mean(
        row[field]
        for row in rows
        if row["source"] == source
        and row["k"] == k
        and (role is None or row["role"] == role)
    )


def _paired_context_metric(rows, source, k, metric):
    grouped = defaultdict(dict)
    for row in rows:
        if row["source"] == source and row["k"] == k:
            grouped[row["episode_id"]][row["condition"]] = row
    values = []
    for conditions in grouped.values():
        if not {"none", "orthogonal", "cue_high", "cue_low"} <= set(
            conditions
        ):
            continue
        if metric == "high_low_gain":
            high = conditions["cue_high"]["gain"]
            low = conditions["cue_low"]["gain"]
            if high is not None and low is not None:
                values.append(high - low)
        elif metric == "orthogonal_movement":
            none = conditions["none"]["gain"]
            orthogonal = conditions["orthogonal"]["gain"]
            if none is not None and orthogonal is not None:
                values.append(abs(orthogonal - none))
        else:
            raise ValueError(metric)
    return _mean(values)


def _write_summary(
    path: Path,
    *,
    model_rows,
    baseline_rows,
    models,
    selected_count,
    completed_by_model,
) -> None:
    lines = [
        "# Three-city C2 small-pilot diagnostics",
        "",
        "> This is a design sanity check, not a model comparison or paper result.",
        "",
        "## Completion",
        "",
        "| Model | Parsed | Selected |",
        "|---|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| {_DISPLAY.get(model, model)} | "
            f"{completed_by_model.get(model, 0)} | {selected_count} |"
        )

    lines += [
        "",
        "## No-context recovery",
        "",
        "| Model | k | Forecast MAE | Response error | Pattern accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        for k in (0, 1, 2):
            lines.append(
                f"| {_DISPLAY.get(model, model)} | {k} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'mae', role='none'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'gain_error', role='none'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'type_accuracy', role='none'))} |"
            )

    lines += [
        "",
        "## Context diagnostics",
        "",
        "The high-minus-buffered column is measured in behaviorally implied response "
        "units. It should be positive under selective context use. Orthogonal "
        "movement should be close to zero.",
        "",
        "| Model | k | High − buffered | Orthogonal movement | "
        "Misleading response error |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        for k in (0, 1, 2):
            lines.append(
                f"| {_DISPLAY.get(model, model)} | {k} | "
                f"{_fmt(_paired_context_metric(model_rows, model, k, 'high_low_gain'))} | "
                f"{_fmt(_paired_context_metric(model_rows, model, k, 'orthogonal_movement'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'gain_error', role='misleading'))} |"
            )

    lines += [
        "",
        "## Observation-matched references",
        "",
        "| Forecaster | k | No-context forecast MAE | Pattern accuracy |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "naive": "Naive",
        "frequentist": "Frequentist target-only",
        "two_prototype_exemplar": "Two-prototype empirical",
        "context_oracle": "Context-aware oracle",
        "clairvoyant": "Clairvoyant",
    }
    for method, label in labels.items():
        for k in (0, 1, 2):
            lines.append(
                f"| {label} | {k} | "
                f"{_fmt(_group_mean(baseline_rows, method, k, 'mae', role='none'))} | "
                f"{_fmt(_group_mean(baseline_rows, method, k, 'type_accuracy', role='none'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _plot(
    path: Path,
    *,
    model_rows,
    baseline_rows,
    models,
    completed_by_model,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ks = (0, 1, 2)

    for model in models:
        label = (
            f"{_DISPLAY.get(model, model)} "
            f"({completed_by_model.get(model, 0)}/48 prompts)"
        )
        axes[0, 0].plot(
            ks,
            [
                _group_mean(model_rows, model, k, "mae", role="none")
                for k in ks
            ],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=label,
        )
        axes[0, 1].plot(
            ks,
            [
                _group_mean(
                    model_rows,
                    model,
                    k,
                    "type_accuracy",
                    role="none",
                )
                for k in ks
            ],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=label,
        )
        axes[1, 0].plot(
            ks,
            [
                _paired_context_metric(
                    model_rows,
                    model,
                    k,
                    "high_low_gain",
                )
                for k in ks
            ],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=label,
        )
        axes[1, 1].plot(
            ks,
            [
                _group_mean(
                    model_rows,
                    model,
                    k,
                    "gain_error",
                    role="misleading",
                )
                for k in ks
            ],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=label,
        )

    baseline_styles = {
        "naive": ("Naive", ":", "#777777"),
        "frequentist": ("Frequentist", "-.", "#222222"),
        "two_prototype_exemplar": (
            "Two-prototype empirical",
            "--",
            "#37679a",
        ),
    }
    for method, (label, linestyle, color) in baseline_styles.items():
        axes[0, 0].plot(
            ks,
            [
                _group_mean(
                    baseline_rows,
                    method,
                    k,
                    "mae",
                    role="none",
                )
                for k in ks
            ],
            linestyle=linestyle,
            linewidth=1.5,
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            ks,
            [
                _group_mean(
                    baseline_rows,
                    method,
                    k,
                    "type_accuracy",
                    role="none",
                )
                for k in ks
            ],
            linestyle=linestyle,
            linewidth=1.5,
            color=color,
            label=label,
        )

    for axis in axes.flat:
        axis.set_xticks(ks)
        axis.set_xlabel("City C observations available (k)")
        axis.grid(alpha=0.25)
    axes[0, 0].set_title("A. No-context forecast error")
    axes[0, 0].set_ylabel("Mean absolute error (poll points)")
    axes[0, 1].set_title("B. No-context pattern recovery")
    axes[0, 1].set_ylabel("Behavioral pattern accuracy")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title("C. Selective context use")
    axes[1, 0].set_ylabel("Implied response: high cue − buffered cue")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("D. Override of misleading context")
    axes[1, 1].set_ylabel("Response error under misleading cue")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Three-city C2 pilot — 4 paired episodes; directional check only",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-key", type=Path, default=_ANSWER_KEY)
    parser.add_argument(
        "--responses",
        type=Path,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=_PILOT / "pilot_diagnostics.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=_PILOT / "pilot_summary.md",
    )
    args = parser.parse_args()

    response_paths = args.responses or sorted(
        _PILOT.glob("responses_*.jsonl")
    )
    keys = {
        record["task_id"]: record for record in _read_jsonl(args.answer_key)
    }
    responses, _ = _load_latest_responses(response_paths)
    if not responses:
        raise SystemExit("no parseable model responses found")
    selected_task_ids = {task_id for _, task_id in responses}
    model_rows = _model_rows(responses, keys)
    baseline_rows = _baseline_rows(keys, selected_task_ids)
    models = [
        model
        for model in _DISPLAY
        if any(row["source"] == model for row in model_rows)
    ]
    completed_by_model = {
        model: sum(1 for source, _ in responses if source == model)
        for model in models
    }
    selected_count = max(completed_by_model.values())
    _write_summary(
        args.summary,
        model_rows=model_rows,
        baseline_rows=baseline_rows,
        models=models,
        selected_count=selected_count,
        completed_by_model=completed_by_model,
    )
    _plot(
        args.figure,
        model_rows=model_rows,
        baseline_rows=baseline_rows,
        models=models,
        completed_by_model=completed_by_model,
    )
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
