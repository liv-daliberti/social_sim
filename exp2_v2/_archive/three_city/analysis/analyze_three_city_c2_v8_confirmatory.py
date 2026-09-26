#!/usr/bin/env python3
"""Confirmatory analysis and paper figures for three-city C2 v8."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v8 import (
    CASES_BY_PREFIX,
    HIGH_TYPE,
    PREFIX_LADDER,
    PROMPT_ARMS,
)

_DATA = _ROOT / "data" / "three_city_c2_v8"
_RUN = _DATA / "full_k1_5_similarity"
_RESPONSES = _RUN / "responses"
_ANSWER_KEY = _RUN / "design" / "answer_key_c2_v8.jsonl"
_OUTDIR = _RUN / "analysis"
_START_K = 1
_PRIMARY_K = 2
_CONVERGENCE_K = 5
_EXPECTED_PER_CELL = 120 * 4 * len(PREFIX_LADDER)
_BOOTSTRAP = 10_000
_BOOTSTRAP_SEED = 20260729

_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}
_MODEL_ORDER = tuple(_DISPLAY)
_MODEL_STYLE = {
    "claude-opus-4-8": ("#4C78A8", "o"),
    "DeepSeek-V4-Pro": ("#7A5195", "s"),
    "gpt-5.4": ("#3F3F3F", "^"),
}
_ARM_LABEL = {
    "blind": "Structure-blind",
    "hint": "Relevance hint",
    "strong_hint": "Continuous-pooling hint",
}
_ARM_COLOR = {
    "blind": "#4A4A4A",
    "hint": "#168A72",
    "strong_hint": "#267CB5",
}
_TARGET = "#E68613"
_ABC = "#4C9A4A"
_ORACLE = "#8A8A8A"
_GRID = "#D8D8D8"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_responses(
    paths: Iterable[Path],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for record in _read_jsonl(path):
            key = (record["model"], record["arm"], record["task_id"])
            priority = (
                2
                if record.get("predicted_poll") is not None
                else (1 if record.get("response_received") is True else 0)
            )
            previous = latest.get(key)
            previous_priority = (
                -1
                if previous is None
                else (
                    2
                    if previous.get("predicted_poll") is not None
                    else (
                        1
                        if previous.get("response_received") is True
                        else 0
                    )
                )
            )
            if priority >= previous_priority:
                latest[key] = record
    return latest


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(clean) if clean else float("nan")


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(probability * len(ordered))),
    )
    return ordered[index]


def _ci(values: Sequence[float]) -> list[float]:
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _score_rows(
    responses: Mapping[tuple[str, str, str], Mapping[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (model, arm, task_id), response in responses.items():
        if task_id not in keys or response.get("predicted_poll") is None:
            continue
        key = keys[task_id]
        prediction = float(response["predicted_poll"])
        gold = float(key["gold"]["expected_poll"])
        baselines = key["baselines"]
        rows.append(
            {
                "model": model,
                "arm": arm,
                "task_id": task_id,
                "episode_id": key["episode_id"],
                "condition": key["condition"],
                "k": int(key["k"]),
                "target_type": key["gold"]["target_type"],
                "prediction": prediction,
                "gold": gold,
                "absolute_error": abs(prediction - gold),
                "target_only": float(
                    baselines["target_only"]["predicted_poll"]
                ),
                "abc_shrinkage": float(
                    baselines["abc_shrinkage"]["predicted_poll"]
                ),
                "privileged_structure": float(
                    baselines["privileged_structure_ceiling"][
                        "predicted_poll"
                    ]
                ),
                "privileged_context": float(
                    baselines["privileged_context_oracle"][
                        "predicted_poll"
                    ]
                ),
                "rationale": response.get("rationale", ""),
            }
        )
    return rows


def _paired_structure(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    k: int,
    arms: Sequence[str] = PROMPT_ARMS,
) -> list[dict[str, Any]]:
    selected = {
        (row["arm"], row["episode_id"]): row
        for row in rows
        if row["model"] == model
        and row["condition"] == "none"
        and row["k"] == k
        and row["arm"] in arms
    }
    episodes = sorted(
        {
            episode
            for _, episode in selected
            if all((arm, episode) in selected for arm in arms)
        }
    )
    return [
        {
            "episode_id": episode,
            **{arm: selected[(arm, episode)] for arm in arms},
        }
        for episode in episodes
    ]


def _paired_context(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    k: int,
    arms: Sequence[str] = PROMPT_ARMS,
) -> list[dict[str, Any]]:
    conditions = ("none", "orthogonal", "cue_high", "cue_low")
    selected = {
        (row["arm"], row["condition"], row["episode_id"]): row
        for row in rows
        if row["model"] == model
        and row["k"] == k
        and row["arm"] in arms
        and row["condition"] in conditions
    }
    episodes = sorted(
        {
            episode
            for _, _, episode in selected
            if all(
                (arm, condition, episode) in selected
                for arm in arms
                for condition in conditions
            )
        }
    )
    return [
        {
            "episode_id": episode,
            **{
                arm: {
                    condition: selected[(arm, condition, episode)]
                    for condition in conditions
                }
                for arm in arms
            },
        }
        for episode in episodes
    ]


def _paired_context_override(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    arms: Sequence[str] = PROMPT_ARMS,
) -> list[dict[str, Any]]:
    conditions = ("none", "orthogonal", "cue_high", "cue_low")
    ks = (_START_K, _CONVERGENCE_K)
    selected = {
        (
            row["arm"],
            row["k"],
            row["condition"],
            row["episode_id"],
        ): row
        for row in rows
        if row["model"] == model
        and row["k"] in ks
        and row["arm"] in arms
        and row["condition"] in conditions
    }
    episodes = sorted(
        {
            episode
            for _, _, _, episode in selected
            if all(
                (arm, k, condition, episode) in selected
                for arm in arms
                for k in ks
                for condition in conditions
            )
        }
    )
    return [
        {
            "episode_id": episode,
            **{
                arm: {
                    k: {
                        condition: selected[(arm, k, condition, episode)]
                        for condition in conditions
                    }
                    for k in ks
                }
                for arm in arms
            },
        }
        for episode in episodes
    ]


def _beta(rows: Sequence[Mapping[str, Any]]) -> float:
    x = [
        float(row["abc_shrinkage"]) - float(row["target_only"])
        for row in rows
    ]
    y = [
        float(row["prediction"]) - float(row["target_only"])
        for row in rows
    ]
    if len(x) < 3:
        return float("nan")
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator < 1e-12:
        return float("nan")
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y)
    ) / denominator


def _structure_statistics(
    paired: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
    include_beta: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(paired), "arms": {}}
    for arm in PROMPT_ARMS:
        rows = [pair[arm] for pair in paired]
        mae = _mean(row["absolute_error"] for row in rows)
        target_improvement = _mean(
            abs(float(row["target_only"]) - float(row["gold"]))
            - float(row["absolute_error"])
            for row in rows
        )
        result["arms"][arm] = {
            "mae": mae,
            "target_only_improvement": target_improvement,
        }
        if include_beta:
            result["arms"][arm]["beta"] = _beta(rows)
    if len(paired) < 3:
        for arm in PROMPT_ARMS:
            result["arms"][arm]["mae_ci"] = [float("nan")] * 2
            result["arms"][arm]["target_only_improvement_ci"] = [
                float("nan")
            ] * 2
            if include_beta:
                result["arms"][arm]["beta_ci"] = [float("nan")] * 2
        result["contrasts"] = {}
        return result

    samples = {
        arm: {
            "mae": [],
            "target_only_improvement": [],
            "beta": [],
        }
        for arm in PROMPT_ARMS
    }
    contrasts = {
        "hint_minus_blind": {"mae_improvement": [], "beta_increase": []},
        "strong_minus_blind": {
            "mae_improvement": [],
            "beta_increase": [],
        },
    }
    rng = random.Random(seed)
    for _ in range(draws):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        estimates: dict[str, dict[str, float]] = {}
        for arm in PROMPT_ARMS:
            rows = [pair[arm] for pair in sample]
            estimates[arm] = {
                "mae": _mean(row["absolute_error"] for row in rows),
                "target_only_improvement": _mean(
                    abs(float(row["target_only"]) - float(row["gold"]))
                    - float(row["absolute_error"])
                    for row in rows
                ),
                "beta": _beta(rows) if include_beta else float("nan"),
            }
            for metric, value in estimates[arm].items():
                if math.isfinite(value):
                    samples[arm][metric].append(value)
        for name, arm in (
            ("hint_minus_blind", "hint"),
            ("strong_minus_blind", "strong_hint"),
        ):
            contrasts[name]["mae_improvement"].append(
                estimates["blind"]["mae"] - estimates[arm]["mae"]
            )
            if include_beta:
                value = estimates[arm]["beta"] - estimates["blind"]["beta"]
                if math.isfinite(value):
                    contrasts[name]["beta_increase"].append(value)

    for arm in PROMPT_ARMS:
        result["arms"][arm]["mae_ci"] = _ci(samples[arm]["mae"])
        result["arms"][arm]["target_only_improvement_ci"] = _ci(
            samples[arm]["target_only_improvement"]
        )
        if include_beta:
            result["arms"][arm]["beta_ci"] = _ci(
                samples[arm]["beta"]
            )
    result["contrasts"] = {}
    for name, arm in (
        ("hint_minus_blind", "hint"),
        ("strong_minus_blind", "strong_hint"),
    ):
        result["contrasts"][name] = {
            "mae_improvement": (
                result["arms"]["blind"]["mae"]
                - result["arms"][arm]["mae"]
            ),
            "mae_improvement_ci": _ci(
                contrasts[name]["mae_improvement"]
            ),
        }
        if include_beta:
            result["contrasts"][name].update(
                {
                    "beta_increase": (
                        result["arms"][arm]["beta"]
                        - result["arms"]["blind"]["beta"]
                    ),
                    "beta_increase_ci": _ci(
                        contrasts[name]["beta_increase"]
                    ),
                }
            )
    return result


def _context_values(
    cell: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    none = float(cell["none"]["prediction"])
    high = float(cell["cue_high"]["prediction"])
    low = float(cell["cue_low"]["prediction"])
    orthogonal = float(cell["orthogonal"]["prediction"])
    target_type = cell["none"]["target_type"]
    wrong = low if target_type == HIGH_TYPE else high
    return {
        "relevant_movement": 0.5 * (abs(high - none) + abs(low - none)),
        "orthogonal_movement": abs(orthogonal - none),
        "cue_direction": 1.0 if high > low else (0.5 if high == low else 0.0),
        "misleading_displacement": abs(wrong - none),
    }


def _context_statistics(
    paired: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "relevant_movement",
        "orthogonal_movement",
        "cue_direction",
        "misleading_displacement",
    )
    result: dict[str, Any] = {"n": len(paired), "arms": {}}
    values = {
        arm: [_context_values(pair[arm]) for pair in paired]
        for arm in PROMPT_ARMS
    }
    for arm in PROMPT_ARMS:
        result["arms"][arm] = {
            metric: _mean(row[metric] for row in values[arm])
            for metric in metrics
        }
    if len(paired) < 3:
        for arm in PROMPT_ARMS:
            for metric in metrics:
                result["arms"][arm][metric + "_ci"] = [float("nan")] * 2
        return result
    samples = {
        arm: {metric: [] for metric in metrics}
        for arm in PROMPT_ARMS
    }
    rng = random.Random(seed)
    for _ in range(draws):
        indexes = [rng.randrange(len(paired)) for _ in paired]
        for arm in PROMPT_ARMS:
            for metric in metrics:
                samples[arm][metric].append(
                    _mean(values[arm][index][metric] for index in indexes)
                )
    for arm in PROMPT_ARMS:
        for metric in metrics:
            result["arms"][arm][metric + "_ci"] = _ci(
                samples[arm][metric]
            )
    return result


def _override_statistics(
    paired: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    per_arm: dict[str, list[float]] = {arm: [] for arm in PROMPT_ARMS}
    for pair in paired:
        for arm in PROMPT_ARMS:
            start = _context_values(pair[arm][_START_K])[
                "misleading_displacement"
            ]
            end = _context_values(pair[arm][_CONVERGENCE_K])[
                "misleading_displacement"
            ]
            per_arm[arm].append(start - end)
    result = {
        "n": len(paired),
        "arms": {
            arm: {"override_reduction": _mean(per_arm[arm])}
            for arm in PROMPT_ARMS
        },
    }
    if len(paired) < 3:
        for arm in PROMPT_ARMS:
            result["arms"][arm]["override_reduction_ci"] = [
                float("nan")
            ] * 2
        return result
    rng = random.Random(seed)
    samples = {arm: [] for arm in PROMPT_ARMS}
    for _ in range(draws):
        indexes = [rng.randrange(len(paired)) for _ in paired]
        for arm in PROMPT_ARMS:
            samples[arm].append(
                _mean(per_arm[arm][index] for index in indexes)
            )
    for arm in PROMPT_ARMS:
        result["arms"][arm]["override_reduction_ci"] = _ci(samples[arm])
    return result


def _baseline_mae(
    keys: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
    k: int,
) -> float:
    selected = [
        row
        for row in keys.values()
        if row["condition"] == "none" and int(row["k"]) == k
    ]
    return _mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in selected
    )


def _oracle_context_curves(
    keys: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[float]]:
    grouped = defaultdict(dict)
    for row in keys.values():
        grouped[(row["episode_id"], int(row["k"]))][row["condition"]] = row
    relevant = []
    misleading = []
    for k in PREFIX_LADDER:
        r_values = []
        w_values = []
        for (episode, group_k), conditions in grouped.items():
            if group_k != k or len(conditions) != 4:
                continue
            prediction = {
                condition: float(
                    row["baselines"]["privileged_context_oracle"][
                        "predicted_poll"
                    ]
                )
                for condition, row in conditions.items()
            }
            none = prediction["none"]
            high = prediction["cue_high"]
            low = prediction["cue_low"]
            r_values.append(
                0.5 * (abs(high - none) + abs(low - none))
            )
            target_type = conditions["none"]["gold"]["target_type"]
            wrong = low if target_type == HIGH_TYPE else high
            w_values.append(abs(wrong - none))
        relevant.append(_mean(r_values))
        misleading.append(_mean(w_values))
    return {"relevant_movement": relevant, "misleading_displacement": misleading}


def _collection_audit(
    response_dir: Path,
    *,
    models: Sequence[str],
    allow_incomplete: bool,
) -> dict[str, Any]:
    manifests = [
        json.loads(path.read_text())
        for path in sorted(response_dir.glob("responses_*.manifest.json"))
    ]
    by_key = {
        (record.get("model"), record.get("arm")): record
        for record in manifests
    }
    failures = []
    cells = {}
    for model in models:
        cells[model] = {}
        for arm in PROMPT_ARMS:
            record = by_key.get((model, arm))
            if record is None:
                failures.append(f"missing manifest: {model}/{arm}")
                cells[model][arm] = {}
                continue
            cells[model][arm] = {
                key: record.get(key)
                for key in (
                    "status",
                    "selected_model_calls",
                    "responses_received",
                    "successfully_parsed",
                    "endpoint",
                    "updated_at",
                )
            }
            if record.get("status") != "complete":
                failures.append(
                    f"incomplete: {model}/{arm}={record.get('status')}"
                )
            if record.get("selected_model_calls") != _EXPECTED_PER_CELL:
                failures.append(
                    f"wrong call count: {model}/{arm}="
                    f"{record.get('selected_model_calls')}"
                )
            if record.get("responses_received") != _EXPECTED_PER_CELL:
                failures.append(
                    f"missing received responses: {model}/{arm}="
                    f"{record.get('responses_received')}"
                )
    if failures and not allow_incomplete:
        raise SystemExit(
            "collection audit failed: " + "; ".join(failures[:10])
        )
    return {
        "valid": not failures,
        "allow_incomplete": allow_incomplete,
        "failures": failures,
        "cells": cells,
    }


def _make_structure_figure(
    path: Path,
    *,
    curves: Mapping[str, Any],
    keys: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    interim: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(9.6, 2.75),
        sharex=True,
        sharey=True,
    )
    ks = list(PREFIX_LADDER)
    target = [
        _baseline_mae(keys, method="target_only", k=k) for k in ks
    ]
    abc = [
        _baseline_mae(keys, method="abc_shrinkage", k=k) for k in ks
    ]
    common_upper = 0.0
    for axis, arm, panel in zip(axes, PROMPT_ARMS, ("A", "B", "C")):
        for model in models:
            color, marker = _MODEL_STYLE[model]
            estimates = [
                curves[model][str(k)]["arms"][arm]["mae"]
                for k in ks
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
            color=_TARGET,
            linestyle=(0, (3, 2)),
            linewidth=1.35,
            zorder=3,
        )
        axis.plot(
            ks,
            abc,
            color=_ABC,
            linestyle=(0, (3, 2)),
            linewidth=1.35,
            zorder=3,
        )
        axis.axvline(_PRIMARY_K, color="#BBBBBB", linewidth=0.7, zorder=1)
        axis.axvline(
            _CONVERGENCE_K,
            color="#DDDDDD",
            linewidth=0.7,
            zorder=1,
        )
        axis.set_title(
            f"({panel}) {_ARM_LABEL[arm]}",
            fontsize=9.2,
            fontweight="bold",
            color=_ARM_COLOR[arm],
            pad=5,
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8)
        axis.grid(color=_GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.5)
        common_upper = max(common_upper, axis.get_ylim()[1])
    axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8)
    for axis in axes:
        axis.set_ylim(0, common_upper)
    handles = [
        Line2D(
            [0],
            [0],
            color=_MODEL_STYLE[model][0],
            marker=_MODEL_STYLE[model][1],
            linewidth=1.7,
            markersize=4.5,
            label=_DISPLAY[model],
        )
        for model in models
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=_TARGET,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                label="Use City C only",
            ),
            Line2D(
                [0],
                [0],
                color=_ABC,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                label="Combine A/B/C (matched)",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(handles),
        frameon=False,
        fontsize=6.6,
        columnspacing=0.75,
        handlelength=1.7,
        handletextpad=0.35,
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Round 1 begins with one completed City C case. Round 2 is the "
            "sparse anchor; rounds 1--5 show 1, 2, 4, 8, and 16 completed C cases."
        ),
        ha="center",
        fontsize=6.7,
        color="#555555",
    )
    if interim:
        figure.text(
            0.5,
            1.115,
            "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#A61B1B",
        )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _make_context_figure(
    path: Path,
    *,
    context: Mapping[str, Any],
    oracle: Mapping[str, Sequence[float]],
    models: Sequence[str],
    interim: bool,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.6, 5.0),
        sharex=True,
        sharey="row",
    )
    ks = list(PREFIX_LADDER)
    for column, arm in enumerate(PROMPT_ARMS):
        top = axes[0, column]
        bottom = axes[1, column]
        for model in models:
            color, marker = _MODEL_STYLE[model]
            relevant = [
                context[model][str(k)]["arms"][arm][
                    "relevant_movement"
                ]
                for k in ks
            ]
            orthogonal = [
                context[model][str(k)]["arms"][arm][
                    "orthogonal_movement"
                ]
                for k in ks
            ]
            misleading = [
                context[model][str(k)]["arms"][arm][
                    "misleading_displacement"
                ]
                for k in ks
            ]
            top.plot(
                ks,
                relevant,
                color=color,
                marker=marker,
                markerfacecolor="white",
                linewidth=1.6,
                markersize=4,
                zorder=4,
            )
            top.plot(
                ks,
                orthogonal,
                color=color,
                linestyle=(0, (1, 2)),
                linewidth=1.1,
                alpha=0.75,
                zorder=3,
            )
            bottom.plot(
                ks,
                misleading,
                color=color,
                marker=marker,
                markerfacecolor="white",
                linewidth=1.6,
                markersize=4,
                zorder=4,
            )
        top.plot(
            ks,
            oracle["relevant_movement"],
            color=_ORACLE,
            linestyle=(0, (4, 2)),
            linewidth=1.15,
        )
        bottom.plot(
            ks,
            oracle["misleading_displacement"],
            color=_ORACLE,
            linestyle=(0, (4, 2)),
            linewidth=1.15,
        )
        top.set_title(
            _ARM_LABEL[arm],
            fontsize=9,
            fontweight="bold",
            color=_ARM_COLOR[arm],
            pad=4,
        )
        for axis in (top, bottom):
            axis.set_xticks(ks)
            axis.grid(color=_GRID, linewidth=0.55)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", labelsize=7.5)
        bottom.set_xlabel("Evidence round $k$", fontsize=8)
    axes[0, 0].set_ylabel(
        "Context movement\n(poll points)",
        fontsize=8,
    )
    axes[1, 0].set_ylabel(
        "Misleading-cue displacement\n(poll points)",
        fontsize=8,
    )
    figure.text(
        0.015,
        0.96,
        "(A) Relevant context should move forecasts more than orthogonal prose",
        fontsize=9.5,
        fontweight="bold",
    )
    figure.text(
        0.015,
        0.49,
        "(B) Diagnostic City C evidence should override a misleading context",
        fontsize=9.5,
        fontweight="bold",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=_MODEL_STYLE[model][0],
            marker=_MODEL_STYLE[model][1],
            linewidth=1.6,
            markersize=4,
            label=_DISPLAY[model],
        )
        for model in models
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.6,
                label="Relevant cue",
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=(0, (1, 2)),
                linewidth=1.1,
                label="Orthogonal prose",
            ),
            Line2D(
                [0],
                [0],
                color=_ORACLE,
                linestyle=(0, (4, 2)),
                linewidth=1.15,
                label="Privileged 80% benchmark",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=len(handles),
        frameon=False,
        fontsize=6.2,
        columnspacing=0.65,
        handlelength=1.6,
        handletextpad=0.35,
    )
    if interim:
        figure.text(
            0.5,
            1.09,
            "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#A61B1B",
        )
    figure.tight_layout(rect=(0.03, 0.02, 1, 0.93), h_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _format(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.2f}"


def _write_primary_csv(
    path: Path,
    *,
    primary: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "arm",
                "n",
                "mae",
                "mae_ci_low",
                "mae_ci_high",
                "target_only_improvement",
                "target_only_improvement_ci_low",
                "target_only_improvement_ci_high",
                "beta",
                "beta_ci_low",
                "beta_ci_high",
            ),
        )
        writer.writeheader()
        for model in models:
            for arm in PROMPT_ARMS:
                row = primary[model]["arms"][arm]
                writer.writerow(
                    {
                        "model": model,
                        "arm": arm,
                        "n": primary[model]["n"],
                        "mae": row["mae"],
                        "mae_ci_low": row["mae_ci"][0],
                        "mae_ci_high": row["mae_ci"][1],
                        "target_only_improvement": row[
                            "target_only_improvement"
                        ],
                        "target_only_improvement_ci_low": row[
                            "target_only_improvement_ci"
                        ][0],
                        "target_only_improvement_ci_high": row[
                            "target_only_improvement_ci"
                        ][1],
                        "beta": row["beta"],
                        "beta_ci_low": row["beta_ci"][0],
                        "beta_ci_high": row["beta_ci"][1],
                    }
                )


def _write_summary(
    path: Path,
    *,
    primary: Mapping[str, Any],
    override: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    lines = [
        "# Three-city C2 v8 five-round graded-similarity matched-estimator summary",
        "",
        "## Main no-target-cue structure test at round 2",
        "",
        "| Model | Prompt arm | n | MAE [95% CI] | Improvement over C only | beta [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        for arm in PROMPT_ARMS:
            row = primary[model]["arms"][arm]
            lines.append(
                "| "
                + " | ".join(
                    (
                        _DISPLAY[model],
                        _ARM_LABEL[arm],
                        str(primary[model]["n"]),
                        (
                            f"{_format(row['mae'])} "
                            f"[{_format(row['mae_ci'][0])}, "
                            f"{_format(row['mae_ci'][1])}]"
                        ),
                        (
                            f"{_format(row['target_only_improvement'])} "
                            f"[{_format(row['target_only_improvement_ci'][0])}, "
                            f"{_format(row['target_only_improvement_ci'][1])}]"
                        ),
                        (
                            f"{_format(row['beta'])} "
                            f"[{_format(row['beta_ci'][0])}, "
                            f"{_format(row['beta_ci'][1])}]"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Misleading-context override from round 1 to round 5",
            "",
            "| Model | Prompt arm | n | Reduction in misleading-cue displacement [95% CI] |",
            "|---|---|---:|---:|",
        ]
    )
    for model in models:
        for arm in PROMPT_ARMS:
            row = override[model]["arms"][arm]
            lines.append(
                "| "
                + " | ".join(
                    (
                        _DISPLAY[model],
                        _ARM_LABEL[arm],
                        str(override[model]["n"]),
                        (
                            f"{_format(row['override_reduction'])} "
                            f"[{_format(row['override_reduction_ci'][0])}, "
                            f"{_format(row['override_reduction_ci'][1])}]"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            (
                "Positive improvement means lower error than the City-C-only "
                "regression. beta=0 follows City C only; beta=1 matches the "
                "prompt-visible continuous A/B/C shrinkage estimate. Positive override "
                "means misleading-context displacement is smaller at k=5."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-key", type=Path, default=_ANSWER_KEY)
    parser.add_argument("--responses-dir", type=Path, default=_RESPONSES)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--bootstrap-draws", type=int, default=_BOOTSTRAP)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--interim", action="store_true")
    args = parser.parse_args()

    keys = {
        row["task_id"]: row for row in _read_jsonl(args.answer_key)
    }
    responses = _latest_responses(
        sorted(args.responses_dir.glob("responses_*.jsonl"))
    )
    rows = _score_rows(responses, keys)
    models = [
        model
        for model in _MODEL_ORDER
        if any(row["model"] == model for row in rows)
    ]
    if not models:
        raise SystemExit("no parsed responses available")
    audit = _collection_audit(
        args.responses_dir,
        models=models,
        allow_incomplete=args.allow_incomplete or args.interim,
    )

    primary = {}
    curves = {}
    context = {}
    override = {}
    for model_index, model in enumerate(models):
        primary[model] = _structure_statistics(
            _paired_structure(rows, model=model, k=_PRIMARY_K),
            draws=args.bootstrap_draws,
            seed=_BOOTSTRAP_SEED + 1_000 * model_index,
            include_beta=True,
        )
        curves[model] = {}
        context[model] = {}
        for k_index, k in enumerate(PREFIX_LADDER):
            curves[model][str(k)] = _structure_statistics(
                _paired_structure(rows, model=model, k=k),
                draws=args.bootstrap_draws,
                seed=(
                    _BOOTSTRAP_SEED
                    + 10_000
                    + 1_000 * model_index
                    + 10 * k_index
                ),
                include_beta=False,
            )
            context[model][str(k)] = _context_statistics(
                _paired_context(rows, model=model, k=k),
                draws=args.bootstrap_draws,
                seed=(
                    _BOOTSTRAP_SEED
                    + 20_000
                    + 1_000 * model_index
                    + 10 * k_index
                ),
            )
        override[model] = _override_statistics(
            _paired_context_override(rows, model=model),
            draws=args.bootstrap_draws,
            seed=_BOOTSTRAP_SEED + 30_000 + 1_000 * model_index,
        )

    timestamp = datetime.now(timezone.utc)
    status = (
        "interim_incomplete_not_confirmatory"
        if args.interim or not audit["valid"]
        else "final_confirmatory"
    )
    oracle = _oracle_context_curves(keys)
    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = (
        "_" + timestamp.strftime("%Y%m%dT%H%M%SZ")
        if args.interim
        else ""
    )
    structure_path = args.outdir / f"three_city_c2_v8_structure{suffix}.png"
    context_path = args.outdir / f"three_city_c2_v8_context{suffix}.png"
    _make_structure_figure(
        structure_path,
        curves=curves,
        keys=keys,
        models=models,
        interim=status != "final_confirmatory",
    )
    _make_context_figure(
        context_path,
        context=context,
        oracle=oracle,
        models=models,
        interim=status != "final_confirmatory",
    )

    result = {
        "experiment": "three_city_c2_v8_full_k1_5_similarity_matched",
        "status": status,
        "created_at": timestamp.isoformat(),
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "unit": "episode",
            "interval": "percentile_95",
            "pairing": "complete parsed cells required by each contrast",
        },
        "primary_condition": "none",
        "primary_k": _PRIMARY_K,
        "convergence_k": _CONVERGENCE_K,
        "cases_by_round": {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER},
        "models": models,
        "collection_audit": audit,
        "primary_structure_at_k2": primary,
        "structure_curves": curves,
        "context_curves": context,
        "context_override_k1_to_k5": override,
        "privileged_context_benchmark_curves": oracle,
        "figures": [structure_path.name, context_path.name],
        "analyzer_sha256": _file_sha256(Path(__file__)),
    }
    result_path = args.outdir / f"confirmatory_results{suffix}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not args.interim:
        _write_primary_csv(
            args.outdir / "confirmatory_primary.csv",
            primary=primary,
            models=models,
        )
        _write_summary(
            args.outdir / "confirmatory_summary.md",
            primary=primary,
            override=override,
            models=models,
        )
    print(f"Wrote {structure_path} and {structure_path.with_suffix('.pdf')}")
    print(f"Wrote {context_path} and {context_path.with_suffix('.pdf')}")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
