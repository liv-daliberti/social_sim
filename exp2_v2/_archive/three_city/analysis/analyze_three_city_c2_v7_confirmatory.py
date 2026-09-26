#!/usr/bin/env python3
"""Confirmatory paired analysis and paper figure for three-city C2 v7."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v7 import PREFIX_LADDER

_DATA = _ROOT / "data" / "three_city_c2_v7"
_RESPONSES = _DATA / "confirmatory"
_ANSWER_KEY = _DATA / "answer_key_c2_v7.jsonl"
_OUTDIR = _DATA / "confirmatory"
_PRIMARY_CONDITION = "relevant"
_PRIMARY_K = 2
_BOOTSTRAP = 10_000
_BOOTSTRAP_SEED = 20260728

_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}
_MODEL_ORDER = tuple(_DISPLAY)
_BLIND = "#4A4A4A"
_HINT = "#168A72"
_NAIVE = "#B28B00"
_TARGET = "#E68613"
_HIERARCHICAL = "#5A9E52"
_GRID = "#D8D8D8"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _latest_responses(paths: Iterable[Path]) -> dict[tuple[str, str, str], dict]:
    latest: dict[tuple[str, str, str], dict] = {}
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


def _percentile(
    values: Sequence[float],
    probability: float,
) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(probability * len(ordered))),
    )
    return ordered[index]


def _ci(values: Sequence[float]) -> tuple[float, float]:
    return (_percentile(values, 0.025), _percentile(values, 0.975))


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(clean) if clean else float("nan")


def _beta(rows: Sequence[Mapping[str, float]]) -> float:
    x = [
        float(row["hierarchical"]) - float(row["target_only"])
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


def _paired_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    condition: str,
    k: int,
) -> list[dict[str, Any]]:
    selected = {
        (row["arm"], row["episode_id"]): row
        for row in rows
        if row["model"] == model
        and row["condition"] == condition
        and row["k"] == k
    }
    episodes = sorted(
        {
            episode
            for arm, episode in selected
            if ("blind", episode) in selected
            and ("hint", episode) in selected
        }
    )
    return [
        {
            "episode_id": episode,
            "blind": selected[("blind", episode)],
            "hint": selected[("hint", episode)],
        }
        for episode in episodes
    ]


def _bootstrap_primary(
    paired: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(paired) < 3:
        return {
            "n": len(paired),
            "blind_mae": float("nan"),
            "hint_mae": float("nan"),
            "mae_improvement": float("nan"),
            "mae_improvement_ci": [float("nan"), float("nan")],
            "blind_beta": float("nan"),
            "blind_beta_ci": [float("nan"), float("nan")],
            "hint_beta": float("nan"),
            "hint_beta_ci": [float("nan"), float("nan")],
            "beta_improvement": float("nan"),
            "beta_improvement_ci": [float("nan"), float("nan")],
        }
    blind_rows = [pair["blind"] for pair in paired]
    hint_rows = [pair["hint"] for pair in paired]
    blind_mae = _mean(row["absolute_error"] for row in blind_rows)
    hint_mae = _mean(row["absolute_error"] for row in hint_rows)
    blind_beta = _beta(blind_rows)
    hint_beta = _beta(hint_rows)
    rng = random.Random(seed)
    mae_differences = []
    beta_differences = []
    blind_betas = []
    hint_betas = []
    size = len(paired)
    for _ in range(_BOOTSTRAP):
        sample = [paired[rng.randrange(size)] for _ in range(size)]
        sampled_blind = [pair["blind"] for pair in sample]
        sampled_hint = [pair["hint"] for pair in sample]
        mae_differences.append(
            _mean(row["absolute_error"] for row in sampled_blind)
            - _mean(row["absolute_error"] for row in sampled_hint)
        )
        sampled_blind_beta = _beta(sampled_blind)
        sampled_hint_beta = _beta(sampled_hint)
        if (
            math.isfinite(sampled_blind_beta)
            and math.isfinite(sampled_hint_beta)
        ):
            blind_betas.append(sampled_blind_beta)
            hint_betas.append(sampled_hint_beta)
            beta_differences.append(
                sampled_hint_beta - sampled_blind_beta
            )
    return {
        "n": len(paired),
        "blind_mae": blind_mae,
        "hint_mae": hint_mae,
        "mae_improvement": blind_mae - hint_mae,
        "mae_improvement_ci": list(_ci(mae_differences)),
        "blind_beta": blind_beta,
        "blind_beta_ci": list(_ci(blind_betas)),
        "hint_beta": hint_beta,
        "hint_beta_ci": list(_ci(hint_betas)),
        "beta_improvement": hint_beta - blind_beta,
        "beta_improvement_ci": list(_ci(beta_differences)),
    }


def _bootstrap_arm_metric(
    paired: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    metric: str,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    rows = [pair[arm] for pair in paired]
    if metric == "mae":
        estimate = _mean(row["absolute_error"] for row in rows)
        statistic = lambda sample: _mean(
            row["absolute_error"] for row in sample
        )
    elif metric == "beta":
        estimate = _beta(rows)
        statistic = _beta
    else:
        raise ValueError(metric)
    if len(rows) < 3:
        return estimate, (float("nan"), float("nan"))
    rng = random.Random(seed)
    samples = []
    for _ in range(_BOOTSTRAP):
        draw = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        value = statistic(draw)
        if math.isfinite(value):
            samples.append(value)
    return estimate, _ci(samples)


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
                "prediction": prediction,
                "gold": gold,
                "absolute_error": abs(prediction - gold),
                "naive": float(baselines["naive"]["predicted_poll"]),
                "target_only": float(
                    baselines["target_only"]["predicted_poll"]
                ),
                "hierarchical": float(
                    baselines["empirical_hierarchical"]["predicted_poll"]
                ),
                "pooled_references": float(
                    baselines["pooled_references"]["predicted_poll"]
                ),
                "rationale": response.get("rationale", ""),
            }
        )
    return rows


def _baseline_mae(
    keys: Mapping[str, Mapping[str, Any]],
    *,
    condition: str,
    k: int,
    method: str,
) -> float:
    selected = [
        key
        for key in keys.values()
        if key["condition"] == condition and int(key["k"]) == k
    ]
    return _mean(
        abs(
            float(key["baselines"][method]["predicted_poll"])
            - float(key["gold"]["expected_poll"])
        )
        for key in selected
    )


def _curve_results(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for model_index, model in enumerate(models):
        curves[model] = {}
        for k in PREFIX_LADDER:
            paired = _paired_rows(
                rows,
                model=model,
                condition=_PRIMARY_CONDITION,
                k=k,
            )
            curves[model][str(k)] = {"n": len(paired)}
            for arm_index, arm in enumerate(("blind", "hint")):
                estimate, ci = _bootstrap_arm_metric(
                    paired,
                    arm=arm,
                    metric="mae",
                    seed=(
                        _BOOTSTRAP_SEED
                        + 1000 * model_index
                        + 100 * k
                        + arm_index
                    ),
                )
                curves[model][str(k)][arm] = {
                    "mae": estimate,
                    "ci": list(ci),
                }
    return curves


def _secondary_results(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
) -> dict[str, Any]:
    """Descriptive secondary estimates for every frozen condition and prefix."""
    results: dict[str, Any] = {}
    for model in models:
        results[model] = {}
        for condition in ("relevant", "none", "orthogonal"):
            results[model][condition] = {}
            for k in PREFIX_LADDER:
                paired = _paired_rows(
                    rows,
                    model=model,
                    condition=condition,
                    k=k,
                )
                blind = [pair["blind"] for pair in paired]
                hint = [pair["hint"] for pair in paired]
                blind_mae = _mean(
                    row["absolute_error"] for row in blind
                )
                hint_mae = _mean(
                    row["absolute_error"] for row in hint
                )
                blind_beta = _beta(blind)
                hint_beta = _beta(hint)
                results[model][condition][str(k)] = {
                    "n": len(paired),
                    "blind_mae": blind_mae,
                    "hint_mae": hint_mae,
                    "mae_improvement": blind_mae - hint_mae,
                    "blind_beta": blind_beta,
                    "hint_beta": hint_beta,
                    "beta_improvement": hint_beta - blind_beta,
                }
    return results


def _context_weight_sensitivity(
    keys: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for k in PREFIX_LADDER:
        selected = [
            key
            for key in keys.values()
            if key["condition"] == "relevant" and int(key["k"]) == k
        ]
        weight_keys = tuple(
            selected[0]["baselines"][
                "empirical_hierarchical_context_sensitivity"
            ]
        )
        results[str(k)] = {
            weight: _mean(
                abs(
                    key["baselines"][
                        "empirical_hierarchical_context_sensitivity"
                    ][weight]["predicted_poll"]
                    - key["gold"]["expected_poll"]
                )
                for key in selected
            )
            for weight in weight_keys
        }
    return results


def _draw_schematic(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_title(
        "(A) Latent structure is available—but undisclosed",
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    boxes = (
        (0.02, 0.56, 0.26, 0.22, "CITY A", "background + cases\nresponse pattern 1"),
        (0.37, 0.56, 0.26, 0.22, "CITY B", "background + cases\nresponse pattern 2"),
        (0.71, 0.56, 0.26, 0.22, "CITY C", "target background\n0–4 cases"),
    )
    colors = ("#707070", "#707070", "#587B9B")
    for (x, y, width, height, title, text), color in zip(boxes, colors):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height * 0.70,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            x + width / 2,
            y + height * 0.33,
            text,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#444444",
            transform=axis.transAxes,
        )
    axis.text(
        0.325,
        0.67,
        "+",
        ha="center",
        va="center",
        fontsize=13,
        color="#777777",
        transform=axis.transAxes,
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.63, 0.67),
            (0.71, 0.67),
            arrowstyle="-|>",
            mutation_scale=9,
            color="#777777",
            linewidth=1,
            transform=axis.transAxes,
        )
    )
    axis.text(
        0.5,
        0.46,
        "Forecast the next +8-news poll",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        transform=axis.transAxes,
    )
    blind = FancyBboxPatch(
        (0.05, 0.13),
        0.40,
        0.25,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1,
        edgecolor=_BLIND,
        facecolor="#F7F7F7",
        transform=axis.transAxes,
    )
    hint = FancyBboxPatch(
        (0.55, 0.13),
        0.40,
        0.25,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=_HINT,
        facecolor="#EEF7F3",
        transform=axis.transAxes,
    )
    axis.add_patch(blind)
    axis.add_patch(hint)
    axis.text(
        0.25,
        0.32,
        "STRUCTURE-BLIND",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=_BLIND,
        transform=axis.transAxes,
    )
    axis.text(
        0.25,
        0.235,
        "neutral forecast request",
        ha="center",
        va="center",
        fontsize=7,
        transform=axis.transAxes,
    )
    axis.text(
        0.75,
        0.32,
        "RELEVANCE HINT",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=_HINT,
        transform=axis.transAxes,
    )
    axis.text(
        0.75,
        0.225,
        "“When forecasting City C,\n"
        "consider whether the earlier cities’\n"
        "response patterns and background\n"
        "descriptions are informative.”",
        ha="center",
        va="center",
        fontsize=5.4,
        transform=axis.transAxes,
    )
    axis.text(
        0.30,
        0.06,
        "Naive",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=_NAIVE,
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.06,
        "City C only",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=_TARGET,
        transform=axis.transAxes,
    )
    axis.text(
        0.75,
        0.06,
        "Hierarchical",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=_HIERARCHICAL,
        transform=axis.transAxes,
    )


def _draw_beta_panel(
    axis: plt.Axes,
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
) -> None:
    axis.set_title(
        "(B) Structure use at the preregistered $k=2$",
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    y_positions = list(reversed(range(len(models))))
    all_limits = []
    paired_counts = []
    for model_index, (model, y) in enumerate(zip(models, y_positions)):
        paired = _paired_rows(
            rows,
            model=model,
            condition=_PRIMARY_CONDITION,
            k=_PRIMARY_K,
        )
        paired_counts.append(len(paired))
        for arm_index, (arm, offset, color, marker) in enumerate(
            (
                ("blind", 0.12, _BLIND, "o"),
                ("hint", -0.12, _HINT, "s"),
            )
        ):
            estimate, ci = _bootstrap_arm_metric(
                paired,
                arm=arm,
                metric="beta",
                seed=(
                    _BOOTSTRAP_SEED
                    + 10_000
                    + 1000 * model_index
                    + arm_index
                ),
            )
            all_limits.extend(ci)
            axis.errorbar(
                estimate,
                y + offset,
                xerr=[
                    [max(0.0, estimate - ci[0])],
                    [max(0.0, ci[1] - estimate)],
                ],
                fmt=marker,
                color=color,
                markerfacecolor=(
                    "white" if arm == "blind" else color
                ),
                markeredgewidth=1.2,
                markersize=5.5,
                linewidth=1.2,
                capsize=2.5,
                label=(
                    "Structure-blind"
                    if model_index == 0 and arm == "blind"
                    else (
                        "Relevance hint"
                        if model_index == 0 and arm == "hint"
                        else None
                    )
                ),
            )
    axis.axvline(
        0,
        color=_TARGET,
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        label="City C only ($\\beta=0$)",
    )
    axis.axvline(
        1,
        color=_HIERARCHICAL,
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        label="Hierarchical ($\\beta=1$)",
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [
            f"{_DISPLAY[model]} ($n={count}$)"
            for model, count in zip(models, paired_counts)
        ],
        fontsize=8,
    )
    finite_limits = [value for value in all_limits if math.isfinite(value)]
    lower = min([-0.2] + finite_limits)
    upper = max([1.2] + finite_limits)
    margin = 0.08 * max(1.0, upper - lower)
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_xlabel("Structure-use coefficient $\\beta$", fontsize=8.5)
    axis.grid(axis="x", color=_GRID, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)


def _draw_mae_panel(
    axis: plt.Axes,
    *,
    model: str,
    model_index: int,
    rows: Sequence[dict[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
) -> None:
    ks = list(PREFIX_LADDER)
    paired_counts = [
        len(
            _paired_rows(
                rows,
                model=model,
                condition=_PRIMARY_CONDITION,
                k=k,
            )
        )
        for k in ks
    ]
    for arm_index, (arm, color, marker, label) in enumerate(
        (
            ("blind", _BLIND, "o", "Structure-blind"),
            ("hint", _HINT, "s", "Relevance hint"),
        )
    ):
        estimates = []
        lows = []
        highs = []
        for k in ks:
            paired = _paired_rows(
                rows,
                model=model,
                condition=_PRIMARY_CONDITION,
                k=k,
            )
            estimate, ci = _bootstrap_arm_metric(
                paired,
                arm=arm,
                metric="mae",
                seed=(
                    _BOOTSTRAP_SEED
                    + 20_000
                    + 1000 * model_index
                    + 100 * k
                    + arm_index
                ),
            )
            estimates.append(estimate)
            lows.append(ci[0])
            highs.append(ci[1])
        axis.plot(
            ks,
            estimates,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.7,
            markerfacecolor=("white" if arm == "blind" else color),
            markeredgewidth=1,
            label=label,
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
    target = [
        _baseline_mae(
            keys,
            condition=_PRIMARY_CONDITION,
            k=k,
            method="target_only",
        )
        for k in ks
    ]
    hierarchy = [
        _baseline_mae(
            keys,
            condition=_PRIMARY_CONDITION,
            k=k,
            method="empirical_hierarchical",
        )
        for k in ks
    ]
    axis.plot(
        ks,
        target,
        color=_TARGET,
        linestyle=(0, (3, 2)),
        linewidth=1.3,
        label="City C only",
        zorder=3,
    )
    axis.plot(
        ks,
        hierarchy,
        color=_HIERARCHICAL,
        linestyle=(0, (3, 2)),
        linewidth=1.3,
        label="Hierarchical",
        zorder=3,
    )
    axis.axvline(
        _PRIMARY_K,
        color="#BBBBBB",
        linewidth=0.8,
        zorder=1,
    )
    if len(set(paired_counts)) == 1:
        count_label = f"paired $n={paired_counts[0]}$ at each $k$"
    else:
        count_label = (
            f"paired $n={min(paired_counts)}$–{max(paired_counts)}$ across $k$"
        )
    axis.set_title(
        f"{_DISPLAY[model]}\n{count_label}",
        fontsize=8.3,
        pad=4,
    )
    axis.set_xticks(ks)
    axis.set_xlabel("City C cases $k$", fontsize=8)
    if model_index == 0:
        axis.set_ylabel("Forecast MAE (poll points)", fontsize=8)
    axis.grid(color=_GRID, linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=7.5)
    axis.set_ylim(bottom=0)
    if model_index == 0:
        axis.legend(
            fontsize=6.4,
            frameon=False,
            loc="upper right",
            ncol=2,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.4,
        )


def _make_figure(
    path: Path,
    *,
    rows: Sequence[dict[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(7.25, 5.55))
    grid = figure.add_gridspec(
        2,
        6,
        height_ratios=[1.02, 1.0],
        hspace=0.76,
        wspace=0.70,
    )
    schematic = figure.add_subplot(grid[0, :3])
    beta_axis = figure.add_subplot(grid[0, 3:])
    _draw_schematic(schematic)
    _draw_beta_panel(beta_axis, rows, models)

    figure.text(
        0.02,
        0.465,
        "(C) Forecast accuracy with relevant context as City C evidence accumulates",
        fontsize=10,
        fontweight="bold",
    )
    mae_axes = []
    for index, model in enumerate(models):
        axis = figure.add_subplot(grid[1, 2 * index : 2 * index + 2])
        mae_axes.append(axis)
        _draw_mae_panel(
            axis,
            model=model,
            model_index=index,
            rows=rows,
            keys=keys,
        )
    common_upper = max(axis.get_ylim()[1] for axis in mae_axes)
    for axis in mae_axes:
        axis.set_ylim(0, common_upper)

    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.96,
        bottom=0.095,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_summary(
    path: Path,
    *,
    primary: Mapping[str, Mapping[str, Any]],
    completion: Mapping[str, Mapping[str, Mapping[str, float]]],
    figure_name: str,
) -> None:
    lines = [
        "# Three-city C2 v7 confirmatory results",
        "",
        "The primary comparison was frozen before v7 model calls: truthful "
        f"mechanism-relevant background at `k={_PRIMARY_K}`, paired by episode.",
        "",
        "## Completion",
        "",
        "| Model | Blind responses / parsed | Hint responses / parsed | "
        "Primary paired episodes |",
        "|---|---:|---:|---:|",
    ]
    for model in _MODEL_ORDER:
        if model not in primary:
            continue
        blind = completion[model]["blind"]
        hint = completion[model]["hint"]
        lines.append(
            f"| {_DISPLAY[model]} | "
            f"{blind['responses_received']:.0f} / {blind['parsed']:.0f} "
            f"({blind['parse_rate']:.1%}) | "
            f"{hint['responses_received']:.0f} / {hint['parsed']:.0f} "
            f"({hint['parse_rate']:.1%}) | {primary[model]['n']} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered outcomes",
            "",
            "Positive MAE improvement means the hint reduced error. Positive "
            "beta improvement means the hint moved forecasts toward the "
            "public-data hierarchical reference.",
            "",
            "| Model | Blind MAE | Hint MAE | MAE improvement [95% CI] | "
            "Blind β [95% CI] | Hint β [95% CI] | "
            "β improvement [95% CI] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in _MODEL_ORDER:
        if model not in primary:
            continue
        result = primary[model]
        mae_ci = result["mae_improvement_ci"]
        blind_beta_ci = result["blind_beta_ci"]
        hint_beta_ci = result["hint_beta_ci"]
        beta_ci = result["beta_improvement_ci"]
        lines.append(
            f"| {_DISPLAY[model]} | {result['blind_mae']:.3f} | "
            f"{result['hint_mae']:.3f} | "
            f"{result['mae_improvement']:+.3f} "
            f"[{mae_ci[0]:+.3f}, {mae_ci[1]:+.3f}] | "
            f"{result['blind_beta']:.3f} "
            f"[{blind_beta_ci[0]:+.3f}, {blind_beta_ci[1]:+.3f}] | "
            f"{result['hint_beta']:.3f} "
            f"[{hint_beta_ci[0]:+.3f}, {hint_beta_ci[1]:+.3f}] | "
            f"{result['beta_improvement']:+.3f} "
            f"[{beta_ci[0]:+.3f}, {beta_ci[1]:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "Intervals are deterministic 10,000-draw percentile bootstraps "
            "over episodes. Models are separate replications; no pooled "
            "model-level p-value is reported.",
            "",
            "## Preregistered interpretation rule",
            "",
        ]
    )
    for model in _MODEL_ORDER:
        if model not in primary:
            continue
        result = primary[model]
        strict_support = (
            result["mae_improvement_ci"][0] > 0
            and result["beta_improvement_ci"][0] > 0
        )
        label = (
            "meets"
            if strict_support
            else "does not meet"
        )
        lines.append(
            f"- **{_DISPLAY[model]}:** {label} the frozen elicitation-effect "
            "criterion (both hint-effect intervals entirely above zero)."
        )
    lines.extend(
        [
            "",
            "Blind beta levels are interpreted directly and with their "
            "intervals; failure to reject zero is not treated as proof of no "
            "spontaneous structure use.",
            "",
            "Full descriptive results by context condition and prefix, plus "
            "the frozen context-weight sensitivity, are in "
            "`confirmatory_results.json`.",
            "",
            "## Figure",
            "",
            f"Main paper figure: `{figure_name}` (PDF and PNG).",
            "",
            "Suggested caption: **A minimal relevance cue tests whether latent "
            "structure is available or spontaneously used.** (A) The two "
            "reference cities exhibit distinct, undisclosed response patterns; "
            "blind and hint prompts differ by one sentence. (B) Structure-use "
            "coefficient at the preregistered two-case prefix, where 0 denotes "
            "City-C-only behavior and 1 the public-data hierarchical reference. "
            "(C) Forecast MAE under truthful mechanism-relevant context as "
            "target evidence accumulates. Lines and 95% intervals use the same "
            "paired episode set at each point. Panel-specific paired episode "
            "counts are shown explicitly; ranges are shown when counts differ "
            "across prefixes.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _write_latex(
    path: Path,
    primary: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "% Auto-generated by analyze_three_city_c2_v7_confirmatory.py",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Blind MAE & Hint MAE & $\\Delta$MAE & "
        "Blind $\\beta$ & Hint $\\beta$ & $\\Delta\\beta$ \\\\",
        "\\midrule",
    ]
    for model in _MODEL_ORDER:
        if model not in primary:
            continue
        result = primary[model]
        lines.append(
            f"{_DISPLAY[model]} & {result['blind_mae']:.2f} & "
            f"{result['hint_mae']:.2f} & "
            f"{result['mae_improvement']:+.2f} & "
            f"{result['blind_beta']:.2f} & "
            f"{result['hint_beta']:.2f} & "
            f"{result['beta_improvement']:+.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-key", type=Path, default=_ANSWER_KEY)
    parser.add_argument("--responses-dir", type=Path, default=_RESPONSES)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    args = parser.parse_args()

    keys = {
        row["task_id"]: row for row in _read_jsonl(args.answer_key)
    }
    response_paths = sorted(args.responses_dir.glob("responses_*.jsonl"))
    responses = _latest_responses(response_paths)
    rows = _score_rows(responses, keys)
    models = [
        model
        for model in _MODEL_ORDER
        if any(row["model"] == model for row in rows)
    ]
    if not models:
        raise SystemExit("no parsed v7 confirmatory responses found")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "confirmatory_scored_rows.jsonl"
    with rows_path.open("w") as handle:
        for row in sorted(
            rows,
            key=lambda item: (
                item["model"],
                item["task_id"],
                item["arm"],
            ),
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    primary = {}
    completion = {}
    for model_index, model in enumerate(models):
        paired = _paired_rows(
            rows,
            model=model,
            condition=_PRIMARY_CONDITION,
            k=_PRIMARY_K,
        )
        primary[model] = _bootstrap_primary(
            paired,
            seed=_BOOTSTRAP_SEED + 1000 * model_index,
        )
        completion[model] = {}
        for arm in ("blind", "hint"):
            arm_responses = [
                response
                for (response_model, response_arm, _), response
                in responses.items()
                if response_model == model and response_arm == arm
            ]
            responses_received = sum(
                response.get("predicted_poll") is not None
                or response.get("response_received") is True
                for response in arm_responses
            )
            parsed = sum(
                response.get("predicted_poll") is not None
                for response in arm_responses
            )
            completion[model][arm] = {
                "expected": len(keys),
                "responses_received": responses_received,
                "parsed": parsed,
                "response_rate": responses_received / len(keys),
                "parse_rate": parsed / responses_received
                if responses_received
                else float("nan"),
            }

    curves = _curve_results(rows, models)
    secondary = _secondary_results(rows, models)
    baseline_curves = {
        condition: {
            str(k): {
                method: _baseline_mae(
                    keys,
                    condition=condition,
                    k=k,
                    method=method,
                )
                for method in (
                    "naive",
                    "target_only",
                    "pooled_references",
                    "empirical_hierarchical",
                    "dgp_oracle",
                )
            }
            for k in PREFIX_LADDER
        }
        for condition in ("relevant", "none", "orthogonal")
    }
    results = {
        "experiment": "three_city_c2_v7_confirmatory",
        "primary_condition": _PRIMARY_CONDITION,
        "primary_k": _PRIMARY_K,
        "bootstrap_draws": _BOOTSTRAP,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "primary": primary,
        "completion": completion,
        "relevant_condition_curves": curves,
        "secondary_by_condition_and_k": secondary,
        "baseline_curves": baseline_curves,
        "hierarchical_context_weight_sensitivity": (
            _context_weight_sensitivity(keys)
        ),
    }
    results_path = args.outdir / "confirmatory_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )

    figure_path = args.outdir / "three_city_c2_v7_main.png"
    _make_figure(
        figure_path,
        rows=rows,
        keys=keys,
        models=models,
    )
    _write_summary(
        args.outdir / "confirmatory_summary.md",
        primary=primary,
        completion=completion,
        figure_name=figure_path.name,
    )
    _write_latex(
        args.outdir / "confirmatory_table.tex",
        primary,
    )
    print(json.dumps(primary, indent=2, sort_keys=True))
    print(f"Wrote {results_path}")
    print(f"Wrote {figure_path} and {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
