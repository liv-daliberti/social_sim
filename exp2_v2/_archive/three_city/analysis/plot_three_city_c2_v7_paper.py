#!/usr/bin/env python3
"""Paper presentation with models overlaid and prompt arms faceted.

The statistical estimands come from the frozen v7 analyzer. This separate
presentation layer was requested during collection and does not alter task
generation, stopping, scoring, or the preregistered bootstrap. When available,
the post-specified strong-structure arm is added as a clearly labeled positive
control.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

import analyze_three_city_c2_v7_confirmatory as frozen

_DATA = _ROOT / "data" / "three_city_c2_v7"
_STRONG_DATA = _ROOT / "data" / "three_city_c2_v7_strong_hint"
_MODEL_STYLE = {
    "claude-opus-4-8": ("#4C78A8", "o"),
    "DeepSeek-V4-Pro": ("#7A5195", "s"),
    "gpt-5.4": ("#3F3F3F", "^"),
}
_ARM_STYLE = {
    "blind": {
        "label": "Structure-blind",
        "color": frozen._BLIND,
        "marker": "o",
        "fill": "white",
    },
    "hint": {
        "label": "Relevance hint",
        "color": frozen._HINT,
        "marker": "s",
        "fill": frozen._HINT,
    },
    "strong_hint": {
        "label": "Strong structure hint",
        "color": "#267CB5",
        "marker": "D",
        "fill": "#267CB5",
    },
}


def _paired_arm_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    condition: str,
    k: int,
    arms: Sequence[str],
) -> list[dict[str, Any]]:
    """Return episode-paired rows common to every displayed arm."""
    selected = {
        (row["arm"], row["episode_id"]): row
        for row in rows
        if row["model"] == model
        and row["condition"] == condition
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


def _draw_arm_panel(
    axis: plt.Axes,
    *,
    arm: str,
    rows: Sequence[dict[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    comparison_arms: Sequence[str],
) -> dict[str, list[int]]:
    ks = list(frozen.PREFIX_LADDER)
    counts: dict[str, list[int]] = {}
    for model_index, model in enumerate(models):
        color, marker = _MODEL_STYLE[model]
        estimates = []
        lows = []
        highs = []
        counts[model] = []
        for k in ks:
            paired = _paired_arm_rows(
                rows,
                model=model,
                condition=frozen._PRIMARY_CONDITION,
                k=k,
                arms=comparison_arms,
            )
            counts[model].append(len(paired))
            estimate, ci = frozen._bootstrap_arm_metric(
                paired,
                arm=arm,
                metric="mae",
                seed=(
                    frozen._BOOTSTRAP_SEED
                    + 40_000
                    + 2_000 * model_index
                    + 100 * k
                    + comparison_arms.index(arm)
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
            linewidth=1.7,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
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
        frozen._baseline_mae(
            keys,
            condition=frozen._PRIMARY_CONDITION,
            k=k,
            method="target_only",
        )
        for k in ks
    ]
    hierarchy = [
        frozen._baseline_mae(
            keys,
            condition=frozen._PRIMARY_CONDITION,
            k=k,
            method="empirical_hierarchical",
        )
        for k in ks
    ]
    axis.plot(
        ks,
        target,
        color=frozen._TARGET,
        linestyle=(0, (3, 2)),
        linewidth=1.35,
        zorder=3,
    )
    axis.plot(
        ks,
        hierarchy,
        color=frozen._HIERARCHICAL,
        linestyle=(0, (3, 2)),
        linewidth=1.35,
        zorder=3,
    )
    axis.axvline(
        frozen._PRIMARY_K,
        color="#BBBBBB",
        linewidth=0.8,
        zorder=1,
    )
    title = _ARM_STYLE[arm]["label"]
    if arm == "strong_hint":
        title += "\n(post-specified positive control)"
    axis.set_title(
        title,
        fontsize=8.5,
        fontweight="bold",
        color=_ARM_STYLE[arm]["color"],
        pad=4,
    )
    axis.set_xticks(ks)
    axis.set_xticklabels(["0\nbackground only", "1", "2", "4"])
    axis.set_xlabel("Completed City C cases $k$", fontsize=8)
    axis.grid(color=frozen._GRID, linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=7.5)
    axis.set_ylim(bottom=0)
    return counts


def _draw_beta_panel(
    axis: plt.Axes,
    *,
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    arms: Sequence[str],
) -> dict[str, int]:
    y_positions = list(reversed(range(len(models))))
    offsets = {
        arm: offset
        for arm, offset in zip(
            arms,
            (
                [0.12, -0.12]
                if len(arms) == 2
                else [0.18, 0.0, -0.18]
            ),
        )
    }
    all_limits: list[float] = []
    counts: dict[str, int] = {}
    for model_index, (model, y) in enumerate(zip(models, y_positions)):
        paired = _paired_arm_rows(
            rows,
            model=model,
            condition=frozen._PRIMARY_CONDITION,
            k=frozen._PRIMARY_K,
            arms=arms,
        )
        counts[model] = len(paired)
        for arm_index, arm in enumerate(arms):
            style = _ARM_STYLE[arm]
            estimate, ci = frozen._bootstrap_arm_metric(
                paired,
                arm=arm,
                metric="beta",
                seed=(
                    frozen._BOOTSTRAP_SEED
                    + 50_000
                    + 1_000 * model_index
                    + arm_index
                ),
            )
            all_limits.extend(ci)
            axis.errorbar(
                estimate,
                y + offsets[arm],
                xerr=[
                    [max(0.0, estimate - ci[0])],
                    [max(0.0, ci[1] - estimate)],
                ],
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor=style["fill"],
                markeredgewidth=1.1,
                markersize=5.2,
                linewidth=1.15,
                capsize=2.3,
                label=style["label"] if model_index == 0 else None,
            )
    axis.axvline(
        0,
        color=frozen._TARGET,
        linestyle=(0, (3, 2)),
        linewidth=1.2,
    )
    axis.axvline(
        1,
        color=frozen._HIERARCHICAL,
        linestyle=(0, (3, 2)),
        linewidth=1.2,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [
            f"{frozen._DISPLAY[model]} ($n={counts[model]}$)"
            for model in models
        ],
        fontsize=8,
    )
    finite_limits = [value for value in all_limits if math.isfinite(value)]
    lower = min([-0.2] + finite_limits)
    upper = max([1.2] + finite_limits)
    margin = 0.08 * max(1.0, upper - lower)
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_xlabel(
        "Cross-city structure-use coefficient $\\beta$",
        fontsize=8.5,
    )
    axis.grid(axis="x", color=frozen._GRID, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)
    axis.legend(
        loc="lower right",
        ncol=len(arms),
        frameon=False,
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    return counts


def _count_summary(counts: Mapping[str, Sequence[int]]) -> str:
    labels = []
    short = {
        "claude-opus-4-8": "Claude",
        "DeepSeek-V4-Pro": "DeepSeek",
        "gpt-5.4": "GPT-5.4",
    }
    for model in frozen._MODEL_ORDER:
        if model not in counts:
            continue
        values = counts[model]
        value = (
            str(values[0])
            if len(set(values)) == 1
            else f"{min(values)}–{max(values)}"
        )
        labels.append(f"{short[model]}: {value}")
    return "Paired episode n across k — " + "; ".join(labels)


def _make_figure(
    path: Path,
    *,
    rows: Sequence[dict[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    arms: Sequence[str],
    interim: bool,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, int]]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(7.6, 5.5))
    grid = figure.add_gridspec(
        2,
        len(arms),
        height_ratios=[1.02, 1.0],
        hspace=1.10,
        wspace=0.42,
    )
    beta_axis = figure.add_subplot(grid[0, :])
    beta_counts = _draw_beta_panel(
        beta_axis,
        rows=rows,
        models=models,
        arms=arms,
    )
    beta_axis.set_title(
        "(A) Cross-city structure use at $k=2$",
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    figure.text(
        0.02,
        0.478,
        (
            "(B) Forecast error when City C's mechanism-linked background "
            "signals its response pattern"
        ),
        fontsize=9.6,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.455,
        (
            "At $k=0$, models see A/B cases and City C's background; "
            "only completed City C cases are absent."
        ),
        fontsize=6.8,
        color="#555555",
    )
    arm_axes = []
    counts: dict[str, dict[str, list[int]]] = {}
    for column, arm in enumerate(arms):
        axis = figure.add_subplot(grid[1, column])
        arm_axes.append(axis)
        counts[arm] = _draw_arm_panel(
            axis,
            arm=arm,
            rows=rows,
            keys=keys,
            models=models,
            comparison_arms=arms,
        )
    arm_axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8)
    common_upper = max(axis.get_ylim()[1] for axis in arm_axes)
    for axis in arm_axes:
        axis.set_ylim(0, common_upper)

    handles = []
    for model in models:
        values = counts[arms[0]][model]
        count = (
            str(values[0])
            if len(set(values)) == 1
            else f"{min(values)}–{max(values)}"
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=_MODEL_STYLE[model][0],
                marker=_MODEL_STYLE[model][1],
                linewidth=1.7,
                markersize=4.5,
                label=f"{frozen._DISPLAY[model]} ($n={count}$)",
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=frozen._TARGET,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                label="Use City C cases only",
            ),
            Line2D(
                [0],
                [0],
                color=frozen._HIERARCHICAL,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                label="A/B/C regression",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.451),
        ncol=len(handles),
        frameon=False,
        fontsize=6.1,
        columnspacing=0.65,
        handlelength=1.7,
        handletextpad=0.35,
    )
    if interim:
        figure.text(
            0.5,
            0.993,
            "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#A61B1B",
        )
        top = 0.93
    else:
        top = 0.965
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=top,
        bottom=0.09,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return counts, beta_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_DATA / "answer_key_c2_v7.jsonl",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=_DATA / "confirmatory",
    )
    parser.add_argument(
        "--strong-responses-dir",
        type=Path,
        default=_STRONG_DATA / "confirmatory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_DATA / "confirmatory",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--interim", action="store_true")
    args = parser.parse_args()

    frozen._BOOTSTRAP = args.bootstrap_draws
    keys = {
        row["task_id"]: row
        for row in frozen._read_jsonl(args.answer_key)
    }
    responses = frozen._latest_responses(
        sorted(args.responses_dir.glob("responses_*.jsonl"))
    )
    responses.update(
        frozen._latest_responses(
            sorted(args.strong_responses_dir.glob("responses_*.jsonl"))
        )
    )
    rows = frozen._score_rows(responses, keys)
    models = [
        model
        for model in frozen._MODEL_ORDER
        if any(row["model"] == model for row in rows)
    ]
    if not models:
        raise SystemExit("no parsed responses available")
    arms = [
        arm
        for arm in ("blind", "hint", "strong_hint")
        if any(row["arm"] == arm for row in rows)
    ]
    if arms[:2] != ["blind", "hint"]:
        raise SystemExit("both original v7 arms are required")

    timestamp = datetime.now(timezone.utc)
    if args.interim:
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        stem = f"interim_{len(arms)}arm_facets_{stamp}"
    else:
        stem = f"three_city_c2_v7_main_{len(arms)}arm_facets"
    figure_path = args.outdir / f"{stem}.png"
    counts, beta_counts = _make_figure(
        figure_path,
        rows=rows,
        keys=keys,
        models=models,
        arms=arms,
        interim=args.interim,
    )
    metadata = {
        "status": (
            "interim_incomplete_not_confirmatory"
            if args.interim
            else "final_presentation"
        ),
        "created_at": timestamp.isoformat(),
        "bootstrap_draws": args.bootstrap_draws,
        "models": models,
        "arms": arms,
        "strong_hint_status": (
            "postspecified_positive_control"
            if "strong_hint" in arms
            else "not_displayed_no_parsed_responses"
        ),
        "common_paired_counts_by_arm_model_and_k": {
            arm: {
                model: {
                    str(k): count
                    for k, count in zip(
                        frozen.PREFIX_LADDER,
                        counts[arm][model],
                    )
                }
                for model in models
            }
            for arm in arms
        },
        "common_paired_counts_at_primary_k": beta_counts,
        "figure": figure_path.name,
        "frozen_analyzer_sha256": frozen._file_sha256(Path(frozen.__file__))
        if hasattr(frozen, "_file_sha256")
        else None,
    }
    metadata_path = args.outdir / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {figure_path} and {figure_path.with_suffix('.pdf')}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
