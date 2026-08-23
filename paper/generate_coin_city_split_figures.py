#!/usr/bin/env python3
"""Generate the paper-ready Coin City prompt and results figures."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
)
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"

ARMS = ("baseline", "abc_no_context", "abc_context")
# Additive misleading-cue arm. It is not part of the frozen three-arm contract,
# so panel (d) renders only the deployments whose wrong-context file is complete
# and every other panel is computed exactly as before.
WRONG_ARM = "abc_wrong_context"
WRONG_ARM_EXPECTED_RECORDS = 1_250
MODELS = (
    "claude-opus-4-8",
    "gpt-5.6-sol",
    "DeepSeek-V4-Pro",
    "FW-Kimi-K3",
    "gemini-3.6-flash",
    "claude-opus-5",
)
MODEL_LABELS = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "gpt-5.6-sol": "GPT-5.6",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "FW-Kimi-K3": "Kimi K3",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "claude-opus-5": "Claude Opus 5",
}
MODEL_PANEL_LABELS = {
    "claude-opus-4-8": "Claude",
    "gpt-5.6-sol": "GPT-5.6",
    "DeepSeek-V4-Pro": "DeepSeek",
    "FW-Kimi-K3": "Kimi K3",
    "gemini-3.6-flash": "Gemini",
    "claude-opus-5": "Opus 5",
}
FIGURE_MODELS = (
    "claude-opus-4-8",
    "gpt-5.6-sol",
    "DeepSeek-V4-Pro",
    "FW-Kimi-K3",
    "gemini-3.6-flash",
    "claude-opus-5",
)
MODEL_COLORS = {
    "claude-opus-4-8": "#7A5195",
    "gpt-5.6-sol": "#2C7FB8",
    "DeepSeek-V4-Pro": "#D95F59",
}
MODEL_MARKERS = {
    "claude-opus-4-8": "o",
    "gpt-5.6-sol": "s",
    "DeepSeek-V4-Pro": "^",
}
ARM_PANEL_LABELS = {
    "baseline": "City C only",
    "abc_no_context": "A/B/C, no C context",
    "abc_context": "A/B/C + C context",
    "abc_wrong_context": "A/B/C + wrong C context",
}
ARM_TITLE_COLORS = {
    "baseline": "#D89000",
    "abc_no_context": "#168F72",
    "abc_context": "#2C7FB8",
    "abc_wrong_context": "#B5405F",
}
CITY_C_OLS_COLOR = "#E69F00"
BEST_LLM_COLOR = "#6A3D9A"
INDIVIDUAL_LLM_COLOR = "#A8ADB4"
ANALOGICAL_OLS_COLOR = "#333333"

ARM_LABELS = {
    "baseline": "Target only",
    "abc_no_context": "References, no target context",
    "abc_context": "References + target context",
}
ARM_COLORS = {
    "baseline": "#555555",
    "abc_no_context": "#0072B2",
    "abc_context": "#D55E00",
}
ABC_COLOR = "#009E73"
CITY_CARD_EDGE = "#6C8EBF"
CITY_CARD_HIGHLIGHT_FACE = "#DAE8FC"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_latest_responses(model: str) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        path = RESPONSES / f"responses_{model}_{arm}.jsonl"
        latest: dict[str, dict] = {}
        for row in read_jsonl(path):
            old = latest.get(row["task_id"])
            if old is None or row.get("predicted_poll") is not None:
                latest[row["task_id"]] = row
        result[arm] = latest
    return result


def load_wrong_context_responses(model: str) -> dict[str, dict] | None:
    """Misleading-arm responses, or None until that deployment's run is complete."""
    path = RESPONSES / f"responses_{model}_{WRONG_ARM}.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if len(rows) < WRONG_ARM_EXPECTED_RECORDS:
        return None
    latest: dict[str, dict] = {}
    for row in rows:
        old = latest.get(row["task_id"])
        if old is None or row.get("predicted_poll") is not None:
            latest[row["task_id"]] = row
    return latest


def _fit_through_origin(rows: list[dict]) -> float:
    """OLS slope of displayed poll change on news, with no intercept."""
    news = np.asarray([row["net_news"] for row in rows], dtype=float)
    change = np.asarray(
        [row["ending_poll"] - row["starting_poll"] for row in rows],
        dtype=float,
    )
    return float(np.dot(news, change) / np.dot(news, news))


def _label_selected_reference(episode: dict, *, invert: bool) -> list[dict]:
    """Reference rows selected by City C's displayed response-regime label."""
    strong_city = episode["strong_reference_city"]
    weak_city = "B" if strong_city == "A" else "A"
    selected_city = strong_city if episode["target_strong"] else weak_city
    if invert:
        selected_city = weak_city if selected_city == strong_city else strong_city
    return episode["reference_a" if selected_city == "A" else "reference_b"]


def compute_analogical_regression_metrics(
    episodes: dict[int, dict] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate the deterministic label-conditioned regression policy.

    The policy selects the reference city assigned to City C's displayed regime,
    pools its four rows with the visible City C prefix, and fits a through-origin
    slope. The inverted condition applies the same policy to the misleading label.
    """
    if episodes is None:
        episodes = {
            row["episode"]: row for row in read_jsonl(DESIGN / "episodes.jsonl")
        }
    result: dict[str, dict[str, dict[str, float]]] = {}
    for k in range(5):
        result[str(k)] = {}
        for label, invert in (("correct", False), ("inverted", True)):
            slopes = np.asarray(
                [
                    _fit_through_origin(
                        _label_selected_reference(episode, invert=invert)
                        + episode["target"][:k]
                    )
                    for episode in episodes.values()
                ],
                dtype=float,
            )
            targets = np.asarray(
                [episode["target_slope"] for episode in episodes.values()],
                dtype=float,
            )
            predictions = np.asarray(
                [
                    episode["query_starting_poll"]
                    + episode["query_net_news"] * slope
                    for episode, slope in zip(episodes.values(), slopes)
                ],
                dtype=float,
            )
            truths = np.asarray(
                [episode["gold_expected_poll"] for episode in episodes.values()],
                dtype=float,
            )
            result[str(k)][label] = {
                "forecast_mae": float(np.mean(np.abs(predictions - truths))),
                "slope_mae": float(np.mean(np.abs(slopes - targets))),
                "rho": float(np.corrcoef(slopes, targets)[0, 1]),
            }
    return result


def compute_metrics() -> dict:
    scores = {row["task_id"]: row for row in read_jsonl(DESIGN / "scoring_key.jsonl")}
    episodes = {row["episode"]: row for row in read_jsonl(DESIGN / "episodes.jsonl")}
    analogical = compute_analogical_regression_metrics(episodes)
    validation = json.loads((DESIGN / "validation.json").read_text())
    frozen_analogical = validation["benchmark_mae"]["structural_diagnostic"]
    for k in range(5):
        if not np.isclose(
            analogical[str(k)]["correct"]["forecast_mae"],
            frozen_analogical[str(k)],
            atol=1e-12,
        ):
            raise RuntimeError(f"Analogical-regression benchmark mismatch for k={k}")
    frozen = json.loads((RUN / "live" / "progress.json").read_text())

    all_metrics: dict[str, dict] = {}
    expected_parsed = {
        "claude-opus-4-8": 3_750,
        "gpt-5.6-sol": 3_749,
        "DeepSeek-V4-Pro": 3_750,
        "FW-Kimi-K3": 3_750,
        "gemini-3.6-flash": 3_750,
        "claude-opus-5": 3_748,
    }
    for model in FIGURE_MODELS:
        responses = load_latest_responses(model)
        metrics: dict[str, dict] = {}
        for k in range(5):
            task_ids = {
                task_id
                for task_id, row in scores.items()
                if int(row["c_cases"]) == k
            }
            common = sorted(
                task_id
                for task_id in task_ids
                if all(
                    responses[arm].get(task_id, {}).get("predicted_poll") is not None
                    for arm in ARMS
                )
            )
            truths = np.asarray(
                [scores[task_id]["gold_expected_poll"] for task_id in common],
                dtype=float,
            )
            arm_mae: dict[str, float] = {}
            arm_rho: dict[str, float] = {}
            wrong = load_wrong_context_responses(model)
            if wrong is not None:
                wrong_ids = [
                    task_id
                    for task_id in common
                    if wrong.get(task_id, {}).get("predicted_poll") is not None
                ]
                wrong_truths = np.asarray(
                    [scores[task_id]["gold_expected_poll"] for task_id in wrong_ids],
                    dtype=float,
                )
                wrong_predictions = np.asarray(
                    [wrong[task_id]["predicted_poll"] for task_id in wrong_ids],
                    dtype=float,
                )
                arm_mae[WRONG_ARM] = float(
                    np.mean(np.abs(wrong_predictions - wrong_truths))
                )
                wrong_implied = [
                    (prediction - episodes[scores[task_id]["episode"]]["query_starting_poll"])
                    / episodes[scores[task_id]["episode"]]["query_net_news"]
                    for task_id, prediction in zip(wrong_ids, wrong_predictions)
                ]
                wrong_target = [
                    episodes[scores[task_id]["episode"]]["target_slope"]
                    for task_id in wrong_ids
                ]
                arm_rho[WRONG_ARM] = float(
                    np.corrcoef(wrong_implied, wrong_target)[0, 1]
                )
            for arm in ARMS:
                predictions = np.asarray(
                    [responses[arm][task_id]["predicted_poll"] for task_id in common],
                    dtype=float,
                )
                arm_mae[arm] = float(np.mean(np.abs(predictions - truths)))

                implied = []
                target = []
                for task_id, prediction in zip(common, predictions):
                    episode = episodes[scores[task_id]["episode"]]
                    implied.append(
                        (prediction - episode["query_starting_poll"])
                        / episode["query_net_news"]
                    )
                    target.append(episode["target_slope"])
                arm_rho[arm] = float(np.corrcoef(implied, target)[0, 1])

            baseline_stats: dict[str, dict[str, float]] = {}
            for baseline_name in ("baseline", "abc_no_context"):
                baseline_ids = [
                    task_id
                    for task_id in common
                    if scores[task_id]["baselines"][baseline_name] is not None
                ]
                if not baseline_ids:
                    baseline_stats[baseline_name] = {
                        "mae": float("nan"),
                        "rho": float("nan"),
                    }
                    continue
                baseline_predictions = np.asarray(
                    [
                        scores[task_id]["baselines"][baseline_name]
                        for task_id in baseline_ids
                    ],
                    dtype=float,
                )
                baseline_truths = np.asarray(
                    [scores[task_id]["gold_expected_poll"] for task_id in baseline_ids],
                    dtype=float,
                )
                baseline_implied = np.asarray(
                    [
                        (prediction - episodes[scores[task_id]["episode"]]["query_starting_poll"])
                        / episodes[scores[task_id]["episode"]]["query_net_news"]
                        for task_id, prediction in zip(baseline_ids, baseline_predictions)
                    ],
                    dtype=float,
                )
                baseline_targets = np.asarray(
                    [
                        episodes[scores[task_id]["episode"]]["target_slope"]
                        for task_id in baseline_ids
                    ],
                    dtype=float,
                )
                baseline_rho = (
                    float(np.corrcoef(baseline_implied, baseline_targets)[0, 1])
                    if np.std(baseline_implied) > 0 and np.std(baseline_targets) > 0
                    else float("nan")
                )
                baseline_stats[baseline_name] = {
                    "mae": float(np.mean(np.abs(baseline_predictions - baseline_truths))),
                    "rho": baseline_rho,
                }

            metrics[str(k)] = {
                "n": len(common),
                "mae": arm_mae,
                "rho": arm_rho,
                "target_ols_mae": baseline_stats["baseline"]["mae"],
                "target_ols_rho": baseline_stats["baseline"]["rho"],
                "abc_ols_mae": baseline_stats["abc_no_context"]["mae"],
                "abc_ols_rho": baseline_stats["abc_no_context"]["rho"],
                "analogical_regression": analogical[str(k)],
            }

            if model == "claude-opus-4-8":
                expected = frozen["curves"][str(k)]
                # A record recovered by _archive/final_repairs/eval/reparse_whitespace_keys.py adds an
                # episode the frozen run scored as unparseable, so the guard is
                # applied to the pre-recovery subset: nothing else may drift.
                frozen_ids = [
                    task_id
                    for task_id in common
                    if not any(
                        responses[arm][task_id].get("reparsed_whitespace_key")
                        for arm in ARMS
                    )
                ]
                frozen_truths = np.asarray(
                    [scores[task_id]["gold_expected_poll"] for task_id in frozen_ids],
                    dtype=float,
                )
                for arm in ARMS:
                    reported = expected["arms"][arm]["llm_mae"][model]
                    frozen_predictions = np.asarray(
                        [
                            responses[arm][task_id]["predicted_poll"]
                            for task_id in frozen_ids
                        ],
                        dtype=float,
                    )
                    frozen_mae = float(
                        np.mean(np.abs(frozen_predictions - frozen_truths))
                    )
                    if not np.isclose(frozen_mae, reported, atol=1e-10):
                        raise RuntimeError(
                            f"MAE mismatch for k={k}, arm={arm}: "
                            f"{frozen_mae} != {reported}"
                        )
                frozen_ols_ids = [
                    task_id
                    for task_id in frozen_ids
                    if scores[task_id]["baselines"]["abc_no_context"] is not None
                ]
                frozen_ols_mae = float(
                    np.mean(
                        np.abs(
                            np.asarray(
                                [
                                    scores[task_id]["baselines"]["abc_no_context"]
                                    for task_id in frozen_ols_ids
                                ],
                                dtype=float,
                            )
                            - np.asarray(
                                [
                                    scores[task_id]["gold_expected_poll"]
                                    for task_id in frozen_ols_ids
                                ],
                                dtype=float,
                            )
                        )
                    )
                )
                if not np.isclose(
                    frozen_ols_mae,
                    expected["abc_no_context_regression_mae"],
                    atol=1e-10,
                ):
                    raise RuntimeError(f"ABC OLS mismatch for k={k}")

        parsed = sum(
            row.get("predicted_poll") is not None
            for arm in responses.values()
            for row in arm.values()
        )
        if parsed != expected_parsed[model]:
            raise RuntimeError(
                f"Expected exactly {expected_parsed[model]:,} parsed "
                f"{MODEL_LABELS[model]} responses, found {parsed:,}"
            )
        all_metrics[model] = metrics

    return all_metrics


def save_figure(fig: plt.Figure, stem: Path) -> None:
    paper_figures = ROOT / "paper" / "figures"
    iclr_figures = ROOT / "paper" / "ICLR" / "figures"
    if stem.parent in {paper_figures, iclr_figures}:
        output_stem = paper_figures / stem.name
        mirror_stem = iclr_figures / stem.name
        for suffix, kwargs in (
            (".pdf", {"facecolor": "white"}),
            (".png", {"dpi": 220, "facecolor": "white"}),
        ):
            path = output_stem.with_suffix(suffix)
            mirror = mirror_stem.with_suffix(suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            mirror.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, **kwargs)
            shutil.copyfile(path, mirror)
    else:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
        fig.savefig(
            stem.with_suffix(".png"),
            dpi=220,
            facecolor="white",
        )
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    edge: str,
    face: str = "white",
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(patch)
    return patch


def add_city_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    context: str,
    bold_phrase: str,
    rows: list[dict],
    edge: str,
    highlight: bool = False,
) -> None:
    face = CITY_CARD_HIGHLIGHT_FACE if highlight else "#FFFFFF"
    add_box(ax, (x, y), width, height, edge=edge, face=face, linewidth=1.05)
    ax.text(
        x + 0.014,
        y + height - 0.035,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#222222",
    )
    placeholder = bold_phrase.replace(" ", "\N{NO-BREAK SPACE}")
    wrapped = textwrap.fill(
        context.replace(bold_phrase, placeholder),
        width=max(27, int(116 * width)),
    )
    math_phrase = bold_phrase.replace(" ", r"\ ")
    wrapped = wrapped.replace(placeholder, rf"$\mathbf{{{math_phrase}}}$")
    ax.text(
        x + 0.014,
        y + height - 0.105,
        wrapped,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        linespacing=1.08,
        color="#3F3F3F",
    )

    values = [
        [
            str(row["pair"]),
            f"{row['starting_poll']:.1f}",
            f"{row['net_news']:+d}",
            f"{row['ending_poll'] - row['starting_poll']:+.1f}",
            f"{row['ending_poll']:.1f}",
        ]
        for row in rows
    ]
    table_height = 0.37 * height
    table = ax.table(
        cellText=values,
        colLabels=["Pair", "Start", "News", "Delta", "End"],
        cellLoc="center",
        colLoc="center",
        bbox=[x + 0.012, y + 0.025, width - 0.024, table_height],
        zorder=2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.7)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#C9CDD2")
        if row_index == 0:
            cell.set_facecolor("#F0F2F4")
            cell.set_text_props(fontweight="bold", color="#333333")
        else:
            cell.set_facecolor(face)


def draw_arm_header(ax: plt.Axes, letter: str, title: str, color: str) -> None:
    ax.text(
        0.0,
        0.99,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": color,
            "edgecolor": color,
            "linewidth": 0.0,
        },
    )
    ax.text(
        0.052,
        0.99,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        fontweight="bold",
        color="#222222",
    )


def make_prompt_figure(output_dir: Path) -> None:
    components = json.loads((DESIGN / "example_prompt_components.json").read_text())
    episode = read_jsonl(DESIGN / "episodes.jsonl")[components["episode"]]
    manifest = json.loads((DESIGN / "manifest.json").read_text())

    national = (
        "Background: Residents generally encounter campaign developments through "
        "national news coverage, where campaign stories receive sustained attention "
        "and tend to produce larger immediate poll movements."
    )
    local = (
        "Background: Residents generally encounter campaign developments through "
        "local news coverage, where campaign stories compete with local issues and "
        "tend to produce smaller immediate poll movements."
    )
    target_context = manifest["context_sentences"]["national_news"]

    fig = plt.figure(figsize=(7.2, 3.75), facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.16, 2.10],
        hspace=0.08,
        left=0.03,
        right=0.985,
        top=0.98,
        bottom=0.025,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(2)]
    for ax in axes:
        ax.set_axis_off()

    shared = (
        "Each row is a separate polling case, not a time series. Positive net news "
        "favors the candidate and negative net news harms the candidate. Poll change "
        "is the ending poll minus the starting poll. Within a city, typical "
        "responsiveness to net news is stable across cases, although individual end "
        "polls remain noisy. Rows with the same Pair label have equal-sized news in "
        "opposite directions."
    )

    query = (
        "A new City C case begins with a poll of 53.9 and has net news +5. "
        "What poll do you predict at the end of this new case? Think carefully "
        "about the expected poll, but do not provide step-by-step working. Respond "
        'with one JSON object only: {"rationale": "one short sentence", '
        '"predicted_poll": <number from 0 to 100>}'
    )
    shared_prompt = "\n\n".join(
        [
            textwrap.fill(shared, width=118),
            textwrap.fill(f"Query: {query}", width=118),
        ]
    )

    axes[0].text(
        0.0,
        0.98,
        "Shared prompt in every arm",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        color="#222222",
    )
    add_box(
        axes[0],
        (0.0, 0.03),
        1.0,
        0.82,
        edge="#B7BCC2",
        face="#F7F8F9",
    )
    axes[0].text(
        0.02,
        0.44,
        shared_prompt,
        transform=axes[0].transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        linespacing=1.10,
        color="#333333",
    )

    ax = axes[1]
    ax.text(
        0.0,
        0.99,
        "Illustrated arm: References + target context",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        fontweight="bold",
        color="#222222",
    )
    context_specs = [
        ("CITY A", national, "national news coverage", episode["reference_a"], False),
        ("CITY B", local, "local news coverage", episode["reference_b"], False),
        (
            "CITY C  ·  TARGET CONTEXT",
            target_context,
            "national news coverage",
            episode["target"],
            True,
        ),
    ]
    for index, (title, context, bold_phrase, rows, highlight) in enumerate(context_specs):
        add_city_card(
            ax,
            x=0.009 + 0.337 * index,
            y=0.06,
            width=0.313,
            height=0.81,
            title=title,
            context=context,
            bold_phrase=bold_phrase,
            rows=rows,
            edge=CITY_CARD_EDGE,
            highlight=highlight,
        )

    save_figure(fig, output_dir / "exp2_coin_city_prompt_arms")


def make_results_figure(output_dir: Path, metrics: dict) -> None:
    x = np.arange(5, dtype=float)
    wrong_models = [
        model
        for model in FIGURE_MODELS
        if WRONG_ARM in metrics[model]["0"]["mae"]
    ]
    panels = list(ARMS) + ([WRONG_ARM] if wrong_models else [])
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(7.2 if len(panels) == 3 else 9.4, 2.95),
        facecolor="white",
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.075 if len(panels) == 3 else 0.058,
        right=0.995,
        bottom=0.165,
        top=0.775,
        wspace=0.14,
    )
    reference_metrics = metrics["DeepSeek-V4-Pro"]
    reference_styles = (
        ("target_ols", CITY_C_OLS_COLOR, "D", "City C OLS"),
        ("abc_ols", ABC_COLOR, "X", "A/B/C OLS"),
    )

    def model_curve(model: str, arm: str) -> list[float]:
        return [metrics[model][str(k)]["mae"][arm] for k in range(5)]

    def best_overall_model() -> str:
        """The single deployment with the lowest mean MAE across the frozen arms."""
        return min(
            FIGURE_MODELS,
            key=lambda model: float(
                np.mean(
                    [metrics[model][str(k)]["mae"][arm] for arm in ARMS for k in range(5)]
                )
            ),
        )

    best_model = best_overall_model()

    for column_index, arm in enumerate(panels):
        ax = axes[column_index]
        panel_models = wrong_models if arm == WRONG_ARM else list(FIGURE_MODELS)

        for model in panel_models:
            ax.plot(
                x,
                model_curve(model, arm),
                color=INDIVIDUAL_LLM_COLOR,
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
        if best_model in panel_models:
            ax.plot(
                x,
                model_curve(best_model, arm),
                color=BEST_LLM_COLOR,
                marker="o",
                markersize=5.0,
                linewidth=2.4,
                zorder=4,
            )

        for prefix, color, marker, _ in reference_styles:
            ax.plot(
                x,
                [reference_metrics[str(k)][f"{prefix}_mae"] for k in range(5)],
                color=color,
                linestyle="--",
                marker=marker,
                markersize=4.4,
                linewidth=1.6,
                zorder=3,
            )

        if arm in ("abc_context", WRONG_ARM):
            label = "correct" if arm == "abc_context" else "inverted"
            ax.plot(
                x,
                [
                    reference_metrics[str(k)]["analogical_regression"][label][
                        "forecast_mae"
                    ]
                    for k in range(5)
                ],
                color=ANALOGICAL_OLS_COLOR,
                linestyle="--",
                marker="P",
                markersize=4.6,
                linewidth=1.8,
                zorder=5,
            )

        ax.set_title(
            f"({chr(ord('a') + column_index)}) {ARM_PANEL_LABELS[arm]}",
            loc="left",
            fontweight="bold",
            fontsize=8.3,
            color=ARM_TITLE_COLORS[arm],
        )
        if column_index == 0:
            ax.set_ylabel("MAE (poll points)")
        ax.set_xlabel("Observed City C cases ($k$)")
        ax.set_xticks(x)
        ax.grid(axis="y", color="#D9DCE0", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)

    mae_values = [
        metrics[model][str(k)]["mae"][arm]
        for arm in panels
        for model in (wrong_models if arm == WRONG_ARM else FIGURE_MODELS)
        for k in range(5)
    ]
    mae_values.extend(
        reference_metrics[str(k)][f"{prefix}_mae"]
        for prefix, _, _, _ in reference_styles
        for k in range(5)
    )
    mae_values.extend(
        reference_metrics[str(k)]["analogical_regression"][label]["forecast_mae"]
        for label in ("correct", "inverted")
        for k in range(5)
    )
    axes[0].set_ylim(
        min(1.4, float(np.nanmin(mae_values)) - 0.15),
        float(np.nanmax(mae_values)) + 0.2,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=BEST_LLM_COLOR,
            marker="o",
            linewidth=2.4,
            markersize=5.0,
            label=f"{MODEL_LABELS[best_model]} (best overall)",
        ),
        Line2D(
            [0],
            [0],
            color=INDIVIDUAL_LLM_COLOR,
            linewidth=1.0,
            label="Other deployments",
        ),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=color,
            linestyle="--",
            marker=marker,
            linewidth=1.6,
            markersize=4.4,
            label=label,
        )
        for _, color, marker, label in reference_styles
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=ANALOGICAL_OLS_COLOR,
            linestyle="--",
            marker="P",
            linewidth=1.8,
            markersize=4.6,
            label="Label-conditioned analogical OLS",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.8,
    )

    save_figure(fig, output_dir / "exp2_coin_city_results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper" / "figures",
    )
    args = parser.parse_args()

    metrics = compute_metrics()
    make_prompt_figure(args.output_dir)
    make_results_figure(args.output_dir, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
