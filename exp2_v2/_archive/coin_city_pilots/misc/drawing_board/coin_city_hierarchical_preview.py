#!/usr/bin/env python3
"""Simulation-only preview of a city-scale carnival-coin analogue.

This file makes no model or network calls.  It generates three statistical
benchmarks from one hierarchical data-generating process:

1. City C observations only.
2. A/B/C pooling with equal prior odds on the two demonstrated regimes.
3. The same A/B/C model with a probabilistic natural-language cue that shifts
   the prior odds toward one demonstrated regime.

The reference influence is obtained from Bayesian updating; there is no fixed
or hand-coded k-dependent pooling weight.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "coin_city_hierarchical_preview"

SEED = 20260805
MONTE_CARLO_EPISODES = 100_000
STARTING_POLL = 50.0
NEWS = 8
ROUNDS = np.array([1, 2, 3, 4, 5])
CASES = np.array([1, 2, 4, 8, 16])
REFERENCE_CASES = 6

# End-of-case poll means for the two latent response regimes.  Equivalently,
# their news-response slopes are 0.25 and 1.00 poll points per news point.
LOW_ENDPOINT_MEAN = 52.0
HIGH_ENDPOINT_MEAN = 58.0
CITY_ENDPOINT_SD = 0.75
CASE_NOISE_SD = 4.5
CUE_RELIABILITY = 0.80

COLORS = {
    "c_only": "#E68613",
    "pooled": "#168A72",
    "context": "#267CB5",
    "misleading": "#B44E5A",
}


def _mixture_posterior_mean(
    target_mean: np.ndarray,
    cases: int,
    reference_low_mean: np.ndarray,
    reference_high_mean: np.ndarray,
    prior_high: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior endpoint mean and posterior P(high regime)."""
    target_mean = np.asarray(target_mean, dtype=float)
    prior_high = np.broadcast_to(np.asarray(prior_high, dtype=float), target_mean.shape)

    # Relative to a noisy reference-city mean, another city from the same
    # regime varies because both cities differ from the regime center and the
    # reference sample itself is noisy.
    component_prior_var = (
        2.0 * CITY_ENDPOINT_SD**2
        + CASE_NOISE_SD**2 / REFERENCE_CASES
    )
    target_sampling_var = CASE_NOISE_SD**2 / float(cases)
    marginal_var = component_prior_var + target_sampling_var

    posterior_var = 1.0 / (
        1.0 / component_prior_var + 1.0 / target_sampling_var
    )
    low_posterior_mean = posterior_var * (
        reference_low_mean / component_prior_var
        + target_mean / target_sampling_var
    )
    high_posterior_mean = posterior_var * (
        reference_high_mean / component_prior_var
        + target_mean / target_sampling_var
    )

    eps = 1e-12
    log_low = (
        np.log(np.clip(1.0 - prior_high, eps, 1.0))
        - 0.5 * (target_mean - reference_low_mean) ** 2 / marginal_var
    )
    log_high = (
        np.log(np.clip(prior_high, eps, 1.0))
        - 0.5 * (target_mean - reference_high_mean) ** 2 / marginal_var
    )
    normalizer = np.maximum(log_low, log_high)
    low_weight = np.exp(log_low - normalizer)
    high_weight = np.exp(log_high - normalizer)
    posterior_high = high_weight / (low_weight + high_weight)
    posterior_mean = (
        (1.0 - posterior_high) * low_posterior_mean
        + posterior_high * high_posterior_mean
    )
    return posterior_mean, posterior_high


def simulate() -> dict:
    rng = np.random.default_rng(SEED)
    n = MONTE_CARLO_EPISODES

    target_high = rng.integers(0, 2, size=n).astype(bool)
    cue_correct = rng.random(n) < CUE_RELIABILITY
    cue_high = np.where(cue_correct, target_high, ~target_high)

    reference_low_truth = LOW_ENDPOINT_MEAN + rng.normal(
        0.0, CITY_ENDPOINT_SD, size=n
    )
    reference_high_truth = HIGH_ENDPOINT_MEAN + rng.normal(
        0.0, CITY_ENDPOINT_SD, size=n
    )
    target_truth = np.where(target_high, HIGH_ENDPOINT_MEAN, LOW_ENDPOINT_MEAN)
    target_truth = target_truth + rng.normal(0.0, CITY_ENDPOINT_SD, size=n)

    reference_low_rows = reference_low_truth[:, None] + rng.normal(
        0.0, CASE_NOISE_SD, size=(n, REFERENCE_CASES)
    )
    reference_high_rows = reference_high_truth[:, None] + rng.normal(
        0.0, CASE_NOISE_SD, size=(n, REFERENCE_CASES)
    )
    reference_low_mean = reference_low_rows.mean(axis=1)
    reference_high_mean = reference_high_rows.mean(axis=1)

    target_rows = target_truth[:, None] + rng.normal(
        0.0, CASE_NOISE_SD, size=(n, int(CASES.max()))
    )
    target_cumulative = np.cumsum(target_rows, axis=1)

    records = []
    conditional = {"aligned": [], "misleading": []}
    for round_number, cases in zip(ROUNDS, CASES):
        target_mean = target_cumulative[:, cases - 1] / float(cases)
        pooled, pooled_p_high = _mixture_posterior_mean(
            target_mean,
            int(cases),
            reference_low_mean,
            reference_high_mean,
            0.5,
        )
        cue_prior_high = np.where(
            cue_high, CUE_RELIABILITY, 1.0 - CUE_RELIABILITY
        )
        context, context_p_high = _mixture_posterior_mean(
            target_mean,
            int(cases),
            reference_low_mean,
            reference_high_mean,
            cue_prior_high,
        )

        estimators = {
            "c_only": target_mean,
            "pooled": pooled,
            "context": context,
        }
        row = {
            "k": int(round_number),
            "city_c_cases": int(cases),
            "mae": {},
            "mc_95_ci": {},
            "mean_posterior_probability_true_regime": {
                "pooled": float(
                    np.mean(np.where(target_high, pooled_p_high, 1.0 - pooled_p_high))
                ),
                "context": float(
                    np.mean(np.where(target_high, context_p_high, 1.0 - context_p_high))
                ),
            },
        }
        for name, estimate in estimators.items():
            error = np.abs(estimate - target_truth)
            se = error.std(ddof=1) / math.sqrt(n)
            row["mae"][name] = float(error.mean())
            row["mc_95_ci"][name] = [
                float(error.mean() - 1.96 * se),
                float(error.mean() + 1.96 * se),
            ]

        for label, mask in (
            ("aligned", cue_correct),
            ("misleading", ~cue_correct),
        ):
            error = np.abs(context[mask] - target_truth[mask])
            conditional[label].append(
                {
                    "k": int(round_number),
                    "city_c_cases": int(cases),
                    "n": int(mask.sum()),
                    "mae": float(error.mean()),
                }
            )
        records.append(row)

    return {
        "experiment": "coin_city_hierarchical_preview",
        "status": "simulation_only_no_model_calls",
        "seed": SEED,
        "episodes": MONTE_CARLO_EPISODES,
        "parameters": {
            "starting_poll": STARTING_POLL,
            "news": NEWS,
            "rounds": ROUNDS.tolist(),
            "city_c_cases": CASES.tolist(),
            "reference_cases_per_city": REFERENCE_CASES,
            "low_endpoint_mean": LOW_ENDPOINT_MEAN,
            "high_endpoint_mean": HIGH_ENDPOINT_MEAN,
            "city_endpoint_sd": CITY_ENDPOINT_SD,
            "case_noise_sd": CASE_NOISE_SD,
            "cue_reliability": CUE_RELIABILITY,
        },
        "curves": records,
        "context_curve_by_cue_status": conditional,
    }


def _render_curve(result: dict) -> None:
    curves = result["curves"]
    x = np.arange(len(curves))
    labels = [f"k={row['k']}\n{row['city_c_cases']} C cases" for row in curves]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 8.0))
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.47, top=0.77, wspace=0.14
    )
    fig.suptitle(
        "A city-scale analogue of the carnival coin",
        y=0.975,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.895, "Simulation-only statistical benchmarks — no LLM outputs",
        ha="center", fontsize=10.5, color="#555555",
    )

    ax = axes[0]
    series = (
        ("c_only", "Baseline: City C evidence only"),
        ("pooled", "A/B/C hierarchical estimate, no context"),
        ("context", "A/B/C hierarchical estimate + context"),
    )
    for name, label in series:
        y = [row["mae"][name] for row in curves]
        ax.plot(
            x, y, marker="o", markersize=6, linewidth=2.4,
            color=COLORS[name], label=label,
        )
    ax.set_title("A. Context helps most when City C evidence is sparse", loc="left")
    ax.set_ylabel("MAE of expected City C poll (points)")
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    pooled = [row["mae"]["pooled"] for row in curves]
    aligned = [
        row["mae"] for row in result["context_curve_by_cue_status"]["aligned"]
    ]
    misleading = [
        row["mae"] for row in result["context_curve_by_cue_status"]["misleading"]
    ]
    ax.plot(
        x, pooled, marker="o", linewidth=2.2, color=COLORS["pooled"],
        label="No contextual cue",
    )
    ax.plot(
        x, aligned, marker="o", linewidth=2.2, color=COLORS["context"],
        label="Cue points toward the true regime (80% of episodes)",
    )
    ax.plot(
        x, misleading, marker="o", linewidth=2.2,
        color=COLORS["misleading"], linestyle="--",
        label="Cue points toward the other regime (20%)",
    )
    ax.set_title("B. Accumulating City C evidence can override context", loc="left")
    ax.set_ylabel("MAE of A/B/C forecast (points)")
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.6)

    fig.text(
        0.5, 0.385,
        "Statistical estimators shown above (benchmarks only — these equations are not given to the LLM)",
        ha="center", fontsize=10.5, fontweight="bold", color="#333333",
    )
    fig.text(
        0.5, 0.315,
        r"City C evidence:  $\bar y_{C,k}=n_k^{-1}\sum_{i=1}^{n_k}y_{Ci}$,  "
        r"$s_k^2=\sigma^2/n_k$"
        "\n"
        r"Within regime $j$:  $m_{j,k}="
        r"\frac{\bar y_j/V_0+\bar y_{C,k}/s_k^2}{1/V_0+1/s_k^2}$,  "
        r"$\omega_{j,k}(\pi)\propto\pi_j\,\mathcal{N}"
        r"(\bar y_{C,k};\bar y_j,V_0+s_k^2)$",
        ha="center", va="center", fontsize=10.2,
        bbox={
            "boxstyle": "round,pad=0.55", "facecolor": "#FFF4B8",
            "edgecolor": "#D4B547", "linewidth": 1.0,
        },
    )
    estimator_boxes = (
        (
            0.18, COLORS["c_only"], "Baseline: City C evidence only",
            r"$\hat\theta_{C,k}=\bar y_{C,k}$",
        ),
        (
            0.50, COLORS["pooled"], "A/B/C, no context",
            r"$\hat\theta_k=\sum_j\omega_{j,k}(1/2)m_{j,k}$",
        ),
        (
            0.82, COLORS["context"], "A/B/C + context",
            r"$\hat\theta_k=\sum_j\omega_{j,k}(\pi_{cue})m_{j,k}$,  "
            r"$\pi_{cue}=(.8,.2)$",
        ),
    )
    for xpos, color, heading, equation in estimator_boxes:
        fig.text(
            xpos, 0.225, heading,
            ha="center", va="center", fontsize=9.6,
            fontweight="bold", color=color,
        )
        fig.text(
            xpos, 0.17, equation,
            ha="center", va="center", fontsize=9.4, color="#222222",
            bbox={
                "boxstyle": "round,pad=0.55", "facecolor": "#FFFBE6",
                "edgecolor": color, "linewidth": 1.5,
            },
        )
    fig.text(
        0.5,
        0.035,
        "The clue changes prior odds between two demonstrated response regimes; "
        "its influence declines through ordinary Bayesian updating, not a fixed k-dependent weight.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    output = OUTDIR / "benchmark_curves.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _table(values: list[float]) -> str:
    lines = [
        "| Case | Starting poll | Net news | Poll at end of case |",
        "|---:|---:|---:|---:|",
    ]
    for index, value in enumerate(values, start=1):
        lines.append(f"| {index} | 50.0 | +8 | {value:.1f} |")
    return "\n".join(lines)


def _representative_episode() -> dict:
    rng = np.random.default_rng(314159)
    reference_a_truth = HIGH_ENDPOINT_MEAN + rng.normal(0.0, CITY_ENDPOINT_SD)
    reference_b_truth = LOW_ENDPOINT_MEAN + rng.normal(0.0, CITY_ENDPOINT_SD)
    target_truth = HIGH_ENDPOINT_MEAN + rng.normal(0.0, CITY_ENDPOINT_SD)
    return {
        "reference_a_truth": float(reference_a_truth),
        "reference_b_truth": float(reference_b_truth),
        "target_truth": float(target_truth),
        "reference_a": (
            reference_a_truth
            + rng.normal(0.0, CASE_NOISE_SD, size=REFERENCE_CASES)
        ).tolist(),
        "reference_b": (
            reference_b_truth
            + rng.normal(0.0, CASE_NOISE_SD, size=REFERENCE_CASES)
        ).tolist(),
        "target": (
            target_truth + rng.normal(0.0, CASE_NOISE_SD, size=2)
        ).tolist(),
        "cue": "high",
        "cue_is_correct": True,
    }


SHARED_PROMPT_INTRO = (
    "You are forecasting a post-news local-election poll.\n"
    "Each table row is a separate polling case, not a time series. Every "
    "case began with a poll of 50.0. Net news +8 then occurred, and the "
    "poll at the end of that case was measured. End-of-case polls vary "
    "because polling and local opinion are not perfectly stable."
)
SHARED_PROMPT_CLOSING = (
    "A new City C case begins with a poll of 50.0 and has net news +8.\n"
    "What poll do you predict at the end of this new case?\n\n"
    "Think carefully about the expected poll, but do not provide step-by-step "
    "working. Respond with one JSON object only:\n"
    '{"rationale": "one short sentence", "predicted_poll": <number from 0 to 100>}'
)
BACKGROUND_A = (
    "Background: Residents followed national campaign developments through "
    "immediate alerts and direct candidate feeds, with little local filtering."
)
BACKGROUND_B = (
    "Background: Residents encountered national campaign developments through "
    "neighborhood briefings where local organizers interpreted outside political news."
)
BACKGROUND_C_CONTEXT = (
    "Background: Most voters followed national campaign events through live "
    "updates and direct campaign feeds rather than local intermediaries."
)
BACKGROUND_C_NONE = "No additional background information is available."


def _prompt_sections(
    episode: dict,
    *,
    include_references: bool,
    include_context: bool,
) -> str:
    sections = []
    if include_references:
        sections.extend(
            [
                "CITY A\n" + BACKGROUND_A + "\n" + _table(episode["reference_a"]),
                "CITY B\n" + BACKGROUND_B + "\n" + _table(episode["reference_b"]),
            ]
        )
    c_background = BACKGROUND_C_CONTEXT if include_context else BACKGROUND_C_NONE
    sections.append("CITY C\n" + c_background + "\n" + _table(episode["target"]))
    return "\n\n".join(sections)


def _prompt(
    episode: dict,
    *,
    include_references: bool,
    include_context: bool,
) -> str:
    return (
        SHARED_PROMPT_INTRO
        + "\n\n"
        + _prompt_sections(
            episode,
            include_references=include_references,
            include_context=include_context,
        )
        + "\n\n"
        + SHARED_PROMPT_CLOSING
    )


def _display_wrap(value: str, width: int) -> str:
    """Wrap prose for the figure without changing the underlying prompt."""
    rendered = []
    for line in value.splitlines():
        if not line or line.startswith("|"):
            rendered.append(line)
        else:
            rendered.extend(
                textwrap.wrap(
                    line, width=width, break_long_words=False,
                    break_on_hyphens=False,
                ) or [""]
            )
    return "\n".join(rendered)


def _render_prompt_structure(episode: dict) -> None:
    fig = plt.figure(figsize=(16.0, 7.6), facecolor="white")
    fig.suptitle(
        "What the LLM sees in each condition",
        y=0.990, fontsize=19, fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        "Shared wording is shown once; each panel contains the exact condition-specific city information",
        ha="center", fontsize=11, color="#555555",
    )
    fig.text(
        0.5, 0.865, _display_wrap(SHARED_PROMPT_INTRO, 150),
        ha="center", va="top", family="monospace", fontsize=8.4,
        bbox={
            "boxstyle": "round,pad=0.65", "facecolor": "#F1F1F1",
            "edgecolor": "#AAAAAA", "linewidth": 1.0,
        },
    )

    panel_specs = (
        (
            "A. Baseline: City C only", COLORS["c_only"],
            _prompt_sections(
                episode, include_references=False, include_context=False
            ),
        ),
        (
            "B. A/B/C, no City C context", COLORS["pooled"],
            _prompt_sections(
                episode, include_references=True, include_context=False
            ),
        ),
        (
            "C. A/B/C + natural City C context", COLORS["context"],
            _prompt_sections(
                episode, include_references=True, include_context=True
            ),
        ),
    )
    lefts = (0.025, 0.345, 0.665)
    for left, (title, color, content) in zip(lefts, panel_specs):
        ax = fig.add_axes([left, 0.205, 0.31, 0.56])
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=color)
        ax.text(
            0.01, 0.975, _display_wrap(content, 65),
            transform=ax.transAxes, ha="left", va="top",
            family="monospace", fontsize=7.0, linespacing=1.12,
            bbox={
                "boxstyle": "round,pad=0.6", "facecolor": "#FCFCFC",
                "edgecolor": color, "linewidth": 1.3,
            },
        )

    fig.text(
        0.82, 0.18,
        "B → C adds only City C's background sentence shown above.",
        ha="center", va="center", fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.45", "facecolor": "#FFF4B8",
            "edgecolor": "#D4B547", "linewidth": 1.0,
        },
    )
    fig.text(
        0.5, 0.145, _display_wrap(SHARED_PROMPT_CLOSING, 150),
        ha="center", va="top", family="monospace", fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.6", "facecolor": "#F1F1F1",
            "edgecolor": "#AAAAAA", "linewidth": 1.0,
        },
    )
    fig.text(
        0.5, 0.018,
        "Display wrapping is visual only. The complete byte-for-byte prompt proposals are in exact_prompts.md.",
        ha="center", fontsize=9, color="#555555",
    )
    output = OUTDIR / "prompt_structure.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _compose_benchmark_and_prompts() -> None:
    """Stack the benchmark/equation figure over the exact prompt figure."""
    with Image.open(OUTDIR / "benchmark_curves.png") as benchmark_source:
        benchmark = benchmark_source.convert("RGB")
    with Image.open(OUTDIR / "prompt_structure.png") as prompt_source:
        prompts = prompt_source.convert("RGB")

    width = max(benchmark.width, prompts.width)

    def fit_width(image: Image.Image) -> Image.Image:
        if image.width == width:
            return image
        height = round(image.height * width / image.width)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    benchmark = fit_width(benchmark)
    prompts = fit_width(prompts)
    gap = max(30, round(width * 0.012))
    canvas = Image.new(
        "RGB", (width, benchmark.height + gap + prompts.height), "white"
    )
    canvas.paste(benchmark, (0, 0))
    canvas.paste(prompts, (0, benchmark.height + gap))
    output = OUTDIR / "benchmark_curves_with_prompts.png"
    canvas.save(output, optimize=True)
    canvas.save(output.with_suffix(".pdf"), "PDF", resolution=200.0)


def _write_prompt_preview(episode: dict) -> None:
    prompts = {
        "A — Baseline: City C only": _prompt(
            episode, include_references=False, include_context=False
        ),
        "B — A/B/C information, no City C context": _prompt(
            episode, include_references=True, include_context=False
        ),
        "C — A/B/C information plus natural City C context": _prompt(
            episode, include_references=True, include_context=True
        ),
    }
    parts = [
        "# Exact drawing-board prompts",
        "",
        "These are prompt proposals, not frozen tasks. No model has been called.",
        "The numbers in B and C are identical; only City C's background sentence changes.",
        "The text never states that two regimes exist or that City C matches A or B.",
        "",
    ]
    for heading, prompt in prompts.items():
        parts.extend([f"## {heading}", "", "```text", prompt, "```", ""])
    (OUTDIR / "exact_prompts.md").write_text("\n".join(parts))
    separate_prompts = {
        "prompt_a_baseline_city_c_only.txt": prompts[
            "A — Baseline: City C only"
        ],
        "prompt_b_abc_no_context.txt": prompts[
            "B — A/B/C information, no City C context"
        ],
        "prompt_c_abc_plus_context.txt": prompts[
            "C — A/B/C information plus natural City C context"
        ],
    }
    for filename, prompt in separate_prompts.items():
        (OUTDIR / filename).write_text(prompt + "\n")


def _validate(result: dict, episode: dict) -> dict:
    curves = result["curves"]
    c_only = [row["mae"]["c_only"] for row in curves]
    pooled = [row["mae"]["pooled"] for row in curves]
    context = [row["mae"]["context"] for row in curves]
    aligned = [
        row["mae"] for row in result["context_curve_by_cue_status"]["aligned"]
    ]
    misleading = [
        row["mae"] for row in result["context_curve_by_cue_status"]["misleading"]
    ]
    prompt_b = _prompt(episode, include_references=True, include_context=False)
    prompt_c = _prompt(episode, include_references=True, include_context=True)
    no_context = "No additional background information is available."
    contextual = (
        "Background: Most voters followed national campaign events through live "
        "updates and direct campaign feeds rather than local intermediaries."
    )
    forbidden = ("bayes", "prior", "regime", "matches city", "closer to city")
    checks = {
        "city_c_only_mae_strictly_declines": all(
            later < earlier for earlier, later in zip(c_only, c_only[1:])
        ),
        "reference_pooling_beats_city_c_only_every_round": all(
            reference < target for reference, target in zip(pooled, c_only)
        ),
        "context_beats_no_context_every_round": all(
            clue < neutral for clue, neutral in zip(context, pooled)
        ),
        "all_three_converge_by_round_5": max(
            c_only[-1], pooled[-1], context[-1]
        ) - min(c_only[-1], pooled[-1], context[-1]) < 0.10,
        "aligned_context_helps_every_round": all(
            clue < neutral for clue, neutral in zip(aligned, pooled)
        ),
        "misleading_context_hurts_every_round": all(
            clue > neutral for clue, neutral in zip(misleading, pooled)
        ),
        "misleading_context_penalty_declines": all(
            later < earlier
            for earlier, later in zip(
                [bad - neutral for bad, neutral in zip(misleading, pooled)],
                [bad - neutral for bad, neutral in zip(misleading, pooled)][1:],
            )
        ),
        "b_and_c_differ_only_by_city_c_background": (
            prompt_c.replace(contextual, no_context) == prompt_b
        ),
        "prompts_do_not_disclose_hierarchical_model": not any(
            token in (prompt_b + prompt_c).lower() for token in forbidden
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "round_1_mae": {
            "city_c_only": c_only[0],
            "abc_no_context": pooled[0],
            "abc_context": context[0],
        },
        "round_5_mae": {
            "city_c_only": c_only[-1],
            "abc_no_context": pooled[-1],
            "abc_context": context[-1],
        },
        "round_5_spread": max(c_only[-1], pooled[-1], context[-1])
        - min(c_only[-1], pooled[-1], context[-1]),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result = simulate()
    (OUTDIR / "benchmark_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    episode = _representative_episode()
    (OUTDIR / "representative_episode.json").write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n"
    )
    _render_curve(result)
    _write_prompt_preview(episode)
    _render_prompt_structure(episode)
    _compose_benchmark_and_prompts()
    validation = _validate(result, episode)
    (OUTDIR / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    if validation["status"] != "passed":
        raise SystemExit("drawing-board validation failed")
    print(f"Wrote simulation-only preview to {OUTDIR}")


if __name__ == "__main__":
    main()
