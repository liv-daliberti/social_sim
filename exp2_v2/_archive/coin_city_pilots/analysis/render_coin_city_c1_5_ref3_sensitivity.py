#!/usr/bin/env python3
"""Plot the C=1..5, fixed 3A+3B regression sensitivity design."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import engine.coin_city_variable_regression as design


REFERENCE_CASES_PER_CITY = 3
EXTRA_SEED_BASE = 910_000
CONVERGENCE_THRESHOLD = 0.50


def _augment_references(episode: dict) -> dict:
    """Keep C/query fixed and append one fresh displayed row to A and B."""
    revised = dict(episode)
    a_rows = list(episode["reference_a"])
    b_rows = list(episode["reference_b"])
    a_slope = (
        episode["reference_strong_slope"]
        if episode["strong_reference_city"] == "A"
        else episode["reference_weak_slope"]
    )
    b_slope = (
        episode["reference_strong_slope"]
        if episode["strong_reference_city"] == "B"
        else episode["reference_weak_slope"]
    )
    a_rng = np.random.default_rng(
        EXTRA_SEED_BASE + 100 * episode["episode"] + 1
    )
    b_rng = np.random.default_rng(
        EXTRA_SEED_BASE + 100 * episode["episode"] + 2
    )
    a_rows.extend(design._sample_rows(a_rng, a_slope, 1))
    b_rows.extend(design._sample_rows(b_rng, b_slope, 1))
    revised["reference_a"] = a_rows
    revised["reference_b"] = b_rows
    return revised


def _mae(episodes: list[dict], c_cases: int, *, abc: bool) -> float:
    errors = []
    for episode in episodes:
        c_rows = episode["target"][:c_cases]
        rows = (
            episode["reference_a"] + episode["reference_b"] + c_rows
            if abc
            else c_rows
        )
        prediction = design.predict_from_slope(
            episode, design.fit_change_slope(rows)
        )
        errors.append(abs(prediction - episode["gold_expected_poll"]))
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-c", type=int, default=1)
    parser.add_argument("--max-c", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.min_c <= args.max_c <= max(design.CASES_BY_ROUND.values()):
        raise SystemExit("requested C range lies outside the available sequence")
    c_cases = tuple(range(args.min_c, args.max_c + 1))
    outdir = (
        ROOT
        / "data"
        / design.EXPERIMENT
        / f"sensitivity_c{args.min_c}_{args.max_c}_ref3"
    )
    if design.CASE_NOISE_SD != 8.0:
        raise SystemExit("this sensitivity preview is specified at noise SD 8.0")
    episodes = [
        _augment_references(design.make_episode(index))
        for index in range(design.EPISODES)
    ]
    if not all(
        len(episode["reference_a"]) == len(episode["reference_b"])
        == REFERENCE_CASES_PER_CITY
        for episode in episodes
    ):
        raise SystemExit("reference-row count invariant failed")
    city_c = [None if n == 0 else _mae(episodes, n, abc=False) for n in c_cases]
    abc = [_mae(episodes, n, abc=True) for n in c_cases]
    gaps = [None if c is None else c - a for c, a in zip(city_c, abc)]
    final_gap = gaps[-1]
    convergence_passes = (
        final_gap is not None and final_gap <= CONVERGENCE_THRESHOLD
    )

    fig, ax = plt.subplots(figsize=(10.4, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.25)
    fig.suptitle(
        f"Regression sensitivity: {args.min_c}–{args.max_c} City C cases",
        y=0.965,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.915,
        "100 fixed episodes · 3 City A + 3 City B cases held constant · "
        "case-noise SD 8.0 · convergence tolerance 0.50 · no LLM calls",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    x = np.asarray(c_cases)
    city_c_plot = np.asarray(
        [np.nan if value is None else value for value in city_c], dtype=float
    )
    ax.plot(
        x,
        city_c_plot,
        color="#E69F00",
        linestyle="--",
        linewidth=3.0,
        marker="D",
        markersize=7,
        label="City C regression",
    )
    ax.plot(
        x,
        abc,
        color="#1B9E77",
        linestyle="--",
        linewidth=3.0,
        marker="X",
        markersize=8,
        label="A/B/C regression",
    )
    ax.fill_between(x, abc, city_c_plot, color="#D9D9D9", alpha=0.32)
    for n, c_value, abc_value, gap in zip(x, city_c, abc, gaps):
        if c_value is not None:
            ax.annotate(
                f"{c_value:.2f}",
                (n, c_value),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#9A6800",
            )
        ax.annotate(
            f"{abc_value:.2f}",
            (n, abc_value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#126B52",
        )
        if gap is not None:
            ax.text(
                n,
                0.5 * (c_value + abc_value),
                f"gap {gap:.2f}",
                ha="center",
                va="center",
                fontsize=8.4,
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
    ax.set_xticks(x)
    ax.set_xlabel("Number of displayed City C cases", fontsize=11)
    ax.set_ylabel("MAE (poll points)", fontsize=11)
    ymax = max(value for value in city_c if value is not None) * 1.18
    ax.set_ylim(0, ymax)
    if args.min_c == 0:
        ax.text(
            0,
            ymax * 0.88,
            "City C regression\nundefined (no C rows)",
            ha="center",
            va="center",
            fontsize=9,
            color="#9A6800",
            bbox={"facecolor": "#FFF7DE", "edgecolor": "#E69F00", "alpha": 0.9},
        )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.text(
        0.5,
        0.125,
        r"City C:  $\hat\beta_C=\frac{\sum_{i\in C}X_i(Y_i-S_i)}"
        r"{\sum_{i\in C}X_i^2}$"
        "      "
        r"A/B/C:  $\hat\beta_{ABC}=\frac{\sum_{i\in A,B,C}X_i(Y_i-S_i)}"
        r"{\sum_{i\in A,B,C}X_i^2}$",
        ha="center",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FFF1A8",
            "edgecolor": "#C9A227",
        },
    )
    fig.text(
        0.5,
        0.045,
        "Both regressions use displayed rows only; held-out truth is used only to compute MAE.",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "two_regressions.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    result = {
        "status": "sensitivity_preview_no_model_calls",
        "episodes": design.EPISODES,
        "case_noise_sd": design.CASE_NOISE_SD,
        "reference_cases": {"A": 3, "B": 3},
        "city_c_cases": list(c_cases),
        "city_c_regression_mae": dict(zip(map(str, c_cases), city_c)),
        "abc_regression_mae": dict(zip(map(str, c_cases), abc)),
        "gap_city_c_minus_abc": dict(zip(map(str, c_cases), gaps)),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "final_gap": final_gap,
        "convergence_passes": convergence_passes,
        "figure": str(output),
    }
    (outdir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
