#!/usr/bin/env python3
"""Three panels telling the one thing the C2 v6 pilot found.

The six-panel `pilot_v6_diagnostics` figure is organised by metric, which buries
the result. This figure is organised by claim:

1. Their forecasts land on the correct side of the two demonstrated
   values. NOTE: not by itself evidence of inference -- the plain average of
   City C's own cases scores identically, so it is plotted alongside. The
   inference evidence is the k=0 column, where there is nothing to average.
2. Their forecast nevertheless tracks City C's own data alone. In v4 the
   news value varies per case, so that means a fitted slope rather than a
   column average -- averaging is no longer a valid estimator here.
3. So they forecast a spread of values instead of the one their own
   classification implies, and leave most of the achievable accuracy unused.

Reads the scored rows written by `analyze_three_city_c2_v6_pilot.py`; makes no
API calls and does not re-derive any score.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v6 import (
    HIGH_RESPONSE,
    HIGH_TYPE,
    LOW_RESPONSE,
    PREFIX_LADDER,
)

_DATA = _ROOT / "data" / "three_city_c2_v6"
_PILOT = _DATA / "pilot_v6"
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
_TARGET_ONLY = "frequentist"
_CEILING = "target_only_bayes"
_GREEN = "#2b8c6b"
_CEILING_COLOR = "#b03a6a"


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _mean(rows, source, k, field, *, role="none"):
    values = [
        row[field]
        for row in rows
        if row["source"] == source
        and row["k"] == k
        and row["role"] == role
        and row[field] is not None
    ]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=_PILOT / "pilot_v6_rows.jsonl")
    parser.add_argument("--out", type=Path, default=_PILOT / "pilot_v6_findings.png")
    parser.add_argument("--top-k", type=int, default=max(PREFIX_LADDER))
    args = parser.parse_args()

    rows = _read_jsonl(args.rows)
    models = [m for m in _DISPLAY if any(r["source"] == m for r in rows)]
    episodes = len({row["episode_id"] for row in rows})
    ks = [k for k in PREFIX_LADDER if any(r["k"] == k for r in rows)]
    top = args.top_k

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))

    # --- 1. Identification succeeds ------------------------------------
    for model in models:
        axes[0].plot(
            ks,
            [_mean(rows, model, k, "type_accuracy") for k in ks],
            marker="o",
            linewidth=2.2,
            color=_COLORS[model],
            label=_DISPLAY[model],
        )
    axes[0].plot(
        ks,
        [_mean(rows, _CEILING, k, "type_accuracy") for k in ks],
        linestyle=(0, (5, 2)),
        linewidth=1.6,
        color=_CEILING_COLOR,
        label="Best possible",
    )
    axes[0].plot(
        ks,
        [_mean(rows, _TARGET_ONLY, k, "type_accuracy") for k in ks],
        linestyle="none",
        marker="s",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        color="#222222",
        label="City C alone (never reads A or B)",
    )
    axes[0].axhline(0.5, color="black", linewidth=0.8)
    axes[0].annotate(
        "chance",
        xy=(ks[-1], 0.5),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color="#444444",
    )
    axes[0].set_ylim(0.35, 1.04)
    axes[0].set_title(
        "1. The forecast lands on the correct side",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Fraction of cities classified correctly")
    axes[0].legend(fontsize=8, loc="lower right")

    # --- 2. The forecast tracks the target's own data alone ------------
    target_only = [_mean(rows, _TARGET_ONLY, k, "mae") for k in ks]
    ceiling = [_mean(rows, _CEILING, k, "mae") for k in ks]
    shade = [k for k, value in zip(ks, target_only) if value is not None]
    offset = len(ks) - len(shade)
    axes[1].fill_between(
        shade,
        ceiling[offset:],
        target_only[offset:],
        color=_GREEN,
        alpha=0.18,
        linewidth=0,
        zorder=0,
        label="Accuracy the two examples make available",
    )
    for model in models:
        axes[1].plot(
            ks,
            [_mean(rows, model, k, "mae") for k in ks],
            marker="o",
            linewidth=2.2,
            color=_COLORS[model],
            label=_DISPLAY[model],
        )
    axes[1].plot(
        ks,
        target_only,
        linestyle="-.",
        linewidth=1.8,
        color="#222222",
        label="Use City C's own cases only",
    )
    axes[1].plot(
        ks,
        ceiling,
        linestyle=(0, (5, 2)),
        linewidth=1.8,
        color=_CEILING_COLOR,
        label="Best possible",
    )
    axes[1].set_title(
        "2. But the forecast follows City C's own data alone",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylabel("Forecast error (poll points)")
    axes[1].legend(fontsize=8, loc="upper right")

    # --- 3. Two-sided bars centred on the decision boundary -----------
    # Everything is re-expressed as signed distance from the boundary, so the
    # axis line IS the boundary: up means "called it responsive", down means
    # "called it buffered". A bar reaching the dashed line has committed fully to
    # the value its own classification implies. This separates the two ways a
    # forecast can fail: a short bar is hedging toward the boundary, while a bar
    # of the right height with scattered dots is unbiased but noisy.
    rng = random.Random(0)
    midpoint = (LOW_RESPONSE + HIGH_RESPONSE) / 2.0
    ideal = HIGH_RESPONSE - midpoint
    sources = [_CEILING] + models
    short = {"claude-opus-4-8": "Claude", "DeepSeek-V4-Pro": "DeepSeek", "gpt-5.4": "GPT-5.4"}
    source_labels = ["Best\npossible"] + [short[m] for m in models]
    half = 0.34
    for index, source in enumerate(sources):
        colour = _COLORS.get(source, _CEILING_COLOR)
        for city_type in (HIGH_TYPE, "buffered_information"):
            selected = [
                row["response"] - midpoint
                for row in rows
                if row["source"] == source
                and row["k"] == top
                and row["role"] == "none"
                and row["target_type"] == city_type
                and row["response"] is not None
            ]
            if not selected:
                continue
            mean = statistics.mean(selected)
            spread = statistics.pstdev(selected)
            axes[2].bar(
                index,
                mean,
                width=half * 2,
                color=colour,
                alpha=0.32,
                edgecolor=colour,
                linewidth=1.2,
                zorder=1,
            )
            for value in selected:
                wrong = (value > 0) != (city_type == HIGH_TYPE)
                axes[2].scatter(
                    index + rng.uniform(-half * 0.78, half * 0.78),
                    value,
                    s=15,
                    alpha=0.85 if wrong else 0.5,
                    linewidths=1.1 if wrong else 0,
                    color="#c0392b" if wrong else colour,
                    marker="X" if wrong else "o",
                    zorder=4 if wrong else 3,
                )
            # Spread is the whole story here, so label it on the bar.
            axes[2].annotate(
                f"sd {spread:.2f}",
                xy=(index, 0.76 if mean > 0 else -0.76),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#333333",
            )

    axes[2].axhline(0, color="black", linewidth=1.6, zorder=2)
    for level, name in ((ideal, "responsive"), (-ideal, "buffered")):
        axes[2].axhline(
            level, color="black", linewidth=1.0, linestyle="--", zorder=1
        )
        axes[2].annotate(
            f"true value,\n{name} city",
            xy=(-1.15, level),
            ha="left",
            va="center",
            fontsize=7.5,
            color="#444444",
        )
    axes[2].annotate(
        "boundary:\nwhich kind\nof city",
        xy=(-1.15, 0),
        ha="left",
        va="center",
        fontsize=7.5,
        color="#444444",
    )
    axes[2].set_xticks(range(len(sources)))
    axes[2].set_xticklabels(source_labels, fontsize=8.5)
    axes[2].set_xlim(-1.20, len(sources) - 0.45)
    axes[2].set_ylim(-0.92, 0.90)
    axes[2].set_ylabel("Forecast distance from the boundary")
    axes[2].set_xlabel(f"after {top} City C cases, no background")
    axes[2].set_title(
        "3. Right height on average, wrong every time",
        fontsize=12,
        fontweight="bold",
    )

    for axis in axes[:2]:
        axis.set_xticks(ks)
        axis.set_xlabel("City C cases the forecaster has seen (k)")
    for axis in axes:
        axis.grid(alpha=0.25)

    interim = "  —  INTERIM, TOO FEW EPISODES TO READ" if episodes < 24 else ""
    fig.suptitle(
        "Two cities shown, a third forecast: the right side is found, "
        f"the right value is not   ({episodes} episodes){interim}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
