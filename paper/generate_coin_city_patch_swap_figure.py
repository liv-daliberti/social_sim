#!/usr/bin/env python3
"""What the label patch does to the forecast, averaged over episodes.

The arms are counterbalanced -- City C wears KIV in half the episodes and ZOR in
the other half -- so averaging by arm cancels the effect exactly. Episodes are
therefore aligned by the regime the recipient started in: every episode whose
label named the fast reference is one group, every episode whose label named the
slow one is the other. Within each group the patch should push the forecast
toward the donor's regime, and the two groups push in opposite directions.

Read from the frozen generations, so the figure cannot drift from the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = (ROOT / "exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4"
       / "mechanistic_probe/qwen3_14b_symbol_relational_v2")
PATCH = "cross_selected_window"

plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
)

INK = "#333A45"
MUTED = "#6B7280"
TO_SLOW = "#B5405F"
TO_FAST = "#2C7FB8"
GRID = "#D9DCE0"


def load():
    rows = [json.loads(line) for line in (RUN / "activation_patch_generations.jsonl")
            .read_text().splitlines() if line.strip()]
    by: dict[tuple, dict] = {}
    for row in rows:
        by.setdefault((row["episode"], row["c_cases"], row["recipient_arm"]), {})[
            row["condition"]] = row
    pairs: dict[tuple, list] = {}
    for (episode, depth, _arm), conditions in by.items():
        before, after = conditions.get("unpatched"), conditions.get(PATCH)
        if not before or not after:
            continue
        if before.get("implied_slope") is None or after.get("implied_slope") is None:
            continue
        key = (depth, bool(before["recipient_cue_strong"]))
        pairs.setdefault(key, []).append(
            (before["implied_slope"], after["implied_slope"]))
    return pairs


def interval(values, index, draws=5000, seed=20260923):
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))][index] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def build(output_dir: Path) -> Path:
    pairs = load()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.15), facecolor="white", sharex=True,
                             sharey=True)
    fig.subplots_adjust(left=0.165, right=0.985, bottom=0.30, top=0.80, wspace=0.08)

    titles = {0: "(a) no City C cases yet", 4: "(b) after four City C cases"}
    rows = {True: (1.0, "started fast", TO_SLOW), False: (0.0, "started slow", TO_FAST)}

    for ax, depth in zip(axes, (0, 4)):
        ax.set_title(titles[depth], loc="left", fontsize=8.2,
                     fontweight="bold", color=INK, pad=6)
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        for started_strong, (y, _label, color) in rows.items():
            values = pairs[(depth, started_strong)]
            before = float(np.mean([v[0] for v in values]))
            after = float(np.mean([v[1] for v in values]))
            lo, hi = interval(values, 1)
            ax.annotate(
                "", xy=(after, y), xytext=(before, y),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.8,
                                shrinkA=3.2, shrinkB=0, mutation_scale=9),
            )
            ax.plot([lo, hi], [y, y], color=color, linewidth=0.9, alpha=0.5, zorder=1)
            ax.plot([before], [y], marker="o", markersize=5.2, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.3, zorder=3)
            ax.plot([after], [y], marker="o", markersize=5.2, color=color, zorder=3)
            # Before above the point, after below it, so the two never collide
            # however close the patch leaves them.
            ax.text(before, y + 0.20, f"{before:.2f}", ha="center", va="bottom",
                    fontsize=6.9, color=MUTED)
            ax.text(after, y - 0.22, f"{after:.2f}", ha="center", va="top",
                    fontsize=6.9, color=color, fontweight="bold")
        ax.set_ylim(-0.62, 1.66)
        ax.set_yticks([1.0, 0.0])
        ax.set_yticklabels(["started fast", "started slow"], fontsize=7.6)
        for tick, color in zip(ax.get_yticklabels(), (TO_SLOW, TO_FAST)):
            tick.set_color(color)
            tick.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0].set_xlim(0.40, 0.92)
    fig.supxlabel("implied responsiveness of City C  (poll points per news point)",
                  fontsize=7.8, color=INK, y=0.035)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "exp2_causal_patch_swap.pdf"
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()
    print(build(args.output_dir))


if __name__ == "__main__":
    main()
