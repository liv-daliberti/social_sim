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
PROBE = (ROOT / "exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4"
         / "mechanistic_probe")
MODELS = (
    ("Qwen3-14B", "qwen3_14b_symbol_relational_v2", "layers 19\u201321 of 40"),
    ("Llama-3.1-70B", "llama3_1_70b_symbol_relational_v1", "layers 35\u201337 of 80"),
)
# Two sites: the mid-network window the protocol selected, and the label token's
# own embedding. The second is close to editing the prompt, so it bounds what a
# complete swap looks like for that deployment.
SITES = (("cross_selected_window", "mid-network window"),
         ("cross_embedding_only", "label token itself"))

plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
)

INK = "#333A45"
MUTED = "#6B7280"
TO_WEAK = "#B5405F"
TO_STRONG = "#2C7FB8"
GRID = "#D9DCE0"
BAND = "#EDEFF2"
BAND_INK = "#8A93A3"
# The regimes City C is drawn from, from the frozen design: g ~ N(.90, .03^2)
# for a strong-response city and N(.25, .03^2) for a weak one. Two standard
# deviations either side covers essentially every episode.
REGIMES = ((0.250, 0.031, "weak-response\nregime"), (0.900, 0.027, "strong-response\nregime"))


def load(study: str):
    rows = [json.loads(line) for line in (PROBE / study / "activation_patch_generations.jsonl")
            .read_text().splitlines() if line.strip()]
    by: dict[tuple, dict] = {}
    for row in rows:
        by.setdefault((row["episode"], row["c_cases"], row["recipient_arm"]), {})[
            row["condition"]] = row
    pairs: dict[tuple, list] = {}
    for (_episode, depth, _arm), conditions in by.items():
        if depth != 0:                      # only the no-evidence stratum
            continue
        before = conditions.get("unpatched")
        if not before or before.get("implied_slope") is None:
            continue
        for site, _label in SITES:
            after = conditions.get(site)
            if not after or after.get("implied_slope") is None:
                continue
            key = (site, bool(before["recipient_cue_strong"]))
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
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), facecolor="white")
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.26, top=0.78, wspace=0.42)

    directions = {True: ("strong-response", TO_WEAK), False: ("weak-response", TO_STRONG)}
    ticks, labels = [], []

    for ax, (name, study, window) in zip(axes, MODELS):
        pairs = load(study)
        episodes = len(pairs[(SITES[0][0], True)]) + len(pairs[(SITES[0][0], False)])
        ax.set_title(f"{name}\n{window}  \u00b7  {episodes} matched pairs", loc="left",
                     fontsize=8.0, fontweight="bold", color=INK, pad=5, linespacing=1.45)
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        for centre, sd, band_label in REGIMES:
            ax.axvspan(centre - 2 * sd, centre + 2 * sd, color=BAND, zorder=0)
            ax.text(centre, 4.62, band_label, ha="center", va="top", fontsize=6.3,
                    color=BAND_INK, linespacing=1.3)
        ticks, labels = [], []
        for site_index, (site, site_label) in enumerate(SITES):
            for offset, (started_strong, (label, color)) in enumerate(directions.items()):
                y = 3.0 - site_index * 1.95 - offset * 0.62
                values = pairs[(site, started_strong)]
                before = float(np.mean([v[0] for v in values]))
                after = float(np.mean([v[1] for v in values]))
                lo, hi = interval(values, 1)
                ax.annotate("", xy=(after, y), xytext=(before, y),
                            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                            shrinkA=3.0, shrinkB=0, mutation_scale=8))
                ax.plot([lo, hi], [y, y], color=color, linewidth=0.8, alpha=0.5, zorder=1)
                ax.plot([before], [y], marker="o", markersize=4.4, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=1.2, zorder=3)
                ax.plot([after], [y], marker="o", markersize=4.4, color=color, zorder=3)
                ax.text(after, y - 0.20, f"{after:.2f}", ha="center", va="top",
                        fontsize=6.4, color=color, fontweight="bold")
                ticks.append(y)
                labels.append(label)
            ax.text(0.0, 3.0 - site_index * 1.95 + 0.40, site_label,
                    transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=7.2, color=INK, fontweight="bold", clip_on=False)
        ax.set_ylim(-0.02, 4.72)
        ax.set_xlim(0.15, 1.42)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7.0)
        for tick, color in zip(ax.get_yticklabels(), (TO_WEAK, TO_STRONG) * len(SITES)):
            tick.set_color(color)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7.2)
        ax.spines[["top", "right", "left"]].set_visible(False)

    fig.supxlabel("implied responsiveness of City C, no City C cases yet "
                  "(poll points per news point); shaded bands are the regimes "
                  "City C is drawn from", fontsize=7.4, color=INK, y=0.03)

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
