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
MODELS = {
    "qwen3_14b": ("Qwen3-14B", "qwen3_14b_symbol_relational_v2",
                  "exp2_causal_patch_swap.pdf", (0.15, 1.02)),
    "llama3_1_70b": ("Llama-3.1-70B", "llama3_1_70b_symbol_relational_v1",
                     "exp2_causal_patch_swap_llama.pdf", (0.15, 1.60)),
}
# Two sites: the mid-network window the protocol selected, and the label token's
# own embedding. The second is close to editing the prompt, so it bounds what a
# complete swap looks like for that deployment.
SITES = (("cross_selected_window", "mid-network\nwindow"),
         ("cross_embedding_only", "label token"))

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
# Each band wears its regime's hue, washed out so it reads as ground rather
# than as a mark: the series and the bands then agree that red is the
# strong-response regime and blue the weak one.
REGIMES = ((0.250, 0.031, "weak-response outcome", TO_STRONG),
           (0.900, 0.027, "strong-response outcome", TO_WEAK))


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


def build(output_dir: Path, key: str = "qwen3_14b") -> Path:
    name, study, filename, ylim = MODELS[key]
    pairs = load(study)
    fig, ax = plt.subplots(figsize=(3.45, 3.75), facecolor="white")
    fig.subplots_adjust(left=0.195, right=0.975, bottom=0.115, top=0.835)

    # Which reference the label pointed at, and therefore which city's cases the
    # forecast should follow.
    series = {True: ("strong-response\ndata", TO_WEAK), False: ("weak-response\ndata", TO_STRONG)}
    positions = {(0, True): 0.55, (0, False): 1.35, (1, True): 2.75, (1, False): 3.55}

    for centre, sd, band_label, band_colour in REGIMES:
        ax.axhspan(centre - 2 * sd, centre + 2 * sd, color=band_colour, alpha=0.11,
                   linewidth=0, zorder=0)
        # Sitting the label inside its own band ties the two together without a
        # leader line, and frees the right margin the outside labels needed.
        ax.text(0.14, centre, band_label, ha="left", va="center", fontsize=6.7,
                color=band_colour, alpha=0.9, fontweight="bold")

    episodes = len(pairs[(SITES[0][0], True)]) + len(pairs[(SITES[0][0], False)])
    # A guide at each starting level. The point of the result is that an arrow
    # ends where the other one began, which is hard to see across a gap and
    # obvious against a line.
    for started_strong, (_label, colour) in series.items():
        start = float(np.mean([v[0] for v in pairs[(SITES[0][0], started_strong)]]))
        ax.axhline(start, color=colour, linewidth=0.7, linestyle=(0, (2.2, 2.0)),
                   alpha=0.45, zorder=0)
    for site_index, (site, site_label) in enumerate(SITES):
        for started_strong, (label, color) in series.items():
            x = positions[(site_index, started_strong)]
            values = pairs[(site, started_strong)]
            before = float(np.mean([v[0] for v in values]))
            after = float(np.mean([v[1] for v in values]))
            lo, hi = interval(values, 1)
            ax.annotate("", xy=(x, after), xytext=(x, before),
                        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.9,
                                        shrinkA=3.2, shrinkB=0, mutation_scale=9))
            ax.plot([x, x], [lo, hi], color=color, linewidth=0.9, alpha=0.5, zorder=1)
            ax.plot([x], [before], marker="o", markersize=5.0, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.3, zorder=3)
            ax.plot([x], [after], marker="o", markersize=5.0, color=color, zorder=3)

    ax.set_xlim(0.0, 4.10)
    ax.set_ylim(*ylim)
    ax.set_xticks([(positions[(0, True)] + positions[(0, False)]) / 2,
                   (positions[(1, True)] + positions[(1, False)]) / 2])
    ax.set_xticklabels([label for _site, label in SITES], fontsize=7.4,
                       fontweight="bold", linespacing=1.3)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.set_ylabel("implied responsiveness of City C\n(poll points per news point)",
                  fontsize=7.4, color=INK, linespacing=1.4)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Colour says which reference the label pointed at; the marker says whether
    # the forecast is the unpatched one or the patched one. Both are needed to
    # read a single arrow, so both are in the legend.
    handles = [plt.Line2D([], [], color=color, linewidth=1.9, label=label.replace("\n", " "))
               for label, color in series.values()]
    handles += [
        plt.Line2D([], [], color=MUTED, linestyle="none", marker="o", markersize=5.0,
                   markerfacecolor="white", markeredgewidth=1.3,
                   label="before the patch"),
        plt.Line2D([], [], color=MUTED, linestyle="none", marker="o", markersize=5.0,
                   label="after the patch"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.01),
              ncol=2, frameon=False, fontsize=6.7, handlelength=1.5,
              columnspacing=1.0, handletextpad=0.55, labelspacing=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    parser.add_argument("--model", choices=sorted(MODELS), default=None)
    args = parser.parse_args()
    for key in ([args.model] if args.model else sorted(MODELS)):
        print(build(args.output_dir, key))


if __name__ == "__main__":
    main()
