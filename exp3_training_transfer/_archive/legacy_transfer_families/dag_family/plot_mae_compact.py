#!/usr/bin/env python3
"""Compact endpoint next-poll MAE figure for Experiment 3.

Shows the completed family-trained and structureless-control arms against their
shared untrained checkpoint on each held-out causal graph.

    /usr/bin/python3 plot_mae_compact.py

Writes paper/figures/exp3_next_poll_mae.{pdf,png}.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from transfer_report import MARGIN_MIN, compute_refs, read_dump, summarize

REPORTS = HERE / "reports"
OUT = HERE.parent.parent / "paper" / "figures" / "exp3_next_poll_mae"

RUNS = {
    "29893776": ("family", 42),
    "29893778": ("family", 43),
    "29899438": ("family", 44),
    "29893777": ("control", 42),
    "29893779": ("control", 43),
    "29893781": ("control", 44),
}
ORDER = ["direct", "two_news_feedback", "mediators", "chain4"]
NICE = {
    "direct": "Direct",
    "two_news_feedback": "Two-news",
    "mediators": "Mediators",
    "chain4": "Chain-4",
}

TEAL = "#0e9384"
AMBER = "#b45309"
GRAY = "#737b85"
INK = "#171a1f"
MUTED = "#656c75"
GRID = "#e1e5ea"


def eval_dir(jobid: str) -> Path:
    for run_dir in sorted(REPORTS.glob(f"pilot_*_j{jobid}")):
        candidates = list(run_dir.glob("**/eval_results"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"no evaluation directory for job {jobid}")


def pool(cells, structure, prefixes):
    pooled = {}
    for k in prefixes:
        for seed, cell in cells.get((structure, k), {}).items():
            pooled[(seed, k)] = cell
    return pooled


def main() -> None:
    dumps = {}
    for jobid, (arm, seed) in RUNS.items():
        paths = sorted(
            (p for p in eval_dir(jobid).glob("*.json") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        dumps[(arm, seed)] = {int(p.stem): p for p in paths}

    steps_by_arm = defaultdict(lambda: defaultdict(set))
    for (arm, seed), steps in dumps.items():
        for step in steps:
            steps_by_arm[step][arm].add(seed)
    paired = {
        step: sorted(arms.get("family", set()) & arms.get("control", set()))
        for step, arms in steps_by_arm.items()
    }
    endpoint = max(step for step, seeds in paired.items() if step > 0 and len(seeds) >= 2)
    seeds = paired[endpoint]
    if endpoint != 301 or seeds != [42, 43, 44]:
        raise RuntimeError(
            f"refusing incomplete endpoint: step={endpoint}, paired seeds={seeds}"
        )

    # The step-0 checkpoint and evaluation set are shared across paired runs.
    base_cells, _ = read_dump(dumps[("family", seeds[0])][0])
    prefixes = sorted({k for (_structure, k) in base_cells})
    cities = defaultdict(set)
    for (structure, _k), per_city in base_cells.items():
        cities[structure].update(per_city)
    refs = compute_refs(
        {structure: sorted(city_seeds) for structure, city_seeds in cities.items()},
        tuple(prefixes),
    )

    informative = {}
    for structure in ORDER:
        informative[structure] = [
            k
            for k in prefixes
            if summarize(refs[("prior_blind", structure, k)])["pi"]
            - summarize(refs[("oracle", structure, k)])["pi"]
            >= MARGIN_MIN
        ] or list(prefixes)

    means = {"family": [], "control": [], "untrained": []}
    raw = defaultdict(list)
    for structure in ORDER:
        ks = informative[structure]
        base_mae = summarize(pool(base_cells, structure, ks))["pi"]
        means["untrained"].append(base_mae)
        raw[("untrained", structure)].append(base_mae)

        for arm in ("family", "control"):
            values = []
            for seed in seeds:
                cells, _ = read_dump(dumps[(arm, seed)][endpoint])
                values.append(summarize(pool(cells, structure, ks))["pi"])
            means[arm].append(float(np.mean(values)))
            raw[(arm, structure)].extend(values)

    fig, ax = plt.subplots(figsize=(3.45, 2.42))
    y = np.arange(len(ORDER), dtype=float)
    height = 0.205
    series = [
        ("family", "Family-trained", TEAL, -height),
        ("control", "Structureless", AMBER, 0.0),
        ("untrained", "Untrained", GRAY, height),
    ]

    max_value = max(max(values) for values in means.values())
    for key, label, color, offset in series:
        vals = np.asarray(means[key])
        bars = ax.barh(
            y + offset,
            vals,
            height=height * 0.88,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                value + 0.14,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=6.0,
                color=INK,
            )

    ax.set_yticks(y, [NICE[s] for s in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, max_value + 1.25)
    ax.set_xlabel("Next-poll MAE (poll points)  ·  lower is better", fontsize=6.8, color=MUTED)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.tick_params(axis="x", labelsize=6.4, colors=MUTED, length=0)
    ax.tick_params(axis="y", labelsize=6.8, colors=INK, length=0, pad=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handlelength=1.15,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.20, top=0.88)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT.with_suffix(f".{suffix}"),
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
    plt.close(fig)

    print(f"endpoint dump step={endpoint}; paired seeds={seeds}")
    for i, structure in enumerate(ORDER):
        print(
            f"{structure:20s} family={means['family'][i]:.2f}  "
            f"control={means['control'][i]:.2f}  "
            f"untrained={means['untrained'][i]:.2f}"
        )
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
