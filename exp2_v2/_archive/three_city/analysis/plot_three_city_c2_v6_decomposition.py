#!/usr/bin/env python3
"""Split the v6 gap into arithmetic cost and sibling neglect, using the solo arm.

The headline gap between a model and the ceiling has two sources that the blind
arm alone cannot separate:

  1. failing to recover City C's own response from City C's own cases;
  2. recovering it, then declining to combine it with the two examples.

Only (2) is the inductive-inference claim. The solo arm shows City C and nothing
else, so a model's error there is (1) with no siblings available to neglect.
That makes the split measurable:

    arithmetic cost = solo error - a perfect target-only fit
    sibling neglect = (blind error - ceiling) - arithmetic cost

Everything is computed on episodes present in BOTH arms, so the subtraction is
paired episode by episode rather than comparing different samples.

Note on the standard findings figure: it is misleading for the solo arm. Its
panels are framed around "the two kinds of city", and the solo prompt never
shows them, so a solo model has no way to know two kinds exist. This figure is
the one to read for that arm.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v6 import PREFIX_LADDER

_DATA = _ROOT / "data"
_DISPLAY = {
    "claude-opus-4-8": "Claude",
    "DeepSeek-V4-Pro": "DeepSeek",
    "gpt-5.4": "GPT-5.4",
}
_COLORS = {
    "claude-opus-4-8": "#7b4ab5",
    "DeepSeek-V4-Pro": "#168a86",
    "gpt-5.4": "#d65f45",
}
_ARITH = "#c98b2e"
_SIBLING = "#2b8c6b"


def _read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mae(rows, source, k):
    values = [
        r["mae"]
        for r in rows
        if r["source"] == source
        and r["k"] == k
        and r["role"] == "none"
        and r["mae"] is not None
    ]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind",
        type=Path,
        default=_DATA / "three_city_c2_v6" / "pilot_v6" / "pilot_v6_rows.jsonl",
    )
    parser.add_argument(
        "--solo",
        type=Path,
        default=_DATA
        / "three_city_c2_v6_solo"
        / "pilot_v6_solo"
        / "pilot_v6_solo_rows.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DATA
        / "three_city_c2_v6_solo"
        / "pilot_v6_solo"
        / "decomposition.png",
    )
    args = parser.parse_args()

    blind, solo = _read(args.blind), _read(args.solo)
    common = {r["episode_id"] for r in blind} & {r["episode_id"] for r in solo}
    if not common:
        raise SystemExit("no episodes are present in both arms yet")
    blind = [r for r in blind if r["episode_id"] in common]
    solo = [r for r in solo if r["episode_id"] in common]

    models = [m for m in _DISPLAY if any(r["source"] == m for r in blind)]
    ks = [k for k in PREFIX_LADDER if any(r["k"] == k for r in blind)]
    top = max(ks)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # --- 1. the arithmetic floor -------------------------------------
    for model in models:
        axes[0].plot(
            ks,
            [_mae(solo, model, k) for k in ks],
            marker="s",
            linestyle="--",
            linewidth=2,
            markerfacecolor="white",
            color=_COLORS[model],
            label=f"{_DISPLAY[model]} — City C alone",
        )
        axes[0].plot(
            ks,
            [_mae(blind, model, k) for k in ks],
            marker="o",
            linewidth=2,
            color=_COLORS[model],
            label=f"{_DISPLAY[model]} — with both examples",
        )
    axes[0].plot(
        ks,
        [_mae(blind, "frequentist", k) for k in ks],
        linestyle="-.",
        linewidth=1.8,
        color="#222222",
        label="perfect fit to City C alone",
    )
    axes[0].plot(
        ks,
        [_mae(blind, "target_only_bayes", k) for k in ks],
        linestyle=(0, (5, 2)),
        linewidth=1.8,
        color="#b03a6a",
        label="best possible",
    )
    axes[0].set_title(
        "1. Showing the examples barely helps", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Forecast error (poll points)")

    # --- 2. the split, at the top of the ladder -----------------------
    ceiling = _mae(blind, "target_only_bayes", top)
    target_only = _mae(blind, "frequentist", top)
    positions = range(len(models))
    arithmetic, sibling = [], []
    for model in models:
        arith = _mae(solo, model, top) - target_only
        arithmetic.append(max(arith, 0.0))
        sibling.append(max((_mae(blind, model, top) - ceiling) - arith, 0.0))
    axes[1].bar(
        positions, arithmetic, 0.55, color=_ARITH, alpha=0.85,
        label="cost of not recovering City C's own response",
    )
    axes[1].bar(
        positions, sibling, 0.55, bottom=arithmetic, color=_SIBLING, alpha=0.85,
        label="cost of not using the two examples",
    )
    for index, (a, s) in enumerate(zip(arithmetic, sibling)):
        axes[1].annotate(
            f"{100*a/(a+s):.0f}% / {100*s/(a+s):.0f}%",
            xy=(index, a + s),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#333333",
        )
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels([_DISPLAY[m] for m in models], fontsize=9)
    axes[1].set_ylabel("Gap from best possible (poll points)")
    axes[1].set_title(
        f"2. What the gap is made of (k={top})", fontsize=12, fontweight="bold"
    )
    axes[1].margins(y=0.16)

    # --- 3. does adding the examples help at all? ---------------------
    for model in models:
        axes[2].plot(
            ks,
            [
                (_mae(solo, model, k) or 0) - (_mae(blind, model, k) or 0)
                for k in ks
            ],
            marker="o",
            linewidth=2.2,
            color=_COLORS[model],
            label=_DISPLAY[model],
        )
    axes[2].axhline(0, color="black", linewidth=1.0)
    axes[2].axhline(
        target_only - ceiling,
        color="#b03a6a",
        linestyle=(0, (5, 2)),
        linewidth=1.6,
        label="what the examples are worth",
    )
    axes[2].set_ylabel("Solo error − error with the examples")
    axes[2].set_title(
        "3. Benefit actually taken from the examples",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].legend(fontsize=8)

    for axis in (axes[0], axes[2]):
        axis.set_xticks(ks)
        axis.set_xlabel("City C cases the forecaster has seen (k)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=1)
    axes[1].legend(
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=1,
        frameon=False,
    )

    fig.suptitle(
        "Is the gap failing to do the arithmetic, or failing to use the "
        f"examples?   ({len(common)} paired episodes)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
