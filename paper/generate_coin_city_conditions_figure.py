#!/usr/bin/env python3
"""The five Coin City conditions as one compact schematic.

Each condition adds information to the one before it, so the figure is a matrix:
rows are the conditions (a)-(e), columns are the three things a prompt can
carry. A reader should be able to find any single condition and read off what it
supplies without consulting the caption.

Only the cue column is coloured, because only the cue varies in kind rather than
in presence: correct, inverted, or arbitrary-but-recoverable. Observation
columns are present or absent and need no hue.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

INK = "#333A45"
MUTED = "#6B7280"
ABSENT_EDGE = "#C9CDD4"
PRESENT_FILL = "#E8EEF4"
PRESENT_EDGE = "#4F7EA8"
CUE_STYLES = {
    "correct": ("#DCE9F5", "#2C7FB8", "national / local"),
    "wrong": ("#F7E1E7", "#B5405F", "national / local, inverted"),
    "arbitrary": ("#EBE3F2", "#7A5195", "KIV / ZOR"),

}

COLUMNS = ("City C cases", "City A/B cases", "Context cue")
# (label, short name, city C, references, cue kind)
ROWS = (
    ("(a)", "target only", True, False, None),
    ("(b)", "add references", True, True, None),
    ("(c)", "add correct cue", True, True, "correct"),
    ("(d)", "add wrong cue", True, True, "wrong"),
    ("(e)", "add arbitrary cue", True, True, "arbitrary"),
)


def _cell(ax, x, y, w, h, *, fill, edge, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.045",
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.8,
            linestyle=(0, (2.4, 1.8)) if dashed else "solid",
            mutation_aspect=0.42,
        )
    )


def build(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 2.05), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.75)
    ax.axis("off")

    lefts = (2.70, 5.08, 7.46)
    width = 2.24
    height = 0.66
    gap = 0.95

    for x, title in zip(lefts, COLUMNS):
        ax.text(
            x + width / 2,
            5.12,
            title,
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=INK,
        )

    for index, (tag, name, has_c, has_ab, cue) in enumerate(ROWS):
        y = 4.28 - index * gap
        ax.text(0.02, y + height / 2, tag, ha="left", va="center",
                fontsize=8.8, fontweight="bold", color=INK)
        ax.text(0.50, y + height / 2, name, ha="left", va="center",
                fontsize=7.7, color=MUTED)

        for column, present in enumerate((has_c, has_ab)):
            if present:
                _cell(ax, lefts[column], y, width, height,
                      fill=PRESENT_FILL, edge=PRESENT_EDGE)
                label = "four rows" if column else "zero to four rows"
                ax.text(lefts[column] + width / 2, y + height / 2, label,
                        ha="center", va="center", fontsize=7.4, color=INK)
            else:
                _cell(ax, lefts[column], y, width, height,
                      fill="white", edge=ABSENT_EDGE, dashed=True)
                ax.text(lefts[column] + width / 2, y + height / 2, "not shown",
                        ha="center", va="center", fontsize=7.2, color=ABSENT_EDGE)

        if cue is None:
            _cell(ax, lefts[2], y, width, height,
                  fill="white", edge=ABSENT_EDGE, dashed=True)
            ax.text(lefts[2] + width / 2, y + height / 2, "not shown",
                    ha="center", va="center", fontsize=7.2, color=ABSENT_EDGE)
        else:
            fill, edge, text = CUE_STYLES[cue]
            _cell(ax, lefts[2], y, width, height, fill=fill, edge=edge)
            ax.text(lefts[2] + width / 2, y + height / 2, text,
                    ha="center", va="center", fontsize=6.7, color=INK)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "exp2_coin_city_conditions.pdf"
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
