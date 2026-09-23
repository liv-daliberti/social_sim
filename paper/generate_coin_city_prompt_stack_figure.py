#!/usr/bin/env python3
"""What a Coin City model is shown, in the order it is shown it.

The question comes first, because it is what the reader needs in order to know
what the tables are for. Then one City C case as an example, then the two
reference cities, then the hint.

Content is copied from the frozen design (design/example_prompt_abc_context.txt),
not retyped, so the figure shows the prompt the models actually received. Rows
are reduced -- one case for City C, two for each reference, where the design
supplies four -- and each block says so, because those counts are part of the
manipulation rather than part of the design.
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
    {"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
)

INK = "#333A45"
MUTED = "#6B7280"
QUESTION_FILL = "#FFF6E3"
QUESTION_EDGE = "#D9A441"
CARD_FILL = "#EDF3F8"
CARD_EDGE = "#4F7EA8"
HINT_FILL = "#DCE9F5"
HINT_EDGE = "#2C7FB8"
MONO = {"family": "monospace", "fontsize": 6.6}

HEADER = "Start    News   Change     End"
CITY_C = ("   57.8      +8     +9.4    67.2",)
CITY_A = ("   45.4      -8     -6.5    38.9",
          "   45.4      +8    +12.2    57.6")
CITY_B = ("   35.0     -10     -5.6    29.4",
          "   35.0     +10     +9.3    44.3")


def box(ax, x, y, w, h, fill, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=fill, edgecolor=edge, linewidth=0.9, mutation_aspect=0.5,
        )
    )


def table(ax, x, y_top, name, rows, note, LINE, PAD):
    ax.text(x + PAD, y_top, name, ha="left", va="top",
            fontsize=7.4, fontweight="bold", color=INK)
    ax.text(x + PAD, y_top - LINE * 1.3, HEADER, ha="left", va="top", color=MUTED, **MONO)
    for index, row in enumerate(rows):
        ax.text(x + PAD, y_top - LINE * (2.3 + index), row,
                ha="left", va="top", color=INK, **MONO)
    ax.text(x + PAD, y_top - LINE * (2.5 + len(rows)), note,
            ha="left", va="top", fontsize=6.4, color=MUTED, style="italic")


def build(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.55), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.05)
    ax.axis("off")

    left, width = 0.08, 9.84
    LINE, PAD = 0.34, 0.20
    top = 7.90

    # The question, first.
    height = PAD * 2 + LINE * 1.9
    box(ax, left, top - height, width, height, QUESTION_FILL, QUESTION_EDGE)
    ax.text(left + PAD, top - PAD,
            "A new City C case is starting with a poll of 53.9 and has net news of +5.\n"
            "What do you predict the poll will be at the end of the week?",
            ha="left", va="top", fontsize=7.6, color=INK, fontweight="bold", linespacing=1.5)
    top -= height + 0.34

    # City C, one case, as an example of how a case reads.
    height = PAD * 2 + LINE * 4.0
    box(ax, left, top - height, width, height, CARD_FILL, CARD_EDGE)
    table(ax, left, top - PAD, "CITY C  ·  the city being forecast",
          CITY_C, "one earlier case, shown as an example", LINE, PAD)
    ax.text(left + width - PAD, top - PAD - LINE * 1.9,
            "← example: a poll of 57.8 met +8 net news and ended at 67.2",
            ha="right", va="top", fontsize=6.6, color=MUTED)
    top -= height + 0.34

    # The two reference cities, side by side.
    half = (width - 0.24) / 2
    height = PAD * 2 + LINE * 5.0
    for offset, (name, rows) in enumerate((("CITY A", CITY_A), ("CITY B", CITY_B))):
        x = left + offset * (half + 0.24)
        box(ax, x, top - height, half, height, CARD_FILL, CARD_EDGE)
        table(ax, x, top - PAD, name, rows, "two of its earlier cases", LINE, PAD)
    top -= height + 0.34

    # The hint.
    height = PAD * 2 + LINE * 1.9
    box(ax, left, top - height, width, height, HINT_FILL, HINT_EDGE)
    ax.text(left + PAD, top - PAD, "HINT", ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=HINT_EDGE)
    ax.text(left + PAD + 0.92, top - PAD,
            "City C residents generally encounter campaign developments through\n"
            "national news coverage — the same as City A.",
            ha="left", va="top", fontsize=7.2, color=INK, linespacing=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "exp2_coin_city_prompt_stack.pdf"
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()
    print(build(args.output_dir))


if __name__ == "__main__":
    main()
