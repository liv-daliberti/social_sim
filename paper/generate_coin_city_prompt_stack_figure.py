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
QUESTION_FILL = "#FFF2CC"   # draw.io yellow
QUESTION_EDGE = "#D6B656"
C_FILL = "#DAE8FC"          # draw.io blue, the city being forecast
C_EDGE = "#6C8EBF"
REF_FILL = "#D5E8D4"        # draw.io green, the two reference cities
REF_EDGE = "#82B366"
HINT_FILL = "#E1D5E7"       # draw.io purple, the hint
HINT_EDGE = "#9673A6"
MONO = {"family": "monospace", "fontsize": 6.2}
TAG_INK = "#5A7D3F"
HIGHLIGHT = "#FFE9A8"

def _row(start, news, change, end):
    return f"{start:>10}{news:>11}{change:>12}{end:>11}"


HEADER = _row("Start", "News", "Change", "End")
CITY_C = (_row("57.8", "+8", "+9.4", "67.2"),)
CITY_A = (_row("45.4", "-8", "-6.5", "38.9"),
          _row("45.4", "+8", "+12.2", "57.6"))
CITY_B = (_row("35.0", "-10", "-5.6", "29.4"),
          _row("35.0", "+10", "+9.3", "44.3"))


def box(ax, x, y, w, h, fill, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=fill, edgecolor=edge, linewidth=0.9, mutation_aspect=0.5,
        )
    )


TABLE_W = 8.82
RULE = "#6B7280"


def _rule(ax, x, y, width, lw):
    ax.plot([x, x + width], [y, y], color=RULE, linewidth=lw,
            solid_capstyle="butt", zorder=3)


def measure(ax, artist):
    """Width of a drawn text in data units."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    return ax.transData.inverted().transform(bbox.corners())[2, 0] - \
        ax.transData.inverted().transform(bbox.corners())[0, 0]


def table(ax, x, y_top, name, rows, tag, LINE, PAD, tag_colour=TAG_INK):
    """One city's cases, ruled top / under-header / bottom.

    The y axis is on a different scale from the x axis, so every rule is placed
    a full LINE clear of the text above it rather than a fraction of one.
    """
    left = x + PAD
    title = ax.text(left, y_top, name, ha="left", va="top",
                    fontsize=7.4, fontweight="bold", color=INK)
    if tag is not None:
        width = measure(ax, title)
        ax.text(left + width + 0.16, y_top - LINE * 0.06, tag,
                ha="left", va="top", fontsize=6.6, color=tag_colour,
                fontweight="bold")

    top_rule = y_top - LINE * 1.30
    _rule(ax, left, top_rule, TABLE_W, 1.0)
    ax.text(left, top_rule - LINE * 0.26, HEADER, ha="left", va="top",
            color=MUTED, **MONO)

    head_rule = top_rule - LINE * 1.30
    _rule(ax, left, head_rule, TABLE_W, 0.7)
    for index, row in enumerate(rows):
        ax.text(left, head_rule - LINE * (0.26 + index), row,
                ha="left", va="top", color=INK, **MONO)

    bottom_rule = head_rule - LINE * (len(rows) + 0.30)
    _rule(ax, left, bottom_rule, TABLE_W, 1.0)
    return bottom_rule


def build(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(3.3, 4.36), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(1.42, 13.70)
    ax.axis("off")

    left, width = 0.30, 9.40
    LINE, PAD = 0.40, 0.22
    top = 13.58

    # The question, first.
    height = PAD * 2 + LINE * 4.0
    box(ax, left, top - height, width, height, QUESTION_FILL, QUESTION_EDGE)
    ax.text(left + PAD, top - PAD,
            "Prompt:  City C has a new starting poll\n"
            "of 53.9 and has net news of +5.\n"
            "What do you predict the poll will be\n"
            "at the end of the week?",
            ha="left", va="top", fontsize=6.9, color=INK,
            fontweight="bold", linespacing=1.5)
    top -= height + 0.30

    cards = (
        ("CITY C  \u00b7  being forecast", CITY_C, "prior samples", C_FILL, C_EDGE),
        ("CITY A", CITY_A, "national news", REF_FILL, REF_EDGE),
        ("CITY B", CITY_B, "local news", REF_FILL, REF_EDGE),
    )
    for name, rows, tag, fill, edge in cards:
        height = PAD * 2 + LINE * (len(rows) + 2.9)
        box(ax, left, top - height, width, height, fill, edge)
        table(ax, left, top - PAD, name, rows, tag, LINE, PAD, edge)
        top -= height + 0.30

    # The hint.
    height = PAD * 2 + LINE * 4.0
    box(ax, left, top - height, width, height, HINT_FILL, HINT_EDGE)
    ax.text(left + PAD, top - PAD, "HINT", ha="left", va="top",
            fontsize=6.9, fontweight="bold", color=HINT_EDGE)
    ax.text(left + PAD, top - PAD - LINE * 1.02,
            "City C residents generally encounter",
            ha="left", va="top", fontsize=6.9, color=INK)
    third = top - PAD - LINE * 2.02
    ax.text(left + PAD, third, "campaign developments through",
            ha="left", va="top", fontsize=6.9, color=INK)
    phrase = ax.text(left + PAD, third - LINE, "national news coverage",
                     ha="left", va="top", fontsize=6.9, color=INK,
                     fontweight="bold", zorder=3)
    span = measure(ax, phrase)
    ax.add_patch(
        FancyBboxPatch(
            (left + PAD - 0.03, third - LINE - LINE * 0.60), span + 0.07, LINE * 0.64,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=HIGHLIGHT, edgecolor="none", zorder=1.5, mutation_aspect=0.5,
        )
    )
    ax.text(left + PAD + span + 0.02, third - LINE, ".", ha="left", va="top",
            fontsize=6.9, color=INK)

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
