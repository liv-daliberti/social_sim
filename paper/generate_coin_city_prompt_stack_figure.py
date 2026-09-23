#!/usr/bin/env python3
"""One Coin City prompt, with every condition's addition shown in place.

The conditions differ by what they append to a single prompt, so the figure
shows that prompt once and marks each block with the condition that introduces
it. Two cases per city are drawn rather than four; the design supplies four, and
the City C prefix varies from zero to four.

Content is copied from the frozen design (design/example_prompt_abc_context.txt
and tasks_abc_wrong_context.jsonl), not retyped, so the figure shows the prompt
the models actually received.
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
SHARED_EDGE = "#9AA3AE"
SHARED_FILL = "#F4F5F7"
ADD_EDGE = "#4F7EA8"
ADD_FILL = "#EDF3F8"
CUE = {
    "c": ("#DCE9F5", "#2C7FB8"),
    "d": ("#F7E1E7", "#B5405F"),
    "e": ("#EBE3F2", "#7A5195"),
}
MONO = {"family": "monospace", "fontsize": 6.4}

HEADER = "Case   Start    News   Change     End"
CITY_C = ("   1    57.8      +8     +9.4    67.2",
          "   2    57.8      -8     -1.8    56.0")
CITY_A = ("   1    45.4      -8     -6.5    38.9",
          "   2    45.4      +8    +12.2    57.6")
CITY_B = ("   1    35.0     -10     -5.6    29.4",
          "   2    35.0     +10     +9.3    44.3")


def box(ax, x, y, w, h, fill, edge, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=fill, edgecolor=edge, linewidth=0.8,
            linestyle=(0, (2.4, 1.8)) if dashed else "solid",
            mutation_aspect=0.5,
        )
    )


def chip(ax, x, y, text, edge, fill):
    box(ax, x, y, 0.78, 0.30, fill, edge)
    ax.text(x + 0.39, y + 0.15, text, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=INK)


def build(output_dir: Path) -> Path:
    # One unit of the y axis is deliberately larger than a text line so nothing
    # inside a block can collide: LINE is the line height every stacked text in
    # this figure steps by.
    fig, ax = plt.subplots(figsize=(6.6, 5.25), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.25)
    ax.axis("off")

    left = 1.05
    width = 8.85
    LINE = 0.30
    PAD = 0.16
    top = 11.02

    def block(y_top, height, fill, edge):
        box(ax, left, y_top - height, width, height, fill, edge)
        return y_top - PAD

    # Shared framing and the query: present in every condition.
    height = PAD * 2 + LINE * 3.4
    cursor = block(top, height, SHARED_FILL, SHARED_EDGE)
    ax.text(left + PAD, cursor,
            "Each row is a separate polling case, not a time series. Positive net news favors the "
            "candidate.\nWithin a city, its typical responsiveness is stable across cases, although "
            "end polls remain noisy.",
            ha="left", va="top", fontsize=6.6, color=MUTED, linespacing=1.5)
    ax.text(left + PAD, cursor - LINE * 2.3,
            "A new City C case begins with a poll of 53.9 and has net news +5. "
            "What poll do you predict?",
            ha="left", va="top", fontsize=6.9, color=INK, fontweight="bold")
    ax.text(left - 0.12, top - height / 2, "every\ncondition", ha="right", va="center",
            fontsize=6.6, color=MUTED, linespacing=1.35)
    top -= height + 0.30

    # City C cases.
    height = PAD * 2 + LINE * 5.0
    cursor = block(top, height, ADD_FILL, ADD_EDGE)
    ax.text(left + PAD, cursor, "CITY C  \u00b7  the city being forecast",
            ha="left", va="top", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(left + PAD, cursor - LINE * 1.25, HEADER, ha="left", va="top", color=MUTED, **MONO)
    for row_index, row in enumerate(CITY_C):
        ax.text(left + PAD, cursor - LINE * (2.2 + row_index), row,
                ha="left", va="top", color=INK, **MONO)
    ax.text(left + PAD, cursor - LINE * 4.3, "zero to four rows shown, by condition",
            ha="left", va="top", fontsize=6.3, color=MUTED, style="italic")
    chip(ax, 0.12, top - height / 2 - 0.15, "(a)", ADD_EDGE, ADD_FILL)
    top -= height + 0.30

    # Reference cities, side by side.
    half = (width - 0.22) / 2
    height = PAD * 2 + LINE * 5.0
    for offset, (name, rows) in enumerate((("CITY A", CITY_A), ("CITY B", CITY_B))):
        x = left + offset * (half + 0.22)
        box(ax, x, top - height, half, height, ADD_FILL, ADD_EDGE)
        cursor = top - PAD
        ax.text(x + PAD, cursor, name, ha="left", va="top",
                fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + PAD, cursor - LINE * 1.25, HEADER, ha="left", va="top", color=MUTED, **MONO)
        for row_index, row in enumerate(rows):
            ax.text(x + PAD, cursor - LINE * (2.2 + row_index), row,
                    ha="left", va="top", color=INK, **MONO)
        ax.text(x + PAD, cursor - LINE * 4.3, "four rows shown",
                ha="left", va="top", fontsize=6.3, color=MUTED, style="italic")
    chip(ax, 0.12, top - height / 2 - 0.15, "(b)", ADD_EDGE, ADD_FILL)
    top -= height + 0.42

    ax.text(left, top, "Context cue, appended to the City C block",
            ha="left", va="top", fontsize=7.0, fontweight="bold", color=INK)
    top -= LINE * 1.25

    cues = (
        ("(c)", "c",
         "Background: City C residents generally encounter campaign developments\n"
         "through national news coverage.",
         "names the strong-response city"),
        ("(d)", "d",
         "Background: City C residents generally encounter campaign developments\n"
         "through local news coverage.",
         "names the weak-response city instead"),
        ("(e)", "e",
         "Background label: KIV.          (City A is KIV, City B is ZOR)",
         "label has no fixed meaning; recoverable only from A and B"),
    )
    for tag, key, text, note in cues:
        fill, edge = CUE[key]
        height = PAD * 2 + LINE * 3.2
        cursor = block(top, height, fill, edge)
        ax.text(left + PAD, cursor, text, ha="left", va="top",
                fontsize=6.6, color=INK, linespacing=1.5)
        ax.text(left + PAD, cursor - LINE * 2.5, note, ha="left", va="top",
                fontsize=6.3, color=MUTED, style="italic")
        chip(ax, 0.12, top - height / 2 - 0.15, tag, edge, fill)
        top -= height + 0.22

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
