#!/usr/bin/env python3
"""Render the dataset-card analytics figure (2x2) from fig_data.txt sections.

Panels: (1) markets created per month, (2) top categories, (3) volume
distribution (log-decade buckets), (4) per-creation-cohort coverage lines.
Colors/marks follow the dataviz reference palette (light surface).

Usage: render_card_figures.py FIG_DATA.txt OUT.png [--partial]
  --partial : render even if the COHORT section is missing (3 panels + note)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"


def parse_sections(path):
    sections, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith("==="):
            cur = line.strip("=")
            sections[cur] = []
        elif cur and "|" in line:
            sections[cur].append(line.split("|"))
    return sections


def fmt_count(v):
    if v >= 1_000_000:
        return f"{v/1e6:.1f}M"
    if v >= 1_000:
        return f"{v/1e3:.0f}K"
    return str(int(v))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def main():
    data_path, out_path = sys.argv[1], sys.argv[2]
    partial = "--partial" in sys.argv
    s = parse_sections(data_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # ── Panel 1: markets created per month ────────────────────────────────
    ax = axes[0][0]
    months = [(m, int(c)) for m, c in s["MONTHLY"]]
    xs = range(len(months))
    vals = [c for _, c in months]
    ax.bar(xs, vals, width=0.85, color=BLUE, linewidth=0)
    style_axes(ax)
    ax.set_title("Markets created per month", color=INK, fontsize=12, loc="left", pad=10)
    tick_idx = [i for i, (m, _) in enumerate(months) if m.endswith(("-01",)) and int(m[:4]) >= 2021]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([months[i][0][:4] for i in tick_idx])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_count(v)))
    peak = max(range(len(vals)), key=lambda i: vals[i])
    ax.annotate(
        f"{months[peak][0]}: {fmt_count(vals[peak])}",
        xy=(peak, vals[peak]), xytext=(peak - len(months) * 0.32, vals[peak] * 0.96),
        color=SECONDARY, fontsize=9,
    )

    # ── Panel 2: top categories ───────────────────────────────────────────
    ax = axes[0][1]
    cats = [(c, int(n)) for c, n in s["CATEGORY"]]
    top, other = cats[:8], sum(n for _, n in cats[8:])
    labels = [c for c, _ in top] + ["Other"]
    values = [n for _, n in top] + [other]
    ys = range(len(labels))[::-1]
    ax.barh(list(ys), values, height=0.72, color=BLUE, linewidth=0)
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.yaxis.grid(False)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.set_title("Markets by event category", color=INK, fontsize=12, loc="left", pad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_count(v)))
    for y, v in zip(ys, values):
        ax.text(v + max(values) * 0.01, y, fmt_count(v), va="center", color=SECONDARY, fontsize=9)
    ax.set_xlim(0, max(values) * 1.14)

    # ── Panel 3: volume distribution ──────────────────────────────────────
    ax = axes[1][0]
    vol = {int(b): int(n) for b, n in s["VOLUME"]}
    order = [-99] + sorted(k for k in vol if k != -99)
    labels3, values3 = [], []
    for b in order:
        if b == -99:
            labels3.append("$0")
        elif b <= 0:
            labels3.append("<$1" if b < 0 else "$1–10")
        else:
            labels3.append({1: "$10–100", 2: "$100–1K", 3: "$1K–10K", 4: "$10K–100K",
                            5: "$100K–1M", 6: "$1M–10M", 7: "$10M–100M", 8: "$100M–1B",
                            9: "$1B+"}.get(b, f"1e{b}"))
        values3.append(vol[b])
    # merge "<$1" duplicates (negative decades)
    merged = {}
    for l, v in zip(labels3, values3):
        merged[l] = merged.get(l, 0) + v
    labels3, values3 = list(merged.keys()), list(merged.values())
    ax.bar(range(len(labels3)), values3, width=0.8, color=BLUE, linewidth=0)
    style_axes(ax)
    ax.set_xticks(range(len(labels3)))
    ax.set_xticklabels(labels3, rotation=35, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_count(v)))
    ax.set_title("Lifetime trading volume per market (USDC)", color=INK, fontsize=12, loc="left", pad=10)

    # ── Panel 4: per-cohort coverage ──────────────────────────────────────
    ax = axes[1][1]
    if "COHORT" in s and s["COHORT"]:
        rows = [(m, int(t), int(c), int(r)) for m, t, c, r in s["COHORT"]]
        rows = [r for r in rows if r[1] >= 200]
        xs4 = range(len(rows))
        pc_c = [100 * c / t for _, t, c, _ in rows]
        pc_r = [100 * r / t for _, t, _, r in rows]
        ax.plot(xs4, pc_c, color=BLUE, linewidth=2, label="has real price data")
        ax.plot(xs4, pc_r, color=AQUA, linewidth=2, label="resolved (price-pinned)")
        leg = ax.legend(loc="lower left", frameon=False, fontsize=9.5)
        for t in leg.get_texts():
            t.set_color(SECONDARY)
        style_axes(ax)
        tick4 = [i for i, r in enumerate(rows) if r[0].endswith("-01") and int(r[0][:4]) >= 2021]
        ax.set_xticks(tick4)
        ax.set_xticklabels([rows[i][0][:4] for i in tick4])
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title("Coverage by market-creation cohort", color=INK, fontsize=12, loc="left", pad=10)
    elif partial:
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.5, "(coverage panel rendered after candle backfill)",
                ha="center", va="center", color=MUTED, fontsize=11)
    else:
        raise SystemExit("COHORT section missing; pass --partial to render without it")

    for ax_row in axes:
        for ax in ax_row:
            ax.title.set_fontweight("semibold")
    fig.suptitle("Polymarket Full Market Dataset — snapshot analytics",
                 color=INK, fontsize=14, fontweight="semibold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
