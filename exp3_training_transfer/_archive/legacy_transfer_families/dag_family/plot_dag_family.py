"""plot_dag_family.py — the Experiment-3 world family, drawn straight from each structure's A/B/C
matrices (so the picture can't drift from the actual worlds). Paper-ready: clean display names,
cycle-safe layering, curved feedback edges, a muted print/CVD-safe palette.

Nodes: observed news x_j (top), latent opinions o_i (middle, layered by shortest path from the news),
the observed poll p (bottom). Edges: B (news->opinion), A off-diagonal (opinion->opinion), C
(opinion->poll); A diagonal = the phi self-loop (carry-over). The single hidden gain g is on the one
dashed teal edge, boxed. Training worlds fill the top block; held-out worlds sit below a divider.

    python3 plot_dag_family.py
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse, FancyBboxPatch, Arc, Rectangle
from matplotlib.lines import Line2D, Line2D as _L
from matplotlib.patches import Patch
from catalog import TRAIN, TEST

# ── palette: grayscale + one teal accent + one amber accent (CVD-safe, prints cleanly) ─────────────
OBS_FC, NODE_EC = "#d6dbe2", "#2b2f36"      # observed node fill / all node outlines
LAT_FC = "#ffffff"                          # latent node fill
EDGE = "#4a4f57"                            # structural edges
PHI = "#9aa0a8"                             # phi self-loop (recessive)
TEAL, TEAL_FILL = "#0e9384", "#d5efeb"      # hidden gain g
INK, MUTED = "#171a1f", "#7c828a"           # title ink / subtitle
TRAIN_BG, TRAIN_EC = "#f6f7f9", "#e4e8ed"   # training card
TEST_BG, TEST_EC, TEST_INK = "#fff6e9", "#d98a1f", "#b5730f"   # held-out card + ink

_NICE = {"hidden_phi": "Hidden φ", "direct": "Direct", "lagged": "Lagged",
         "inhibitory": "Inhibitory", "mediators": "Mediators", "echo": "Echo", "skip": "Skip",
         "chain2": "Chain-2", "chain3": "Chain-3", "chain4": "Chain-4",
         "chain2_hidden_phi": "Chain-2 (hidden φ)"}


def nice(name: str) -> str:
    if name in _NICE:
        return _NICE[name]
    s = name
    for a, b in [("two_news", "Two-news"), ("three_news", "Three-news"),
                 ("four_news", "Four-news"), ("five_news", "Five-news")]:
        s = s.replace(a, b)
    for a, b in [("_2stage", " 2-stage"), ("_split", " split"), ("_confounder", " confounder"),
                 ("_chain", " chain"), ("_feedback", " feedback"), ("_tree", " tree")]:
        s = s.replace(a, b)
    s = s.replace("_", " ")
    return s[:1].upper() + s[1:]


def latent_levels(A, B):
    """Level = 1 + shortest hops from a news-driven opinion (BFS, so cycles stay finite)."""
    d, m = A.shape[0], B.shape[1]
    lvl = [None] * d
    cur = [i for i in range(d) if any(B[i, j] != 0 for j in range(m))]
    for i in cur:
        lvl[i] = 1
    L = 1
    while cur:
        L += 1; nxt = []
        for j in cur:
            for i in range(d):                       # edge o_j -> o_i is A[i, j]
                if i != j and A[i, j] != 0 and lvl[i] is None:
                    lvl[i] = L; nxt.append(i)
        cur = nxt
    return [l or 1 for l in lvl]


def positions(st):
    A, B = st.A, st.B
    m, d = st.m, st.d
    pos = {}
    xs = list(np.linspace(-1.85, 1.85, m)) if m > 1 else [0.0]
    for j in range(m):
        pos[("x", j)] = (xs[j], 3.6)
    lvl = latent_levels(A, B); L = max(lvl)
    byl = defaultdict(list)
    for i in range(d):
        byl[lvl[i]].append(i)
    gap = min(0.98, 1.95 / max(L - 1, 1))          # comfortable per-level gap; centered on the panel
    top = 1.95 + (L - 1) * gap / 2
    for l, idxs in byl.items():
        xx = list(np.linspace(-1.0, 1.0, len(idxs))) if len(idxs) > 1 else [0.0]
        y = top - (l - 1) * gap
        for k, i in enumerate(idxs):
            pos[("o", i)] = (float(xx[k]), y)
    pos[("p", 0)] = (0.0, 0.3)
    return pos


def arrow(ax, a, b, color=EDGE, dashed=False, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color,
                                 ls="--" if dashed else "-", shrinkA=12, shrinkB=12, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def selfloop(ax, xy):
    x, y = xy
    ax.add_patch(Arc((x - 0.34, y + 0.10), 0.46, 0.46, angle=0, theta1=95, theta2=350, color=PHI, lw=1.1, zorder=1))
    ax.annotate("", (x - 0.19, y + 0.29), (x - 0.36, y + 0.29),
                arrowprops=dict(arrowstyle="-|>", color=PHI, lw=1.1))
    ax.text(x - 0.62, y + 0.06, r"$\phi$", fontsize=7.5, color=PHI, ha="center", va="center", zorder=2)


def node(ax, xy, kind, label):
    x, y = xy
    if kind == "o":
        ax.add_patch(Ellipse((x, y), 0.56, 0.44, fc=LAT_FC, ec=NODE_EC, lw=1.3, zorder=3))
    else:
        ax.add_patch(Circle((x, y), 0.23, fc=OBS_FC, ec=NODE_EC, lw=1.3, zorder=3))
    ax.text(x, y, label, fontsize=8.5, ha="center", va="center", zorder=4, color=INK)


def draw(ax, st, is_test=False):
    A, B, C = st.A, st.B, st.C
    m, d = st.m, st.d
    pos = positions(st)
    kind, hi, hj = st.hidden
    for i in range(d):
        for j in range(m):
            if B[i, j] != 0:
                h = kind == "B" and (i, j) == (hi, hj)
                arrow(ax, pos[("x", j)], pos[("o", i)], color=TEAL if h else EDGE, dashed=h)
        for j in range(d):
            if i != j and A[i, j] != 0:
                h = kind == "A" and (i, j) == (hi, hj)
                yi, yj = pos[("o", i)][1], pos[("o", j)][1]
                if A[j, i] != 0:               # mutual pair -> curve the two apart
                    rad = 0.26 if i < j else -0.26
                elif yi > yj:                  # forward / downward
                    rad = 0.0
                else:                          # back / upward edge -> curve out
                    rad = 0.3
                arrow(ax, pos[("o", j)], pos[("o", i)], color=TEAL if h else EDGE, dashed=h, rad=rad)
        if A[i, i] != 0:
            selfloop(ax, pos[("o", i)])
        if C[0, i] != 0:
            arrow(ax, pos[("o", i)], pos[("p", 0)])
    # hidden gain g, boxed near the dashed edge's midpoint
    src = pos[("x", hj)] if kind == "B" else pos[("o", hj)]
    dst = pos[("o", hi)]
    gx, gy = (src[0] + dst[0]) / 2 + 0.4, (src[1] + dst[1]) / 2
    ax.add_patch(FancyBboxPatch((gx - 0.14, gy - 0.14), 0.28, 0.28, boxstyle="round,pad=0.02",
                                fc=TEAL_FILL, ec=TEAL, lw=1.2, ls="--", zorder=5))
    ax.text(gx, gy, "$g$", fontsize=8.5, color=TEAL, ha="center", va="center", zorder=6, fontweight="bold")
    for j in range(m):
        node(ax, pos[("x", j)], "x", f"$x_{j+1}$" if m > 1 else "$x$")
    for i in range(d):
        node(ax, pos[("o", i)], "o", f"$o_{i+1}$" if d > 1 else "$o$")
    node(ax, pos[("p", 0)], "p", "$p$")
    ax.set_title(nice(st.name), fontsize=10.5, color=(TEST_INK if is_test else INK),
                 fontweight="bold", pad=5)
    ax.set_xlim(-2.25, 2.45); ax.set_ylim(-0.1, 4.05); ax.axis("off"); ax.set_aspect("equal")
    ax.patch.set_visible(False)


def main():
    from matplotlib.gridspec import GridSpec
    cols = 5; sub = 2 * cols                                   # 2x sub-columns so short rows can center
    ntr, nte = len(TRAIN), len(TEST)
    trows = -(-ntr // cols)
    fig = plt.figure(figsize=(3.3 * cols, 3.25 * (trows + 1) + 0.8))
    gs = GridSpec(trows + 2, sub, figure=fig, height_ratios=[1] * trows + [0.42, 1],
                  hspace=0.42, wspace=0.5, left=0.015, right=0.985, top=0.945, bottom=0.065)
    panels = []

    def place(rowidx, structs, is_test):
        loff = (sub - 2 * len(structs)) // 2                   # center a short row in the sub-grid
        for c, st in enumerate(structs):
            ax = fig.add_subplot(gs[rowidx, loff + 2 * c: loff + 2 * c + 2])
            draw(ax, st, is_test); panels.append((ax, is_test))

    for r in range(trows):
        place(r, TRAIN[r * cols:(r + 1) * cols], False)        # full rows fill; the short last row centers
    place(trows + 1, TEST, True)

    fig.suptitle(f"The Experiment 3 world family: {ntr} training structures, {nte} held-out for transfer",
                 fontsize=16, fontweight="bold", color=INK, y=0.985)

    leg = [Patch(fc=TRAIN_BG, ec=TRAIN_EC, label=f"training world  ({ntr})"),
           Patch(fc=TEST_BG, ec=TEST_EC, lw=1.6, label=f"held-out world  ({nte})"),
           Circle((0, 0), 1, fc=OBS_FC, ec=NODE_EC, label="observed: news $x$, poll $p$"),
           Ellipse((0, 0), 1, 1, fc=LAT_FC, ec=NODE_EC, label="latent opinion $o$"),
           _L([0], [0], color=TEAL, ls="--", marker="s", mfc=TEAL_FILL, mec=TEAL,
              label="hidden gain $g$ (inferred from polls)"),
           _L([0], [0], color=PHI, label=r"$\phi$: carry-over / mean-reversion")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.002), handletextpad=0.6, columnspacing=1.8)

    # cards behind panels (positions final immediately with explicit GridSpec margins)
    for ax, is_test in panels:
        p = ax.get_position(); px, pb, pt = 0.006, 0.006, 0.001
        fig.add_artist(Rectangle((p.x0 - px, p.y0 - pb), p.width + 2 * px, p.height + pb + pt,
                                 transform=fig.transFigure, zorder=0,
                                 fc=(TEST_BG if is_test else TRAIN_BG),
                                 ec=(TEST_EC if is_test else TRAIN_EC),
                                 lw=(2.4 if is_test else 0.9)))
    train_bot = min(a.get_position().y0 for a, t in panels if not t)
    test_top = max(a.get_position().y1 for a, t in panels if t)
    ymid = (train_bot + test_top) / 2
    fig.add_artist(_L([0.05, 0.95], [ymid, ymid], transform=fig.transFigure,
                      color=TEST_EC, lw=1.3, ls=(0, (6, 4)), zorder=1))
    fig.text(0.5, ymid, "  held-out worlds · never trained on, only tested  ", ha="center", va="center",
             fontsize=11.5, color=TEST_INK, style="italic", fontweight="bold", zorder=2,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))

    out = "/n/fs/similarity/social_sim/paper/figures/exp3_dag_family_all"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}.png + .pdf  ({ntr} train + {nte} held-out)")


if __name__ == "__main__":
    main()
