"""plot_curves_heatmap.py — learning-curve HEATMAP: models (rows) x training step (cols).

Reads reports/curves.csv (written by curve_from_eval_dag.py), averages seeds, and draws two
annotated heatmaps side by side: held-out forecast MAE and recovery MAE. Cells show the number and
are colored low=good (green) -> high=bad (red), normalized between the oracle ceiling and the
untrained level, so the left->right green-ing IS the learning. Reference rows (oracle / naive /
persistence) are appended for context.

    python3 curve_from_eval_dag.py            # refresh reports/curves.csv first
    python3 plot_curves_heatmap.py
"""
# !! SUPERSEDED (2026-07-09).  Reads reports/curves.csv, which is written by curve_from_eval_dag.py
# from the pre-h* scale-ladder runs: they trained the DEGENERATE one-step probe (a news-ignoring
# constant collects ~70% of the oracle reward) on the OLD catalog (explosive `two_news_feedback`,
# `skip` still in the training mix). Those dumps also carry `predicted_poll`, which multi-shock
# completions no longer emit, so this cannot be re-run against current dumps.
#
# There is no h* scale ladder yet -- only Qwen3-4B family+control. To revive this figure: port
# curve_from_eval_dag.py to the multi-shock schema (parse poll_A..poll_D, honour `horizon`) and run
# a new ladder. For the current result use plot_transfer_curves.py / plot_per_week.py.

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
CSV = HERE / "reports" / "curves.csv"
OUTDIR = Path("/n/fs/similarity/social_sim/paper/figures")

# reference rows shown on the heatmap: oracle (ceiling) + naive frequentist (stat bar) only
REF = {"oracle": (2.04, 0.14), "naive freq.": (4.14, 0.32)}  # 13-world held-out (4 structs)
# explicit parameter counts -> rows/lines always sorted smallest -> largest
_SIZE = {"Qwen3-4B": 4, "Qwen2.5-7B": 7, "Llama-3.1-8B": 8, "Qwen2.5-14B": 14,
         "Mistral-Small-24B": 24, "Qwen2.5-32B": 32, "Qwen2.5-72B": 72}
MODEL_ORDER = sorted(_SIZE, key=_SIZE.get)   # only those present in the CSV are drawn


def load():
    """(model,step) -> (mean fmae over seeds, mean rmae over seeds), using structure=='_mean'."""
    f, r = defaultdict(list), defaultdict(list)
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            if row["structure"] != "_mean":
                continue
            key = (row["model"], int(row["step"]))
            if row["fmae"]: f[key].append(float(row["fmae"]))
            if row["rmae"]: r[key].append(float(row["rmae"]))
    fm = {k: np.mean(v) for k, v in f.items()}
    rm = {k: np.mean(v) for k, v in r.items()}
    return fm, rm


def matrix(valmap, models, steps):
    M = np.full((len(models), len(steps)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(steps):
            if (m, s) in valmap:
                M[i, j] = valmap[(m, s)]
    return M


def draw(ax, M, models, steps, title, vlo, vhi, refvals, fmt="{:.2f}"):
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("#e8e8e8")
    norm = Normalize(vmin=vlo, vmax=vhi)
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("training step", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(models)):
        for j in range(len(steps)):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                # white text on dark (extreme) cells, black otherwise
                t = norm(v); dark = t < 0.18 or t > 0.82
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7.5, color="white" if dark else "black")
    # reference rows appended below with a separating gap
    y0 = len(models) + 0.4
    for name, rv in refvals.items():
        if rv is None:
            continue
        ax.text(-0.6, y0, name, ha="right", va="center", fontsize=8, style="italic", color="#333")
        ax.add_patch(plt.Rectangle((-0.5, y0 - 0.4), len(steps), 0.8, facecolor=cmap(norm(rv)),
                                   edgecolor="white", lw=0.5, clip_on=False))
        ax.text((len(steps) - 1) / 2, y0, fmt.format(rv), ha="center", va="center",
                fontsize=7.5, color="white" if (norm(rv) < 0.18 or norm(rv) > 0.82) else "black")
        y0 += 1
    ax.set_ylim(y0 - 0.5, -0.5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="lower = better")


def main():
    fm, rm = load()
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in fm)]
    steps = sorted({k[1] for k in fm if k[1] % 25 == 0})   # drop OAT's dup final eval (step 251)
    if not models or not steps:
        print("no curve data yet in", CSV); return
    Mf = matrix(fm, models, steps)
    Mr = matrix(rm, models, steps)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(9, 1.0 * len(steps) + 3), 0.6 * (len(models) + 4) + 2))
    draw(a1, Mf, models, steps, "Held-out forecast MAE (poll pts)", 2.0, 8.0,
         {k: v[0] for k, v in REF.items()})
    draw(a2, Mr, models, steps, r"Held-out recovery MAE $|\hat g-g|$", 0.13, 0.55,
         {k: v[1] for k, v in REF.items()})
    fig.suptitle("Forecasting training regularizes over-reading across models "
                 "(darker green = closer to oracle)", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"exp3_heatmap.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {OUTDIR}/exp3_heatmap.png + .pdf")
    print(f"models={models}\nsteps={steps}")


if __name__ == "__main__":
    main()
