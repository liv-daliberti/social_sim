"""plot_curves_lines.py — learning curves as LINES with a 3-seed confidence band.

Reads reports/curves.csv (curve_from_eval_dag.py). One line per model; where a model has >=2 seeds
(Qwen3-4B: 42/43/44) the line is the seed mean and a shaded 95% CI (Student-t) band shows the spread.
Two panels: held-out forecast MAE and recovery MAE, with oracle / naive / persistence reference lines.

    python3 curve_from_eval_dag.py     # refresh reports/curves.csv first
    python3 plot_curves_lines.py
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

HERE = Path(__file__).resolve().parent
CSV = HERE / "reports" / "curves.csv"
OUTDIR = Path("/n/fs/similarity/social_sim/paper/figures")

REF = {"oracle": (2.04, 0.14), "naive freq.": (4.14, 0.32), "persistence": (None, None)}  # 13-world held-out (4 structs)
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}   # two-sided 95% Student-t by n (df=n-1)

# fixed colors + explicit param counts -> plotted/legended smallest -> largest
_SIZE = {"Qwen3-4B": 4, "Qwen2.5-7B": 7, "Llama-3.1-8B": 8, "Qwen2.5-14B": 14,
         "Mistral-Small-24B": 24, "Qwen2.5-32B": 32, "Qwen2.5-72B": 72}
_COLOR = {
    "Qwen3-4B": "#d62728", "Qwen2.5-7B": "#ff7f0e", "Llama-3.1-8B": "#1f77b4",
    "Qwen2.5-14B": "#9467bd", "Mistral-Small-24B": "#8c564b",
    "Qwen2.5-32B": "#17becf", "Qwen2.5-72B": "#7f7f7f",
}
STYLE = [(m, _COLOR[m]) for m in sorted(_SIZE, key=_SIZE.get)]   # smallest -> largest


def load():
    # model -> metric -> step -> [seed values]
    d = defaultdict(lambda: {"fmae": defaultdict(list), "rmae": defaultdict(list)})
    for r in csv.DictReader(open(CSV)):
        if r["structure"] != "_mean" or int(r["step"]) % 25 != 0:
            continue
        s = int(r["step"])
        if r["fmae"]: d[r["model"]]["fmae"][s].append(float(r["fmae"]))
        if r["rmae"]: d[r["model"]]["rmae"][s].append(float(r["rmae"]))
    return d


def series(stepmap):
    """steps, means, ci_halfwidth (95% t) — sorted by step."""
    steps = sorted(stepmap)
    means = np.array([np.mean(stepmap[s]) for s in steps])
    ci = []
    for s in steps:
        v = stepmap[s]; n = len(v)
        ci.append(T95.get(n, 1.96) * np.std(v, ddof=1) / np.sqrt(n) if n >= 2 else 0.0)
    return np.array(steps), means, np.array(ci)


def draw(ax, d, metric, title, refs, ylab):
    for name, color in STYLE:
        if name not in d or not d[name][metric]:
            continue
        steps, mean, ci = series(d[name][metric])
        multiseed = any(len(d[name][metric][s]) >= 2 for s in steps)
        lbl = name + (" (3 seeds, 95% CI)" if multiseed else "")
        if len(steps) == 1:
            ax.plot(steps, mean, "o", color=color, label=lbl, ms=7)
        else:
            ax.plot(steps, mean, "-", color=color, label=lbl, lw=2, marker="o", ms=3)
            if np.any(ci > 0):
                ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.25, lw=0)
    for rn, (rc, style) in {"oracle": ("#2e7d32", ":"), "naive freq.": ("#777", "--"),
                             "persistence": ("#b71c1c", "-.")}.items():
        rv = refs.get(rn)
        if rv is not None:
            ax.axhline(rv, color=rc, ls=style, lw=1.2, alpha=0.8)
            ax.text(ax.get_xlim()[1], rv, f" {rn}", va="center", ha="left", fontsize=7.5, color=rc)
    ax.set_xlabel("training step (eval checkpoint every 25)", fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 301, 25))                      # one tick per eval checkpoint
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.25)


def main():
    d = load()
    if not d:
        print("no curve data in", CSV); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    draw(a1, d, "fmae", "Held-out forecast MAE (poll pts)",
         {k: v[0] for k, v in REF.items()}, r"$\overline{|\hat p_{t+1}-p_{t+1}|}$")
    draw(a2, d, "rmae", r"Held-out recovery MAE $|\hat g-g|$",
         {k: v[1] for k, v in REF.items()}, r"$\overline{|\hat g-g|}$")
    a1.margins(x=0.14); a2.margins(x=0.14)
    a1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle("Forecasting training regularizes over-reading across model families and scales",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"exp3_curves.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {OUTDIR}/exp3_curves.png + .pdf")
    for name, _ in STYLE:
        if name in d and d[name]["fmae"]:
            steps, mean, ci = series(d[name]["fmae"])
            tag = f" (95% CI up to +/-{ci.max():.2f})" if np.any(ci > 0) else ""
            print(f"  {name:18s} steps {steps[0]}..{steps[-1]}  {mean[0]:.2f} -> {mean[-1]:.2f}{tag}")


if __name__ == "__main__":
    main()
