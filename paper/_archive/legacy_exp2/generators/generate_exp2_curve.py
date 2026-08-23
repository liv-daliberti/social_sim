#!/usr/bin/env python3
"""Experiment 2 — recovery learning curve.

Recovery ρ(ĝ_k, g) vs. number of observed weeks k, one line per forecaster.
The Bayes oracle (its posterior-mean ĝ_k) is the ceiling; each agent's curve
shows how fast it infers the latent g as evidence accumulates. At k=0 there is
no evidence, so recovery starts near 0.

Reads exp2_simulated_worlds/biased_news/data/recovery_curve/results_*.jsonl
(output of eval/run_recovery_curve.py). Run from repo root:
    python paper/generate_exp2_curve.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REC = Path(__file__).resolve().parent.parent / "exp2_simulated_worlds" / "biased_news" / "data" / "recovery_curve"
_OUT = Path(__file__).resolve().parent / "figures"
_OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Arial"],
    "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
    "savefig.bbox": "tight",
})
STYLE = {  # slug -> (label, color); Qwen purple, Llama blue, frontier orange (matches Exp 1), size by shade
    "qwen2.5-7b":   ("Qwen 7B",      "#c4b5fd"),
    "qwen2.5-14b":  ("Qwen 14B",     "#a78bfa"),
    "qwen2.5-32b":  ("Qwen 32B",     "#8b5cf6"),
    "qwen2.5-72b":  ("Qwen 72B",     "#6d28d9"),
    "llama3.1-8b":  ("Llama 3.1 8B", "#93c5fd"),
    "llama3.1-70b": ("Llama 3.1 70B","#3b82f6"),
    "llama3.3-70b": ("Llama 3.3 70B","#1d4ed8"),
    "gpt-5.4":         ("GPT-5.4",         "#ea580c"),
    "claude-opus-4-8": ("Claude Opus 4.8", "#dc2626"),
    "DeepSeek-V4-Pro": ("DeepSeek V4-Pro", "#0d9488"),
}
C_BAYES = "#6b7280"             # Bayes oracle ceiling (medium-dark grey)
C_NAIVE = "#9ca3af"             # OLS forecaster (light-medium grey, dashed)
C_NBAYES = "#374151"            # Bayes forecaster (dark grey, dash-dot)
_GGRID = [0.1 + 0.9 * i / 40 for i in range(41)]
_GPRIOR = sum(_GGRID) / len(_GGRID)     # range midpoint (0.55) — the no-information guess
ORDER = ["gpt-5.4", "claude-opus-4-8", "DeepSeek-V4-Pro", "qwen2.5-72b", "llama3.3-70b",
         "llama3.1-70b", "qwen2.5-32b", "qwen2.5-14b", "llama3.1-8b", "qwen2.5-7b"]
MIN_N = 20   # don't plot a model until it has enough cities to be meaningful


def _ols(xs, ys):
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    d = sum((x - mx) ** 2 for x in xs)
    return None if d < 1e-9 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / d


def _naive_diffs(r, k):
    """Weekly responses over the first k weeks: the consecutive poll changes p_t - p_{t-1} aligned
    with that week's news. No o_0=50 anchor (the baseline prompt never states the start), so the
    forecaster uses only what it can see — giving k-1 changes from k observed weeks."""
    nws, pl = r["news"], r["polls"]
    return nws[1:k], [pl[i] - pl[i - 1] for i in range(1, k)]


def naive_ghat(r, k):
    """Matched-knowledge baseline: free-intercept OLS slope of the weekly poll change on news —
    uses only the observed data (no priors, noise scales, baseline, or φ). The free intercept needs
    two changes, so an estimate needs k>=3; with fewer it returns the range midpoint (ρ=0)."""
    if k > len(r["polls"]):
        return None
    if k < 3:
        return _GPRIOR
    n, d = _naive_diffs(r, k)
    s = _ols(n, d)
    return _GPRIOR if s is None else s


def naive_bayes_ghat(r, k, sig=3.0):
    """Naive Bayes baseline: FREE-INTERCEPT posterior-mean g from Δp = α + g·news with a uniform
    g∈[0.1,1] prior — like the baseline OLS it does not assume the disclosed structure (no
    zero-intercept, baseline, or φ); the intercept α is profiled out by de-meaning x and Δp within
    the window (a flat prior on α). A free intercept needs two changes (k>=3); with fewer it returns
    the prior mean (no evidence → ρ=0)."""
    if k > len(r["polls"]):
        return None
    if k < 3:
        return sum(_GGRID) / len(_GGRID)
    n, d = _naive_diffs(r, k)
    mn = sum(n) / len(n); md = sum(d) / len(d)
    nc = [x - mn for x in n]; dc = [y - md for y in d]
    if sum(x * x for x in nc) < 1e-9:
        return sum(_GGRID) / len(_GGRID)
    lls = [-sum((dt - g * nt) ** 2 for dt, nt in zip(dc, nc)) / (2 * sig * sig) for g in _GGRID]
    m = max(lls); w = [math.exp(l - m) for l in lls]; t = sum(w) or 1.0
    return sum(_GGRID[i] * w[i] for i in range(len(_GGRID))) / t


_SHOCKS = [-10, -5, 5, 10]


def forecast_ghat(r, k, gfn):
    """Task-matched reference: estimate g from history via gfn, then FORECAST next week's
    poll at each test shock — clip(last_poll + g·shock, 0, 100) — and read g back as the slope
    of those forecasts: the identical predict-then-read-slope pipeline used for the LLM."""
    gest = gfn(r, k)
    if gest is None:
        return None
    L = r["polls"][k - 1] if k >= 1 else 50.0
    preds = [max(0.0, min(100.0, L + gest * s)) for s in _SHOCKS]
    return _ols(_SHOCKS, preds)


def corr(a, b):
    p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(p) < 3:
        return None
    a, b = zip(*p); ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x-ma)**2 for x in a); vb = sum((y-mb)**2 for y in b)
    if va < 1e-9 or vb < 1e-9:        # no variation (e.g. k=0) → no recovery
        return 0.0
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / (va*vb)**.5


def _shade_inference(ax, ybot=-0.05, legend=True):
    """Good/bad anchored on the no-inference floor: green = positive recovery (doing real
    inference, ρ>0), red = ρ≤0 (no better than a fixed-gain shortcut). Reference curves
    (oracle, regression baselines) overlay as graded markers, not as the good/bad boundary."""
    ax.axhspan(0.0, 1.0, color="#22c55e", alpha=0.08, zorder=0,
               label=r"Doing inference ($\rho>0$)" if legend else None)
    ax.axhspan(ybot, 0.0, color="#ef4444", alpha=0.10, zorder=0,
               label=r"No inference ($\rho\leq 0$)" if legend else None)


def main():
    by = defaultdict(list)
    for p in sorted(_REC.glob("results_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line); by[r["model"]].append(r)
                except Exception:
                    pass
    if not by:
        print("No curve records in", _REC); return

    ks = sorted({c["k"] for r in next(iter(by.values())) for c in r["curve"]})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # everything is plotted from k=1 (k=0 has no history, so its readout is no-evidence noise)
    rks = [k for k in ks if k >= 1]
    # Bayes oracle ceiling (posterior-mean ĝ_k); use the most-populated model's cities
    ref = max(by.values(), key=len)
    oracle = [corr([r["g"] for r in ref],
                   [next(c["bayes_g"] for c in r["curve"] if c["k"] == k) for r in ref]) for k in rks]
    # forecaster references (read through the same slope pipeline as the LLM) — graded markers
    naive = [corr([r["g"] for r in ref], [forecast_ghat(r, k, naive_ghat) for r in ref]) for k in rks]
    nbayes = [corr([r["g"] for r in ref], [forecast_ghat(r, k, naive_bayes_ghat) for r in ref]) for k in rks]
    # good/bad anchored on the no-inference floor (ρ=0)
    _shade_inference(ax)
    ax.plot(rks, oracle, color=C_BAYES, lw=2.8, marker="o", ms=6, zorder=4, label="Bayes oracle (ceiling)")
    ax.plot(rks, nbayes, color=C_NBAYES, lw=2.2, ls="-.", marker="D", ms=4, zorder=4,
            label="Bayes forecaster")
    ax.plot(rks, naive, color=C_NAIVE, lw=2.2, ls="--", marker="^", ms=5, zorder=4,
            label="OLS forecaster")

    # model curves start at k=1: k=0 has no history (identical prompt), so the readout there is
    # pure sampling noise, not recovery; the references stay pinned at 0 at k=0
    mks = [k for k in ks if k >= 1]
    for slug in ORDER + [m for m in by if m not in ORDER]:
        if slug not in by:
            continue
        recs = by[slug]
        if len(recs) < MIN_N:          # skip models too sparse to plot meaningfully
            continue
        lab, col = STYLE.get(slug, (slug, "#111827"))
        ys = [corr([r["g"] for r in recs],
                   [next(c["lm_g"] for c in r["curve"] if c["k"] == k) for r in recs]) for k in mks]
        ax.plot(mks, ys, color=col, lw=2.4, marker="s", ms=5, zorder=3, label=lab)

    xmax = int(os.environ.get("RECOVERY_XMAX", "0")) or max(ks)   # cap weeks shown (e.g. 5 to zoom early)
    ax.set_xlabel("Weeks observed  $k$")
    ax.set_ylabel(r"Recovery  $\rho(\hat{g}_k,\, g)$")
    ax.set_xlim(0.8, xmax + 0.2); ax.set_ylim(-0.05, 1.0)            # axis starts at k=1 (k=0 carries no evidence)
    ax.set_xticks([k for k in ks if 1 <= k <= xmax])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
              framealpha=0.96, edgecolor="#d1d5db", columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    suffix = f"_k{xmax}" if xmax != max(ks) else ""      # suffixed file when zoomed, so k=10 stays canonical
    for ext, dpi in (("pdf", 300), ("png", 600)):
        p = _OUT / f"exp2_recovery_curve{suffix}.{ext}"; fig.savefig(p, dpi=dpi); print(f"  Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
