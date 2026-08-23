#!/usr/bin/env python3
"""Experiment 2 — recovery learning curve, INFORMATION-PARITY variant.

Three panels, recovery ρ(ĝ_k, g) vs. weeks observed k. The LLM is read against two
INFORMATION-MATCHED forecasting agents — principled algorithms handed ONLY what the parity
prompt gives the model: the g∈[0.1,1] range, that opinion moves ≈g·news (zero-mean drift),
and the noise scales. They get NO advantage the LLM lacks — no mean-reversion φ and no known
o_0=50 baseline (they regress consecutive poll changes the LLM can see). They DO take the prompt at
its word that Δopinion = g·news plus zero-mean drift, so they fit through the origin (no intercept);
one change identifies g, so an estimate needs k>=2. At k=0,1 they answer with the range midpoint, so
recovery is 0 there (a real point, not a gap). They run the IDENTICAL 4-shock predict→slope pipeline:
  (A) behavioural SLOPE readout  vs. the Bayes forecaster (uniform-g prior, no φ)
  (B) behavioural SLOPE readout  vs. the OLS forecaster   (data-only regression)
  (C) directly STATED readout    — the model's reported points-per-news (the forecasters
      have no "stated" rate). Coherent models (GPT, 72B) match the slope panels; the 7B
      collapses.

The two forecasters are the fair bar for "told the structure, who recovers g?": they know
exactly what the prompt states, no more. Panel A also carries the full Bayes ORACLE as a
faint ceiling — it additionally knows mean-reversion φ=0.90, which the parity prompt never
states, so the oracle→forecaster gap is the headroom from that one undisclosed fact, not a
fair target. (The forecasters are read through predict→slope for an honest like-for-like
with the LLM; numerically this equals their direct fit, but the framing is now correct.)

Reads exp2_simulated_worlds/biased_news/data/recovery_curve_parity/results_*.jsonl.
Run from repo root:  python paper/generate_exp2_curve_parity.py
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REC = Path(__file__).resolve().parent.parent / "exp2_simulated_worlds" / "biased_news" / "data" / "recovery_curve_parity"
_OUT = Path(__file__).resolve().parent / "figures"
_OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Arial"],
    "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
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
C_BAYESF = "#4b5563"            # Bayes forecaster (dark grey, solid)
C_OLSF   = "#9ca3af"            # OLS forecaster (medium grey, dashed)
C_ORACLE = "#cbd5e1"           # full Bayes oracle, faint grey-dotted ceiling (knows baseline+φ)
_GGRID = [round(0.1 + 0.9 * i / 40, 4) for i in range(41)]   # g prior grid the forecaster is told
G_LO, G_HI = _GGRID[0], _GGRID[-1]                            # stated g-range the LLM is also given
# Noise on Δp under the forecaster's stated (no-φ) random-walk model: Var = σ_o² + 2σ_survey²
# = 1² + 2·2² = 9, i.e. σ=3 — derived entirely from the prompt's "about 1" / "about 2". The
# recovered ρ is ~invariant to it anyway (sig 1→8 moves ρ by ≤0.02), so it confers no hidden edge.
SIG_FORECASTER = math.sqrt(1.0 ** 2 + 2 * 2.0 ** 2)          # = 3.0
PHI = 0.90                                                   # world mean-reversion, known only to the oracles
ORDER = ["gpt-5.4", "claude-opus-4-8", "DeepSeek-V4-Pro", "qwen2.5-72b", "llama3.3-70b",
         "llama3.1-70b", "qwen2.5-32b", "qwen2.5-14b", "llama3.1-8b", "qwen2.5-7b"]
MIN_N = int(os.environ.get("RECOVERY_MIN_N", "20"))   # don't plot a model until it has enough cities


def _ols(xs, ys):
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    d = sum((x - mx) ** 2 for x in xs)
    return None if d < 1e-9 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / d


def _weekly_diffs(r, k):
    """Fair weekly responses — only what the LLM can see in the prompt table: consecutive poll
    changes p_t − p_{t-1} aligned with that week's news. NO o_0=50 anchor (the parity prompt
    never states the baseline), so the first usable change is week 2's and an estimate needs k>=2
    (with one poll and no baseline, g is unidentifiable — the same position a fair LLM is in)."""
    nws, pl = r["news"], r["polls"]
    return nws[1:k], [pl[i] - pl[i - 1] for i in range(1, k)]


_GPRIOR = sum(_GGRID) / len(_GGRID)     # range midpoint (0.55) — the no-information guess


def naive_ghat(r, k):
    """OLS forecaster's estimate: through-origin slope of Δp on news over the consecutive poll
    changes (g = Σ x·Δp / Σ x²), CLAMPED to the stated range g∈[0.1,1.0]. The parity prompt states
    the relationship is Δopinion = g·news plus a zero-mean drift, so the forecaster takes it at its
    word (no intercept); it does NOT know the o_0=50 baseline or φ. One change identifies g, so an
    estimate needs k>=2; with fewer (k=0,1) it returns the range midpoint (recovery ρ=0, not a gap)."""
    if k > len(r["polls"]):
        return None
    if k < 2:
        return _GPRIOR
    n, d = _weekly_diffs(r, k)
    sx2 = sum(x * x for x in n)
    if sx2 < 1e-9:
        return _GPRIOR
    return max(G_LO, min(G_HI, sum(x * y for x, y in zip(n, d)) / sx2))


def naive_bayes_ghat(r, k, sig=SIG_FORECASTER):
    """Bayes forecaster's estimate: through-origin posterior-mean g from Δp≈g·news over the
    consecutive poll changes, with a uniform g∈[0.1,1] prior and the stated noise scale — taking the
    prompt's zero-mean-drift statement at its word (no intercept), no baseline, no φ. The prior keeps
    g in range. One change identifies g (k>=2); with fewer the posterior is the prior (ρ=0)."""
    if k > len(r["polls"]):
        return None
    if k < 2:
        return _GPRIOR
    n, d = _weekly_diffs(r, k)
    if not n:
        return _GPRIOR
    lls = [-sum((dt - g * nt) ** 2 for dt, nt in zip(d, n)) / (2 * sig * sig) for g in _GGRID]
    m = max(lls); w = [math.exp(l - m) for l in lls]; t = sum(w) or 1.0
    return sum(_GGRID[i] * w[i] for i in range(len(_GGRID))) / t


def ols_oracle_ghat(r, k):
    """OLS ORACLE (frequentist peak): like the Bayes oracle it KNOWS the baseline o_0=50 and the
    mean-reversion φ, so it regresses the φ-corrected change y_t=(p_t-50)-φ(p_{t-1}-50) on news_t
    through the origin (the correction makes it zero-intercept by construction). Identifiable from
    k=1; no prior or optimal filtering, so it tracks just under the Bayes oracle. Clamped to range."""
    if k < 1 or k > len(r["polls"]):
        return None
    pl = r["polls"]
    prev = [50.0] + [pl[i] for i in range(k - 1)]
    y = [(pl[i] - 50.0) - PHI * (prev[i] - 50.0) for i in range(k)]
    n = r["news"][:k]
    sx2 = sum(x * x for x in n)
    return None if sx2 < 1e-9 else max(G_LO, min(G_HI, sum(x * yy for x, yy in zip(n, y)) / sx2))


def naive_baseline50_ghat(r, k):
    """MATCHED forecaster: assumes the o_0=50 baseline the LLM demonstrably uses (its traces back g
    out of (p_1-50)/news), but does NOT know mean-reversion φ. It is the OLS oracle's regression with
    φ:=1 (a random walk anchored at 50): y_t=(p_t-50)-(p_{t-1}-50) with a synthetic poll of 50 at t=0,
    regressed on news_t through the origin. This makes g identifiable from k=1 — unlike the
    baseline-DENIED forecaster (which scores ρ=0 there) — so it is the genuine like-for-like peer at
    the first observation: same assume-50 prior as the model, still no privileged φ. Clamped to range."""
    if k < 1 or k > len(r["polls"]):
        return None
    pl = r["polls"]
    prev = [50.0] + [pl[i] for i in range(k - 1)]
    y = [(pl[i] - 50.0) - 1.0 * (prev[i] - 50.0) for i in range(k)]   # φ:=1 → no mean-reversion
    n = r["news"][:k]
    sx2 = sum(x * x for x in n)
    return None if sx2 < 1e-9 else max(G_LO, min(G_HI, sum(x * yy for x, yy in zip(n, y)) / sx2))


_SHOCKS = [-10, -5, 5, 10]


def forecast_ghat(r, k, gfn):
    """Read a forecaster like the LLM: estimate g via gfn, FORECAST next week's poll at each
    test shock — clip(last_poll + g·shock, 0, 100) — and recover ĝ as the slope of those four
    forecasts. Identical predict→slope pipeline to the model, so the comparison is like-for-like."""
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
    if va < 1e-9 or vb < 1e-9:
        return 0.0
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / (va*vb)**.5


def _zero_floor(ax):
    """Faint ρ=0 no-inference baseline (a constant gain cannot correlate with g, so any
    fixed-gain shortcut sits here). A single reference line, not a good/bad colour wash."""
    ax.axhline(0.0, color="#cbd5e1", lw=1.0, ls=(0, (4, 3)), zorder=0)


def _shade_pass(ax, ks, boundary, legend=False):
    """Green above the fair forecaster boundary (the model infers MORE than a principled same-info
    fit), red below it (worse than a one-line forecaster). The forecaster curve is the boundary."""
    ax.fill_between(ks, boundary, 1.0, color="#22c55e", alpha=0.09, zorder=0,
                    label="Above forecaster" if legend else None)
    ax.fill_between(ks, -0.05, boundary, color="#ef4444", alpha=0.09, zorder=0,
                    label="Below forecaster" if legend else None)


def kcorr(recs, k, key):
    return corr([r["g"] for r in recs],
                [next((c.get(key) for c in r["curve"] if c["k"] == k), None) for r in recs])


_DATE_RE = re.compile(r"^results_(.+)_(\d{4}-\d{2}-\d{2})\.jsonl$")


def load_fair_records(rec_dir=_REC):
    """Per model, load ONLY its newest NON-EMPTY results file — NOT a merge across dates.

    The 2026-06-29 information-parity rerun must SUPERSEDE the pre-fair 2026-06-22/24 runs, not be
    backfilled by them: merging across dates silently refilled each open-weight model to n=250 with
    ~40% old-prompt cities while the frontier rows stayed 100% fair. Taking the newest non-empty file
    fixes that. "Non-empty" also lets models whose newest run is still empty (the deferred Llama-70B
    jobs, whose 06-29 files are 0 bytes) fall back to their last file that has data, preserving those
    rows from the old run until the rerun lands."""
    newest: dict[str, tuple[str, list]] = {}     # slug -> (date, records)
    for p in sorted(rec_dir.glob("results_*.jsonl")):
        mobj = _DATE_RE.match(p.name)
        if not mobj:
            continue
        slug, date = mobj.group(1), mobj.group(2)
        recs = []
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        if not recs:
            continue
        if slug not in newest or date > newest[slug][0]:
            newest[slug] = (date, recs)
    return {slug: recs for slug, (_d, recs) in newest.items()}


def main():
    # per model, use only its newest NON-EMPTY results file (the fair 06-29 rerun supersedes the
    # pre-fair 06-22/24 runs; deferred 70B fall back to old data) — see load_fair_records.
    by = load_fair_records()
    if not by:
        print("No parity curve records in", _REC); return

    # k-range spans ALL models (not just whichever is first in dict order — some frontier files
    # cap at k=5, the open-weight at k=10; using one model made max(ks) and the _k5 suffix flaky).
    ks = sorted({c["k"] for recs in by.values() for r in recs for c in r["curve"]})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 5.4), sharey=True)

    # everything is plotted from k=1 (k=0 has no history, so its readout is no-evidence noise)
    rks = [k for k in ks if k >= 1]
    # information-matched FORECASTERS (no φ), read through the same predict→slope as the LLM;
    # each slope panel also carries its ORACLE (knows baseline+φ) as a faint ceiling/peak
    ref = max(by.values(), key=len); g = [r["g"] for r in ref]
    bayesf  = [corr(g, [forecast_ghat(r, k, naive_bayes_ghat) for r in ref]) for k in rks]
    olsf    = [corr(g, [forecast_ghat(r, k, naive_ghat) for r in ref]) for k in rks]
    oracle  = [kcorr(ref, k, "bayes_g") for k in rks]
    olsorac = [corr(g, [ols_oracle_ghat(r, k) for r in ref]) for k in rks]

    # each forecaster owns a slope panel; green/red shading splits at that forecaster (the fair bar)
    _shade_pass(axA, rks, bayesf, legend=True)
    axA.plot(rks, oracle, color=C_ORACLE, lw=2.0, ls=":", marker="o", ms=3.5, zorder=4,
             label=r"Bayes oracle (knows $\phi$)")
    axA.plot(rks, bayesf, color=C_BAYESF, lw=2.6, marker="o", ms=5, zorder=5,
             label="Bayes forecaster")
    _shade_pass(axB, rks, olsf)
    axB.plot(rks, olsorac, color=C_ORACLE, lw=2.0, ls=":", marker="o", ms=3.5, zorder=4,
             label=r"OLS oracle (knows $\phi$)")
    axB.plot(rks, olsf, color=C_OLSF, lw=2.4, ls="--", marker="^", ms=5, zorder=5,
             label="OLS forecaster")
    _zero_floor(axC)

    # emphasise the sparse-data anchor: at k=1 the same-information forecaster is provably
    # unidentified (one poll, no baseline → g not estimable), so it sits at ρ=0 while the
    # agents already recover signal. A hollow marker + callout on each slope panel.
    for ax, fcol in ((axA, C_BAYESF), (axB, C_OLSF)):
        ax.plot([1], [0.0], marker="o", ms=11, mfc="white", mec=fcol, mew=2.0,
                zorder=7, clip_on=False)
        ax.annotate(r"$k{=}1$: forecaster can't" "\n" r"identify $g$  ($\rho{=}0$)",
                    xy=(1, 0.0), xytext=(1.45, 0.20), fontsize=9, color="#374151",
                    va="center", zorder=8,
                    arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.9))

    # model curves start at k=1: k=0 has no history (identical prompt), so the readout there is
    # pure sampling noise, not recovery; the references stay pinned at 0 at k=0
    mks = [k for k in ks if k >= 1]
    mh, ml = [], []
    for slug in ORDER + [m for m in by if m not in ORDER]:
        if slug not in by:
            continue
        recs = by[slug]
        if len(recs) < MIN_N:
            continue
        lab, col = STYLE.get(slug, (slug, "#111827"))
        slope_ys = [kcorr(recs, k, "lm_g") for k in mks]
        stated_ys = [kcorr(recs, k, "lm_g_stated") for k in mks]
        h, = axA.plot(mks, slope_ys, color=col, lw=2.2, marker="s", ms=4, zorder=3)
        axB.plot(mks, slope_ys, color=col, lw=2.2, marker="s", ms=4, zorder=3)
        axC.plot(mks, stated_ys, color=col, lw=2.2, marker="s", ms=4, zorder=3)
        mh.append(h); ml.append(f"{lab} (n={len(recs)})")

    xmax = int(os.environ.get("RECOVERY_XMAX", "0")) or max(ks)   # cap weeks shown (e.g. 5 to zoom early)
    xticks = [k for k in ks if 1 <= k <= xmax]                    # axis starts at k=1 (k=0 carries no evidence)
    for ax, title in ((axA, r"(A)  Slope $\hat{g}_k$ vs. Bayes forecaster"),
                      (axB, r"(B)  Slope $\hat{g}_k$ vs. OLS forecaster"),
                      (axC, r"(C)  Stated rate $\hat{g}_k$")):
        ax.set_xlabel("Weeks observed  $k$")
        ax.set_title(title, fontsize=12.5, pad=8)
        ax.set_xlim(0.8, xmax + 0.2); ax.set_ylim(-0.05, 1.0); ax.set_xticks(xticks)
    axA.set_ylabel(r"Recovery  $\rho(\hat{g}_k,\, g)$")
    axA.legend(loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#d1d5db")
    axB.legend(loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#d1d5db")
    fig.legend(mh, ml, loc="lower center", ncol=min(len(mh), 8), bbox_to_anchor=(0.5, -0.03),
               fontsize=10, framealpha=0.96, edgecolor="#d1d5db")

    fig.suptitle("Information-parity probe: told the generative structure, who recovers $g$?",
                 fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    suffix = f"_k{xmax}" if xmax != max(ks) else ""      # suffixed file when zoomed, so k=10 stays canonical
    for ext, dpi in (("pdf", 300), ("png", 600)):
        p = _OUT / f"exp2_recovery_curve_parity{suffix}.{ext}"; fig.savefig(p, dpi=dpi); print(f"  Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
