#!/usr/bin/env python3
"""Exp 2 (appendix) — the calibration-vs-informativeness scatter, demoted from the hero figure.

The main-text Figure now carries the two ERROR metrics (latent-recovery |g_hat-g| and next-poll
|p_hat-p|). This appendix scatter keeps the secondary INFORMATIVENESS ($\\rho$) view: at the sparse
anchor k=2, x = rho(g_hat, g) (higher = better ordering), y = mean|g_hat-g| (lower = better
calibration). Every forecaster lands where the thesis predicts -- shortcuts in the poor corners,
the naive frequentist buys rho at the cost of calibration, always-prior is calibrated but blind,
and only the frontier agents sit in the good corner (informative AND calibrated).

Reuses generate_exp2_skill.py's readouts. Run from repo root:
  python paper/generate_exp2_informativeness.py
Outputs paper/figures/exp2_informativeness.{pdf,png}
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp2_skill", _HERE / "generate_exp2_skill.py")
_s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s)

_OUT = _HERE / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

K_SCATTER = 2
MIN_N = _s.MIN_N
_FRONTIER = _s._FRONTIER
STYLE = _s.STYLE
_PRIOR = _s._PRIOR


def main():
    by = _s._m.load_fair_records()
    if not by:
        print("no data"); return
    ref = max(by.values(), key=len)
    k = K_SCATTER

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Arial"],
        "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
    })
    fig, axB = plt.subplots(1, 1, figsize=(7.2, 5.6))

    pts = [
        ("Bayes oracle", _s._rho(ref, k, _s.g_bayes), _s._mae(ref, k, _s.g_bayes), "#1e293b", "o"),
        ("naive frequentist", _s._rho(ref, k, _s.g_freq_unclamped), _s._mae(ref, k, _s.g_freq_unclamped), "#b45309", "^"),
        ("always-prior 0.55", 0.0, _s._mae(ref, k, lambda r, k: _PRIOR), "#94a3b8", "D"),
        ("face value ĝ=1", 0.0, _s._mae(ref, k, lambda r, k: 1.0), "#cbd5e1", "s"),
        ("ignore ĝ=0", 0.0, _s._mae(ref, k, lambda r, k: 0.0), "#cbd5e1", "s"),
    ]
    for lab, x, y, col, mk in pts:
        if x is None or y is None:
            continue
        axB.scatter([x], [y], s=110, color=col, marker=mk, zorder=5, edgecolor="white", lw=1)
        axB.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8.5, color="#334155")
    for slug in by:
        recs = by[slug]
        if len(recs) < MIN_N or slug not in STYLE:
            continue
        lab, col = STYLE[slug]
        frontier = slug in _FRONTIER
        x, y = _s._rho(recs, k, _s.g_model), _s._mae(recs, k, _s.g_model)
        if x is None or y is None:
            continue
        axB.scatter([x], [y], s=150 if frontier else 55, color=col, marker="*" if frontier else "o",
                    zorder=6 if frontier else 4, edgecolor="white", lw=1, alpha=1 if frontier else 0.75)
        if frontier:
            axB.annotate(lab, (x, y), textcoords="offset points", xytext=(7, -10), fontsize=9,
                         color=col, fontweight="bold")

    axB.axhline(_s._mae(ref, k, _s.g_bayes), color="#1e293b", ls=":", lw=1, zorder=1)
    axB.axhspan(0.0, _s._mae(ref, k, _s.g_bayes) + 0.04, xmin=0.60, color="#22c55e", alpha=0.06, zorder=0)
    axB.set_xlabel(r"Informativeness  $\rho(\hat g, g)$   (higher = better $\rightarrow$)")
    axB.set_ylabel(r"Calibration error  $\overline{|\hat g - g|}$   ($\downarrow$ lower = better)")
    axB.set_title(f"Calibration vs. informativeness  (k={k})", fontsize=12.5)
    axB.set_xlim(-0.06, 0.80); axB.set_ylim(0.0, 0.58)
    axB.text(0.97, 0.03, "GOOD corner:\ninformative + calibrated", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=9.5, color="#16a34a", fontweight="bold")

    fig.tight_layout()
    for ext, dpi in (("pdf", 300), ("png", 160)):
        p = _OUT / f"exp2_informativeness.{ext}"; fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
