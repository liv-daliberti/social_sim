#!/usr/bin/env python3
"""Exp 2 — the PRIMARY Experiment 2 figure: two error curves, one story told twice.

The claim is that the frontier agents recover the city's hidden news-responsiveness g from
sparse evidence AND turn that into near-optimal forecasts. So the hero figure carries the two
error metrics side by side, both read as observations k accumulate, both against the same
reference bracket (Bayes-oracle ceiling, overshooting naive-frequentist, always-prior floor,
fixed shortcuts):

  (A) LATENT-RECOVERY error   mean|g_hat - g|            (does it infer the hidden gain?)
  (B) NEXT-POLL forecast error mean|p_hat(s) - mu_k(s)|  (does that yield good forecasts?)

Why error and not rho(g_hat,g). rho is scale/variance-invariant, so an unregularized frequentist
that OVERSHOOTS on sparse data (g_hat=2 when g=0.3) still scores rho~1 by preserving city ORDER --
it rewards the exact failure a good prior avoids, and it is not the teaser's claim (the carnival
coin is per-instance CALIBRATION, not ordering). The rho / informativeness axis is demoted to the
main table (it keeps a rho column) and an appendix scatter (generate_exp2_informativeness.py).

Panel B target. Under held-out shock s at prefix k the ground-truth expected next poll is
  mu_k(s) = clip(50 + phi*(o_k - 50) + g*s, 0, 100),   o_k = opinion_traj[k],
regenerated from each city's seed (episode_id city_%05d -> seed). The four counterfactual shocks
never realize, so the target is the DGP's conditional mean, not a noisy draw: it is the
Bayes-optimal point forecast, and even the oracle (which must filter o_k and g) sits above 0.
This is pure re-analysis of the stored per-shock predictions -- NO re-runs.

Reuses generate_exp2_curve_parity.py (loader, STYLE, corr) + the biased_news engine (mu, oracle,
shortcuts). Run from repo root:
  python paper/generate_exp2_skill.py
Outputs paper/figures/exp2_skill.{pdf,png}
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp2_parity", _HERE / "generate_exp2_curve_parity.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

# biased_news engine — ground-truth next-poll target, Bayes oracle, fixed shortcuts
_BN = _HERE.parent / "exp2_simulated_worlds" / "biased_news"
sys.path.insert(0, str(_BN))
from engine.news_response import generate_episode, compute_bayes, predict_fixed_gain, PHI, OPINION_INIT

_OUT = _HERE / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 3, 4, 5]
MIN_N = 30
_FRONTIER = ("gpt-5.4", "claude-opus-4-8", "DeepSeek-V4-Pro")
_SHOCKS = [-10, -5, 5, 10]           # the stored per-shock predictions align to this order
_PRIOR = 0.55                        # range midpoint — the no-information guess

# extend the shared Qwen/Llama/frontier palette with the Mistral (magenta) family so all
# local models in the table roster also appear in the figure + colour key. (Gemma excluded.)
STYLE = {**_m.STYLE,
    "mistral-small-24b": ("Mistral-Small 24B", "#be185d"),
    "mistral-nemo-12b":  ("Mistral-Nemo 12B",  "#f472b6"),
}
# size-ordered (largest -> smallest) so the bottom colour key reads by scale
ORDER = ["gpt-5.4", "claude-opus-4-8", "DeepSeek-V4-Pro",
         "qwen2.5-72b", "llama3.3-70b", "llama3.1-70b", "qwen2.5-32b",
         "mistral-small-24b", "qwen2.5-14b", "mistral-nemo-12b",
         "llama3.1-8b", "qwen2.5-7b"]


# ── (A) latent-recovery ĝ readouts ────────────────────────────────────────────
def g_model(r, k):
    return next((c.get("lm_g") for c in r["curve"] if c["k"] == k), None)


def g_bayes(r, k):
    return next((c.get("bayes_g") for c in r["curve"] if c["k"] == k), None)


def g_freq_unclamped(r, k):
    """A naive frequentist: through-origin OLS of poll changes on news, NO prior, NO clamp to the
    stated range. This is what overshoots on sparse data. For k>=2 it regresses the consecutive poll
    changes (baseline-denied). At k=1 it has ONE poll, no consecutive change to regress, and no
    baseline of its own -- it does NOT know opinion starts at 50, so a single poll cannot identify the
    gain. Unregularized and with no prior to shrink toward, it has nothing to discount with and reads
    the week's news at face value (g_hat=1): its worst point (~0.46), included so the statistical bar
    exists from the first week; consecutive changes from k>=2 then let it actually estimate the gain."""
    if k == 1:
        return 1.0
    n, d = _m._weekly_diffs(r, k)
    sx2 = sum(x * x for x in n)
    return None if sx2 < 1e-9 else sum(x * y for x, y in zip(n, d)) / sx2


def _mae(recs, k, gfn):
    es = [abs(gfn(r, k) - r["g"]) for r in recs if gfn(r, k) is not None]
    return statistics.mean(es) if es else None


def _rho(recs, k, gfn):
    return _m.corr([r["g"] for r in recs], [gfn(r, k) for r in recs])


# ── (B) next-poll forecast readouts ───────────────────────────────────────────
_TRAJ: dict[int, list] = {}                    # seed -> opinion_traj (regenerated once)


def _traj(r):
    sid = int(r["episode_id"].split("_")[1])
    if sid not in _TRAJ:
        ep = generate_episode(seed=sid, T=10)
        assert abs(ep["g"] - r["g"]) < 1e-6, (sid, ep["g"], r["g"])   # seed↔g alignment guard
        _TRAJ[sid] = ep["opinion_traj"]
    return _TRAJ[sid]


def _mu(r, k, s):
    """Ground-truth expected next poll under shock s at prefix k (clip of the DGP mean)."""
    o_k = _traj(r)[k]                          # opinion_traj[0]=50, [k]=latent after k weeks
    return max(0.0, min(100.0, OPINION_INIT + PHI * (o_k - OPINION_INIT) + r["g"] * s))


def _preds(r, k):
    return next((c.get("preds") for c in r["curve"] if c["k"] == k), None)


def _last(r, k):
    return r["polls"][k - 1] if k >= 1 else 50.0


def p_model(r, k):
    pr = _preds(r, k)
    if not pr:
        return None
    es = [abs(p - _mu(r, k, s)) for p, s in zip(pr, _SHOCKS) if p is not None]
    return statistics.mean(es) if es else None


def p_oracle(r, k):
    return statistics.mean(
        abs(compute_bayes(r["news"][:k], r["polls"][:k], s)["predicted_poll"] - _mu(r, k, s))
        for s in _SHOCKS)


def p_fixed(r, k, g):
    L = _last(r, k)
    return statistics.mean(abs(predict_fixed_gain(L, s, g) - _mu(r, k, s)) for s in _SHOCKS)


def p_freq(r, k):
    g = g_freq_unclamped(r, k)                  # unclamped through-origin slope (overshoots), k>=2
    if g is None:
        return None
    L = _last(r, k)
    return statistics.mean(abs(max(0.0, min(100.0, L + g * s)) - _mu(r, k, s)) for s in _SHOCKS)


def p_prior(r, k):
    L = _last(r, k)
    return statistics.mean(abs(max(0.0, min(100.0, L + _PRIOR * s)) - _mu(r, k, s)) for s in _SHOCKS)


def _pmae(recs, k, pfn):
    es = [pfn(r, k) for r in recs]
    es = [e for e in es if e is not None]
    return statistics.mean(es) if es else None


# ── shared panel painter ──────────────────────────────────────────────────────
def _draw_curves(ax, ref, by, *, oracle, freq, ignore, face, model_fn):
    """Paint one error panel: faint shortcut floor, then the headline trio (naive-frequentist,
    the three frontier LLMs, Bayes oracle). Open-weight models are omitted here to keep the hero
    figure clean — their full roster lives in the appendix scatter/tables."""
    # green "achievable good" band: between the naive-frequentist bar (top) and the Bayes-oracle
    # floor (bottom). Below the oracle is unreachable, so the zone stops there -- a forecaster in
    # the band beats the fair statistical bar without (impossibly) beating the oracle.
    ax.fill_between(KS, freq, oracle, color="#16a34a", alpha=0.16, lw=0, zorder=0)

    ax.plot(KS, face, color="#d1d5db", lw=1.4, zorder=2, label="face value ($g{=}1$)")
    ax.plot(KS, ignore, color="#e5e7eb", lw=1.4, zorder=2, label="ignore news ($g{=}0$)")

    ax.plot(KS, freq, color="#b45309", lw=2.4, ls="--", marker="^", ms=7, zorder=5,
            label="naive frequentist (overshoots)")
    for slug in _FRONTIER:
        recs = by.get(slug)
        if not recs:
            continue
        lab, col = STYLE.get(slug, (slug, "#111827"))
        ax.plot(KS, [model_fn(recs, k) for k in KS], color=col, lw=2.6, marker="s", ms=5,
                zorder=6, label=f"{lab} (n={len(recs)})")
    ax.plot(KS, oracle, color="#1e293b", lw=2.4, ls=":", marker="o", ms=4, zorder=7,
            label="Bayes oracle (ceiling)")


# ── figure ──────────────────────────────────────────────────────────────────
def main():
    by = _m.load_fair_records()
    if not by:
        print("no data"); return
    ref = max(by.values(), key=len)
    print(f"reference city set n={len(ref)}")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Arial"],
        "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
    })
    # side-by-side (A | B): both panels share one legend, drawn once beneath the pair.
    # squat aspect — wide panels, roughly half the earlier height.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 3.9))

    # ---- Panel A: latent-recovery error, mean|g_hat - g| vs k ----------------
    _draw_curves(
        axA, ref, by,
        oracle=[_mae(ref, k, g_bayes) for k in KS],
        freq=[_mae(ref, k, g_freq_unclamped) for k in KS],
        ignore=[_mae(ref, k, lambda r, k: 0.0) for k in KS],
        face=[_mae(ref, k, lambda r, k: 1.0) for k in KS],
        model_fn=lambda recs, k: _mae(recs, k, g_model),
    )
    axA.set_xlabel("Weeks observed  $k$")
    axA.set_ylabel("Recovery error\n" r"$\overline{|\hat g - g|}$")
    axA.set_title("(A)  Recovers the hidden gain $g$", fontsize=12.5)
    axA.set_xticks(KS); axA.set_ylim(0.06, 0.56)

    # ---- Panel B: next-poll forecast error, mean|p_hat - mu| vs k ------------
    _draw_curves(
        axB, ref, by,
        oracle=[_pmae(ref, k, p_oracle) for k in KS],
        freq=[_pmae(ref, k, p_freq) for k in KS],
        ignore=[_pmae(ref, k, lambda r, k: p_fixed(r, k, 0.0)) for k in KS],
        face=[_pmae(ref, k, lambda r, k: p_fixed(r, k, 1.0)) for k in KS],
        model_fn=lambda recs, k: _pmae(recs, k, p_model),
    )
    axB.set_xlabel("Weeks observed  $k$")
    axB.set_ylabel("Next-poll error\n" r"$\overline{|\hat p_{t+1}-p_{t+1}|}$")
    axB.set_title("(B)  Forecasts the next poll near-optimally", fontsize=12.5)
    axB.set_xticks(KS); axB.set_ylim(1.0, 4.5)

    # one shared legend beneath both panels (entries are identical in A and B).
    # the overshoot/regularization reading is carried by the figure caption, not in-panel text.
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.02),
               fontsize=9.5, framealpha=0.95, edgecolor="#d1d5db",
               columnspacing=1.4, handlelength=2.0)
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    for ext, dpi in (("pdf", 300), ("png", 160)):
        p = _OUT / f"exp2_skill.{ext}"; fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)

    # numeric dump so the reframe is auditable without opening the figure
    hdr = "  ".join(f"k={k}" for k in KS)
    print("\n(A) latent-recovery MAE |g_hat-g| (lower=better):")
    print(f"{'':26s} {hdr}")
    def lineA(name, fn, recs=ref):
        print(f"{name:26s} " + "  ".join(f"{(fn(recs,k) if fn(recs,k) is not None else float('nan')):.3f}" for k in KS))
    lineA("Bayes oracle", lambda r,k: _mae(r,k,g_bayes))
    lineA("naive frequentist", lambda r,k: _mae(r,k,g_freq_unclamped))
    lineA("always-prior 0.55", lambda r,k: _mae(r,k, lambda r,k: _PRIOR))
    for slug in ("claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro"):
        if slug in by:
            lineA(slug, lambda r,k,recs=by[slug]: _mae(recs,k,g_model))

    print("\n(B) next-poll MAE |p_hat - mu| (poll pts, lower=better):")
    print(f"{'':26s} {hdr}")
    def lineB(name, fn, recs=ref):
        print(f"{name:26s} " + "  ".join(f"{(fn(recs,k) if fn(recs,k) is not None else float('nan')):.2f}" for k in KS))
    lineB("Bayes oracle", lambda r,k: _pmae(r,k,p_oracle))
    lineB("naive frequentist", lambda r,k: _pmae(r,k,p_freq))
    lineB("always-prior 0.55", lambda r,k: _pmae(r,k,p_prior))
    lineB("face value g=1", lambda r,k: _pmae(r,k, lambda r,k: p_fixed(r,k,1.0)))
    lineB("ignore news g=0", lambda r,k: _pmae(r,k, lambda r,k: p_fixed(r,k,0.0)))
    for slug in ("claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro"):
        if slug in by:
            lineB(slug, lambda r,k,recs=by[slug]: _pmae(recs,k,p_model))


if __name__ == "__main__":
    main()
