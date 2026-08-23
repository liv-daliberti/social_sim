#!/usr/bin/env python3
"""Experiment 2 — recovery + cost summary (clean bar charts).

Two plain bar panels per forecaster (no scatter / per-g lines):
  Left  — Recovery: correlation of the recovered gain ĝ with the true g (higher
          = better latent inference).
  Right — Mean systematic next-poll error |pred − E[poll]| (lower = better),
          averaged over the multi-shock probe.

ĝ for an LLM is the slope of predicted Δpoll on the test shock
(eval/run_recovery_probe.py); for the oracle it is the Kalman posterior mean.
Reads exp2_simulated_worlds/biased_news/data/recovery_probe/results_*.jsonl.
Run from repo root:  python paper/generate_exp2_recovery.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BN  = Path(__file__).resolve().parent.parent / "exp2_simulated_worlds" / "biased_news"
_REC = _BN / "data" / "recovery_probe"
_OUT = Path(__file__).resolve().parent / "figures"
_OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(_BN))
from engine.news_response import generate_episode, compute_bayes, predict_fixed_gain

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Arial"],
    "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#eef2f7", "grid.linewidth": 0.8, "savefig.bbox": "tight",
})
# forecaster order (best recovery first) → (label, color)
C_BAYES="#f59e0b"; C_GPT="#ea580c"; C_72="#059669"; C_7="#7c3aed"; C_FACE="#dc2626"; C_IGN="#2563eb"


def _seed(eid): return int(eid.split("_")[-1])


def corr(a, b):
    p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    a, b = zip(*p); ma, mb = statistics.mean(a), statistics.mean(b)
    d = (sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)) ** .5
    return (sum((x-ma)*(y-mb) for x, y in zip(a, b)) / d) if d else 0.0


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
        print("No probe records in", _REC); return

    # oracle + shortcut error/recovery from a shared episode set (any model's cities)
    ref = by.get("qwen2.5-72b") or next(iter(by.values()))
    bayes_err, face_err, ign_err = [], [], []
    for r in ref:
        ep = generate_episode(seed=_seed(r["episode_id"])); oT = ep["opinion_traj"][-1]; g = r["g"]; last = r["last_poll"]
        for s in r["shocks"]:
            gold = max(0.0, min(100.0, oT + g*s))
            bayes_err.append(abs(compute_bayes(r["news"], r["polls"], s)["predicted_poll"] - gold))
            face_err.append(abs(predict_fixed_gain(last, s, 1.0) - gold))
            ign_err.append(abs(predict_fixed_gain(last, s, 0.0) - gold))
    bayes_rho = corr([r["g"] for r in ref], [r["bayes_g_hat"] for r in ref])

    # per-LLM recovery + systematic error
    def llm_stats(recs):
        g = [r["g"] for r in recs]; sl = [r["lm_g_slope"] for r in recs]
        err = []
        for r in recs:
            ep = generate_episode(seed=_seed(r["episode_id"])); oT = ep["opinion_traj"][-1]
            for s, pr in zip(r["shocks"], r["preds"]):
                if pr is not None:
                    err.append(abs(pr - max(0.0, min(100.0, oT + r["g"]*s))))
        return corr(g, sl), statistics.mean(err)

    rows = [("Bayes\noracle", C_BAYES, bayes_rho, statistics.mean(bayes_err))]
    for key, lab, col in [("gpt-5.4", "GPT-5.4", C_GPT), ("qwen2.5-72b", "Qwen\n72B", C_72),
                          ("qwen2.5-7b", "Qwen\n7B", C_7)]:
        if key in by:
            rho, err = llm_stats(by[key]); rows.append((lab, col, rho, err))
    rows += [("Face\nvalue", C_FACE, 0.0, statistics.mean(face_err)),
             ("Ignore\nnews", C_IGN, 0.0, statistics.mean(ign_err))]

    labels = [r[0] for r in rows]; cols = [r[1] for r in rows]
    rhos = [r[2] for r in rows]; errs = [r[3] for r in rows]
    x = range(len(rows))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axL.bar(x, rhos, color=cols, edgecolor="white", linewidth=0.6, width=0.7)
    for i, v in zip(x, rhos):
        axL.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    axL.set_ylim(0, 1.0); axL.set_ylabel(r"Recovery  $\rho(\hat{g},\,g)$   (higher better)")
    axL.set_title("Did it recover the latent?", fontsize=13, fontweight="bold")

    axR.bar(x, errs, color=cols, edgecolor="white", linewidth=0.6, width=0.7)
    for i, v in zip(x, errs):
        axR.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=11)
    axR.set_ylim(0, max(errs)*1.18); axR.set_ylabel("Mean systematic next-poll error   (lower better)")
    axR.set_title("...and what it costs", fontsize=13, fontweight="bold")

    for ax in (axL, axR):
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.tick_params(length=4)
    fig.tight_layout()
    for ext, dpi in (("pdf", 300), ("png", 600)):
        p = _OUT / f"exp2_recovery.{ext}"; fig.savefig(p, dpi=dpi); print(f"  Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
