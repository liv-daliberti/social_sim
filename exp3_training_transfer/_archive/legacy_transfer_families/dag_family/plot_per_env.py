#!/usr/bin/env python3
"""plot_per_env.py — the per-environment view: pi, eps and rho for EACH held-out world separately.

The aggregate transfer figure pools the four held-out worlds, and that pooling hides two things:

  * pi is dominated by the delayed worlds (chain4 ~12, mediators ~9 poll pts), so the aggregate
    mostly reports how badly the model does where the news has barely reached the poll;
  * eps is NOT commensurable across worlds. The recoverable response gamma* spans 0.10-1.00 in
    `direct` but only 0.022-0.216 in `chain4`, so averaging |ghat - gamma*| across them averages
    quantities on different scales.

Each world therefore gets its own panel and its own y-scale, with its own references computed on the
identical episodes at that world's h* and over its informative prefixes only.

Rows: pi (poll pts) / eps (own gamma* scale) / rho.  Columns: the four held-out worlds.
Lines: family-trained, structureless control (both start from the shared untrained model at step 0).

    /usr/bin/python3 plot_per_env.py    # -> paper/figures/exp3_per_env.{png,pdf}
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transfer_report import GAMMAS, MARGIN_MIN, compute_refs, read_dump, summarize

REPORTS = HERE / "reports"
OUT = HERE.parent.parent / "paper" / "figures" / "exp3_per_env"

RUNS = {
    "29893776": ("family", 42), "29893778": ("family", 43), "29899438": ("family", 44),
    "29893777": ("control", 42), "29893779": ("control", 43), "29893781": ("control", 44),
}
# immediate worlds first, then the delayed ones -- the reader should meet `direct` (the Exp-2 world) first
ORDER = ["direct", "two_news_feedback", "mediators", "chain4"]
NICE = {"direct": "direct  (the Exp-2 world)", "two_news_feedback": "two-news feedback",
        "mediators": "mediators (fork–join)", "chain4": "chain-4 (triple-delayed)"}

TEAL, AMBER = "#0e9384", "#b45309"
INK, MUTED, GRID = "#171a1f", "#7c828a", "#e4e8ed"
REF_DARK, REF_MID = "#2b2f36", "#7c828a"
BOUNDS = {"pi": (0.0, None), "eps": (0.0, None), "rho": (-1.0, 1.0)}


def clamp(a, m):
    lo, hi = BOUNDS[m]
    return np.clip(np.asarray(a, float), -np.inf if lo is None else lo,
                   np.inf if hi is None else hi)


def evdir(jobid):
    for d in sorted(REPORTS.glob(f"pilot_*_j{jobid}")):
        e = list(d.glob("**/eval_results"))
        if e:
            return e[0]
    return None


def pool(cells, st, kk):
    out = {}
    for k in kk:
        for sd, c in cells.get((st, k), {}).items():
            out[(sd, k)] = c
    return out


def main():
    dumps = {}
    for j, (arm, seed) in RUNS.items():
        ev = evdir(j)
        if ev is None:
            continue
        ds = sorted((p for p in ev.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem))
        if ds:
            dumps[(arm, seed)] = {int(p.stem): p for p in ds}

    # paired seeds per step: a seed counts only where BOTH arms reached that step
    steps_arm = defaultdict(lambda: defaultdict(set))
    for (arm, sd), d in dumps.items():
        for s in d:
            steps_arm[s][arm].add(sd)
    paired = {s: sorted(a.get("family", set()) & a.get("control", set())) for s, a in steps_arm.items()}
    paired = {s: sd for s, sd in paired.items() if sd}
    steps = sorted(paired)
    print("paired seeds per step:", paired)

    base_path = dumps[("family", 42)][0]
    base_cells, _ = read_dump(base_path)
    ks = sorted({k for (_s, k) in base_cells})
    cities = defaultdict(set)
    for (st, _k), per in base_cells.items():
        cities[st].update(per.keys())
    print("computing per-world references ...")
    refs = compute_refs({s: sorted(v) for s, v in cities.items()}, tuple(ks))

    info = {}
    for st in ORDER:
        info[st] = [k for k in ks
                    if summarize(refs[("prior_blind", st, k)])["pi"]
                    - summarize(refs[("oracle", st, k)])["pi"] >= MARGIN_MIN] or list(ks)
    print("informative prefixes:", info)

    # model curves: {(arm, struct, metric)} -> {step: [per-seed values]}
    curves = defaultdict(lambda: defaultdict(list))
    cache = {}
    for s in steps:
        for sd in paired[s]:
            for arm in ("family", "control"):
                key = (arm, sd, s)
                if key not in cache:
                    cache[key] = read_dump(dumps[(arm, sd)][s])[0]
                cells = cache[key]
                for st in ORDER:
                    summ = summarize(pool(cells, st, info[st]))
                    for m in ("pi", "eps", "rho"):
                        if m in summ and np.isfinite(summ[m]):
                            curves[(arm, st, m)][s].append(summ[m])

    def ref_val(name, st, m):
        return summarize(pool({(st, k): refs[(name, st, k)] for k in info[st]}, st, info[st])).get(m, np.nan)

    def best_fixed(st):
        return min(summarize(pool({(st, k): refs[(f"fixed_{g:.2f}", st, k)] for k in info[st]},
                                  st, info[st]))["pi"] for g in GAMMAS)

    rows = [("Next-poll error  $\\pi$", "pi"), ("Recovery error  $\\varepsilon$", "eps"),
            ("Informativeness  $\\rho$", "rho")]
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 8.2), sharex=True)

    for r, (rowlab, m) in enumerate(rows):
        for c, st in enumerate(ORDER):
            ax = axes[r][c]
            # per-world references (own scale)
            if m == "pi":
                for nm, val, sty, col in (("oracle", ref_val("oracle", st, "pi"), ":", REF_DARK),
                                          ("prior blind", ref_val("prior_blind", st, "pi"), "--", REF_MID),
                                          ("best fixed", best_fixed(st), (0, (5, 2)), REF_DARK),
                                          ("persistence", ref_val("fixed_0.00", st, "pi"), "-.", REF_MID)):
                    if np.isfinite(val):
                        ax.axhline(val, color=col, ls=sty, lw=1.1, zorder=1)
            elif m == "eps":
                for val, sty, col in ((ref_val("oracle", st, "eps"), ":", REF_DARK),
                                      (ref_val("fixed_0.55", st, "eps"), "-.", REF_MID)):
                    if np.isfinite(val):
                        ax.axhline(val, color=col, ls=sty, lw=1.1, zorder=1)
            else:
                v = ref_val("oracle", st, "rho")
                if np.isfinite(v):
                    ax.axhline(v, color=REF_DARK, ls=":", lw=1.1, zorder=1)
                ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)

            for arm, col, lab in (("family", TEAL, "family-trained"),
                                  ("control", AMBER, "structureless control")):
                d = curves[(arm, st, m)]
                xs = sorted(d)
                if not xs:
                    continue
                mean = np.array([np.mean(d[x]) for x in xs])
                ax.plot(xs, mean, color=col, lw=1.9, marker="o", ms=3.2, label=lab, zorder=4)
                lo, hi = [], []
                for x in xs:
                    v = np.asarray(d[x], float)
                    if v.size >= 2:
                        sem = v.std(ddof=1) / np.sqrt(v.size)
                        t = {2: 12.706, 3: 4.303}.get(v.size, 2.0)
                        lo.append(v.mean() - t * sem); hi.append(v.mean() + t * sem)
                    else:
                        lo.append(np.nan); hi.append(np.nan)
                ax.fill_between(xs, clamp(lo, m), clamp(hi, m), color=col, alpha=0.14, lw=0, zorder=2)

            if r == 0:
                ax.set_title(NICE[st], fontsize=10, color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(rowlab, fontsize=9.5, color=INK)
            if r == 2:
                ax.set_xlabel("Dr. GRPO step", fontsize=8.5, color=MUTED)
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)

    # label the references once, on the first column of each row
    axes[0][0].annotate("persistence", xy=(0.02, ref_val("fixed_0.00", "direct", "pi")),
                        xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                        fontsize=7, color=REF_MID)
    axes[0][0].annotate("best fixed", xy=(0.02, best_fixed("direct")),
                        xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                        fontsize=7, color=REF_DARK)
    axes[0][0].annotate("oracle", xy=(0.02, ref_val("oracle", "direct", "pi")),
                        xycoords=("axes fraction", "data"), xytext=(0, -9), textcoords="offset points",
                        fontsize=7, color=REF_DARK)
    axes[1][0].annotate("always prior", xy=(0.02, ref_val("fixed_0.55", "direct", "eps")),
                        xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                        fontsize=7, color=REF_MID)
    axes[2][0].annotate("oracle", xy=(0.02, ref_val("oracle", "direct", "rho")),
                        xycoords=("axes fraction", "data"), xytext=(0, -9), textcoords="offset points",
                        fontsize=7, color=REF_DARK)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Per-environment transfer: the four held-out worlds, each on its own scale",
                 fontsize=12.5, color=INK, y=1.05)
    nmax = max(len(v) for v in paired.values())
    fig.text(0.5, -0.02,
             "Each world is probed at its own $h^\\star$ and scored over its informative prefixes only. "
             "$\\varepsilon$ is on each world's own response scale ($\\gamma^\\star$ spans $0.10$–$1.00$ "
             "in direct but $0.022$–$0.216$ in chain-4), so $\\varepsilon$ is NOT comparable across "
             f"columns. Both arms use the same paired seeds at each step (n $\\leq$ {nmax}); "
             "bands are 95% t-CI, clipped to each statistic's admissible range.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}.png + .pdf")


if __name__ == "__main__":
    main()
