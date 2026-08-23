#!/usr/bin/env python3
"""plot_per_week.py — the evidence-accumulation view: does the model sharpen as polls arrive?

The Experiment-3 analogue of Experiment 2's sparse-evidence curve. At each prefix length k (weeks of
history observed) we read the same three axes on the held-out worlds, for the untrained base, the
family-trained model, and the structureless control:

    pi   next-poll forecast error, probed at the world's h* (poll pts)
    eps  recovery error |ghat - g|
    rho  informativeness, Spearman rho(ghat, g)

against oracle / prior-blind / best-fixed-gamma references recomputed at each k on the identical
episodes.

SCOPE.  We pool only the held-out worlds whose probe response at h* IS the gain itself
(gamma* == g exactly: `direct`, `two_news_feedback`).  Two reasons, both about not averaging
incommensurable things: (i) in `mediators` and `chain4` the response is an affine image of g, so their
eps lives on a different scale; (ii) those worlds are unidentifiable at small k (the pulse has not
reached the poll), so including them would change *what is being averaged* as k grows and a flat curve
could be mistaken for a flat model.  Fixing the world set keeps the x-axis honest.

    /usr/bin/python3 plot_per_week.py     # -> paper/figures/exp3_per_week.{png,pdf}

(The pre-2026-07-08 one-step version is kept as plot_per_week.py.legacy_onestep.bak; it parses
`predicted_poll`, which multi-shock completions no longer emit.)
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
from lg_dag import G_LO, G_HI, first_response_horizon, response_at_horizon
from catalog import BY_NAME, TEST_NAMES
from transfer_report import GAMMAS, MARGIN_MIN, compute_refs, read_dump, summarize

REPORTS = HERE / "reports"
OUT = HERE.parent.parent / "paper" / "figures" / "exp3_per_week"

# A Student-t interval is unbounded, but these statistics are not: pi and eps are non-negative and
# rho is a correlation. At n=2 (t*=12.7) an unclamped band runs to rho=+1.5, which is impossible and
# unreadable. Clamp each band to its metric's admissible range.
BOUNDS = {"pi": (0.0, None), "eps": (0.0, None), "rho": (-1.0, 1.0)}


def _clamp(vals, metric):
    lo, hi = BOUNDS[metric]
    out = np.asarray(vals, float)
    if lo is not None:
        out = np.maximum(out, lo)
    if hi is not None:
        out = np.minimum(out, hi)
    return out

# the h* generation (same runs as plot_transfer_curves.py)
RUNS = {
    "29893776": ("family", 42), "29893778": ("family", 43), "29899438": ("family", 44),
    "29893777": ("control", 42), "29893779": ("control", 43), "29893781": ("control", 44),
}

# series colours validated with the dataviz six-checks (worst adjacent CVD dE 47.6; both >= 3:1)
TEAL, AMBER = "#0e9384", "#b45309"
INK, MUTED, GRID = "#171a1f", "#7c828a", "#e4e8ed"
REF_DARK, REF_MID = "#2b2f36", "#7c828a"


def find_eval_dir(jobid):
    for d in sorted(REPORTS.glob(f"pilot_*_j{jobid}")):
        ev = list(d.glob("**/eval_results"))
        if ev:
            return ev[0]
    return None


def gain_is_identity(name: str) -> bool:
    """Is the probe response at h* the gain itself, rather than an affine image of it?"""
    st = BY_NAME[name]
    h = first_response_horizon(st)
    return (abs(response_at_horizon(st, G_LO, h) - G_LO) < 1e-9 and
            abs(response_at_horizon(st, G_HI, h) - G_HI) < 1e-9)


def pool(cells: dict, structs, k):
    """Merge {(struct,k): {seed: cell}} across structs at one k, keyed uniquely by (struct, seed)."""
    out = {}
    for st in structs:
        for seed, cell in cells.get((st, k), {}).items():
            out[(st, seed)] = cell
    return out


def main():
    # dumps per (arm, seed), keyed by step
    steps = defaultdict(dict)                      # arm -> seed -> {step: path}
    for jobid, (arm, seed) in RUNS.items():
        ev = find_eval_dir(jobid)
        if ev is None:
            continue
        for d in ev.glob("*.json"):
            if d.stem.isdigit():
                steps[arm][seed] = {**steps[arm].get(seed, {}), int(d.stem): d}
    if not steps.get("family") or not steps.get("control"):
        print("need both arms"); return

    # Pin BOTH arms to the same checkpoint. Taking each run's own last dump would silently average a
    # still-untrained seed (whose latest dump is step 0) into the "trained" curve, flattering the
    # other arm and inflating the band. Use the largest step present in >=2 seeds of each arm.
    # ... and pair the seeds: a seed only counts if BOTH arms reached this step, so the two curves
    # are computed over the same seeds and any difference is the arm, not the sample.
    def paired_seeds(step):
        return sorted(s for s in steps["family"] if step in steps["family"][s]
                      and s in steps["control"] and step in steps["control"][s])
    candidates = sorted({s for a in steps for ss in steps[a].values() for s in ss if s > 0}, reverse=True)
    STEP = next((s for s in candidates if len(paired_seeds(s)) >= 2), None)
    if STEP is None:
        print("no common trained checkpoint reached by >=2 paired seeds yet"); return
    SEEDS = paired_seeds(STEP)
    print(f"comparing both arms at step {STEP} over paired seeds {SEEDS}")

    runs = defaultdict(list)                       # arm -> [(seed, dump_at_STEP)]
    for arm in ("family", "control"):
        for seed in SEEDS:
            runs[arm].append((seed, steps[arm][seed][STEP]))

    base_path = next(ss[0] for ss in steps["family"].values() if 0 in ss)
    base_cells, _ = read_dump(base_path)                  # step 0: same base model for every run
    ks = sorted({k for (_st, k) in base_cells})
    cities = defaultdict(set)
    for (st, _k), per in base_cells.items():
        cities[st].update(per.keys())
    cities = {st: sorted(s) for st, s in cities.items()}

    print("computing references at every prefix ...")
    refs = compute_refs(cities, tuple(ks))

    def identifiable(st, k):
        o = summarize(refs[("oracle", st, k)])["pi"]
        b = summarize(refs[("prior_blind", st, k)])["pi"]
        return (b - o) >= MARGIN_MIN

    structs = [n for n in TEST_NAMES if gain_is_identity(n) and all(identifiable(n, k) for k in ks)]
    dropped = [n for n in TEST_NAMES if n not in structs]
    print(f"pooling {structs}")
    print(f"dropped {dropped}  (response is an affine image of g, and/or unidentifiable at small k)")

    def arm_curves(arm):
        per_k = {k: defaultdict(list) for k in ks}
        for _seed, dN in runs.get(arm, []):
            cells, _ = read_dump(dN)
            for k in ks:
                s = summarize(pool(cells, structs, k))
                for m in ("pi", "eps", "rho"):
                    if m in s and np.isfinite(s[m]):
                        per_k[k][m].append(s[m])
        return per_k

    base_k = {k: summarize(pool(base_cells, structs, k)) for k in ks}
    fam_k, ctl_k = arm_curves("family"), arm_curves("control")

    def ref_curve(name, metric):
        vals = []
        for k in ks:
            s = summarize(pool({(st, k): refs[(name, st, k)] for st in structs}, structs, k))
            vals.append(s.get(metric, np.nan))
        return np.asarray(vals, float)

    def best_fixed_pi():
        # strongest single constant response, chosen per (world, k) -- the sharpest blind bar
        return np.asarray([float(np.mean([min(summarize(refs[(f"fixed_{g:.2f}", st, k)])["pi"]
                                              for g in GAMMAS) for st in structs])) for k in ks])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
    panels = [("Next-poll error  $\\pi$  (poll pts)", "pi"),
              ("Recovery error  $\\varepsilon$", "eps"),
              ("Informativeness  $\\rho(\\hat g, g)$", "rho")]

    for ax, (title, m) in zip(axes, panels):
        ax.plot(ks, ref_curve("oracle", m), color=REF_DARK, ls=":", lw=1.2, zorder=1)
        ax.plot(ks, ref_curve("prior_blind", m), color=REF_MID, ls="--", lw=1.2, zorder=1)
        if m == "pi":
            ax.plot(ks, best_fixed_pi(), color=REF_DARK, ls=(0, (5, 2)), lw=1.2, zorder=1)
        elif m == "eps":
            ax.plot(ks, ref_curve("fixed_0.55", m), color=REF_MID, ls="-.", lw=1.2, zorder=1)

        # the untrained model is the starting point, not a third category: recessive dashed grey
        ax.plot(ks, [base_k[k].get(m, np.nan) for k in ks], color=MUTED, ls="--", lw=1.6,
                marker="o", ms=3.0, label="untrained (step 0)", zorder=3)

        for per_k, color, lab in ((fam_k, TEAL, "family-trained"),
                                  (ctl_k, AMBER, "structureless control")):
            mean = np.array([np.mean(per_k[k][m]) if per_k[k][m] else np.nan for k in ks])
            ax.plot(ks, mean, color=color, lw=2, marker="o", ms=3.5, label=lab, zorder=4)
            lo, hi = [], []
            for k in ks:
                v = np.asarray(per_k[k][m], float)
                if v.size >= 2:
                    sem = v.std(ddof=1) / np.sqrt(v.size)
                    t = {2: 12.706, 3: 4.303}.get(v.size, 2.0)
                    lo.append(v.mean() - t * sem); hi.append(v.mean() + t * sem)
                else:
                    lo.append(np.nan); hi.append(np.nan)
            ax.fill_between(ks, _clamp(lo, m), _clamp(hi, m), color=color, alpha=0.15, lw=0, zorder=2)

        # direct right-edge labels: references are context, so they stay out of the legend.
        # nearly-coincident references (prior-blind vs always-prior on eps) get staggered offsets.
        lbl = [("oracle", ref_curve("oracle", m)[-1], REF_DARK, 3),
               ("prior blind", ref_curve("prior_blind", m)[-1], REF_MID, 3)]
        if m == "pi":
            lbl.append(("best fixed", best_fixed_pi()[-1], REF_DARK, 3))
        elif m == "eps":
            lbl.append(("always prior", ref_curve("fixed_0.55", m)[-1], REF_MID, 3))
        lbl.sort(key=lambda t: t[1])
        prev = None
        for i, (name, v, c, dy) in enumerate(lbl):
            if not np.isfinite(v):
                continue
            if prev is not None and abs(v - prev) < 0.13 * (ax.get_ylim()[1] - ax.get_ylim()[0]):
                dy = -9                                   # push the lower of a colliding pair down
            prev = v
            ax.annotate(name, xy=(1.0, v), xycoords=("axes fraction", "data"),
                        xytext=(-3, dy), textcoords="offset points", ha="right",
                        fontsize=7.5, color=c)

        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlabel("weeks of history observed  $k$", fontsize=9, color=MUTED)
        ax.set_xticks(ks)
        ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=3,
               fontsize=8.5, frameon=False)
    fig.suptitle("Evidence accumulation on the held-out worlds: does the model sharpen as polls arrive?",
                 fontsize=11.5, color=INK, y=1.14)
    fig.text(0.5, -0.07,
             "Pooled over the held-out worlds whose probe response at $h^\\star$ is the gain itself "
             f"({', '.join(structs)}); the rest are dropped so the averaged quantity does not change "
             f"with $k$. Both arms read at Dr. GRPO step {STEP} over the same seeds {SEEDS}; "
             "bands are 95% t-CI.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}.png + .pdf")
    for k in ks:
        f = {m: (np.mean(fam_k[k][m]) if fam_k[k][m] else np.nan) for m in ("pi", "eps", "rho")}
        c = {m: (np.mean(ctl_k[k][m]) if ctl_k[k][m] else np.nan) for m in ("pi", "eps", "rho")}
        print(f"  k={k}:  base pi={base_k[k]['pi']:5.2f} eps={base_k[k].get('eps', float('nan')):.3f}"
              f" | fam pi={f['pi']:5.2f} eps={f['eps']:.3f} rho={f['rho']:+.3f}"
              f" | ctrl pi={c['pi']:5.2f} eps={c['eps']:.3f} rho={c['rho']:+.3f}")


if __name__ == "__main__":
    main()
