#!/usr/bin/env python3
"""plot_transfer_curves.py — the Experiment-3 headline figure: family vs structureless-control
learning curves on the held-out worlds, read the Exp-2 way (pi / eps / rho together).

Three panels vs training step, each with the family arm (teal, 3-seed 95% t-CI band) and the
control arm (amber, same):

    pi   next-poll forecast error on held-out worlds (poll pts; informative (structure,k) cells)
    eps  recovery error |ghat - g| (gain-recovering held-out worlds)
    rho  informativeness, Spearman rho(ghat, g) — the axis the Jul-7 runs collapsed on

Reference lines are computed on the IDENTICAL eval episodes at the SAME probe horizon as the dumps
(the `horizon` field of each reference; defaults to 1 for pre-h* dumps), so the figure is valid for
both generations of runs. The verdict the figure encodes: family-trained shows transfer-as-world-
modeling only where its pi curve dives below the blind constants while rho stays up — and separates
from the control curve, which can only learn format/level-tracking/news-suppression.

    python3 plot_transfer_curves.py            # -> paper/figures/exp3_transfer_curves.{png,pdf}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lg_dag import SHOCKS, filter_all, forecast_at_horizon, generate_episode, true_output_at_horizon
from catalog import BY_NAME, TEST_NAMES
from prompt import parse_forecasts
from transfer_report import GAMMAS, MARGIN_MIN, PRIOR_MEAN, summarize, _pool


def read_dump(path: Path):
    """Schema-tolerant version of transfer_report.read_dump: accepts pre-h* dumps (no `horizon`
    field) because THIS script computes its references at each dump's own probe horizon, so the
    comparison stays internally consistent for both generations of runs."""
    records = json.loads(Path(path).read_text())
    out = defaultdict(dict)
    n_miss = n_tot = 0
    for rec in records:
        ref = json.loads(rec["reference"])
        n_tot += 1
        text = rec["output"]
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        preds = parse_forecasts(text, n=len(ref["targets"]))
        if preds is None:
            n_miss += 1
            continue
        preds = np.asarray(preds)
        shocks = np.asarray(ref["shocks"], float)
        out[(ref["structure"], ref["k"])][ref["seed"]] = {
            "pi": float(np.mean(np.abs(preds - np.asarray(ref["targets"], float)))),
            "ghat": float(np.cov(shocks, preds, bias=True)[0, 1] / np.var(shocks)),
            "g": float(ref["g"]),
            "true_gain": ref.get("true_gain"),
            "recover": bool(ref.get("recover", False)),
        }
    return out, (n_miss / max(n_tot, 1))

REPORTS = HERE / "reports"
OUT = HERE.parent.parent / "paper" / "figures" / "exp3_transfer_curves"

# jobid -> (arm, seed). Update when runs are relaunched.
# 2026-07-09: the h* generation. The previous entries (29892467-472) were trained against the
# degenerate one-step probe and their dumps carry no `horizon` field -- transfer_report.py's guard
# rejects them, and mixing them with h* references would silently produce a wrong figure.
RUNS = {
    "29893776": ("family", 42), "29893778": ("family", 43), "29899438": ("family", 44),
    "29893777": ("control", 42), "29893779": ("control", 43), "29893781": ("control", 44),
}

# series colors validated CVD-safe (dataviz six-checks: worst adjacent dE 47.6)
TEAL, AMBER = "#0e9384", "#b45309"
INK, MUTED, GRID = "#171a1f", "#7c828a", "#e4e8ed"
REF_DARK, REF_MID = "#2b2f36", "#7c828a"


def find_eval_dir(jobid):
    for d in sorted(REPORTS.glob(f"pilot_*_j{jobid}")):
        ev = list(d.glob("**/eval_results"))
        if ev:
            return ev[0]
    return None


def dump_horizons(evdir):
    """{structure: probe horizon} from the first dump (pre-h* dumps carry no field -> 1)."""
    dumps = sorted(evdir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    recs = json.loads(dumps[0].read_text())
    hz = {}
    for r in recs:
        ref = json.loads(r["reference"])
        hz[ref["structure"]] = int(ref.get("horizon", 1))
    return hz


def compute_refs_at(cities_by_struct, ks, horizons):
    """Reference forecasters on the identical episodes, probed at the dumps' own horizon."""
    refs = defaultdict(dict)
    for st_name, seeds in cities_by_struct.items():
        struct = BY_NAME[st_name]
        po, pi_in = struct.probe_output, struct.probe_input
        h = horizons.get(st_name, 1)
        # true response at h: d target / d shock (exact, from the noise-free map)
        def resp(g):
            u = np.zeros(struct.m); u[pi_in] = 1.0
            base = np.full(struct.d, struct.baseline)
            S0 = np.zeros((1, struct.d)) + base
            return float(true_output_at_horizon(struct, S0, 0, g, u, h)[po]
                         - true_output_at_horizon(struct, S0, 0, g, np.zeros(struct.m), h)[po])
        rec = abs(resp(1.0) - resp(0.1)) > 1e-9      # does the h-response vary with g?
        for seed in seeds:
            ep = generate_episode(struct, seed=seed, T=10)
            U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
            for k in ks:
                targets, last = {}, float(Y[k - 1][po])
                for s in SHOCKS:
                    u = np.zeros(struct.m); u[pi_in] = s
                    targets[s] = float(true_output_at_horizon(struct, S, k, g, u, h)[po])

                def probe(filt):
                    xs = np.asarray(SHOCKS, float); ys = []
                    for s in SHOCKS:
                        u = np.zeros(struct.m); u[pi_in] = s
                        ys.append(float(forecast_at_horizon(struct, filt, u, h)[po]))
                    ys = np.asarray(ys)
                    return (float(np.mean([abs(y - targets[s]) for y, s in zip(ys, SHOCKS)])),
                            float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs)))

                common = {"g": g, "true_gain": resp(g), "recover": rec}
                p, gh = probe(filter_all(struct, U[:k], Y[:k]))
                refs[("oracle", st_name, k)][seed] = {"pi": p, "ghat": gh, **common}
                p, gh = probe(filter_all(struct, U[:k], Y[:k], g_grid=np.asarray([PRIOR_MEAN])))
                refs[("prior_blind", st_name, k)][seed] = {"pi": p, "ghat": gh, **common}
                for gam in GAMMAS:
                    refs[(f"fixed_{gam:.2f}", st_name, k)][seed] = {
                        "pi": float(np.mean([abs(last + gam * s - targets[s]) for s in SHOCKS])),
                        "ghat": float(gam), **common}
    return refs


def headline(get_cell, info_ks):
    """Aggregate a row: pi over all 4 held-out structures, eps/rho over the recover ones."""
    pis, epss, rhos = [], [], []
    for st in TEST_NAMES:
        pooled = get_cell(st, info_ks[st])
        if not pooled:
            continue
        s = summarize(pooled)
        pis.append(s["pi"])
        if "eps" in s:
            epss.append(s["eps"]); rhos.append(s.get("rho", float("nan")))
    return (float(np.mean(pis)) if pis else np.nan,
            float(np.mean(epss)) if epss else np.nan,
            float(np.nanmean(rhos)) if rhos else np.nan)


def main():
    # ── model curves ────────────────────────────────────────────────────────────────────────────
    curves = defaultdict(dict)      # (arm, seed) -> {step: (pi, eps, rho)}
    cities, ks_seen, horizons = None, None, None
    refs = info_ks = None
    for jobid, (arm, seed) in RUNS.items():
        ev = find_eval_dir(jobid)
        if ev is None:
            print(f"[{arm} s{seed} j{jobid}] no eval dir yet"); continue
        if cities is None:
            horizons = dump_horizons(ev)
            d0, _ = read_dump(sorted(ev.glob("*.json"), key=lambda p: int(p.stem))[0])
            ks_seen = sorted({k for (_, k) in d0})
            cities = defaultdict(set)
            for (st, _k), per in d0.items():
                cities[st].update(per.keys())
            cities = {st: sorted(s) for st, s in cities.items()}
            print(f"cities: { {s: len(c) for s, c in cities.items()} } ks={ks_seen} horizons={horizons}")
            print("computing references at the dumps' probe horizon ...")
            refs = compute_refs_at(cities, tuple(ks_seen), horizons)
            info_ks = {}
            for st in TEST_NAMES:
                info_ks[st] = [k for k in ks_seen
                               if (summarize(refs[("prior_blind", st, k)])["pi"]
                                   - summarize(refs[("oracle", st, k)])["pi"]) >= MARGIN_MIN] or list(ks_seen)
            print(f"informative ks: {info_ks}")
        for dump in sorted(ev.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9):
            if not dump.stem.isdigit():
                continue
            data, _miss = read_dump(dump)
            curves[(arm, seed)][int(dump.stem)] = headline(
                lambda st, ks, data=data: _pool(data, st, ks), info_ks)
        print(f"[{arm} s{seed} j{jobid}] steps {sorted(curves[(arm, seed)])}")

    if not curves:
        print("no data"); return

    # ── reference headline values (flat lines) ──────────────────────────────────────────────────
    def ref_cell(rname):
        return lambda st, ks: _pool({(s, k): v for (rn, s, k), v in refs.items() if rn == rname}, st, ks)
    R = {name: headline(ref_cell(name), info_ks) for name in ("oracle", "prior_blind")}
    R["always_prior"] = headline(ref_cell("fixed_0.55"), info_ks)
    bf_pis = []
    for st in TEST_NAMES:     # best fixed gamma chosen per structure by pi
        best = min((summarize(_pool({(s, k): v for (rn, s, k), v in refs.items()
                                     if rn == f"fixed_{gam:.2f}"}, st, info_ks[st]))["pi"]
                    for gam in GAMMAS))
        bf_pis.append(best)
    R["best_fixed"] = (float(np.mean(bf_pis)), np.nan, np.nan)

    # ── figure ──────────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
    panels = [("Next-poll error  $\\pi$  (poll pts)", 0), ("Recovery error  $\\varepsilon$", 1),
              ("Informativeness  $\\rho(\\hat g, g)$", 2)]
    series = [("family", TEAL, "family-trained"), ("control", AMBER, "structureless control")]

    steps_all = sorted({s for c in curves.values() for s in c})

    # PAIRED seeds per step: a seed contributes at a step only if BOTH arms reached that step there.
    # Union-of-steps alone is not enough -- while runs are in flight one arm can be ahead on a seed
    # (here control has seed 44 and family does not), so the two lines would be averaged over
    # different seeds and a gap between them could be the sample rather than the arm.
    seeds_at = defaultdict(dict)                       # step -> arm -> {seed: value-tuple}
    for (a, sd), c in curves.items():
        for s, v in c.items():
            seeds_at[s].setdefault(a, {})[sd] = v
    paired = {s: sorted(set(d.get("family", {})) & set(d.get("control", {})))
              for s, d in seeds_at.items()}
    paired = {s: sd for s, sd in paired.items() if sd}
    print("paired seeds per step:", {s: paired[s] for s in sorted(paired)})

    for ax, (title, mi) in zip(axes, panels):
        for arm, color, label in series:
            step_vals = defaultdict(list)
            for s, sds in paired.items():
                for sd in sds:
                    v = seeds_at[s][arm][sd][mi]
                    if np.isfinite(v):
                        step_vals[s].append(v)
            xs = sorted(step_vals)
            if not xs:
                continue
            mean = np.array([np.mean(step_vals[s]) for s in xs])
            ax.plot(xs, mean, color=color, lw=2, marker="o", ms=3.5, label=label, zorder=3)
            TCRIT = {2: 12.706, 3: 4.303}                       # 95% Student-t, df = n-1
            lo, hi = [], []
            for s in xs:
                v = np.asarray(step_vals[s], float)
                if v.size >= 2:
                    sem = v.std(ddof=1) / np.sqrt(v.size)
                    t = TCRIT.get(v.size, 2.0)
                    lo.append(v.mean() - t * sem); hi.append(v.mean() + t * sem)
                else:
                    lo.append(np.nan); hi.append(np.nan)
            # pi/eps are non-negative and rho is a correlation; an unclamped t-band at n=2
            # (t*=12.7) runs outside the admissible range, which is impossible and unreadable.
            bnd = {0: (0.0, None), 1: (0.0, None), 2: (-1.0, 1.0)}[mi]
            clamp = lambda a: np.clip(np.asarray(a, float),
                                      bnd[0] if bnd[0] is not None else -np.inf,
                                      bnd[1] if bnd[1] is not None else np.inf)
            ax.fill_between(xs, clamp(lo), clamp(hi), color=color, alpha=0.15, lw=0, zorder=2)
        # reference lines: recessive grays, distinct dashes, direct right-edge labels
        ref_style = {"oracle": (REF_DARK, ":"), "prior_blind": (REF_MID, "--"),
                     "always_prior": (REF_MID, "-."), "best_fixed": (REF_DARK, (0, (5, 2)))}
        show = {0: ["oracle", "prior_blind", "best_fixed"], 1: ["oracle", "always_prior"],
                2: ["oracle"]}[mi]
        for rname in show:
            v = R[rname][mi]
            if not np.isfinite(v):
                continue
            c, ls = ref_style[rname]
            ax.axhline(v, color=c, ls=ls, lw=1.2, zorder=1)
            ax.annotate(rname.replace("_", " "), xy=(1.0, v), xycoords=("axes fraction", "data"),
                        xytext=(-3, 3), textcoords="offset points", ha="right",
                        fontsize=7.5, color=c)
        if mi == 2:
            ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlabel("Dr. GRPO step", fontsize=9, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)

    # figure-level legend: keeps identity off the plot area, where it collided with the pi curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, fontsize=8.5, frameon=False)
    # report the paired sample used at the most complete checkpoint
    nmax = max((len(sd) for sd in paired.values()), default=0)
    fig.suptitle("Held-out worlds: forecast error, recovery, and informativeness under training  "
                 "(family vs structureless control)", fontsize=11.5, color=INK, y=1.10)
    fig.text(0.5, -0.06,
             f"4 held-out structures, 60 cities each, informative (structure, k) cells only "
             f"(oracle-vs-blind margin ≥ {MARGIN_MIN}); "
             f"both arms averaged over the same paired seeds at each step (n = {nmax}), "
             "95% t-CI bands; ε/ρ over gain-recovering structures.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}.png + .pdf")

    # tidy CSV of the plotted curves
    csv = REPORTS / "transfer_curves.csv"
    with open(csv, "w") as fh:
        fh.write("arm,seed,step,pi,eps,rho\n")
        for (arm, seed), c in sorted(curves.items()):
            for s in sorted(c):
                pi_v, eps_v, rho_v = c[s]
                fh.write(f"{arm},{seed},{s},{pi_v:.4f},{eps_v:.4f},{rho_v:.4f}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
