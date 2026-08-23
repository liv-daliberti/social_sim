#!/usr/bin/env python3
"""plot_probe_diagnostic.py — why Experiment 3 needed the h* probe, and what "success" means.

Three panels, all computed from the world matrices themselves (no run artefacts, so the figure
cannot drift from the catalog):

  (a) Probe informativeness. How much the probed poll's response to news varies with the hidden gain
      g, at the fixed one-step probe vs at each world's first-response horizon h*. A bar at zero
      means the probe is DEGENERATE there: all four shock scenarios share one target, so the answers
      carry no information about g and a constant reply is exactly optimal.

  (b) The reward exploit. What a constant "ignore the news" policy earns, as a share of the oracle's
      reward, under the one-step probe vs under h*. Under h=1 the policy collects most of the pay in
      the delayed worlds -- which is what the Jul-7 runs learned (pi fell while rho collapsed).

  (c) The goalposts. Where every reference forecaster sits on held-out next-poll error pi, so it is
      legible what the trained model must actually beat: not the oracle, but the blind constants.

    /usr/bin/python3 plot_probe_diagnostic.py     # -> paper/figures/exp3_probe_diagnostic.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lg_dag import (G_LO, G_HI, SHOCKS, encodes_gain, filter_all, first_response_horizon,
                    forecast_at_horizon, generate_episode, response_at_horizon,
                    true_output_at_horizon)
from catalog import CATALOG, TEST_NAMES

OUT = HERE.parent.parent / "paper" / "figures" / "exp3_probe_diagnostic"

# dataviz reference palette (light). Validated: worst adjacent CVD dE 74.6 (blue/red), 96.7 (blue/orange).
BLUE, RED, ORANGE = "#2a78d6", "#e34948", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e6e4", "#fcfcfb"

RS, SW, SS = 15.0, 0.5, 0.5          # reward_scale_pts, slope_weight, slope_scale_g (dag_rl.sh)
N_EP, KS = 120, (1, 2, 3, 5, 7, 9)


def _reward(preds, targets, shocks, slope_target):
    level = np.mean([max(0.0, 1.0 - abs(p - t) / RS) for p, t in zip(preds, targets)])
    slope = np.cov(shocks, preds, bias=True)[0, 1] / np.var(shocks)
    return (1 - SW) * level + SW * max(0.0, 1.0 - abs(slope - slope_target) / SS)


def constant_share(struct, h, n=N_EP):
    """Mean reward of a constant 'ignore the news' reply, as a share of the oracle's (which is 1)."""
    xs = np.asarray(SHOCKS, float)
    out = []
    for i in range(n):
        ep = generate_episode(struct, seed=7_000 + i, T=10)
        S, g, Y = ep["S"], ep["g"], ep["Y"]
        k = 1 + (i % 8)
        tg = []
        for s in SHOCKS:
            u = np.zeros(struct.m); u[struct.probe_input] = s
            tg.append(float(true_output_at_horizon(struct, S, k, g, u, h)[struct.probe_output]))
        last = float(Y[k - 1][struct.probe_output])
        st = response_at_horizon(struct, g, h)
        out.append(_reward([last] * 4, tg, xs, st) / max(_reward(tg, tg, xs, st), 1e-9))
    return float(np.mean(out))


MARGIN_MIN = 0.10                              # matches transfer_report.py / plot_transfer_curves.py
GAMMAS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def held_out_bracket(n=60):
    """pi (poll pts) for each reference on the held-out worlds, at each world's h*.

    Restricted to INFORMATIVE (structure, k) cells -- those where the oracle beats the prior-blind
    filter by >= MARGIN_MIN poll points, i.e. where knowing g actually helps. This is the same filter
    transfer_report.py and plot_transfer_curves.py apply, so the bracket here matches the reference
    lines there. Without it, chain4/mediators at k<=3 (news has not reached the poll) drag every
    reference toward persistence and the goalposts read too easy."""
    acc = {k: [] for k in ("oracle", "prior_blind", "naive_freq", "always_prior",
                           "best_fixed", "persistence")}
    for si, name in enumerate(TEST_NAMES):
        st = next(s for s in CATALOG if s.name == name)
        h = first_response_horizon(st)
        cell = {k: {**{kk: [] for kk in acc}, **{f"g{gi}": [] for gi in range(len(GAMMAS))}}
                for k in KS}
        for i in range(n):
            # exactly the held-out cities of make_dataset_dag.build_eval (per-structure seed offset)
            ep = generate_episode(st, seed=90_000_000 + si * 1_000_000 + i, T=10)
            U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
            for k in KS:
                per, tg, last, us = cell[k], [], float(Y[k - 1][st.probe_output]), []
                for s in SHOCKS:
                    u = np.zeros(st.m); u[st.probe_input] = s; us.append(u)
                    tg.append(float(true_output_at_horizon(st, S, k, g, u, h)[st.probe_output]))
                tg = np.asarray(tg)
                mae = lambda p: float(np.mean(np.abs(np.asarray(p) - tg)))

                filt = filter_all(st, U[:k], Y[:k])
                per["oracle"].append(mae([forecast_at_horizon(st, filt, u, h)[st.probe_output] for u in us]))
                fb = filter_all(st, U[:k], Y[:k], g_grid=np.asarray([0.55]))
                per["prior_blind"].append(mae([forecast_at_horizon(st, fb, u, h)[st.probe_output] for u in us]))
                per["persistence"].append(mae([last] * 4))
                per["always_prior"].append(mae([np.clip(last + 0.55 * s, 0, 100) for s in SHOCKS]))
                if k >= 2:
                    dy = np.diff(Y[:k, st.probe_output]).astype(float)
                    x = U[1:k, st.probe_input].astype(float)
                    gf = float((x @ dy) / (x @ x)) if (x @ x) > 0 else 0.55
                    per["naive_freq"].append(mae([np.clip(last + gf * s, 0, 100) for s in SHOCKS]))
                for gi, gam in enumerate(GAMMAS):
                    per[f"g{gi}"].append(mae([np.clip(last + gam * s, 0, 100) for s in SHOCKS]))
        info = [k for k in KS
                if np.mean(cell[k]["prior_blind"]) - np.mean(cell[k]["oracle"]) >= MARGIN_MIN] or list(KS)
        print(f"  {name}: informative ks {info}")
        for kk in acc:
            if kk == "best_fixed":
                continue
            vals = [v for k in info for v in cell[k][kk]]
            if vals:
                acc[kk].append(float(np.mean(vals)))
        # ONE constant gamma per structure (not per episode): the strongest single blind policy,
        # matching transfer_report.py's best_fixed. A per-episode argmin would be hindsight no
        # constant policy could achieve, and would draw the goalpost in the wrong place.
        acc["best_fixed"].append(min(float(np.mean([v for k in info for v in cell[k][f"g{gi}"]]))
                                     for gi in range(len(GAMMAS))))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    names = [s.name for s in CATALOG]
    hstar = {s.name: first_response_horizon(s) for s in CATALOG}
    rng1 = {s.name: abs(response_at_horizon(s, G_HI, 1) - response_at_horizon(s, G_LO, 1)) for s in CATALOG}
    rngh = {s.name: abs(response_at_horizon(s, G_HI, hstar[s.name])
                        - response_at_horizon(s, G_LO, hstar[s.name])) for s in CATALOG}
    order = sorted(names, key=lambda n: (n in TEST_NAMES, -rngh[n]))

    train = [s for s in CATALOG if s.name not in TEST_NAMES]
    share1 = {s.name: constant_share(s, 1) for s in train}
    shareh = {s.name: constant_share(s, hstar[s.name]) for s in train}
    torder = sorted(share1, key=lambda n: -share1[n])
    print("computing held-out bracket ...", flush=True)
    br = held_out_bracket()

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), facecolor=SURFACE,
                             constrained_layout=True)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)

    # ── (a) probe informativeness ────────────────────────────────────────────────────────────────
    ax = axes[0]
    y = np.arange(len(order)); bh = 0.36
    ax.barh(y + bh / 2, [rng1[n] for n in order], bh, color=RED, label="one-step probe (h=1)",
            edgecolor=SURFACE, linewidth=1.2)
    ax.barh(y - bh / 2, [rngh[n] for n in order], bh, color=BLUE, label="response horizon h*",
            edgecolor=SURFACE, linewidth=1.2)
    for i, n in enumerate(order):
        if rngh[n] < 1e-9:                      # g sits on the persistence: no input gain, ever
            ax.text(0.02, i, "$g$ hides in $\\phi$ — no gain to recover",
                    va="center", fontsize=7, color=MUTED, style="italic")
        else:
            if rng1[n] < 1e-9:                  # blind only because the pulse hasn't arrived yet
                ax.text(0.02, i + bh / 2, "blind at h=1", va="center", fontsize=7, color=RED)
            ax.text(rngh[n] + 0.02, i - bh / 2, f"h*={hstar[n]}", va="center",
                    fontsize=7, color=MUTED)
    lab = [f"{n}  ▪" if n in TEST_NAMES else n for n in order]
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("spread of the probe response across $g\\in[0.1,1.0]$", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1.30)
    ax.set_title("(a) The one-step probe was blind in most worlds",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=MUTED)
    ax.text(0.985, 0.985, "▪ held-out world", fontsize=7, color=MUTED, ha="right",
            va="top", transform=ax.transAxes)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)

    # ── (b) the exploit ─────────────────────────────────────────────────────────────────────────
    ax = axes[1]
    y = np.arange(len(torder))
    ax.barh(y + bh / 2, [100 * share1[n] for n in torder], bh, color=RED, label="one-step probe (h=1)",
            edgecolor=SURFACE, linewidth=1.2)
    ax.barh(y - bh / 2, [100 * shareh[n] for n in torder], bh, color=BLUE, label="response horizon h*",
            edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y); ax.set_yticklabels(torder, fontsize=8); ax.invert_yaxis()
    ax.axvline(100, color=MUTED, lw=0.8, ls=":")
    ax.text(101.5, 0, "oracle", fontsize=7.5, color=MUTED, ha="left", va="center")
    a1 = 100 * np.mean(list(share1.values())); ah = 100 * np.mean(list(shareh.values()))
    ax.set_xlabel("reward earned by a constant 'ignore the news' reply\n(% of the oracle's reward)",
                  fontsize=9, color=MUTED)
    ax.set_xlim(0, 112)
    ax.set_title(f"(b) …so ignoring the news paid.  mean {a1:.0f}% → {ah:.0f}%",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)

    # ── (c) goalposts ───────────────────────────────────────────────────────────────────────────
    ax = axes[2]
    rows = [("oracle (ceiling)", br["oracle"]),
            ("prior-blind (knows structure)", br["prior_blind"]),
            ("best fixed γ (hindsight)", br["best_fixed"]),
            ("always-prior γ=0.55", br["always_prior"]),
            ("naive frequentist", br["naive_freq"]),
            ("persistence (floor)", br["persistence"])]
    rows.sort(key=lambda r: r[1])
    yy = np.arange(len(rows))
    vals = [v for _, v in rows]
    # one measure → one hue, light→dark by magnitude (sequential, not categorical)
    ramp = ["#184f95", "#2a78d6", "#3987e5", "#6da7ec", "#86b6ef", "#b7d3f6"]
    ax.barh(yy, vals, 0.55, color=ramp[:len(rows)], edgecolor=SURFACE, linewidth=1.2)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.08, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    bar = min(br["best_fixed"], br["always_prior"])
    ax.axvline(bar, color=ORANGE, lw=1.6)
    ax.text(bar + 0.14, 0.0, "strongest blind constant\n← the model must land here",
            fontsize=7.5, color=ORANGE, ha="left", va="center")
    ax.set_xlabel("held-out next-poll error  π  (poll points, lower is better)", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(vals) * 1.20)
    ax.set_title("(c) What counts as success on held-out worlds",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}.png + .pdf")
    print(f"  constant-policy share: h=1 {a1:.1f}%  ->  h* {ah:.1f}%")
    print("  held-out bracket:", {k: round(v, 2) for k, v in br.items()})


if __name__ == "__main__":
    main()
