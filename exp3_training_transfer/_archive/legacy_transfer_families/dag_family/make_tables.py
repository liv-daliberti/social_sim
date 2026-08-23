#!/usr/bin/env python3
"""make_tables.py — emit the Experiment-3 LaTeX tables from the eval dumps, per environment.

Writes two tabulars that the paper \\input's, so the numbers cannot drift from the runs:

    paper/tables/exp3_refs.tex      the per-world reference bracket
    paper/tables/exp3_transfer.tex  the per-world transfer result (family vs structureless control)

Why per world, and why no pooled row.  Pooling the four held-out worlds is misleading twice over.
(i) pi is dominated by the delayed worlds (chain4 ~12, mediators ~9 poll pts), so a pooled pi mostly
reports how badly the model does where the news has barely reached the poll.  (ii) eps is not
commensurable across worlds: the recoverable response gamma* spans 0.10-1.00 in `direct` but only
0.022-0.216 in `chain4`, so the same absolute |ghat - gamma*| means very different things.

We therefore report a SCALE-NORMALISED recovery error

    eps_tilde = eps / eps(prior-blind)

where prior-blind is the per-structure Kalman filter with g frozen at the range midpoint: it knows the
structure and never infers the gain, so its eps is exactly that world's "don't bother inferring" floor,
expressed on that world's own response scale.  eps_tilde < 1 means inferring g beat not inferring it.
(The `always-prior` reference is NOT a valid floor here: it fixes the *response* at 0.55 regardless of
the world's gamma* range, which is absurd in `chain4` where the true response never exceeds 0.216.)

    /usr/bin/python3 make_tables.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transfer_report import GAMMAS, MARGIN_MIN, compute_refs, read_dump, summarize
from lg_dag import G_LO, G_HI, first_response_horizon, response_at_horizon
from catalog import BY_NAME

REPORTS = HERE / "reports"
TABLES = HERE.parent.parent / "paper" / "tables"

RUNS = {"29893776": ("family", 42), "29893778": ("family", 43), "29899438": ("family", 44),
        "29893777": ("control", 42), "29893779": ("control", 43), "29893781": ("control", 44)}
ORDER = ["direct", "two_news_feedback", "mediators", "chain4"]
NICE = {"direct": r"\texttt{direct} (Exp-2 world)", "two_news_feedback": r"\texttt{two\_news\_feedback}",
        "mediators": r"\texttt{mediators}", "chain4": r"\texttt{chain4}"}


def evdir(j):
    for d in sorted(REPORTS.glob(f"pilot_*_j{j}")):
        e = list(d.glob("**/eval_results"))
        if e:
            return e[0]


def pool(cells, st, kk):
    out = {}
    for k in kk:
        for sd, c in cells.get((st, k), {}).items():
            out[(sd, k)] = c
    return out


def main():
    dumps = {}
    for j, (a, s) in RUNS.items():
        ev = evdir(j)
        if ev is None:
            continue
        ds = sorted((p for p in ev.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem))
        if ds:
            dumps[(a, s)] = {int(p.stem): p for p in ds}

    steps_arm = defaultdict(lambda: defaultdict(set))
    for (a, sd), d in dumps.items():
        for s in d:
            steps_arm[s][a].add(sd)
    paired = {s: sorted(v.get("family", set()) & v.get("control", set())) for s, v in steps_arm.items()}
    STEP = max(s for s, sd in paired.items() if s > 0 and len(sd) >= 2)
    SEEDS = paired[STEP]
    if STEP != 301 or SEEDS != [42, 43, 44]:
        raise RuntimeError(
            f"refusing to emit the endpoint table from incomplete runs: step={STEP}, seeds={SEEDS}"
        )
    print(f"reading step {STEP}, paired seeds {SEEDS}")

    base, _ = read_dump(dumps[("family", SEEDS[0])][0])
    ks = sorted({k for (_s, k) in base})
    cities = defaultdict(set)
    for (st, _k), per in base.items():
        cities[st].update(per.keys())
    print("computing per-world references ...")
    refs = compute_refs({s: sorted(v) for s, v in cities.items()}, tuple(ks))

    def iks(st):
        return [k for k in ks
                if summarize(refs[("prior_blind", st, k)])["pi"]
                - summarize(refs[("oracle", st, k)])["pi"] >= MARGIN_MIN] or list(ks)

    def refc(name, st, kk):
        return summarize(pool({(st, k): refs[(name, st, k)] for k in kk}, st, kk))

    def bestfix(st, kk):
        return min(summarize(pool({(st, k): refs[(f"fixed_{g:.2f}", st, k)] for k in kk}, st, kk))["pi"]
                   for g in GAMMAS)

    def arm(a, st, kk):
        acc = defaultdict(list)
        for sd in SEEDS:
            s = summarize(pool(read_dump(dumps[(a, sd)][STEP])[0], st, kk))
            for m in ("pi", "eps", "rho"):
                if m in s and np.isfinite(s[m]):
                    acc[m].append(s[m])
        return {m: float(np.mean(v)) for m, v in acc.items()}

    TABLES.mkdir(parents=True, exist_ok=True)

    # ── reference bracket, per world ────────────────────────────────────────────────────────────
    lines = [r"\begin{tabular}{l c ccccc c cc}", r"\toprule",
             r" & & \multicolumn{5}{c}{next-poll error $\pi$ (poll pts)} & &"
             r" \multicolumn{2}{c}{recovery $\varepsilon$} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){9-10}",
             r"Held-out world & $h^\star$ & oracle & prior-blind & best fixed $\gamma$ & "
             r"always-prior & persistence & & oracle & prior-blind \\", r"\midrule"]
    for st in ORDER:
        kk = iks(st)
        o, pb = refc("oracle", st, kk), refc("prior_blind", st, kk)
        ap, ps = refc("fixed_0.55", st, kk), refc("fixed_0.00", st, kk)
        lines.append(f"{NICE[st]} & ${first_response_horizon(BY_NAME[st])}$ & "
                     f"${o['pi']:.2f}$ & ${pb['pi']:.2f}$ & ${bestfix(st,kk):.2f}$ & "
                     f"${ap['pi']:.2f}$ & ${ps['pi']:.2f}$ & & "
                     f"${o['eps']:.3f}$ & ${pb['eps']:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "exp3_refs.tex").write_text("% generated by make_tables.py -- do not edit\n"
                                          + "\n".join(lines) + "\n")

    # ── transfer result, per world ──────────────────────────────────────────────────────────────
    lines = [r"\begin{tabular}{l c ccc c cc c cc}", r"\toprule",
             r" & & \multicolumn{3}{c}{$\pi\downarrow$ (poll pts)} & &"
             r" \multicolumn{2}{c}{$\tilde\varepsilon\downarrow$} & &"
             r" \multicolumn{2}{c}{$\rho\uparrow$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){7-8}\cmidrule(lr){10-11}",
             r"Held-out world & $h^\star$ & base & family & control & & family & control & &"
             r" family & control \\", r"\midrule"]
    for st in ORDER:
        kk = iks(st)
        b, f, c = summarize(pool(base, st, kk)), arm("family", st, kk), arm("control", st, kk)
        eb = refc("prior_blind", st, kk)["eps"]
        bold = lambda x, better: (r"\textbf{" + f"{x:.2f}" + "}") if better else f"{x:.2f}"
        lines.append(f"{NICE[st]} & ${first_response_horizon(BY_NAME[st])}$ & "
                     f"${b['pi']:.2f}$ & ${bold(f['pi'], f['pi']<c['pi'])}$ & ${bold(c['pi'], c['pi']<f['pi'])}$ & & "
                     f"${f['eps']/eb:.2f}$ & ${c['eps']/eb:.2f}$ & & "
                     f"${f['rho']:.3f}$ & ${c['rho']:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "exp3_transfer.tex").write_text("% generated by make_tables.py -- do not edit\n"
                                              + "\n".join(lines) + "\n")

    print(f"wrote {TABLES/'exp3_refs.tex'}\nwrote {TABLES/'exp3_transfer.tex'}")
    print(f"\nSTEP={STEP} SEEDS={SEEDS}")
    for st in ORDER:
        kk = iks(st)
        f, c = arm("family", st, kk), arm("control", st, kk)
        eb = refc("prior_blind", st, kk)["eps"]
        print(f"  {st:20s} eps~ fam {f['eps']/eb:.2f}  ctrl {c['eps']/eb:.2f}")


if __name__ == "__main__":
    main()
