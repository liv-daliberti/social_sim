#!/usr/bin/env python3
"""transfer_report.py — the Experiment-3 transfer table, read the Exp-2 way (2026-07-08 audit fix).

Reads MULTI-SHOCK OAT eval dumps (eval_results/<step>.json; references carry `targets`/`shocks`)
and reports, per held-out structure and prefix k, ALL THREE metrics the claim needs:

    pi   = next-poll forecast error  mean_i |pred_i - target_i|   (poll pts; all structures)
    eps  = recovery error            mean_city |ghat - g|          (recover=True structures)
    rho  = informativeness           Spearman rho(ghat, g)         (recover=True structures)
    (+ beta, mean ghat, sd ghat — the shrinkage signature)

against SIX references computed on the IDENTICAL episodes:

    oracle        Kalman filter marginalizing g over the grid (ceiling)
    prior_blind   same filter with g FIXED at 0.55 (structure known, no inference) — the blind bar
    naive_freq    through-origin OLS of consecutive poll changes on the probe news (data-only)
    always_prior  last_poll + 0.55*shock (blind constant, the 'calibrated yet blind' floor)
    best_fixed    the best-in-hindsight constant gamma on this cell (strongest possible constant)
    persistence   predict the last poll (gamma=0)

Per Exp 2's own standard, a positive transfer claim must beat the blind constants on pi AND keep
rho materially above 0 — low eps alone is satisfied by guessing the prior mean (eps=0.225).

Identifiability guard: each (structure, k) cell is annotated with the oracle-vs-prior_blind margin
(pi_blind - pi_oracle). Cells where the margin < MARGIN_MIN carry no usable g signal (e.g. chain4
at k<=3: the news hasn't reached the poll) and are EXCLUDED from the headline per-structure means.

Usage:
    python3 transfer_report.py <eval_results_dir>                       # before=first, after=last dump
    python3 transfer_report.py fam=<dir> ctrl=<dir> [...]               # multiple labelled runs
    python3 transfer_report.py <dir> --csv out.csv --latex table.tex
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lg_dag import (G_GRID, SHOCKS, encodes_gain, filter_all, first_response_horizon,
                    forecast_at_horizon, generate_episode, response_at_horizon,
                    true_output_at_horizon)
from catalog import BY_NAME, TEST_NAMES
from prompt import parse_forecasts

PRIOR_MEAN = 0.55
MARGIN_MIN = 0.10          # poll pts: min oracle-vs-blind margin for a cell to count as informative
GAMMAS = np.round(np.arange(0.0, 1.01, 0.05), 2)


# ── model read-out ────────────────────────────────────────────────────────────────────────────────
def read_dump(path: Path):
    """-> {(structure, k): {seed: {'pi', 'ghat', 'g', 'true_gain', 'recover'}}}, miss_rate

    Guards the probe convention: references must carry `horizon` equal to the structure's
    first-response horizon h*. Dumps written before the 2026-07-08 h* fix used a fixed one-step
    probe, whose targets/slope_target mean something different -- scoring them against h* references
    would silently produce a wrong table."""
    records = json.loads(Path(path).read_text())
    out = defaultdict(dict)
    n_miss = n_tot = 0
    for rec in records:
        ref = json.loads(rec["reference"])
        st_name = ref["structure"]
        h_ref = ref.get("horizon")
        h_exp = first_response_horizon(BY_NAME[st_name])
        if h_ref is None:
            raise SystemExit(
                f"{path}: reference for '{st_name}' has no `horizon` field -- this dump predates the "
                f"h* multi-shock probe (2026-07-08). Its one-step targets are not comparable to the "
                f"h* references this report computes. Re-run with the current dataset.")
        if int(h_ref) != h_exp:
            raise SystemExit(
                f"{path}: '{st_name}' was probed at horizon {h_ref} but its first-response horizon "
                f"is {h_exp}. Dump and catalog disagree; refusing to score.")
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


# ── references on the identical episodes ─────────────────────────────────────────────────────────
def compute_refs(cities_by_struct: dict, ks: tuple):
    """refs[(name, structure, k)][seed] = {'pi', 'ghat', 'g', ...}; one generative pass per city."""
    refs = defaultdict(dict)
    for st_name, seeds in cities_by_struct.items():
        struct = BY_NAME[st_name]
        po, pi_in = struct.probe_output, struct.probe_input
        # probe each structure at its own first-response horizon h*, exactly as the dataset does
        h = first_response_horizon(struct)
        rec = encodes_gain(struct, h)
        tg = lambda g: response_at_horizon(struct, g, h)   # true response at h* (the gain to recover)
        for seed in seeds:
            ep = generate_episode(struct, seed=seed, T=10)
            U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
            for k in ks:
                targets, last = {}, float(Y[k - 1][po])
                for s in SHOCKS:
                    u = np.zeros(struct.m); u[pi_in] = s
                    targets[s] = float(true_output_at_horizon(struct, S, k, g, u, h)[po])

                def probe(filt):
                    preds = {}
                    for s in SHOCKS:
                        u = np.zeros(struct.m); u[pi_in] = s
                        preds[s] = float(forecast_at_horizon(struct, filt, u, h)[po])
                    xs = np.asarray(SHOCKS, float)
                    ys = np.asarray([preds[s] for s in SHOCKS])
                    return (float(np.mean([abs(preds[s] - targets[s]) for s in SHOCKS])),
                            float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs)))

                common = {"g": g, "true_gain": tg(g), "recover": rec}
                p, gh = probe(filter_all(struct, U[:k], Y[:k]))
                refs[("oracle", st_name, k)][seed] = {"pi": p, "ghat": gh, **common}
                p, gh = probe(filter_all(struct, U[:k], Y[:k], g_grid=np.asarray([PRIOR_MEAN])))
                refs[("prior_blind", st_name, k)][seed] = {"pi": p, "ghat": gh, **common}

                if k >= 2:      # naive frequentist: data-only through-origin fit of poll changes
                    dy = np.array([Y[t][po] - Y[t - 1][po] for t in range(1, k)], float)
                    x = np.array([U[t][pi_in] for t in range(1, k)], float)
                    gf = float((x @ dy) / (x @ x)) if (x @ x) > 0 else PRIOR_MEAN
                    refs[("naive_freq", st_name, k)][seed] = {
                        "pi": float(np.mean([abs(last + gf * s - targets[s]) for s in SHOCKS])),
                        "ghat": gf, **common}

                for gam in GAMMAS:  # constant policies anchored at the last poll
                    refs[(f"fixed_{gam:.2f}", st_name, k)][seed] = {
                        "pi": float(np.mean([abs(last + gam * s - targets[s]) for s in SHOCKS])),
                        "ghat": float(gam), **common}
    return refs


# ── summaries ────────────────────────────────────────────────────────────────────────────────────
def _spearman(x, y):
    """x = the forecaster's ghat, y = the true gain. A forecaster whose ghat never varies is BLIND,
    not undefined: it orders no cities, so rho = 0 (the Exp-2 convention for always-prior). nan is
    reserved for the degenerate case where the TRUTH is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(y) == 0:
        return float("nan")
    if np.std(x) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(per_seed: dict) -> dict:
    """{seed: {'pi','ghat','g','true_gain','recover'}} -> {pi, n, [eps, rho, beta, mghat, sghat]}"""
    out = {"pi": float(np.mean([d["pi"] for d in per_seed.values()])), "n": len(per_seed)}
    rec = [(d["ghat"], d["true_gain"]) for d in per_seed.values()
           if d.get("recover") and d.get("true_gain") is not None and d.get("ghat") is not None]
    if len(rec) >= 3:
        gh = np.array([a for a, _ in rec]); gt = np.array([b for _, b in rec])
        out.update(eps=float(np.mean(np.abs(gh - gt))), rho=_spearman(gh, gt),
                   beta=float(np.cov(gt, gh, bias=True)[0, 1] / np.var(gt)) if np.var(gt) > 0 else float("nan"),
                   mghat=float(np.mean(gh)), sghat=float(np.std(gh)))
    return out


def _fmt(s, keys=("pi", "eps", "rho", "beta", "mghat", "sghat")):
    def one(k):
        v = s.get(k)
        return f"{k}={v:.3f}" if isinstance(v, float) and np.isfinite(v) else f"{k}=--"
    return " ".join(one(k) for k in keys)


def _pool(data, st, ks):
    pooled = {}
    for k in ks:
        for seed, d in data.get((st, k), {}).items():
            pooled[(seed, k)] = d
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    help="eval_results dir(s), optionally labelled: fam=<dir> ctrl=<dir>")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--latex", default=None)
    ap.add_argument("--margin-min", type=float, default=MARGIN_MIN)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        label, _, d = spec.rpartition("=")
        runs.append((label or Path(d).parent.parent.name, Path(d)))

    # model rows: base = first run's first dump; per run, trained = last dump
    rows = {}           # row label -> {(structure,k): {seed: {...}}}
    for i, (label, evdir) in enumerate(runs):
        dumps = sorted(evdir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
        dumps = [d for d in dumps if d.stem.isdigit()]
        if not dumps:
            print(f"!! no eval dumps in {evdir}"); continue
        if i == 0:
            data, miss = read_dump(dumps[0])
            rows[f"base (step {dumps[0].stem})"] = data
            print(f"[{label}] base   step {dumps[0].stem}  miss={miss:.3f}")
        data, miss = read_dump(dumps[-1])
        rows[f"{label} (step {dumps[-1].stem})"] = data
        print(f"[{label}] trained step {dumps[-1].stem}  miss={miss:.3f}")

    first = next(iter(rows.values()))
    cities = defaultdict(set)
    ks_seen = sorted({k for (_, k) in first})
    for (st, _k), per in first.items():
        cities[st].update(per.keys())
    cities = {st: sorted(s) for st, s in cities.items()}
    print(f"cities/structure: { {st: len(s) for st, s in cities.items()} }  ks={ks_seen}")

    print("computing references on the identical episodes ...")
    refs = compute_refs(cities, tuple(ks_seen))

    # identifiability margins per cell, and each structure's informative ks
    margins, info_ks = {}, {}
    for st in TEST_NAMES:
        info_ks[st] = []
        for k in ks_seen:
            o = summarize(refs.get(("oracle", st, k), {}))
            b = summarize(refs.get(("prior_blind", st, k), {}))
            m = b["pi"] - o["pi"] if ("pi" in o and "pi" in b) else float("nan")
            margins[(st, k)] = m
            if np.isfinite(m) and m >= args.margin_min:
                info_ks[st].append(k)
        if not info_ks[st]:
            info_ks[st] = list(ks_seen)     # degenerate world: keep everything but say so
            print(f"!! {st}: NO informative prefix (margin < {args.margin_min} everywhere)")

    # ── per-cell detail ──────────────────────────────────────────────────────────────────────────
    for st in TEST_NAMES:
        print(f"\n════ {st} ════  (informative ks: {info_ks[st]}; margin per k: "
              + " ".join(f"k{k}:{margins[(st,k)]:+.2f}" for k in ks_seen) + ")")
        for k in ks_seen:
            tag = "" if k in info_ks[st] else "   [excluded: no g signal]"
            print(f"  k={k}{tag}")
            for label, data in rows.items():
                per = data.get((st, k), {})
                if per:
                    print(f"    {label:24s} {_fmt(summarize(per))}")
            for rname in ("oracle", "prior_blind", "naive_freq", "fixed_0.55", "fixed_0.00"):
                per = refs.get((rname, st, k), {})
                if per:
                    nice = {"fixed_0.55": "always_prior", "fixed_0.00": "persistence"}.get(rname, rname)
                    print(f"    {nice:24s} {_fmt(summarize(per))}")

    # ── headline: per structure over informative ks, then TEST mean ────────────────────────────
    def headline(get_cell):
        per_struct, test_pi, test_eps, test_rho = {}, [], [], []
        for st in TEST_NAMES:
            pooled = get_cell(st, info_ks[st])
            if not pooled:
                continue
            s = summarize(pooled)
            per_struct[st] = s
            test_pi.append(s["pi"])
            if "eps" in s:
                test_eps.append(s["eps"]); test_rho.append(s.get("rho", float("nan")))
        agg = {"pi": float(np.mean(test_pi)) if test_pi else float("nan"),
               "eps": float(np.mean(test_eps)) if test_eps else float("nan"),
               "rho": float(np.nanmean(test_rho)) if test_rho else float("nan")}
        return per_struct, agg

    print("\n════ HEADLINE (informative ks only; eps/rho over recover-structures) ════")
    all_lines = []
    model_rows = [(label, (lambda st, ks, data=data: _pool(data, st, ks))) for label, data in rows.items()]
    ref_rows = [(nice, (lambda st, ks, rname=rname: _pool(
                    {(s, k): v for (rn, s, k), v in refs.items() if rn == rname}, st, ks)))
                for rname, nice in (("oracle", "oracle"), ("prior_blind", "prior_blind"),
                                    ("naive_freq", "naive_freq"), ("fixed_0.55", "always_prior"),
                                    ("fixed_0.00", "persistence"))]
    # best fixed gamma: chosen per structure by pi on its informative ks
    def best_fixed_cell(st, ks):
        best = None
        for gam in GAMMAS:
            pooled = _pool({(s, k): v for (rn, s, k), v in refs.items() if rn == f"fixed_{gam:.2f}"}, st, ks)
            if pooled:
                s = summarize(pooled)
                if best is None or s["pi"] < best[1]["pi"]:
                    best = (gam, s, pooled)
        return best

    for label, get in model_rows + ref_rows:
        per_struct, agg = headline(get)
        cells = "  ".join(f"{st}:{per_struct[st]['pi']:.2f}" for st in TEST_NAMES if st in per_struct)
        line = (f"  {label:24s} PI={agg['pi']:.2f}  EPS={agg['eps']:.3f}  RHO={agg['rho']:.3f}   [{cells}]"
                if np.isfinite(agg["eps"]) else f"  {label:24s} PI={agg['pi']:.2f}   [{cells}]")
        print(line); all_lines.append((label, agg, per_struct))
    bf = {}
    for st in TEST_NAMES:
        b = best_fixed_cell(st, info_ks[st])
        if b:
            bf[st] = b
    if bf:
        pis = [b[1]["pi"] for b in bf.values()]
        gams = " ".join(f"{st}:γ={b[0]:.2f}" for st, b in bf.items())
        print(f"  {'best_fixed (per-struct)':24s} PI={np.mean(pis):.2f}   [{gams}]")

    print("\nVERDICT RULE: a run shows transfer-as-world-modeling only if, vs base, PI improves,"
          "\nPI beats always_prior AND best_fixed, and RHO does not collapse toward 0."
          "\nA run that matches the control run's numbers learned format, not structure.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["row", "structure", "k", "informative", "pi", "eps", "rho", "beta", "mghat", "sghat", "n"])
            for label, data in rows.items():
                for st in TEST_NAMES:
                    for k in ks_seen:
                        per = data.get((st, k), {})
                        if not per:
                            continue
                        s = summarize(per)
                        w.writerow([label, st, k, int(k in info_ks[st])] +
                                   [round(s[c], 4) if isinstance(s.get(c), float) and np.isfinite(s[c]) else ""
                                    for c in ("pi", "eps", "rho", "beta", "mghat", "sghat")] + [s["n"]])
        print(f"wrote {args.csv}")

    if args.latex:
        with open(args.latex, "w") as fh:
            fh.write("% generated by transfer_report.py -- eps/rho/pi transfer table\n")
            fh.write("\\begin{tabular}{l ccc}\n\\toprule\n")
            fh.write("Forecaster & $\\pi\\downarrow$ (poll pts) & $\\varepsilon\\downarrow$ & $\\rho\\uparrow$ \\\\\n\\midrule\n")
            for label, agg, _ in all_lines:
                eps = f"{agg['eps']:.2f}" if np.isfinite(agg["eps"]) else "---"
                rho = f"{agg['rho']:.2f}" if np.isfinite(agg["rho"]) else "---"
                fh.write(f"{label} & {agg['pi']:.2f} & {eps} & {rho} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")
        print(f"wrote {args.latex}")


if __name__ == "__main__":
    main()
