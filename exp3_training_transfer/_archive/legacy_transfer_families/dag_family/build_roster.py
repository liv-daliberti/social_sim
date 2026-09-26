"""build_roster.py — aggregate the untrained-model DAG-family sweep into the Exp-3 roster table.

Reads every data/dag_eval/dageval_<slug>_<date>.jsonl (one per model, latest date per slug),
computes each model's forecast-MAE and recovery-MAE at a chosen k, aggregated over a chosen set of
structures (default the 4 held-out TEST), and places them against the analytic oracle / naive-freq /
persistence bracket. Emits a console table and a LaTeX table.

    python3 build_roster.py                       # TEST structures, k=3
    python3 build_roster.py --structures all --k 3
    python3 build_roster.py --latex reports/roster_test.tex
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import TEST, TRAIN, CATALOG, BY_NAME
from eval import eval_forecaster, fc_oracle, fc_naive_freq, fc_persist

HERE = Path(__file__).resolve().parent
EVDIR = HERE / "data" / "dag_eval"

# size-ordered display roster (slug -> pretty label); slugs are model.replace(':','-')
ROSTER = [
    ("gpt-5.4",            "GPT-5.4"),
    ("claude-opus-4-8",    "Claude Opus 4.8"),
    ("DeepSeek-V4-Pro",    "DeepSeek V4-Pro"),
    ("qwen2.5-72b",        "Qwen2.5-72B"),
    ("llama3.3-70b",       "Llama-3.3-70B"),
    ("llama3.1-70b",       "Llama-3.1-70B"),
    ("qwen2.5-32b",        "Qwen2.5-32B"),
    ("gemma2-27b",         "Gemma-2-27B"),
    ("mistral-small-24b",  "Mistral-Small-24B"),
    ("qwen2.5-14b",        "Qwen2.5-14B"),
    ("gemma2-9b",          "Gemma-2-9B"),
    ("llama3.1-8b",        "Llama-3.1-8B"),
    ("qwen2.5-7b",         "Qwen2.5-7B"),
]


def pick(spec):
    if spec == "test":  return TEST
    if spec == "train": return TRAIN
    if spec == "all":   return CATALOG
    return [BY_NAME[n] for n in spec.split(",")]


def latest_file(slug):
    fs = sorted(EVDIR.glob(f"dageval_{slug}_*.jsonl"))
    return fs[-1] if fs else None


def model_scores(rows, structs, k):
    """(forecast-MAE, recovery-MAE, parse-miss) aggregated over `structs` at prefix k."""
    names = {s.name for s in structs}
    rec_names = {s.name for s in structs if s.recover}
    fes, res, seen, complete = [], [], 0, 0
    for r in rows:
        if r["k"] != k or r["structure"] not in names:
            continue
        seen += 1
        pr, mu = r["preds"], r["mu"]
        ok = [(p, m) for p, m in zip(pr, mu) if p is not None]
        if len(ok) == len(pr):
            complete += 1
        if ok:
            fes.append(float(np.mean([abs(p - m) for p, m in ok])))
        if r["structure"] in rec_names and r.get("true_gain") is not None and all(p is not None for p in pr):
            xs = np.asarray(r["shocks"], float); ys = np.asarray(pr, float)
            gh = float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs))
            res.append(abs(gh - r["true_gain"]))
    miss = 1 - complete / max(seen, 1)
    return (float(np.mean(fes)) if fes else None,
            float(np.mean(res)) if res else None, miss, seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", default="test")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--latex", default=None)
    ap.add_argument("--min-n", type=int, default=8,
                    help="omit models from the LaTeX table with fewer than this many scored episodes")
    args = ap.parse_args()

    structs = pick(args.structures)
    k = args.k

    # analytic reference bracket on the same structures/k (mean over structs; recovery over recover=True)
    def ref_agg(fn):
        per = [eval_forecaster(s, fn, N=25, ks=[k]) for s in structs]
        f = float(np.mean([p["fmae"][k] for p in per]))
        rvals = [p["rmae"][k] for s, p in zip(structs, per) if s.recover and p["rmae"].get(k) is not None]
        return f, (float(np.mean(rvals)) if rvals else None)
    refs = {name: ref_agg(fn) for name, fn in
            (("oracle", fc_oracle), ("naive", fc_naive_freq), ("persist", fc_persist))}

    # models present
    rows_by_slug = {}
    for slug, _ in ROSTER:
        fp = latest_file(slug)
        if fp:
            rows_by_slug[slug] = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]

    def fmt(f, r):
        a = f"{f:.2f}" if f is not None else "  -- "
        b = f"{r:.2f}" if r is not None else " -- "
        return f"{a} / {b}"

    print(f"\nUntrained roster on {args.structures} structures ({[s.name for s in structs]}), k={k}")
    print(f"cols = forecast-MAE / recovery-MAE   (lower better)\n")
    print(f"{'model':20s} {'fMAE / recMAE':>16s}   miss  n")
    print(f"{'-'*20} {'-'*16}   ----  --")
    print(f"{'ORACLE (ceiling)':20s} {fmt(*refs['oracle']):>16s}")
    for slug, label in ROSTER:
        if slug not in rows_by_slug:
            continue
        f, r, miss, n = model_scores(rows_by_slug[slug], structs, k)
        print(f"{label:20s} {fmt(f, r):>16s}   {miss:.2f}  {n}")
    print(f"{'NAIVE (stat bar)':20s} {fmt(*refs['naive']):>16s}")
    print(f"{'PERSISTENCE (floor)':20s} {fmt(refs['persist'][0], None):>16s}")

    if args.latex:
        lines = [
            r"% auto-generated by build_roster.py — untrained roster on the DAG family",
            r"\begin{tabular}{l cc}", r"\toprule",
            r"Model & forecast MAE & recovery MAE \\",
            r"\midrule",
            rf"\emph{{oracle}} (ceiling) & ${refs['oracle'][0]:.2f}$ & "
            + (rf"${refs['oracle'][1]:.2f}$ \\" if refs['oracle'][1] is not None else r"--- \\"),
        ]
        for slug, label in ROSTER:
            if slug not in rows_by_slug:
                continue
            f, r, miss, n = model_scores(rows_by_slug[slug], structs, k)
            if f is None or n < args.min_n:   # not yet evaluated / too few episodes — omit row
                continue
            rc = f"${r:.2f}$" if r is not None else "---"
            lines.append(rf"{label} & ${f:.2f}$ & {rc} \\")
        lines += [
            rf"\emph{{naive frequentist}} & ${refs['naive'][0]:.2f}$ & "
            + (rf"${refs['naive'][1]:.2f}$ \\" if refs['naive'][1] is not None else r"--- \\"),
            rf"\emph{{persistence}} (floor) & ${refs['persist'][0]:.2f}$ & --- \\",
            r"\bottomrule", r"\end{tabular}",
        ]
        outp = Path(args.latex); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(lines) + "\n")
        print(f"\nwrote LaTeX -> {outp}")


if __name__ == "__main__":
    main()
