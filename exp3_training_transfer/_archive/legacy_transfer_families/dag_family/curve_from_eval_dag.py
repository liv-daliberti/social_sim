"""curve_from_eval_dag.py — learning curves from the 250-step / eval-every-25 training runs.

For each training run it reads every <step>.json in the run's eval_results dir (via read_dump from
recovery_from_eval_dag.py), computes the held-out forecast-MAE and recovery-MAE at each eval step, and
writes a tidy CSV (model, seed, step, structure, fmae, rmae; structure "_mean" = mean over held-out).
Qwen3-4B's three seeds are aggregated per step (mean +/- std) for an error band. Feeds the learning-
curve figure.

    python3 curve_from_eval_dag.py                 # all RUNS below -> reports/curves.csv + console
    python3 curve_from_eval_dag.py --csv reports/curves.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery_from_eval_dag import read_dump

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"

# !! SUPERSEDED (2026-07-09 audit). Every jobid below is pre-h*: it trained the degenerate one-step
# probe on the OLD catalog, whose `two_news_feedback` had spectral radius 1.297 (an explosive world
# sold as mean-reverting) and whose training mix still contained `skip` (resting poll ~100). Those
# dumps carry no `horizon` field. reports/curves.csv and the figures built from it (exp3_curves,
# exp3_heatmap, exp3_per_week) are stale and are NOT included in the paper -- do not regenerate them
# as current results. For the live result use plot_transfer_curves.py / transfer_report.py on the
# h* family-vs-control runs instead.
#
# jobid -> (model label, seed). Update when runs are relaunched.
RUNS = {
    # 13-world RICHER-DATA retrain (5400 train rows, k=1..5 eval, 300 steps), TEMP=1.3 REWARD_SCALE=15
    "29852388": ("Qwen3-4B", 42), "29852404": ("Qwen3-4B", 43), "29852408": ("Qwen3-4B", 44),
    "29852401": ("Qwen2.5-7B", 42), "29852405": ("Qwen2.5-7B", 43), "29852409": ("Qwen2.5-7B", 44),
    "29852402": ("Llama-3.1-8B", 42), "29852406": ("Llama-3.1-8B", 43), "29862742": ("Llama-3.1-8B", 44),
    "29862599": ("Qwen2.5-14B", 42), "29862600": ("Qwen2.5-14B", 43), "29862601": ("Qwen2.5-14B", 44),
}


def find_eval_dir(jobid):
    for d in sorted(REPORTS.glob(f"pilot_*_j{jobid}")):
        ev = list(d.glob("**/eval_results"))
        if ev:
            return ev[0]
    return None


def step_curve(evdir):
    """{step: {'fmae': mean_over_heldout, 'rmae': mean, 'miss': ..., 'per': {struct: (fmae,rmae)}}}"""
    out = {}
    for dump in sorted(evdir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9):
        if not dump.stem.isdigit():
            continue
        d = read_dump(dump)
        structs = [k for k in d if not k.startswith("_")]
        fs = [d[s]["fmae"] for s in structs if isinstance(d[s]["fmae"], float)]
        rs = [d[s]["rmae"] for s in structs if isinstance(d[s]["rmae"], float)]
        out[int(dump.stem)] = {
            "fmae": float(np.mean(fs)) if fs else None,
            "rmae": float(np.mean(rs)) if rs else None,
            "miss": d["_miss"],
            "per": {s: (d[s]["fmae"], d[s]["rmae"]) for s in structs},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPORTS / "curves.csv"))
    args = ap.parse_args()

    rows = []           # (model, seed, step, structure, fmae, rmae)
    curves = {}         # (model, seed) -> step_curve
    for jobid, (model, seed) in RUNS.items():
        ev = find_eval_dir(jobid)
        if ev is None:
            print(f"[{model} s{seed} j{jobid}] no eval dir yet")
            continue
        sc = step_curve(ev)
        curves[(model, seed)] = sc
        for step, v in sc.items():
            rows.append((model, seed, step, "_mean", v["fmae"], v["rmae"]))
            for st, (f, r) in v["per"].items():
                rows.append((model, seed, step, st, f, r))
        steps = sorted(sc)
        last = sc[steps[-1]] if steps else {}
        print(f"[{model} s{seed} j{jobid}] steps={steps}  "
              f"latest fMAE={last.get('fmae')}/rec={last.get('rmae')} (miss {last.get('miss')})")

    REPORTS.mkdir(exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seed", "step", "structure", "fmae", "rmae"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3],
                        "" if r[4] is None else round(r[4], 3),
                        "" if r[5] is None else round(r[5], 3)])
    print(f"\nwrote {len(rows)} rows -> {args.csv}")

    # per-model mean curve (4B averaged across seeds) on the held-out mean
    print("\n=== held-out mean forecast-MAE by step (seeds averaged) ===")
    by_model = defaultdict(lambda: defaultdict(list))   # model -> step -> [fmae over seeds]
    by_model_r = defaultdict(lambda: defaultdict(list))
    for (model, seed), sc in curves.items():
        for step, v in sc.items():
            if v["fmae"] is not None: by_model[model][step].append(v["fmae"])
            if v["rmae"] is not None: by_model_r[model][step].append(v["rmae"])
    for model in sorted(by_model):
        steps = sorted(by_model[model])
        cells = []
        for s in steps:
            fs = by_model[model][s]; rs = by_model_r[model].get(s, [])
            fm = np.mean(fs); rm = (np.mean(rs) if rs else float("nan"))
            band = f"±{np.std(fs):.2f}" if len(fs) > 1 else ""
            cells.append(f"s{s}:{fm:.2f}{band}/{rm:.2f}")
        print(f"  {model:14s} " + "  ".join(cells))
    print("  (ref bracket: oracle ~1.7/0.14, naive ~3.85/0.37, persistence ~4.6)")


if __name__ == "__main__":
    main()
