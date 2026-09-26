#!/usr/bin/env python3
"""Read OAT held-out eval dumps and report Experiment-2 latent recovery.

OAT's base evaluate writes ``<save_path>/eval_results/<step>.json`` at every eval:
one record per held-out prompt with the model ``output`` and the ``reference``
(our JSON carrying seed, test_news, last_poll, g).  Because the held-out set has
four shocks per city (make_dataset.py), we recover each city's implied gain as the
four-shock slope ghat = Cov(shock, pred)/Var(shock) -- the exact Experiment-2
read-out -- then report recovery rho(ghat, g), tracking slope beta, and mean ghat.

Point it at the step-0 dump (base model, before training) and the final dump
(after training) to read the before/after recovery that answers Experiment 3.

Usage:
    python recovery_from_eval.py <eval_results_dir>
    python recovery_from_eval.py --before <0.json> --after <NN.json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_PRED_RE = re.compile(r'"?predicted_poll"?\s*:\s*"?(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def _parse_pred(text):
    # OAT dumps `output` as a one-element list of strings.
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    m = None
    for m in _PRED_RE.finditer(text):
        pass
    return max(0.0, min(100.0, float(m.group(1)))) if m else None


def recovery(json_path: Path) -> dict:
    records = json.loads(Path(json_path).read_text())
    by_city: dict[int, dict] = {}
    n_unparsed = 0
    for rec in records:
        ref = json.loads(rec["reference"])
        pred = _parse_pred(rec["output"])
        c = by_city.setdefault(ref["seed"], {"g": ref["g"], "x": [], "y": []})
        if pred is None:
            n_unparsed += 1
            continue
        c["x"].append(float(ref["test_news"]))
        c["y"].append(pred)

    ghat, gtrue = [], []
    for c in by_city.values():
        x, y = np.array(c["x"]), np.array(c["y"])
        if len(x) < 2 or np.var(x) == 0:
            continue
        ghat.append(float(np.cov(x, y, bias=True)[0, 1] / np.var(x)))
        gtrue.append(c["g"])
    ghat, gtrue = np.array(ghat), np.array(gtrue)

    rho = float(np.corrcoef(ghat, gtrue)[0, 1]) if len(ghat) > 1 else float("nan")
    beta = float(np.cov(gtrue, ghat, bias=True)[0, 1] / np.var(gtrue)) if len(ghat) > 1 else float("nan")
    # eps = per-instance calibration ERROR |ghat - g| (the paper's headline metric);
    # bias = signed mean(ghat - g) (positive = overshoots the true gain).
    eps = float(np.mean(np.abs(ghat - gtrue))) if len(ghat) else float("nan")
    bias = float(np.mean(ghat - gtrue)) if len(ghat) else float("nan")
    return {
        "file": str(json_path),
        "n_cities": len(ghat),
        "rho": rho,
        "eps": eps,
        "bias": bias,
        "beta": beta,
        "mean_ghat": float(ghat.mean()) if len(ghat) else float("nan"),
        "mean_gtrue": float(gtrue.mean()) if len(gtrue) else float("nan"),
        "unparsed_rows": n_unparsed,
    }


def _fmt(d: dict) -> str:
    return (f"  n={d['n_cities']:>4}  eps(|ghat-g|)={d['eps']:.3f}  rho={d['rho']:+.3f}  "
            f"bias={d['bias']:+.3f}  beta={d['beta']:+.3f}  "
            f"mean_ghat={d['mean_ghat']:.3f} (true {d['mean_gtrue']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", nargs="?", type=Path,
                    help="eval_results dir; uses lowest-step as before, highest as after")
    ap.add_argument("--before", type=Path)
    ap.add_argument("--after", type=Path)
    args = ap.parse_args()

    if args.eval_dir:
        files = sorted(args.eval_dir.glob("*.json"),
                       key=lambda p: int(re.match(r"(\d+)", p.stem).group(1))
                       if re.match(r"(\d+)", p.stem) else 0)
        if not files:
            raise SystemExit(f"no *.json in {args.eval_dir}")
        before, after = files[0], files[-1]
    else:
        before, after = args.before, args.after

    print("Experiment-2 latent recovery from OAT held-out eval dumps\n")
    if before:
        b = recovery(before)
        print(f"BEFORE  ({Path(before).name}):\n" + _fmt(b))
    if after and Path(after) != Path(before or ""):
        a = recovery(after)
        print(f"AFTER   ({Path(after).name}):\n" + _fmt(a))
        if before:
            print(f"\n  delta rho = {a['rho'] - b['rho']:+.3f}")
    print("\n(Exp-2 same-information forecaster bar at the horizon: rho ~ 0.89.)")


if __name__ == "__main__":
    main()
