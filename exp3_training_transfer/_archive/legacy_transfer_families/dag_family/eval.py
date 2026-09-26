"""eval.py — score any forecaster on the LG DAG family via the four-shock probe (exp2's read-out).

!! LEGACY ONE-STEP PROBE (pre-2026-07-08). This module asks every world for NEXT week's poll, which is
degenerate in the delayed worlds: the pulse has not reached the poll, all four shock targets coincide,
and the recovered slope carries no information about g. It also reads `struct.recover`, the legacy
"one-step gain defined" flag, which disagrees with catalog.RECOVERS_AT_HSTAR for 6 structures.
It is retained only for the open-weight roster path (run_dag_eval.py / build_roster.py), which has not
yet been re-run on the current family. The instrument the paper reports is the h* probe: see
lg_dag.first_response_horizon / true_output_at_horizon, make_dataset_dag.py, and transfer_report.py.

A *forecaster* is a batch function  f(struct, U, Y, next_us) -> [predicted_probe_poll, ...]  that,
given a history, returns its forecast of the probe poll for each of several next-week news vectors.
The reference forecasters (oracle, prior-blind, naive-freq, persistence) and an LLM (prompt -> call
-> parse) all share this interface, so they are scored identically:

  at prefix k we query the four held-out shocks s in {-10,-5,+5,+10} on the probe news channel and
    forecast error  = mean_s |pred(s) - mu_k(s)|   (poll pts; mu_k = ground-truth expected poll)
    recovery ghat   = slope_s(pred(s))              (where a one-step gain exists)

This is the same instrument as Experiment 2, so an LLM's numbers drop straight into the reference
bracket from metrics.py. (Forecast error here is the four-shock version, matching the LLM's queries;
it agrees with the oracle's exp2 poll-MAE.)
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from lg_dag import filter_all, forecast_from, generate_episode, true_next_output, true_recovery_gain, SHOCKS
from catalog import CATALOG, TRAIN, TEST
from prompt import build_prompt, parse_forecast

PRIOR_G = 0.55


# ── reference forecasters (batch: return one probe-poll forecast per next-week news vector) ───────
def fc_oracle(st, U, Y, next_us, grid=None):
    filt = filter_all(st, U, Y) if grid is None else filter_all(st, U, Y, grid)
    po = st.probe_output
    return [float(forecast_from(st, filt, nu)[po]) for nu in next_us]


def fc_prior_blind(st, U, Y, next_us):
    return fc_oracle(st, U, Y, next_us, grid=np.array([PRIOR_G]))


def fc_persist(st, U, Y, next_us):
    last = float(Y[len(Y) - 1][st.probe_output]) if len(Y) >= 1 else float(st.baseline)
    return [last] * len(next_us)


def fc_naive_freq(st, U, Y, next_us):
    po, pi, k = st.probe_output, st.probe_input, len(Y)
    last = float(Y[k - 1][po]) if k >= 1 else float(st.baseline)
    if k < 2:
        return [last] * len(next_us)
    du = U[1:k, pi]; dy = np.diff(Y[:k, po]); den = float((du * du).sum())
    b = 0.0 if den < 1e-9 else float((du * dy).sum() / den)
    return [float(np.clip(last + b * nu[pi], 0, 100)) for nu in next_us]


REFERENCES = {"oracle": fc_oracle, "prior_blind": fc_prior_blind,
              "naive_freq": fc_naive_freq, "persist": fc_persist}


# ── LLM forecaster: build prompt -> call -> parse, one call per shock ─────────────────────────────
def make_model_forecaster(call_model: Callable[[str], str], reason_cap: int = 400):
    def fc(st, U, Y, next_us) -> List[Optional[float]]:
        return [parse_forecast(call_model(build_prompt(st, U, Y, nu, reason_cap))) for nu in next_us]
    return fc


# ── scoring ──────────────────────────────────────────────────────────────────────────────────────
def _shock_inputs(st):
    us = []
    for s in SHOCKS:
        u = np.zeros(st.m); u[st.probe_input] = s
        us.append(u)
    return us


def score_episode(st, forecaster, ep, k):
    U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
    next_us = _shock_inputs(st)
    mu = np.array([true_next_output(st, S, k, g, u)[st.probe_output] for u in next_us])
    preds = forecaster(st, U[:k], Y[:k], next_us)
    errs = [abs(p - m) for p, m in zip(preds, mu) if p is not None]
    fmae = float(np.mean(errs)) if errs else None
    ghat = None
    if st.recover and all(p is not None for p in preds):
        xs = np.asarray(SHOCKS, float); ys = np.asarray(preds, float)
        ghat = float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs))
    return fmae, ghat


def eval_forecaster(st, forecaster, N=150, ks=(1, 3, 5), seed0=20000):
    fm = {k: [] for k in ks}; rm = {k: [] for k in ks}
    for i in range(N):
        ep = generate_episode(st, seed=seed0 + i, T=10)
        gt = true_recovery_gain(st, ep["g"]) if st.recover else None
        for k in ks:
            fmae, ghat = score_episode(st, forecaster, ep, k)
            if fmae is not None:
                fm[k].append(fmae)
            if ghat is not None and gt is not None:
                rm[k].append(abs(ghat - gt))
    r = lambda d: {k: (round(float(np.mean(v)), 3) if v else None) for k, v in d.items()}
    return {"fmae": r(fm), "rmae": r(rm)}


def _mean(structs, res, metric, k):
    vals = [res[s.name][metric][k] for s in structs if res[s.name][metric][k] is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def run_references(N=150, ks=(3,)):
    """Baseline bracket via the four-shock instrument (sanity check; agrees with metrics.py)."""
    res = {}
    for st in CATALOG:
        res[st.name] = {ref: eval_forecaster(st, fn, N=N, ks=ks) for ref, fn in REFERENCES.items()}
    k = ks[0]
    print(f"\n=== four-shock BASELINE — forecast err (poll pts) / recovery |ghat-g| at k={k} ===")
    print(f"{'':18s} {'oracle':>13s} {'prior_blind':>13s} {'naive_freq':>13s} {'persist':>8s}")
    for st in CATALOG:
        r = res[st.name]; sp = "TEST" if st in TEST else "    "
        def cell(ref):
            f = r[ref]["fmae"][k]; g = r[ref]["rmae"][k]
            return f"{f:>5.2f}" + (f"/{g:.2f}" if g is not None else "/ -- ")
        print(f"{st.name:14s}{sp:4s} {cell('oracle'):>13s} {cell('prior_blind'):>13s} "
              f"{cell('naive_freq'):>13s} {r['persist']['fmae'][k]:>8.2f}")
    print("-" * 70)
    def splitmean(grp, ref, metric):
        vals = [res[s.name][ref][metric][k] for s in grp if res[s.name][ref][metric][k] is not None]
        return round(float(np.mean(vals)), 2) if vals else None
    for lbl, grp in (("TRAIN", TRAIN), ("TEST", TEST)):
        print(f"  {lbl} forecast  oracle={splitmean(grp,'oracle','fmae')}  blind={splitmean(grp,'prior_blind','fmae')}  "
              f"naive={splitmean(grp,'naive_freq','fmae')}  persist={splitmean(grp,'persist','fmae')}")
        print(f"  {lbl} recovery  oracle={splitmean(grp,'oracle','rmae')}  naive={splitmean(grp,'naive_freq','rmae')}")
    return res


if __name__ == "__main__":
    run_references()
