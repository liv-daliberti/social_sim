"""metrics.py — reference forecasters + the pre-training baseline harness for the LG DAG family.

The two headline metrics (both lower-is-better), aggregated per structure and over the TRAIN / TEST
splits:
  - forecast error  : |y_hat_{k+1} - E[y_{k+1}]|  (poll points; the universal transfer metric)
  - recovery error  : |g_hat - g_true|            (only where a one-step input->output gain exists)

Reference forecasters bracket any model:
  - ORACLE       : Kalman filter that knows the structure and marginalizes the hidden gain g. Ceiling.
  - PRIOR-BLIND  : knows the structure but fixes g at the prior mean (0.55) and never infers it --
                   isolates the *value of inferring the latent* (oracle - blind gap).
  - NAIVE-ARX    : structure-AGNOSTIC data fit y_t ~ c + phi*y_{t-1} + b*u_t (OLS on the history);
                   the fair statistical bar a trained model should beat. Needs k>=3.
  - PERSISTENCE  : predict last output y_k. Simplest data-only floor.
For recovery: ORACLE (posterior-mean implied gain), NAIVE-FREQ (through-origin OLS of Delta y on the
probe input, exp2's naive frequentist generalized), and PRIOR (implied gain at g=0.55).
"""
from __future__ import annotations

import numpy as np

from lg_dag import (compute_bayes, generate_episode, true_next_output, filter_all, forecast_from,
                    oracle_recovery, true_recovery_gain, SHOCKS)
from catalog import CATALOG, TRAIN, TEST, BY_NAME

PRIOR_G = 0.55


# ── forecast references: predict y_{k+1} given history U[:k], Y[:k] and next input u_next ─────────
def f_oracle(st, U, Y, u_next, k):
    return compute_bayes(st, U, Y, u_next)["pred"][st.probe_output]


def f_prior_blind(st, U, Y, u_next, k):
    return compute_bayes(st, U, Y, u_next, g_grid=np.array([PRIOR_G]))["pred"][st.probe_output]


def f_persist(st, U, Y, u_next, k):
    return float(Y[k - 1][st.probe_output]) if k >= 1 else st.baseline


def f_naive_freq(st, U, Y, u_next, k):
    """Data-only 'naive frequentist' forecast (exp2's bar, generalized): estimate the one-step
    input->output response by through-origin OLS of Delta y on the probe input, then predict
    y_k + b_hat * u_next[probe]. One parameter, so it stays stable on sparse data (unlike a full
    ARX fit) while still over-reading a noisy response the way exp2's frequentist does. Needs k>=2."""
    po, pi = st.probe_output, st.probe_input
    if k < 2:
        return None
    du = U[1:k, pi]; dy = np.diff(Y[:k, po])
    denom = float((du * du).sum())
    if denom < 1e-9:
        return None
    b = float((du * dy).sum() / denom)
    return float(np.clip(Y[k - 1, po] + b * u_next[pi], 0, 100))


# ── recovery references (only where st.recover): implied one-step input->output gain ─────────────
def r_oracle(st, U, Y, k):
    return oracle_recovery(st, U, Y)


def r_naive_freq(st, U, Y, k):
    """exp2's naive frequentist: through-origin OLS of Delta y on the probe input over the history."""
    if k < 2:
        return None
    po, pi = st.probe_output, st.probe_input
    du = U[1:k, pi]; dy = np.diff(Y[:k, po])
    denom = float((du * du).sum())
    return None if denom < 1e-9 else float((du * dy).sum() / denom)


def r_prior(st, U, Y, k):
    return true_recovery_gain(st, PRIOR_G)


F_REFS = {"oracle": f_oracle, "prior_blind": f_prior_blind, "naive_freq": f_naive_freq, "persist": f_persist}
R_REFS = {"oracle": r_oracle, "naive_freq": r_naive_freq, "prior": r_prior}


# ── per-structure evaluation ─────────────────────────────────────────────────────────────────────
def eval_structure(st, N=150, ks=(1, 3, 5), seed0=20000):
    """Mean forecast error and (where defined) recovery error for every reference, at each k.
    The full-grid Kalman filter is computed ONCE per (episode, k) and reused for both the oracle
    forecast and the oracle recovery."""
    po = st.probe_output
    fe = {ref: {k: [] for k in ks} for ref in F_REFS}
    re = {ref: {k: [] for k in ks} for ref in R_REFS}
    for i in range(N):
        ep = generate_episode(st, seed=seed0 + i, T=10)
        U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
        for k in ks:
            u_next = U[k]
            tgt = true_next_output(st, S, k, g, u_next)[po]
            filt = filter_all(st, U[:k], Y[:k])                       # expensive, computed once
            fe["oracle"][k].append(abs(forecast_from(st, filt, u_next)[po] - tgt))
            fe["prior_blind"][k].append(abs(f_prior_blind(st, U[:k], Y[:k], u_next, k) - tgt))
            nvf = f_naive_freq(st, U[:k], Y[:k], u_next, k)
            if nvf is not None:
                fe["naive_freq"][k].append(abs(nvf - tgt))
            fe["persist"][k].append(abs(f_persist(st, U[:k], Y[:k], u_next, k) - tgt))
            if st.recover:
                gt = true_recovery_gain(st, g)
                ig = oracle_recovery(st, U[:k], Y[:k], filt=filt)     # reuse the same filter
                if ig is not None:
                    re["oracle"][k].append(abs(ig - gt))
                nf = r_naive_freq(st, U[:k], Y[:k], k)
                if nf is not None:
                    re["naive_freq"][k].append(abs(nf - gt))
                re["prior"][k].append(abs(r_prior(st, U[:k], Y[:k], k) - gt))
    agg = lambda d: {ref: {k: (round(float(np.mean(v)), 3) if v else None) for k, v in kv.items()}
                     for ref, kv in d.items()}
    return {"forecast": agg(fe), "recovery": agg(re), "recover": st.recover}


def _mean_over(structs, key, ref, k, results):
    vals = [results[s.name][key][ref][k] for s in structs
            if results[s.name][key][ref][k] is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def main(N=200):
    results = {st.name: eval_structure(st, N=N) for st in CATALOG}

    def fmt(v, w=5):
        return (f"{v:>{w}.2f}" if isinstance(v, float) else f"{'--':>{w}}")

    print(f"\n=== PRE-TRAINING BASELINE — forecast error |y_hat - E[y]| (poll pts, k=3) ===")
    print(f"{'structure':18s} {'split':5s} {'oracle':>6s} {'blind':>6s} {'n.freq':>6s} {'persist':>7s}")
    for st in CATALOG:
        r = results[st.name]["forecast"]; sp = "TEST" if st in TEST else "train"
        print(f"{st.name:18s} {sp:5s} {fmt(r['oracle'][3])} {fmt(r['prior_blind'][3])} "
              f"{fmt(r['naive_freq'][3])} {fmt(r['persist'][3],7)}")
    print("-" * 52)
    for lbl, grp in (("TRAIN mean", TRAIN), ("TEST mean", TEST)):
        print(f"{lbl:18s} {'':5s} "
              f"{fmt(_mean_over(grp,'forecast','oracle',3,results),6)} "
              f"{fmt(_mean_over(grp,'forecast','prior_blind',3,results),6)} "
              f"{fmt(_mean_over(grp,'forecast','naive_freq',3,results),6)} "
              f"{fmt(_mean_over(grp,'forecast','persist',3,results),7)}")

    print(f"\n=== recovery error |g_hat - g| (k=3; structures with a one-step gain) ===")
    print(f"{'structure':18s} {'split':5s} {'oracle':>6s} {'naive_freq':>10s} {'prior':>6s}")
    for st in CATALOG:
        if not results[st.name]["recover"]:
            continue
        r = results[st.name]["recovery"]; sp = "TEST" if st in TEST else "train"
        print(f"{st.name:18s} {sp:5s} {fmt(r['oracle'][3])} {fmt(r['naive_freq'][3],10)} {fmt(r['prior'][3])}")
    rec_tr = [s for s in TRAIN if s.recover]; rec_te = [s for s in TEST if s.recover]
    print("-" * 48)
    for lbl, grp in (("TRAIN mean", rec_tr), ("TEST mean", rec_te)):
        if grp:
            print(f"{lbl:18s} {'':5s} {fmt(_mean_over(grp,'recovery','oracle',3,results),6)} "
                  f"{fmt(_mean_over(grp,'recovery','naive_freq',3,results),10)} "
                  f"{fmt(_mean_over(grp,'recovery','prior',3,results),6)}")
    return results


if __name__ == "__main__":
    main()
