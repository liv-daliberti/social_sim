"""cross_oracle_check.py — observational-distinctness check over the 20 LG-DAG structures.

Two structures are REDUNDANT (observationally equivalent) iff a forecaster/oracle that only sees the
observables (news, polls) cannot tell them apart. Operationalized: apply structure A's Kalman oracle
to structure B's DATA and measure next-poll forecast MAE. If A's (mis-specified) oracle forecasts B's
data about as well as B's OWN oracle, A and B are observationally equivalent.

Cross-application is only defined within an interface group (same # observed news m, # polls p) --
structures with different (m,p) are trivially distinct. For each B we report B's native oracle MAE vs
the best WRONG oracle; a large positive margin means B is a genuinely distinct forecasting problem.
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from catalog import CATALOG, BY_NAME
from lg_dag import generate_episode, filter_all, forecast_from, true_next_output

N = 50            # episodes per structure
T = 10
KS = (1, 3, 5)    # prefix lengths
SEED0 = 90000

# pre-generate each structure's held-out data once
DATA = {s.name: [generate_episode(s, seed=SEED0 + i, T=T) for i in range(N)] for s in CATALOG}


def cross_mae(A, B):
    """MAE of structure A's oracle forecasting structure B's data (natural next-news forecast)."""
    errs = []
    for ep in DATA[B.name]:
        U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
        for k in KS:
            if k >= len(U):
                continue
            u_next = U[k]
            tgt = float(true_next_output(B, S, k, g, u_next)[B.probe_output])
            pred = float(forecast_from(A, filter_all(A, U[:k], Y[:k]), u_next)[A.probe_output])
            errs.append(abs(pred - tgt))
    return float(np.mean(errs))


def main():
    groups = defaultdict(list)
    for s in CATALOG:
        groups[(s.m, s.p)].append(s)

    print(f"cross-oracle distinctness  (N={N} episodes/struct, prefixes k={KS}, natural next-poll MAE)\n")
    redundant = []
    for (m, p), structs in sorted(groups.items()):
        if len(structs) < 2:
            print(f"[m={m} p={p}]  {structs[0].name}: unique interface -> trivially distinct\n")
            continue
        names = [s.name for s in structs]
        M = np.array([[cross_mae(A, B) for B in structs] for A in structs])  # M[i,j]=A_i on B_j data
        print(f"[interface m={m} p={p}]  native oracle vs BEST WRONG oracle "
              f"(margin = how much worse the best other-structure oracle is):")
        print(f"  {'structure':16s} {'native':>7} {'best-wrong':>11} {'(which)':>16} {'margin':>8}")
        for j, B in enumerate(structs):
            nat = M[j, j]
            others = sorted((M[i, j], names[i]) for i in range(len(structs)) if i != j)
            bw, who = others[0]
            margin = 100 * (bw - nat) / nat
            tag = "  <== REDUNDANT" if margin < 5 else ("  (borderline)" if margin < 12 else "")
            if margin < 5:
                redundant.append((names[j], who, nat, bw, margin))
            print(f"  {names[j]:16s} {nat:7.2f} {bw:11.2f} {who:>16} {margin:6.1f}%{tag}")
        print()

    # spotlight the confounder specifically
    if "confounder_gain" in BY_NAME and "chain1_gain" in BY_NAME:
        cf, direct = BY_NAME["confounder_gain"], BY_NAME["chain1_gain"]
        nat = cross_mae(cf, cf); dir_on_cf = cross_mae(direct, cf)
        print(f"CONFOUNDER SPOTLIGHT: confounder's own oracle {nat:.2f} vs DIRECT oracle on confounder "
              f"data {dir_on_cf:.2f}  (margin {100*(dir_on_cf-nat)/nat:.1f}%)")
        print("  -> large margin = the confounder mode IS observable (distinct); ~0 = it collapses to direct.\n")

    print("SUMMARY:", f"{len(redundant)} structure(s) look observationally redundant (margin < 5%)"
          if redundant else "no redundant structures — every structure's own oracle is meaningfully best.")
    for nm, who, nat, bw, mg in redundant:
        print(f"  - {nm}: matched by {who} ({nat:.2f} vs {bw:.2f}, {mg:.1f}%)")


if __name__ == "__main__":
    main()
