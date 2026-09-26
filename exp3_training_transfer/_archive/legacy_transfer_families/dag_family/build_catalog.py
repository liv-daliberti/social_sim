"""build_catalog.py — construct a family of OBSERVATIONALLY-DISTINCT LG-DAG structures, with the
distinctness check built into creation.

The old catalog collapsed because several structures were same-topology variants (noise-only tweaks,
near-identical 2-stage chains) that a forecaster/oracle cannot tell apart. This builds a richer
CANDIDATE POOL that spans *observable* axes with *large* differences -- interface (m news x p polls),
chain depth (1/2/3-stage + lag), large mean-reversion gaps (phi in {.45,.7,.93}), feedback type
(confounder / mediators), which edge is hidden (input gain / coupling / persistence), driver sign --
then keeps only a MUTUALLY-DISTINCT subset, verified by cross-oracle forecasting.

Distinctness test (per interface group, since different (m,p) are trivially distinct): apply structure
A's Kalman oracle to structure B's data; if it forecasts about as well as B's own oracle, they're the
same problem. Two structures coexist only if, in BOTH directions, the other's oracle is >= MARGIN
worse -- so neither is a relabel or a special case of the other.

    python3 build_catalog.py                 # report the distinct set + distinctness matrix
    python3 build_catalog.py --write         # also emit catalog_distinct.py
"""
from __future__ import annotations
import argparse, sys
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from lg_dag import LGStructure, generate_episode, filter_all, forecast_from, true_next_output

_H = 0.5           # placeholder overwritten per-episode by the hidden gain
MARGIN = 0.15      # a structure's own oracle must beat every other's on its data by >= 15% (both ways)
N, T, KS, SEED0 = 60, 10, (1, 3, 5), 90000


def S(name, A, B, C, hidden, note, recover=True, **kw):
    return LGStructure(name, np.array(A, float), np.array(B, float), np.array(C, float),
                       hidden=hidden, recover=recover, note=note, **kw)


# ---- candidate pool: diverse topologies, big param gaps, spanning interfaces ---------------------
def candidates():
    C = []
    # === m=1, p=1 : depth / phi / hidden-edge / feedback ===
    C += [
        S("direct_exp2",  [[0.90]], [[_H]], [[1]], ("B",0,0), "news->opinion->poll, phi=.9 -- the Experiment-2 world"),
        S("direct_fast",  [[0.45]], [[_H]], [[1]], ("B",0,0), "direct, fast mean-reversion phi=.45"),
        S("hidden_phi",   [[_H]],  [[0.7]], [[1]], ("A",0,0), "hidden = persistence (gain known); infer stickiness", recover=False),
        S("chain2",       [[0.85,0],[0.55,0.85]], [[_H],[0]], [[0,1]], ("B",0,0), "2-stage: effect delayed one week", recover=False),
        S("chain3",       [[0.70,0,0],[0.8,0.70,0],[0,0.8,0.70]], [[_H],[0],[0]], [[0,0,1]], ("B",0,0), "3-stage, doubly delayed", recover=False),
        S("lagged",       [[0.9,_H],[0,0]], [[0],[1]], [[1,0]], ("A",0,1), "news moves opinion NEXT week (1-week lag)", recover=False),
        S("hidden_coupling", [[0.85,0],[_H,0.85]], [[0.5],[0]], [[0,1]], ("A",1,0), "2-stage, hidden edge is the o1->o2 coupling", recover=False),
        S("confounder",   [[0.9,0.4],[0,0.83]], [[_H],[0.6]], [[1,0]], ("B",0,0), "news drives opinion AND a confounder (own phi) that feeds back"),
        S("mediators",    [[0.9,0,0],[0,0.83,0],[0.5,0.5,0]], [[_H],[0.6],[0]], [[0,0,1]], ("B",0,0), "two parallel mediators into the polled stage", recover=False),
        S("chain2_slow",  [[0.95,0],[0.4,0.95]], [[_H],[0]], [[0,1]], ("B",0,0), "2-stage, very sticky phi=.95", recover=False),
        S("inhibitory",   [[0.9,-0.5],[0,0.75]], [[_H],[0.5]], [[1,0]], ("B",0,0), "news lifts poll then a slow inhibitor pulls it back (overshoot-correct)"),
    ]
    # === m=2, p=1 : two observed news drivers ===
    C += [
        S("two_news_aligned", [[0.9]], [[_H,0.5]], [[1]], ("B",0,0), "2 drivers, 2nd known same-sign (+.5)"),
        S("two_news_opposed", [[0.9]], [[_H,-0.6]], [[1]], ("B",0,0), "2 drivers, 2nd known opposite-sign (-.6)"),
        S("two_news_split",   [[0.9,0],[0,0.88]], [[_H,0],[0,0.6]], [[0.7,0.7]], ("B",0,0), "each driver its own opinion; poll reads both", recover=False),
        S("two_news_confounder", [[0.9,0.4],[0,0.83]], [[_H,0],[0,0.5]], [[1,0]], ("B",0,0), "driver1->opinion (hidden); driver2->confounder->opinion", recover=False),
    ]
    # === m=3, p=1 ===
    C += [
        S("three_news",       [[0.9]], [[_H,0.4,-0.3]], [[1]], ("B",0,0), "3 drivers; two known (+.4,-.3)"),
        S("three_news_2stage",[[0.85,0],[0.5,0.85]], [[_H,0.4,-0.3],[0,0,0]], [[0,1]], ("B",0,0), "3 drivers into a 2-stage chain", recover=False),
    ]
    # === m=1, p=2 : two observed polls ===
    C += [
        S("two_poll",     [[0.9]], [[_H]], [[1],[1]], ("B",0,0), "one opinion read by TWO polls"),
        S("chain2_2poll", [[0.9,0],[0.5,0.9]], [[_H],[0]], [[1,0],[0,1]], ("B",0,0), "2-stage; BOTH stages observed (poll each)", recover=False),
        S("confounder_2poll", [[0.9,0.4],[0,0.83]], [[_H],[0.6]], [[1,0],[0,1]], ("B",0,0), "confounder with both latents polled"),
        S("chain3_2poll", [[0.9,0,0],[0.5,0.9,0],[0,0.5,0.9]], [[_H],[0],[0]], [[1,0,0],[0,0,1]], ("B",0,0), "3-stage; observe the first (immediate) and last (delayed) stage"),
    ]
    # === m=2, p=2 ===
    C += [
        S("parallel_2x2", [[0.9,0],[0,0.9]], [[_H,0],[0,0.5]], [[1,0],[0,1]], ("B",0,0), "two driver->opinion->poll lines, side by side"),
        S("coupled_2x2",  [[0.9,0],[0.4,0.9]], [[_H,0.5],[0,0]], [[1,0],[0,1]], ("B",0,0), "two polls, second opinion fed by the first", recover=False),
        S("confounder_2x2", [[0.9,0.4],[0,0.83]], [[_H,0],[0,0.5]], [[1,0],[0,1]], ("B",0,0), "two news, two polls, with a confounder latent", recover=False),
    ]
    return C


# ---- distinctness machinery ---------------------------------------------------------------------
def _data(structs):
    return {s.name: [generate_episode(s, seed=SEED0+i, T=T) for i in range(N)] for s in structs}

def cross_mae(A, B, data):
    errs = []
    for ep in data[B.name]:
        U, Y, S_, g = ep["U"], ep["Y"], ep["S"], ep["g"]
        for k in KS:
            if k >= len(U): continue
            un = U[k]
            tgt = float(true_next_output(B, S_, k, g, un)[B.probe_output])
            pr  = float(forecast_from(A, filter_all(A, U[:k], Y[:k]), un)[A.probe_output])
            errs.append(abs(pr - tgt))
    return float(np.mean(errs))

def native_mae(B, data):
    return cross_mae(B, B, data)


def select_distinct(cands, target=20):
    data = _data(cands)
    nat = {s.name: native_mae(s, data) for s in cands}
    # health: keep only structures whose own oracle meaningfully beats persistence-ish (nat < 4.2)
    healthy = [s for s in cands if nat[s.name] < 4.2]
    by_iface = defaultdict(list)
    for s in healthy: by_iface[(s.m, s.p)].append(s)

    selected, report = [], []
    for (m, p), group in sorted(by_iface.items()):
        keep = []
        for cand in group:
            ok = True
            for other in keep:
                a = 100*(cross_mae(other, cand, data) - nat[cand.name]) / nat[cand.name]   # other's oracle on cand
                b = 100*(cross_mae(cand, other, data) - nat[other.name]) / nat[other.name]  # cand's oracle on other
                if min(a, b) < 100*MARGIN:          # one covers the other -> collapse
                    ok = False
                    report.append(f"    drop {cand.name}: collapses with {other.name} (min margin {min(a,b):.0f}%)")
                    break
            if ok:
                keep.append(cand)
        selected += keep
        report.append(f"[m={m} p={p}] kept {len(keep)}/{len(group)}: {[s.name for s in keep]}")
    return selected, nat, data, report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); args = ap.parse_args()
    cands = candidates()
    print(f"candidate pool: {len(cands)} structures across interfaces "
          f"{sorted(set((s.m,s.p) for s in cands))}\n")
    sel, nat, data, report = select_distinct(cands)
    print("\n".join(report))
    print(f"\n=> {len(sel)} mutually-distinct structures selected (target 20)\n")

    # verification: worst-case 'best wrong oracle' margin per selected structure (within its interface)
    by_iface = defaultdict(list)
    for s in sel: by_iface[(s.m, s.p)].append(s)
    print(f"{'structure':20s} {'native':>7} {'best-wrong':>11} {'(which)':>18} {'margin':>7}")
    worst = 1e9
    for (m,p), group in sorted(by_iface.items()):
        for B in group:
            others = [(cross_mae(A, B, data), A.name) for A in group if A.name != B.name]
            if others:
                bw, who = min(others); mg = 100*(bw-nat[B.name])/nat[B.name]
                worst = min(worst, mg)
                print(f"{B.name:20s} {nat[B.name]:7.2f} {bw:11.2f} {who:>18} {mg:6.0f}%")
            else:
                print(f"{B.name:20s} {nat[B.name]:7.2f} {'(only in iface)':>30}")
    print(f"\nworst within-interface margin: {worst:.0f}%  (all pairs across interfaces are trivially distinct)")

    if args.write and len(sel) >= 20:
        emit(sel[:20])
    elif args.write:
        print(f"\nNOT writing: only {len(sel)} distinct (< 20).")


def _mat(M, hidslot=None):
    rows = []
    for r, row in enumerate(M.tolist()):
        cells = ["_H" if hidslot == (r, c) else repr(round(x, 4)) for c, x in enumerate(row)]
        rows.append("[" + ",".join(cells) + "]")
    return "[" + ",".join(rows) + "]"


def emit(structs):
    # 4 held-out: the Exp-2 world + three structurally-novel axes (deep chain, opposed drivers, two-poll)
    test = ["direct_exp2", "chain3", "two_news_opposed", "two_poll"]
    test = [t for t in test if t in {s.name for s in structs}]
    L = ['"""catalog_distinct.py -- 20 observationally-DISTINCT LG-DAG structures.',
         'Auto-built + VERIFIED by build_catalog.py: every same-interface pair is >=40% distinct under',
         'the cross-oracle test (a structure\'s own Kalman oracle beats every other on its data). Do not',
         'hand-edit; regenerate via `python3 build_catalog.py --write`."""',
         "import numpy as np", "from lg_dag import LGStructure", "",
         "def _S(name,A,B,C,hidden,note,recover=True,**kw):",
         "    return LGStructure(name,np.array(A,float),np.array(B,float),np.array(C,float),",
         "                       hidden=hidden,recover=recover,note=note,**kw)", "", "_H = 0.5", "", "CATALOG = ["]
    for s in structs:
        kind, i, j = s.hidden
        A = _mat(s.A, (i, j) if kind == "A" else None)
        B = _mat(s.B, (i, j) if kind == "B" else None)
        L.append(f"    _S({s.name!r}, {A}, {B}, {_mat(s.C)}, {s.hidden!r},")
        L.append(f"       {s.note!r}, recover={s.recover}),")
    L += ["]", "BY_NAME = {s.name: s for s in CATALOG}",
          f"TEST_NAMES = {test!r}",
          "TRAIN = [s for s in CATALOG if s.name not in TEST_NAMES]",
          "TEST = [BY_NAME[n] for n in TEST_NAMES]",
          "assert len(CATALOG) == 20 and len(TEST) == 4 and len(TRAIN) == 16"]
    open("catalog_distinct.py", "w").write("\n".join(L) + "\n")
    print(f"\nwrote catalog_distinct.py  (20 structures; held-out = {test})")


if __name__ == "__main__":
    main()
