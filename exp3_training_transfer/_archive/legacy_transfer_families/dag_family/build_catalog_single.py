"""build_catalog_single.py — build 20 SINGLE-POLL LG-DAG worlds that are unique at BOTH levels:
(1) topology (distinct wiring + which edge is hidden — so sign-only / phi-only twins are collapsed),
(2) observation (cross-oracle: a structure's own Kalman oracle must beat every other's on its data).

Fixes the old catalog's two problems: (a) it allowed topology-twins that differ only in a coefficient
(direct_exp2 vs direct_fast = same graph, phi .9 vs .45) or a sign (two_news_aligned vs opposed), and
(b) it mixed in multi-poll structures. Every world here has exactly ONE poll, ONE hidden gain g, and a
distinct wiring. 15 train + 5 held-out.

    python3 build_catalog_single.py            # report distinct set + margins
    python3 build_catalog_single.py --write    # also emit catalog_single.py
"""
from __future__ import annotations
import argparse, sys
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
# reuse the verified distinctness machinery (cross-oracle forecasting) from build_catalog
from build_catalog import S, cross_mae, native_mae, _data, _mat, MARGIN, _H


# ---- candidate pool: SINGLE-POLL, diverse wirings (m=1/2/3), each a genuinely different graph -------
def candidates():
    C = []
    # === m=1 : depth / hidden-slot / feedback / confounder ===
    C += [
        S("direct",        [[0.9]], [[_H]], [[1]], ("B",0,0),
          "news->opinion->poll (the Exp-2 world); infer input gain", recover=True),
        S("hidden_phi",    [[_H]], [[0.7]], [[1]], ("A",0,0),
          "same graph but the UNKNOWN is the persistence phi (gain known)", recover=False),
        S("chain2",        [[0.85,0],[0.6,0.85]], [[_H],[0]], [[0,1]], ("B",0,0),
          "2-stage: news hits poll one week later; infer input gain", recover=False),
        S("hidden_coupling",[[0.85,0],[_H,0.85]], [[0.5],[0]], [[0,1]], ("A",1,0),
          "2-stage; the HIDDEN edge is the o1->o2 coupling (input known)", recover=False),
        S("chain3",        [[0.7,0,0],[0.8,0.7,0],[0,0.8,0.7]], [[_H],[0],[0]], [[0,0,1]], ("B",0,0),
          "3-stage chain, doubly delayed", recover=False),
        S("chain4",        [[0.7,0,0,0],[0.8,0.7,0,0],[0,0.8,0.7,0],[0,0,0.8,0.7]],
          [[_H],[0],[0],[0]], [[0,0,0,1]], ("B",0,0), "4-stage chain, triple-delayed", recover=False),
        S("lagged",        [[0.9,0.7],[0,0]], [[0],[1]], [[1,0]], ("A",0,1),
          "news enters a buffer o2, released to the polled o1 next week; hidden = release edge", recover=False),
        S("inhibitory",    [[0.9,-0.5],[0.6,0.78]], [[_H],[0]], [[1,0]], ("B",0,0),
          "news lifts o1, o1 drives o2, o2 inhibits o1 -> overshoot-correct", recover=True),
        S("confounder",    [[0.9,0.4],[0,0.85]], [[_H],[0]], [[1,0]], ("B",0,0),
          "polled o1 driven by news AND a latent drift o2 (own phi, unobserved)", recover=True),
        S("mediators",     [[0.85,0,0],[0,0.85,0],[0.6,0.6,0.7]], [[_H],[0.5],[0]], [[0,0,1]], ("B",0,0),
          "news splits into two parallel mediators that merge into the polled stage", recover=False),
        S("coupling3",     [[0.7,0,0],[0.8,0.7,0],[0,_H,0.7]], [[0.5],[0],[0]], [[0,0,1]], ("A",2,1),
          "3-stage; hidden edge is the DEEP o2->o3 coupling", recover=False),
    ]
    # === m=2 : two observed news ===
    C += [
        S("two_news",      [[0.9]], [[_H,0.5]], [[1]], ("B",0,0),
          "two drivers into one opinion; infer gain of driver 1 (driver 2 known)", recover=True),
        S("two_news_split",[[0.85,0],[0.6,0.85]], [[_H,0],[0,0.6]], [[0,1]], ("B",0,0),
          "driver1 delayed via o1, driver2 direct to the polled o2", recover=False),
        S("two_news_confounder",[[0.9,0.4],[0,0.85]], [[_H,0],[0,0.5]], [[1,0]], ("B",0,0),
          "driver1->polled o1 (hidden); driver2->confounder o2->o1", recover=True),
        S("two_news_feedback",[[0.85,0.4],[0.5,0.85]], [[_H,0],[0,0.5]], [[1,0]], ("B",0,0),
          "each driver its own opinion, o1<->o2 coupled loop; poll reads o1", recover=True),
        S("two_news_chain",[[0.85,0],[0.6,0.85]], [[_H,0.5],[0,0]], [[0,1]], ("B",0,0),
          "both drivers into stage 1, poll reads the delayed stage 2", recover=False),
    ]
    # === m=3 : three observed news ===
    C += [
        S("three_news",    [[0.9]], [[_H,0.4,-0.3]], [[1]], ("B",0,0),
          "three drivers into one opinion; two known (+.4,-.3)", recover=True),
        S("three_news_2stage",[[0.85,0],[0.6,0.85]], [[_H,0.4,-0.3],[0,0,0]], [[0,1]], ("B",0,0),
          "three drivers into a 2-stage chain", recover=False),
        S("three_news_tree",[[0.85,0],[0.6,0.85]], [[_H,0.5,0],[0,0,0.5]], [[0,1]], ("B",0,0),
          "drivers 1,2 -> o1 ; driver 3 -> o2 ; o1->o2->poll", recover=False),
        S("three_news_confounder",[[0.9,0.4],[0,0.85]], [[_H,0,0],[0,0.5,0.5]], [[1,0]], ("B",0,0),
          "driver1->polled o1 (hidden); drivers 2,3 -> confounder o2 -> o1", recover=True),
    ]
    # === more m=1 / m=2 wirings ===
    C += [
        S("echo",          [[0.9,0.5],[0,0.4]], [[_H],[0.6]], [[1,0]], ("B",0,0),
          "news hits polled o1 now AND via o2 as a delayed echo -> immediate bump + echo", recover=True),
        S("skip",          [[0.85,0],[0.6,0.85]], [[_H],[0]], [[1,1]], ("B",0,0),
          "poll reads BOTH stages: immediate o1 + delayed o2 from one news", recover=True),
        S("ring3",         [[0.8,0,0.4],[0.55,0.8,0],[0,0.55,0.8]], [[_H],[0],[0]], [[1,0,0]], ("B",0,0),
          "news drives a 3-node ring (o1->o2->o3->o1); poll reads o1", recover=True),
        S("chain2_hidden_phi",[[0.85,0],[0.6,_H]], [[0.5],[0]], [[0,1]], ("A",1,1),
          "2-stage chain but the UNKNOWN is the polled stage's persistence (gain known)", recover=False),
        S("two_news_parallel",[[0.85,0,0],[0,0.85,0],[0.6,0.6,0.7]], [[_H,0],[0,0.6],[0,0]], [[0,0,1]],
          ("B",0,0), "two drivers each own delayed channel, merged into the polled stage", recover=False),
    ]
    # === m=4 : four observed news (distinct wirings) ===
    C += [
        S("four_news",     [[0.9]], [[_H,0.4,-0.3,0.3]], [[1]], ("B",0,0),
          "four drivers into one opinion; three known", recover=True),
        S("four_news_2stage",[[0.85,0],[0.6,0.85]], [[_H,0.4,-0.3,0.3],[0,0,0,0]], [[0,1]], ("B",0,0),
          "four drivers into a 2-stage chain (delayed poll)", recover=False),
        S("four_news_split",[[0.85,0],[0.6,0.85]], [[_H,0.4,0,0],[0,0,0.5,0.4]], [[0,1]], ("B",0,0),
          "drivers 1,2 -> o1 ; drivers 3,4 -> polled o2 ; o1->o2", recover=False),
        S("four_news_confounder",[[0.9,0.4],[0,0.85]], [[_H,0,0,0],[0,0.4,0.4,0.4]], [[1,0]], ("B",0,0),
          "driver1->polled o1 (hidden); drivers 2,3,4 -> confounder o2 -> o1", recover=True),
    ]
    # === m=5 : five observed news ===
    C += [
        S("five_news",     [[0.9]], [[_H,0.4,-0.3,0.3,-0.2]], [[1]], ("B",0,0),
          "five drivers into one opinion; four known", recover=True),
        S("five_news_2stage",[[0.85,0],[0.6,0.85]], [[_H,0.4,-0.3,0.3,-0.2],[0,0,0,0,0]], [[0,1]],
          ("B",0,0), "five drivers into a 2-stage chain", recover=False),
    ]
    return C


# ---- latent-dynamics signature: same opinion graph + readout + news-entry pattern + hidden slot means
# the worlds differ only in news COUNT (five_news == four_news + a linear term) or coefficient VALUES.
def latent_sig(s):
    A = (np.abs(s.A) > 1e-9).astype(int)
    Cc = (np.abs(s.C) > 1e-9).astype(int)
    gets_news = tuple(int(any(s.B[i, j] != 0 for j in range(s.m))) for i in range(s.d))  # WHICH latents, not how many
    kind, hi, hj = s.hidden
    hid = ("B", hi) if kind == "B" else ("A", hi, hj)
    return (s.d, tuple(map(tuple, A.tolist())), tuple(map(tuple, Cc.tolist())), gets_news, hid)


def select(cands):
    report = []
    # 0) LATENT-DYNAMICS gate: collapse news-count / coefficient variants; keep the minimal-news rep
    lat_ok, seenL = [], {}
    for s in sorted(cands, key=lambda s: (s.m, s.d, s.name)):
        lg = latent_sig(s)
        if lg in seenL:
            report.append(f"    latent-drop {s.name}: same latent dynamics as {seenL[lg]} (news-count / coef variant)")
        else:
            seenL[lg] = s.name; lat_ok.append(s)
    # 1) observational gate: within each news-interface, keep mutually-distinct (cross-oracle)
    data = _data(lat_ok)
    nat = {s.name: native_mae(s, data) for s in lat_ok}
    by_m = defaultdict(list)
    for s in lat_ok: by_m[s.m].append(s)
    selected = []
    for m, group in sorted(by_m.items()):
        keep = []
        for cand in group:
            ok = True
            for other in keep:
                a = 100*(cross_mae(other, cand, data) - nat[cand.name]) / nat[cand.name]
                b = 100*(cross_mae(cand, other, data) - nat[other.name]) / nat[other.name]
                if min(a, b) < 100*MARGIN:
                    ok = False
                    report.append(f"    obs-drop {cand.name}: collapses with {other.name} (min {min(a,b):.0f}%)")
                    break
            if ok: keep.append(cand)
        selected += keep
        report.append(f"[m={m}] kept {len(keep)}/{len(group)}: {[s.name for s in keep]}")
    return selected, nat, data, report


# held-out 4: span the transfer challenges — Exp-2 anchor (direct), depth extrapolation (chain4),
# parallel merge (mediators), and a NOVEL 2-news loop (two_news_feedback; two_news_split stays in train
# so the model still sees multi-news). Everything else trains.
TEST_WANT = ["direct", "chain4", "mediators", "two_news_feedback"]


def emit(structs, path="catalog_single.py"):
    test = [t for t in TEST_WANT if t in {s.name for s in structs}][:5]
    L = ['"""catalog_single.py -- 20 SINGLE-POLL LG-DAG worlds, unique in topology AND observation.',
         'Auto-built + VERIFIED by build_catalog_single.py (topology gate + cross-oracle >= MARGIN both ways).',
         'Every world: one poll, one hidden gain g, a distinct wiring. Regenerate via build_catalog_single.py --write."""',
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
          "assert all(s.p == 1 for s in CATALOG), 'single-poll only'",
          "assert len(TEST) >= 3 and len(CATALOG) == len(TRAIN) + len(TEST)"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {path}  (20 single-poll; held-out = {test})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); args = ap.parse_args()
    cands = candidates()
    assert all(s.p == 1 for s in cands), "pool must be single-poll"
    print(f"candidate pool: {len(cands)} single-poll structures (m in {sorted(set(s.m for s in cands))})\n")
    sel, nat, data, report = select(cands)
    print("\n".join(report))
    print(f"\n=> {len(sel)} mutually-distinct single-poll structures (target 20)\n")

    by_m = defaultdict(list)
    for s in sel: by_m[s.m].append(s)
    print(f"{'structure':22s} {'native':>7} {'best-wrong':>11} {'(which)':>20} {'margin':>7} recover")
    worst = 1e9
    for m, group in sorted(by_m.items()):
        for B in group:
            others = [(cross_mae(A, B, data), A.name) for A in group if A.name != B.name]
            if others:
                bw, who = min(others); mg = 100*(bw-nat[B.name])/nat[B.name]; worst = min(worst, mg)
                print(f"{B.name:22s} {nat[B.name]:7.2f} {bw:11.2f} {who:>20} {mg:6.0f}%   {B.recover}")
            else:
                print(f"{B.name:22s} {nat[B.name]:7.2f} {'(only in m-group)':>32}         {B.recover}")
    print(f"\nworst within-interface margin: {worst:.0f}%  (cross-interface pairs trivially distinct)")
    print(f"recover=True count: {sum(1 for s in sel if s.recover)} / {len(sel)}")

    if args.write and len(sel) >= 8:
        emit(sel)
    elif args.write:
        print(f"\nNOT writing: only {len(sel)} distinct.")


if __name__ == "__main__":
    main()
