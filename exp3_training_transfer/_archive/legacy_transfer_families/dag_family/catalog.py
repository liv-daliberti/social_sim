"""catalog.py -- 12 SINGLE-POLL LG-DAG worlds (8 train / 4 held-out), unique in topology AND observation.
Auto-built + VERIFIED by build_catalog_single.py (topology gate + cross-oracle >= MARGIN both ways).
Every world: one poll, one hidden gain g, a distinct wiring. Regenerate via build_catalog_single.py --write.

NOTE (2026-07-08): `skip` was removed. Its poll summed two baseline-50 latents (C=[1,1]), so it rested
at ~100 and saturated >50% of its polls at the ceiling -- a degenerate training world whose target is
often pinned regardless of g. Held-out set is unchanged (skip was train-only).

NOTE (2026-07-08, audit): `two_news_feedback` coupling tamed (A was [[.85,.4],[.5,.85]], spectral
radius 1.297 -- an EXPLOSIVE world sold as mean-reverting; its polls wandered ~36 pts and the 4-shock
probe saturated at the clip bounds) and `inhibitory` tamed (was [[.9,-.5],[.6,.78]], |eig|=1.001
marginal). Import-time assertions below now enforce stability at both ends of the g range and a
mid-range poll baseline for every world, so a bad edit fails fast instead of poisoning a training run.

NOTE (2026-07-09): deep-chain DC amplification tamed (chain4 coupling .8->.6, chain3 .8->.65,
mediators merge .6->.45): at the old couplings 3-5% of drawn-g polls and reward targets clipped
(chain4 5.4%, week-10 poll sd 31 pts). Tamed: clip <=0.4% while the oracle-vs-prior-blind
identifiability margin at k=7/9 stays well above the 0.10 reporting threshold (chain4 +.22/+.36,
chain3 +.45/+.38, mediators +.44/+.47)."""
import numpy as np
from lg_dag import LGStructure

def _S(name,A,B,C,hidden,note,recover=True,**kw):
    return LGStructure(name,np.array(A,float),np.array(B,float),np.array(C,float),
                       hidden=hidden,recover=recover,note=note,**kw)

_H = 0.5

CATALOG = [
    _S('direct', [[0.9]], [[_H]], [[1.0]], ('B', 0, 0),
       'news->opinion->poll (the Exp-2 world); infer input gain', recover=True),
    _S('hidden_phi', [[_H]], [[0.7]], [[1.0]], ('A', 0, 0),
       'same graph but the UNKNOWN is the persistence phi (gain known)', recover=False),
    _S('chain2', [[0.85,0.0],[0.6,0.85]], [[_H],[0.0]], [[0.0,1.0]], ('B', 0, 0),
       '2-stage: news hits poll one week later; infer input gain', recover=False),
    _S('chain2_hidden_phi', [[0.85,0.0],[0.6,_H]], [[0.5],[0.0]], [[0.0,1.0]], ('A', 1, 1),
       "2-stage chain but the UNKNOWN is the polled stage's persistence (gain known)", recover=False),
    _S('echo', [[0.9,0.5],[0.0,0.4]], [[_H],[0.6]], [[1.0,0.0]], ('B', 0, 0),
       'news hits polled o1 now AND via o2 as a delayed echo -> immediate bump + echo', recover=True),
    _S('inhibitory', [[0.85,-0.5],[0.6,0.7]], [[_H],[0.0]], [[1.0,0.0]], ('B', 0, 0),
       'news lifts o1, o1 drives o2, o2 inhibits o1 -> overshoot-correct', recover=True),
    _S('lagged', [[0.9,_H],[0.0,0.0]], [[0.0],[1.0]], [[1.0,0.0]], ('A', 0, 1),
       'news enters a buffer o2, released to the polled o1 next week; hidden = release edge', recover=False),
    _S('chain3', [[0.7,0.0,0.0],[0.65,0.7,0.0],[0.0,0.65,0.7]], [[_H],[0.0],[0.0]], [[0.0,0.0,1.0]], ('B', 0, 0),
       '3-stage chain, doubly delayed', recover=False),
    _S('mediators', [[0.85,0.0,0.0],[0.0,0.85,0.0],[0.45,0.45,0.7]], [[_H],[0.5],[0.0]], [[0.0,0.0,1.0]], ('B', 0, 0),
       'news splits into two parallel mediators that merge into the polled stage', recover=False),
    _S('chain4', [[0.7,0.0,0.0,0.0],[0.6,0.7,0.0,0.0],[0.0,0.6,0.7,0.0],[0.0,0.0,0.6,0.7]], [[_H],[0.0],[0.0],[0.0]], [[0.0,0.0,0.0,1.0]], ('B', 0, 0),
       '4-stage chain, triple-delayed', recover=False),
    _S('two_news_feedback', [[0.8,0.2],[0.1,0.8]], [[_H,0.0],[0.0,0.5]], [[1.0,0.0]], ('B', 0, 0),
       'each driver its own opinion, o1<->o2 coupled loop; poll reads o1', recover=True),
    _S('two_news_split', [[0.85,0.0],[0.6,0.85]], [[_H,0.0],[0.0,0.6]], [[0.0,1.0]], ('B', 0, 0),
       'driver1 delayed via o1, driver2 direct to the polled o2', recover=False),
]
BY_NAME = {s.name: s for s in CATALOG}
TEST_NAMES = ['direct', 'chain4', 'mediators', 'two_news_feedback']
TRAIN = [s for s in CATALOG if s.name not in TEST_NAMES]
TEST = [BY_NAME[n] for n in TEST_NAMES]
assert all(s.p == 1 for s in CATALOG), 'single-poll only'
assert len(TEST) >= 3 and len(CATALOG) == len(TRAIN) + len(TEST)

# ── health gates (2026-07-08 audit): stability + observable baseline ─────────────────────────────
# Every world must be stable across the whole hidden-gain range. Hidden-persistence slots (g on A's
# diagonal) touch spectral radius exactly 1.0 at the measure-zero endpoint g=G_HI=1.0 -- every
# realized draw g<1 is strictly stable -- so those are allowed <=1; everything else must be <1.
from lg_dag import G_LO, G_HI
for _s in CATALOG:
    for _g in (G_LO, G_HI):
        _A, _ = _s.with_gain(_g)
        _rho = float(np.max(np.abs(np.linalg.eigvals(_A))))
        _lim = 1.0 + 1e-9 if _s.hidden[0] == 'A' else 1.0 - 1e-9
        assert _rho <= _lim, f'{_s.name}: unstable dynamics, spectral radius {_rho:.3f} at g={_g}'
    _b = _s.C @ np.full(_s.d, _s.baseline)
    assert np.all((_b >= 25.0) & (_b <= 75.0)), \
        f'{_s.name}: poll baseline {_b} too close to the [0,100] clip bounds'
del _s, _g, _A, _rho, _lim, _b

# ── the h* instrument's notion of "recoverable" ──────────────────────────────────────────────────
# CAUTION: LGStructure.recover is the LEGACY flag -- "is a ONE-STEP input->output gain defined?" --
# and is read only by the pre-h* roster path (eval.py, run_dag_eval.py, build_roster.py). Under the
# h* probe the right question is whether the response AT h* varies with g, which is True for 6 more
# structures (the delayed chains: their response is simply deferred, not absent). make_dataset_dag.py
# and transfer_report.py use THIS map, via the `recover` field they write into each reference.
# Do not read `struct.recover` when working with the h* instrument.
from lg_dag import encodes_gain, first_response_horizon        # noqa: E402
PROBE_HORIZON = {s.name: first_response_horizon(s) for s in CATALOG}
RECOVERS_AT_HSTAR = {s.name: encodes_gain(s, PROBE_HORIZON[s.name]) for s in CATALOG}
