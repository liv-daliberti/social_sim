"""catalog.py — the family of 20 linear-Gaussian DAG structures for Experiment 3.

Each is a distinct small LG world (see lg_dag.py). They span: mean-reversion speed, observation
noise, number of driver channels, number of observed outputs, chain depth (direct / 2-stage /
3-stage / lagged), which single edge is hidden (input gain, inter-latent coupling, or persistence),
and confounding. The shared task is FORECAST the next output; a per-structure Kalman filter is the
oracle.

`recover=True` marks structures where the driver moves the observed output in ONE step, so a clean
input->output gain recovery is defined (the exp2 read-out); deeper/delayed/hidden-elsewhere
structures are `recover=False` and scored on forecast error only.

TRAIN / TEST split (16 / 4): the 4 held-out structures are structurally novel relative to the rest,
and include `chain1_gain` -- the Experiment-2 biased-news world -- so Exp 3 asks whether training on
16 OTHER structures improves recovery on the Exp-2 world it never trained on.
"""
from __future__ import annotations

import numpy as np

from lg_dag import LGStructure


def _S(name, A, B, C, hidden, note, recover=True, **kw):
    return LGStructure(name, np.array(A, float), np.array(B, float), np.array(C, float),
                       hidden=hidden, recover=recover, note=note, **kw)


# placeholder 0.5 in the hidden slot is overwritten per-episode by with_gain(g)
_H = 0.5

CATALOG = [
    # ── direct-gain: driver moves the observed poll in one step (recover=True) ─────────────────
    _S("chain1_gain",   [[0.90]], [[_H]], [[1]], ("B", 0, 0),
       "news -> opinion -> poll  (the Experiment-2 world)"),
    _S("reactive_gain", [[0.60]], [[_H]], [[1]], ("B", 0, 0),
       "fast mean-reversion (phi=0.6): opinion snaps back quickly"),
    _S("sticky_gain",   [[0.95]], [[_H]], [[1]], ("B", 0, 0),
       "slow mean-reversion (phi=0.95): near random walk"),
    _S("noisy_gain",    [[0.90]], [[_H]], [[1]], ("B", 0, 0),
       "high survey noise (sigma_y=4): recovery harder", sigma_y=4.0),
    _S("quiet_gain",    [[0.90]], [[_H]], [[1]], ("B", 0, 0),
       "low noise (sigma_s=0.5, sigma_y=1): cleaner signal", sigma_s=0.5, sigma_y=1.0),
    _S("two_news_gain", [[0.90]], [[_H, 0.5]], [[1]], ("B", 0, 0),
       "two news channels; channel-2 effect (+0.5) known, channel-1 gain hidden"),
    _S("two_news_control", [[0.90]], [[_H, -0.6]], [[1]], ("B", 0, 0),
       "a second, oppositely-signed known driver (-0.6) competes with the hidden one"),
    _S("three_news_gain", [[0.90]], [[_H, 0.4, -0.3]], [[1]], ("B", 0, 0),
       "three drivers (+0.4, -0.3 known); infer channel-1 gain amid two knowns"),
    _S("two_poll_gain", [[0.90]], [[_H]], [[1], [1]], ("B", 0, 0),
       "one opinion read by TWO independent polls: more observation per week"),
    _S("parallel_gain", [[0.90, 0.0], [0.0, 0.90]], [[_H], [0.5]], [[1, 0], [0, 1]], ("B", 0, 0),
       "two opinions from the same news (o1 hidden gain, o2 known); both polled"),
    _S("confounder_gain", [[0.90, 0.4], [0.0, 0.90]], [[_H], [0.6]], [[1, 0]], ("B", 0, 0),
       "news drives o1 (hidden) and a confounder o2 (known) that feeds back into o1"),

    # ── forecast-only: deep / delayed / hidden-elsewhere (recover=False) ───────────────────────
    _S("chain2_gain",   [[0.90, 0.0], [0.5, 0.90]], [[_H], [0.0]], [[0, 1]], ("B", 0, 0),
       "news -> o1 -> o2 -> poll (2-stage; effect is delayed one week)", recover=False),
    _S("chain2_coupling", [[0.90, 0.0], [_H, 0.90]], [[0.5], [0.0]], [[0, 1]], ("A", 1, 0),
       "2-stage, but the HIDDEN edge is the o1->o2 coupling (news gain known)", recover=False),
    _S("chain3_gain",   [[0.70, 0, 0], [0.8, 0.70, 0], [0, 0.8, 0.70]], [[_H], [0], [0]], [[0, 0, 1]],
       ("B", 0, 0), "news -> o1 -> o2 -> o3 -> poll (3-stage, doubly delayed)", recover=False),
    _S("chain1_phi",    [[_H]], [[0.6]], [[1]], ("A", 0, 0),
       "hidden = PERSISTENCE phi (news gain known 0.6); infer stickiness", recover=False),
    _S("lagged_gain",   [[0.90, _H], [0.0, 0.0]], [[0.0], [1.0]], [[1, 0]], ("A", 0, 1),
       "news enters a carrier and moves opinion NEXT week (pure 1-week lag)", recover=False),
    _S("chain2_sticky", [[0.95, 0.0], [0.35, 0.95]], [[_H], [0.0]], [[0, 1]], ("B", 0, 0),
       "2-stage with slow reversion (phi=0.95)", recover=False),
    _S("chain2_noisy",  [[0.90, 0.0], [0.5, 0.90]], [[_H], [0.0]], [[0, 1]], ("B", 0, 0),
       "2-stage with high survey noise (sigma_y=4)", recover=False, sigma_y=4.0),
    _S("two_mediator",  [[0.90, 0, 0], [0, 0.90, 0], [0.5, 0.5, 0.0]], [[_H], [0.6], [0]],
       [[0, 0, 1]], ("B", 0, 0),
       "news -> o1 (hidden) and -> o2 (known), both mediate into the polled o3", recover=False),
    _S("chain2_phi2",   [[0.90, 0.0], [0.5, _H]], [[0.7], [0.0]], [[0, 1]], ("A", 1, 1),
       "2-stage; hidden = the polled stage's own persistence (news + coupling known)", recover=False),
]

BY_NAME = {s.name: s for s in CATALOG}

# 4 held-out TEST structures (structurally novel axes): the exp2 world, the deepest chain, an
# opposed-driver world, and a hidden-persistence world. The other 16 are TRAIN.
TEST_NAMES = ["chain1_gain", "chain3_gain", "two_news_control", "chain1_phi"]
TRAIN = [s for s in CATALOG if s.name not in TEST_NAMES]
TEST = [BY_NAME[n] for n in TEST_NAMES]

assert len(CATALOG) == 20 and len(TRAIN) == 16 and len(TEST) == 4
