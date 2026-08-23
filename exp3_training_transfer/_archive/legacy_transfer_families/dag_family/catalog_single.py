"""catalog_single.py -- 20 SINGLE-POLL LG-DAG worlds, unique in topology AND observation.
Auto-built + VERIFIED by build_catalog_single.py (topology gate + cross-oracle >= MARGIN both ways).
Every world: one poll, one hidden gain g, a distinct wiring. Regenerate via build_catalog_single.py --write."""
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
    _S('inhibitory', [[0.9,-0.5],[0.6,0.78]], [[_H],[0.0]], [[1.0,0.0]], ('B', 0, 0),
       'news lifts o1, o1 drives o2, o2 inhibits o1 -> overshoot-correct', recover=True),
    _S('lagged', [[0.9,_H],[0.0,0.0]], [[0.0],[1.0]], [[1.0,0.0]], ('A', 0, 1),
       'news enters a buffer o2, released to the polled o1 next week; hidden = release edge', recover=False),
    _S('skip', [[0.85,0.0],[0.6,0.85]], [[_H],[0.0]], [[1.0,1.0]], ('B', 0, 0),
       'poll reads BOTH stages: immediate o1 + delayed o2 from one news', recover=True),
    _S('chain3', [[0.7,0.0,0.0],[0.8,0.7,0.0],[0.0,0.8,0.7]], [[_H],[0.0],[0.0]], [[0.0,0.0,1.0]], ('B', 0, 0),
       '3-stage chain, doubly delayed', recover=False),
    _S('mediators', [[0.85,0.0,0.0],[0.0,0.85,0.0],[0.6,0.6,0.7]], [[_H],[0.5],[0.0]], [[0.0,0.0,1.0]], ('B', 0, 0),
       'news splits into two parallel mediators that merge into the polled stage', recover=False),
    _S('chain4', [[0.7,0.0,0.0,0.0],[0.8,0.7,0.0,0.0],[0.0,0.8,0.7,0.0],[0.0,0.0,0.8,0.7]], [[_H],[0.0],[0.0],[0.0]], [[0.0,0.0,0.0,1.0]], ('B', 0, 0),
       '4-stage chain, triple-delayed', recover=False),
    _S('two_news_feedback', [[0.85,0.4],[0.5,0.85]], [[_H,0.0],[0.0,0.5]], [[1.0,0.0]], ('B', 0, 0),
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
