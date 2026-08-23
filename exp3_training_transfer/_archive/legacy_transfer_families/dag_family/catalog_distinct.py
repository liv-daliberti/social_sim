"""catalog_distinct.py -- 20 observationally-DISTINCT LG-DAG structures.
Auto-built + VERIFIED by build_catalog.py: every same-interface pair is >=40% distinct under
the cross-oracle test (a structure's own Kalman oracle beats every other on its data). Do not
hand-edit; regenerate via `python3 build_catalog.py --write`."""
import numpy as np
from lg_dag import LGStructure

def _S(name,A,B,C,hidden,note,recover=True,**kw):
    return LGStructure(name,np.array(A,float),np.array(B,float),np.array(C,float),
                       hidden=hidden,recover=recover,note=note,**kw)

_H = 0.5

CATALOG = [
    _S('direct_exp2', [[0.9]], [[_H]], [[1.0]], ('B', 0, 0),
       'news->opinion->poll, phi=.9 -- the Experiment-2 world', recover=True),
    _S('direct_fast', [[0.45]], [[_H]], [[1.0]], ('B', 0, 0),
       'direct, fast mean-reversion phi=.45', recover=True),
    _S('hidden_phi', [[_H]], [[0.7]], [[1.0]], ('A', 0, 0),
       'hidden = persistence (gain known); infer stickiness', recover=False),
    _S('chain2', [[0.85,0.0],[0.55,0.85]], [[_H],[0.0]], [[0.0,1.0]], ('B', 0, 0),
       '2-stage: effect delayed one week', recover=False),
    _S('chain3', [[0.7,0.0,0.0],[0.8,0.7,0.0],[0.0,0.8,0.7]], [[_H],[0.0],[0.0]], [[0.0,0.0,1.0]], ('B', 0, 0),
       '3-stage, doubly delayed', recover=False),
    _S('lagged', [[0.9,_H],[0.0,0.0]], [[0.0],[1.0]], [[1.0,0.0]], ('A', 0, 1),
       'news moves opinion NEXT week (1-week lag)', recover=False),
    _S('confounder', [[0.9,0.4],[0.0,0.83]], [[_H],[0.6]], [[1.0,0.0]], ('B', 0, 0),
       'news drives opinion AND a confounder (own phi) that feeds back', recover=True),
    _S('inhibitory', [[0.9,-0.5],[0.0,0.75]], [[_H],[0.5]], [[1.0,0.0]], ('B', 0, 0),
       'news lifts poll then a slow inhibitor pulls it back (overshoot-correct)', recover=True),
    _S('two_poll', [[0.9]], [[_H]], [[1.0],[1.0]], ('B', 0, 0),
       'one opinion read by TWO polls', recover=True),
    _S('chain2_2poll', [[0.9,0.0],[0.5,0.9]], [[_H],[0.0]], [[1.0,0.0],[0.0,1.0]], ('B', 0, 0),
       '2-stage; BOTH stages observed (poll each)', recover=False),
    _S('confounder_2poll', [[0.9,0.4],[0.0,0.83]], [[_H],[0.6]], [[1.0,0.0],[0.0,1.0]], ('B', 0, 0),
       'confounder with both latents polled', recover=True),
    _S('two_news_aligned', [[0.9]], [[_H,0.5]], [[1.0]], ('B', 0, 0),
       '2 drivers, 2nd known same-sign (+.5)', recover=True),
    _S('two_news_opposed', [[0.9]], [[_H,-0.6]], [[1.0]], ('B', 0, 0),
       '2 drivers, 2nd known opposite-sign (-.6)', recover=True),
    _S('two_news_split', [[0.9,0.0],[0.0,0.88]], [[_H,0.0],[0.0,0.6]], [[0.7,0.7]], ('B', 0, 0),
       'each driver its own opinion; poll reads both', recover=False),
    _S('two_news_confounder', [[0.9,0.4],[0.0,0.83]], [[_H,0.0],[0.0,0.5]], [[1.0,0.0]], ('B', 0, 0),
       'driver1->opinion (hidden); driver2->confounder->opinion', recover=False),
    _S('parallel_2x2', [[0.9,0.0],[0.0,0.9]], [[_H,0.0],[0.0,0.5]], [[1.0,0.0],[0.0,1.0]], ('B', 0, 0),
       'two driver->opinion->poll lines, side by side', recover=True),
    _S('coupled_2x2', [[0.9,0.0],[0.4,0.9]], [[_H,0.5],[0.0,0.0]], [[1.0,0.0],[0.0,1.0]], ('B', 0, 0),
       'two polls, second opinion fed by the first', recover=False),
    _S('confounder_2x2', [[0.9,0.4],[0.0,0.83]], [[_H,0.0],[0.0,0.5]], [[1.0,0.0],[0.0,1.0]], ('B', 0, 0),
       'two news, two polls, with a confounder latent', recover=False),
    _S('three_news', [[0.9]], [[_H,0.4,-0.3]], [[1.0]], ('B', 0, 0),
       '3 drivers; two known (+.4,-.3)', recover=True),
    _S('three_news_2stage', [[0.85,0.0],[0.5,0.85]], [[_H,0.4,-0.3],[0.0,0.0,0.0]], [[0.0,1.0]], ('B', 0, 0),
       '3 drivers into a 2-stage chain', recover=False),
]
BY_NAME = {s.name: s for s in CATALOG}
TEST_NAMES = ['direct_exp2', 'chain3', 'two_news_opposed', 'two_poll']
TRAIN = [s for s in CATALOG if s.name not in TEST_NAMES]
TEST = [BY_NAME[n] for n in TEST_NAMES]
assert len(CATALOG) == 20 and len(TEST) == 4 and len(TRAIN) == 16
