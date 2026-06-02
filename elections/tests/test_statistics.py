"""
Monte-Carlo statistical validation of the election DAG simulator.

For each node, we fix its parent nodes to values that trigger a specific CPT row,
run N_RUNS simulations, and verify the empirical distribution of that node is
within the Hoeffding bound of the declared CPT probabilities.

Hoeffding's inequality (one-sided, per state):
  P(|p̂ - p| > ε)  ≤  2 · exp(-2 · n · ε²)

At n = N_RUNS = 1000, α = ALPHA = 0.01:
  ε ≈ 0.0515  (≈ 5 percentage points)

This means: if the simulator is correct, all states should land within ±5pp of
their declared probability in 99% of runs.  A failure here indicates either a
CPT transcription error or a bug in _find_dist / _sample.

We also run a χ² goodness-of-fit test (α=0.01) on the full multinomial as a
complementary check — it is more sensitive when deviations are spread across
many states.
"""

import sys, os, math, random, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.dag import simulate, NODES, _find_dist

# ── Statistical parameters ────────────────────────────────────────────────────
N_RUNS = 1000
ALPHA  = 0.01   # per-state Hoeffding significance level

def _hoeffding_eps(n=N_RUNS, alpha=ALPHA) -> float:
    return math.sqrt(-math.log(alpha / 2) / (2 * n))

EPS = _hoeffding_eps()   # ≈ 0.0515


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_empirical(node_id: str, overrides: dict, n: int = N_RUNS, seed: int = 42) -> dict:
    """Run n simulations and return empirical state frequencies for node_id."""
    random.seed(seed)
    counts: dict = collections.Counter()
    for _ in range(n):
        counts[simulate(overrides=overrides)["states"][node_id]] += 1
    return {s: counts.get(s, 0) / n for s in NODES[node_id]["states"]}


def hoeffding_check(node_id: str, expected: dict, observed: dict, ctx: str = "") -> None:
    """Fail if any state deviates beyond the Hoeffding bound."""
    fails = []
    for state in NODES[node_id]["states"]:
        p_exp = expected.get(state, 0.0)
        p_obs = observed.get(state, 0.0)
        dev   = abs(p_obs - p_exp)
        if dev > EPS:
            fails.append(
                f"  state={state!r}: declared={p_exp:.3f}, "
                f"empirical={p_obs:.3f}, |Δ|={dev:.4f} > ε={EPS:.4f}"
            )
    if fails:
        pytest.fail(
            f"[{node_id}] Hoeffding violation"
            + (f" ({ctx})" if ctx else "") + f"\n"
            + "\n".join(fails)
        )


def chi2_check(node_id: str, expected: dict, observed_counts: dict,
               n: int = N_RUNS, ctx: str = "") -> None:
    """χ² goodness-of-fit against declared distribution. α = 0.001 (very lenient)."""
    from scipy import stats as _stats  # optional dependency

    obs, exp = [], []
    for state in NODES[node_id]["states"]:
        o = observed_counts.get(state, 0)
        e = expected.get(state, 0.0) * n
        if e > 0:
            obs.append(o); exp.append(e)

    if len(obs) < 2:
        return  # not enough cells to test

    # scale expected to sum to observed total
    total_obs = sum(obs)
    exp_scaled = [e * total_obs / sum(exp) for e in exp]

    chi2, p = _stats.chisquare(obs, f_exp=exp_scaled)
    if p < 0.001:
        pytest.fail(
            f"[{node_id}] χ²={chi2:.2f}, p={p:.4f} < 0.001"
            + (f" ({ctx})" if ctx else "")
        )


def statistical_test(node_id: str, overrides: dict, ctx: str = "") -> None:
    """Full statistical check: Hoeffding + optional χ²."""
    node = NODES[node_id]
    parent_states = {p: overrides[p] for p in node.get("parents", []) if p in overrides}
    expected = _find_dist(node, parent_states)
    observed = run_empirical(node_id, overrides)
    hoeffding_check(node_id, expected, observed, ctx)
    # χ² is optional (scipy might not be installed)
    try:
        obs_counts = {s: round(f * N_RUNS) for s, f in observed.items()}
        chi2_check(node_id, expected, obs_counts, N_RUNS, ctx)
    except ImportError:
        pass


# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (node_id, overrides_dict, human_readable_description)

ROOT_CASES = [
    ("A1", {}, "prior — no parents"),
    ("A2", {}, "prior — no parents"),
    ("A3", {}, "prior — no parents"),
]

CAMPAIGN_CASES = [
    # B1: Blue candidate quality | parent: A3
    ("B1", {"A3": "Blue-leaning"}, "partisan advantage → strong blue candidate likely"),
    ("B1", {"A3": "Toss-up"},      "toss-up baseline"),
    ("B1", {"A3": "Red-leaning"},  "partisan disadvantage"),
    # B2: Red candidate quality | parent: A3
    ("B2", {"A3": "Red-leaning"},  "partisan advantage → strong red candidate likely"),
    ("B2", {"A3": "Toss-up"},      "toss-up baseline"),
    ("B2", {"A3": "Blue-leaning"}, "partisan disadvantage"),
    # B3: Ground game | parents: A3, A2
    ("B3", {"A3": "Blue-leaning", "A2": "High"},   "high-trust blue-lean → blue GOTV advantage"),
    ("B3", {"A3": "Toss-up",      "A2": "Medium"}, "neutral ground"),
    ("B3", {"A3": "Red-leaning",  "A2": "Low"},    "low-trust red-lean → red GOTV advantage"),
]

EVENT_CASES = [
    # C1: Major event | parents: A1, A2
    ("C1", {"A1": "Strong",  "A2": "High"},   "stable conditions → event unlikely"),
    ("C1", {"A1": "Neutral", "A2": "Medium"}, "moderate conditions"),
    ("C1", {"A1": "Weak",    "A2": "Low"},    "unstable → event very likely"),
    # C2: Event type | parent: C1
    ("C2", {"C1": "Yes"}, "event occurs → type distribution"),
    ("C2", {"C1": "No"},  "no event → always None"),
    # C3: Event target | parent: C2
    ("C3", {"C2": "Scandal"},            "scandal can help either side"),
    ("C3", {"C2": "Geopolitical shock"}, "geo shock mostly ambiguous"),
    ("C3", {"C2": "None"},               "no event → always Ambiguous"),
    # C4: Event severity | parent: C2
    ("C4", {"C2": "Geopolitical shock"}, "geo shock tends severe"),
    ("C4", {"C2": "Cultural event"},     "cultural events often low-severity"),
    ("C4", {"C2": "None"},               "no event → always Low"),
]

NEWS_CASES = [
    # E1: News type | parents: C2, A1
    # When C2 is an event type, the score-1 C2-only rule wins regardless of A1.
    # When C2=None, A1 drives the agenda via score-2 rules — so A1 must be specified.
    ("E1", {"C2": "Scandal"},                       "scandal → scandal news dominant"),
    ("E1", {"C2": "Geopolitical shock"},             "geo shock → foreign crisis coverage"),
    ("E1", {"C2": "None", "A1": "Weak"},            "no event, weak economy → economy-heavy news"),
    ("E1", {"C2": "None", "A1": "Neutral"},         "no event, neutral economy → campaign-heavy news"),
    ("E1", {"C2": "None", "A1": "Strong"},          "no event, strong economy → campaign + other issues"),
    # E2: News reliability | parents: C4, A2
    ("E2", {"C4": "High",   "A2": "High"},          "high severity + high trust → official coverage"),
    ("E2", {"C4": "Low",    "A2": "Low"},           "low severity + low trust → rumour-heavy"),
    ("E2", {"C4": "Medium", "A2": "Medium"},        "moderate scenario"),
    # E3: News tone | parents: C3, E2, B1, B2, A3
    # Score-3 rules fire when C3 + B-candidate quality + E2 all specified.
    # Score-2 rules fire when only C3 + E2 specified (no B1/B2 in parents).
    # Score-1 C3-only rules fire for Ambiguous with even candidates.
    ("E3", {"C3":"Helps Blue","B1":"Strong","E2":"Official"},
     "strong blue cand + confirmed blue event → strongly blue-favorable"),
    ("E3", {"C3":"Helps Blue","B1":"Average","E2":"Official"},
     "avg blue cand + confirmed blue event → blue-favorable"),
    ("E3", {"C3":"Helps Blue","B1":"Average","E2":"Rumor"},
     "blue event + rumour grade → muted blue tone"),
    ("E3", {"C3":"Helps Red","B2":"Strong","E2":"Official"},
     "strong red cand + confirmed red event → strongly red-favorable"),
    ("E3", {"C3":"Helps Red","B2":"Average","E2":"Official"},
     "avg red cand + confirmed red event → red-favorable"),
    ("E3", {"C3":"Ambiguous","B1":"Strong","B2":"Weak"},
     "ambiguous event, strong blue vs weak red → blue-leaning tone"),
    ("E3", {"C3":"Ambiguous","B1":"Average","B2":"Average","A3":"Toss-up"},
     "ambiguous event, even candidates, toss-up → neutral mixed tone"),
    ("E3", {"C3":"Ambiguous","B1":"Average","B2":"Average","A3":"Blue-leaning"},
     "ambiguous event, even candidates, blue-lean → slightly blue tone"),
    # E4: News volume | parents: C4, B3
    # C4=High has a single score-1 rule (B3 irrelevant).
    # C4=Medium/Low: B3=Even triggers the score-2 rule; other B3 values use score-1.
    ("E4", {"C4": "High"},                          "high severity → wall-to-wall coverage"),
    ("E4", {"C4": "Medium", "B3": "Even"},          "medium severity + even ground game → standard medium volume"),
    ("E4", {"C4": "Medium", "B3": "Blue advantage"},"medium severity + active blue gotv → boosted medium volume"),
    ("E4", {"C4": "Low",    "B3": "Even"},          "low severity + even ground game → quiet cycle"),
    ("E4", {"C4": "Low",    "B3": "Red advantage"}, "low severity + active red gotv → slightly higher volume"),
]

OPINION_CASES = [
    # D1: Blue momentum | parents: B1, B2, C3, E3
    ("D1", {"B1":"Strong", "B2":"Weak",    "E3":"Blue-favorable"}, "strong blue cand + blue news → rising"),
    ("D1", {"B1":"Average","B2":"Average", "C3":"Ambiguous","E3":"Neutral/mixed"}, "neutral all round"),
    ("D1", {"B1":"Weak",   "B2":"Strong",  "E3":"Red-favorable"},  "weak blue + red news → falling"),
    # B1=Average,B2=Weak avoids triggering any 3-condition rule so the 2-cond C3+E3 rule wins
    ("D1", {"B1":"Average","B2":"Weak","C3":"Helps Blue","E3":"Blue-favorable"}, "event+tone helps blue"),
    # D2: Red momentum | parents: B2, B1, C3, E3
    ("D2", {"B2":"Strong", "B1":"Weak",    "E3":"Red-favorable"},  "strong red cand + red news → rising"),
    ("D2", {"B2":"Average","B1":"Average", "C3":"Ambiguous","E3":"Neutral/mixed"}, "neutral all round"),
    ("D2", {"B2":"Weak",   "B1":"Strong",  "E3":"Blue-favorable"}, "weak red + blue news → falling"),
    # D3: Voter uncertainty | parents: A2, E2, E4
    ("D3", {"A2":"Low",  "E2":"Rumor",    "E4":"High"},  "low trust + rumour + high volume → max uncertainty"),
    ("D3", {"A2":"High", "E2":"Official", "E4":"Low"},   "high trust + official + low volume → min uncertainty"),
    ("D3", {"A2":"Medium","E2":"Reported"},               "moderate scenario"),
    # D4: Issue salience | parents: A1, C2, E1
    ("D4", {"A1":"Weak",  "C2":"None"},              "weak economy, no event → economy dominates"),
    ("D4", {"C2":"Scandal","E1":"Scandal"},           "scandal event+news → trust/corruption dominates"),
    ("D4", {"C2":"Geopolitical shock","E1":"Foreign crisis"}, "geo shock → security dominates"),
]

# F and H layers removed from DAG — no FORECAST_CASES or FORECASTER_CASES.

MECHANICS_CASES = [
    # G1: Blue turnout | parents: B3, C2, D1
    ("G1", {"B3":"Blue advantage", "D1":"Rising"},  "blue GOTV + rising momt → high blue turnout"),
    ("G1", {"B3":"Red advantage",  "D1":"Falling"}, "red GOTV + falling momt → low blue turnout"),
    ("G1", {"B3":"Even",           "D1":"Stable"},  "neutral ground + stable → normal blue turnout"),
    # Explicit 3-condition rule: blue GOTV + weather event + rising momentum
    ("G1", {"B3":"Blue advantage", "C2":"Weather event", "D1":"Rising"}, "weather damps blue advantage turnout"),
    # G2: Red turnout | parents: B3, C2, D2
    ("G2", {"B3":"Red advantage",  "D2":"Rising"},  "red GOTV + rising momt → high red turnout"),
    ("G2", {"B3":"Blue advantage", "D2":"Falling"}, "blue GOTV + falling momt → low red turnout"),
    ("G2", {"B3":"Even",           "D2":"Stable"},  "neutral scenario"),
    # G3: Independent split | parents: A3, D4, D1, D2
    # D4="Culture" avoids D4-Economy/Trust rules (score 3) colliding with D1/D2 rules (score 3)
    ("G3", {"A3":"Toss-up","D4":"Culture","D1":"Rising",  "D2":"Falling"}, "toss-up, blue surging → Blue+5 likely"),
    ("G3", {"A3":"Toss-up","D4":"Culture","D1":"Falling", "D2":"Rising"},  "toss-up, red surging → Red+5 likely"),
    ("G3", {"A3":"Toss-up","D4":"Culture","D1":"Stable",  "D2":"Stable"},  "toss-up, flat race → near-even split"),
    # D2="Stable" stops D4-Economy-D2Rising rule from winning; D4="Economy" triggers the 3-cond rule
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Rising","D2":"Stable"},    "economy+blue rising → blue econ voters"),
    # Blue/Red leaning: specify all 4 parents so no ambiguity
    ("G3", {"A3":"Blue-leaning","D4":"Culture","D1":"Rising", "D2":"Stable"}, "blue-lean + rising → blue independents"),
    ("G3", {"A3":"Red-leaning", "D4":"Culture","D2":"Rising", "D1":"Stable"}, "red-lean + rising → red independents"),
]

OUTCOME_CASES = [
    # I1: Vote share category | parents: G1, G2, G3
    # I1 is fully deterministic — each (G1,G2,G3) combo maps to exactly one outcome.
    ("I1", {"G1":"High",   "G2":"Low",    "G3":"Blue +5"},  "high blue turnout + low red + blue split → Blue landslide"),
    ("I1", {"G1":"High",   "G2":"Normal", "G3":"Blue +5"},  "high blue + normal red + blue split → Blue landslide"),
    ("I1", {"G1":"High",   "G2":"Normal", "G3":"Near-even"},"high blue + normal red + even split → Blue narrow win"),
    ("I1", {"G1":"Normal", "G2":"Normal", "G3":"Near-even"},"perfectly balanced → Recount/disputed"),
    ("I1", {"G1":"Normal", "G2":"Normal", "G3":"Red +5"},   "balanced turnout, red split → Red narrow win"),
    ("I1", {"G1":"Low",    "G2":"High",   "G3":"Red +5"},   "low blue + high red + red split → Red landslide"),
    ("I1", {"G1":"Low",    "G2":"High",   "G3":"Near-even"},"low blue + high red + even split → Red landslide"),
    # I2: Winner | parent: I1  (now fully deterministic for landslide/narrow outcomes)
    ("I2", {"I1":"Blue landslide"},   "blue landslide → certain Blue wins"),
    ("I2", {"I1":"Blue narrow win"},  "blue narrow win → certain Blue wins"),
    ("I2", {"I1":"Red narrow win"},   "red narrow win → certain Red wins"),
    ("I2", {"I1":"Red landslide"},    "red landslide → certain Red wins"),
    ("I2", {"I1":"Recount/disputed"}, "recount → 50/50 coin flip"),
]

# ── Tie-breaker cases: every config where two equal-specificity rules match ────
# These verify that the FIRST max-score rule wins (score > best_score).
# They will fail if the comparator is ever reverted to >=.
TIE_CASES = [
    # D1 — C3-rule (rule[3]) beats E3-rule (rule[4] or rule[5])
    ("D1", {"B1":"Strong","B2":"Average","C3":"Helps Blue","E3":"Blue-favorable"},
     "strong blue + C3-helps + blue news: C3 rule must win over E3 rule"),
    ("D1", {"B1":"Strong","B2":"Average","C3":"Helps Blue","E3":"Neutral/mixed"},
     "strong blue + C3-helps + neutral news: C3 rule must win over E3-neutral rule"),

    # D2 — C3-rule (rule[3]) beats E3-rule (rule[4] or rule[5])
    ("D2", {"B2":"Strong","B1":"Average","C3":"Helps Red","E3":"Red-favorable"},
     "strong red + C3-helps + red news: C3 rule must win over E3 rule"),
    ("D2", {"B2":"Strong","B1":"Average","C3":"Helps Red","E3":"Neutral/mixed"},
     "strong red + C3-helps + neutral news: C3 rule must win over E3-neutral rule"),

    # D3 — A2-Low rule (rule[9]) beats E2-Official rule (rule[10])
    ("D3", {"A2":"Low","E2":"Official","E4":"Low"},
     "low trust + official news + low volume: low-trust uncertainty wins"),
    ("D3", {"A2":"Low","E2":"Official","E4":"Medium"},
     "low trust + official news + medium volume: low-trust uncertainty wins"),
    ("D3", {"A2":"Low","E2":"Official","E4":"High"},
     "low trust + official news + high volume: low-trust uncertainty wins"),

    # G1 — weather rule beats momentum rule for same B3 group
    ("G1", {"B3":"Blue advantage","C2":"Weather event","D1":"Stable"},
     "blue gotv + weather + stable momentum: weather rule wins over D1-Stable"),
    ("G1", {"B3":"Blue advantage","C2":"Weather event","D1":"Falling"},
     "blue gotv + weather + falling momentum: weather rule wins over D1-Falling"),
    ("G1", {"B3":"Even","C2":"Weather event","D1":"Rising"},
     "even ground + weather + rising momentum: weather suppression wins over D1-Rising"),
    ("G1", {"B3":"Even","C2":"Weather event","D1":"Stable"},
     "even ground + weather + stable momentum: weather suppression wins over D1-Stable"),
    ("G1", {"B3":"Even","C2":"Weather event","D1":"Falling"},
     "even ground + weather + falling momentum: weather suppression wins over D1-Falling"),

    # G2 — symmetric to G1
    ("G2", {"B3":"Red advantage","C2":"Weather event","D2":"Stable"},
     "red gotv + weather + stable momentum: weather rule wins over D2-Stable"),
    ("G2", {"B3":"Red advantage","C2":"Weather event","D2":"Falling"},
     "red gotv + weather + falling momentum: weather rule wins over D2-Falling"),
    ("G2", {"B3":"Even","C2":"Weather event","D2":"Rising"},
     "even ground + weather + rising momentum: weather suppression wins over D2-Rising"),
    ("G2", {"B3":"Even","C2":"Weather event","D2":"Stable"},
     "even ground + weather + stable momentum: weather suppression wins over D2-Stable"),
    ("G2", {"B3":"Even","C2":"Weather event","D2":"Falling"},
     "even ground + weather + falling momentum: weather suppression wins over D2-Falling"),

    # G3 Toss-up — economy/trust rules: most critical ties (Blue+5 vs Red+5 reversal)
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Rising","D2":"Rising"},
     "toss-up, economy, both rising: D1-economy rule wins → Blue+5 favored"),
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Rising","D2":"Falling"},
     "toss-up, economy, blue rising/red falling: generic-momentum rule wins"),
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Falling","D2":"Rising"},
     "toss-up, economy, red rising/blue falling: generic-momentum rule wins"),
    ("G3", {"A3":"Toss-up","D4":"Trust/corruption","D1":"Rising","D2":"Rising"},
     "toss-up, trust, both rising: D1-trust rule wins → Blue+5 favored"),
    ("G3", {"A3":"Toss-up","D4":"Trust/corruption","D1":"Rising","D2":"Falling"},
     "toss-up, trust, blue rising/red falling: generic-momentum rule wins"),
    ("G3", {"A3":"Toss-up","D4":"Trust/corruption","D1":"Falling","D2":"Rising"},
     "toss-up, trust, red rising/blue falling: generic-momentum rule wins"),
    ("G3", {"A3":"Toss-up","D4":"Security","D1":"Falling","D2":"Rising"},
     "toss-up, security, red rising: generic-momentum rule wins over security-D2 rule"),

    # G3 Blue-leaning — D1-rules listed before D2-rules
    ("G3", {"A3":"Blue-leaning","D4":"Culture","D1":"Rising","D2":"Rising"},
     "blue-lean, both rising: D1-Rising rule wins over D2-Rising"),
    ("G3", {"A3":"Blue-leaning","D4":"Culture","D1":"Stable","D2":"Rising"},
     "blue-lean, stable blue / red rising: D1-Stable rule wins over D2-Rising"),

    # G3 Red-leaning — D2-rules listed before D1-rules
    ("G3", {"A3":"Red-leaning","D4":"Culture","D1":"Rising","D2":"Rising"},
     "red-lean, both rising: D2-Rising rule wins over D1-Rising"),
    ("G3", {"A3":"Red-leaning","D4":"Culture","D1":"Rising","D2":"Stable"},
     "red-lean, blue rising / red stable: D2-Stable rule wins over D1-Rising"),
]

ALL_CASES = (
    ROOT_CASES + CAMPAIGN_CASES + EVENT_CASES + NEWS_CASES +
    OPINION_CASES + MECHANICS_CASES + OUTCOME_CASES + TIE_CASES
)


# ── Parametrized statistical tests ────────────────────────────────────────────

@pytest.mark.parametrize("node_id,overrides,desc", ALL_CASES)
def test_node_distribution(node_id, overrides, desc):
    """
    Fix parent nodes to `overrides`, run N_RUNS simulations, and verify that
    each state's empirical frequency is within the Hoeffding bound of the
    declared CPT probability for that set of parent values.
    """
    statistical_test(node_id, overrides, ctx=desc)


# ── Whole-pipeline smoke test ─────────────────────────────────────────────────

def test_full_pipeline_deterministic():
    """Same seed → identical outcome every time."""
    r1 = simulate(seed=42)
    r2 = simulate(seed=42)
    assert r1["states"] == r2["states"], "Same seed produced different states"


def test_full_pipeline_all_nodes_sampled():
    """Every node must appear in the states dict after a simulation run."""
    from engine.dag import TOPO_ORDER
    r = simulate(seed=0)
    missing = [nid for nid in TOPO_ORDER if nid not in r["states"]]
    assert not missing, f"Missing nodes in states: {missing}"


def test_full_pipeline_distributions_present():
    """distributions dict must contain all nodes after a simulation run."""
    from engine.dag import TOPO_ORDER
    r = simulate(seed=0)
    assert "distributions" in r, "simulate() must return 'distributions'"
    missing = [nid for nid in TOPO_ORDER if nid not in r["distributions"]]
    assert not missing, f"Missing nodes in distributions: {missing}"


def test_full_pipeline_narrative_items_complete():
    """Each narrative item must carry distribution, all_states, and parents."""
    r = simulate(seed=7)
    for phase in r["narrative"]:
        for it in phase["items"]:
            assert "distribution" in it, f"narrative item {it['node']!r} missing 'distribution'"
            assert "all_states"   in it, f"narrative item {it['node']!r} missing 'all_states'"
            assert "parents"      in it, f"narrative item {it['node']!r} missing 'parents'"


def test_override_forces_state():
    """When a node is overridden, its state must equal the forced value."""
    forced = {"A1": "Weak", "A3": "Toss-up", "C1": "Yes", "C2": "Scandal"}
    r = simulate(overrides=forced, seed=5)
    for nid, val in forced.items():
        assert r["states"][nid] == val, \
            f"Override {nid}={val!r} was not honoured; got {r['states'][nid]!r}"


def test_winner_is_valid_state():
    """I2 (Winner) must always be Blue wins or Red wins — no disputed state."""
    from engine.dag import NODES
    valid = set(NODES["I2"]["states"])
    assert valid == {"Blue wins", "Red wins"}, \
        f"I2 states should be exactly Blue/Red wins, got {valid}"
    for seed in range(20):
        r = simulate(seed=seed)
        assert r["states"]["I2"] in valid, \
            f"Seed {seed}: I2={r['states']['I2']!r} is not a valid state"
