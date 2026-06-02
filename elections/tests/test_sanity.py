"""
Fast (no-simulation) structural correctness checks.

- Every CPT row's probabilities sum to 1.0
- TOPO_ORDER lists every node exactly once, parents before children
- _find_dist returns the correct distribution for deterministic inputs
- H→I edge does not exist (forecaster belief is an observer, not a cause)
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.dag import NODES, TOPO_ORDER, DAG_EDGES, _find_dist


# ── 1. CPT probability sums ───────────────────────────────────────────────────

@pytest.mark.parametrize("node_id", list(NODES.keys()))
def test_cpt_rows_sum_to_one(node_id):
    node = NODES[node_id]
    if "prior" in node:
        total = sum(node["prior"].values())
        assert abs(total - 1.0) < 1e-6, \
            f"{node_id} prior sums to {total}, expected 1.0"
    else:
        for i, rule in enumerate(node["cpt"]):
            total = sum(rule["dist"].values())
            assert abs(total - 1.0) < 1e-6, \
                f"{node_id} CPT row {i} (cond={rule['cond']}) sums to {total}"


# ── 2. Topological order ──────────────────────────────────────────────────────

def test_topo_order_contains_all_nodes():
    assert set(TOPO_ORDER) == set(NODES.keys()), \
        "TOPO_ORDER does not contain exactly the same nodes as NODES"

def test_topo_order_no_duplicates():
    assert len(TOPO_ORDER) == len(set(TOPO_ORDER)), \
        "TOPO_ORDER contains duplicate node IDs"

def test_topo_order_parents_before_children():
    """Every node's parents must appear earlier in TOPO_ORDER."""
    position = {nid: i for i, nid in enumerate(TOPO_ORDER)}
    violations = []
    for nid in TOPO_ORDER:
        node_pos = position[nid]
        for parent in NODES[nid].get("parents", []):
            if position[parent] >= node_pos:
                violations.append(f"{parent} (pos {position[parent]}) >= {nid} (pos {node_pos})")
    assert not violations, "Topological order violated:\n" + "\n".join(violations)


# ── 3. DAG causal edge sanity ─────────────────────────────────────────────────

def test_h_does_not_cause_i():
    """H (Forecaster Belief) must not have a causal edge to I (Final Outcome)."""
    assert ("H", "I") not in DAG_EDGES, \
        "H→I edge found: Forecaster Belief does not causally determine the outcome."

def test_edges_reference_valid_groups():
    valid_groups = set(nid[0] for nid in NODES)
    for src, tgt in DAG_EDGES:
        assert src in valid_groups, f"Edge source {src!r} is not a valid group"
        assert tgt in valid_groups, f"Edge target {tgt!r} is not a valid group"


# ── 4. _find_dist unit tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("node_id,parents,expected_state,expected_p", [
    # A nodes are root — _find_dist returns prior regardless
    ("A1", {},                              "Strong",        0.30),
    ("A2", {},                              "High",          0.25),
    ("A3", {},                              "Blue-leaning",  0.35),
    # B nodes
    ("B1", {"A3": "Blue-leaning"},          "Strong",        0.40),
    ("B1", {"A3": "Toss-up"},               "Average",       0.50),
    ("B1", {"A3": "Red-leaning"},           "Weak",          0.25),
    ("B2", {"A3": "Red-leaning"},           "Strong",        0.40),
    ("B3", {"A3": "Blue-leaning","A2":"High"}, "Blue advantage", 0.55),
    ("B3", {"A3": "Red-leaning", "A2":"Low"},  "Red advantage",  0.55),
    # C nodes
    ("C1", {"A1":"Weak",   "A2":"Low"},     "Yes",           0.70),
    ("C1", {"A1":"Strong", "A2":"High"},    "No",            0.80),
    ("C2", {"C1":"Yes"},                    "None",          0.00),
    ("C2", {"C1":"No"},                     "None",          1.00),
    ("C4", {"C2":"Geopolitical shock"},     "High",          0.35),
    ("C4", {"C2":"None"},                   "Low",           1.00),
    # E nodes — E1 has parents C2, A1; E3 has parents C3,E2,B1,B2,A3; E4 has parents C4,B3
    ("E2", {"C4":"High",   "A2":"High"},    "Official",      0.30),
    ("E2", {"C4":"Low",    "A2":"Low"},     "Rumor",         0.45),
    # E3: with no B1 specified, the score-2 rule {C3,E2} wins (B1-specific rules need 3 conds)
    ("E3", {"C3":"Helps Blue","E2":"Official"},  "Blue-favorable", 0.80),
    ("E3", {"C3":"Helps Red", "E2":"Official"},  "Red-favorable",  0.80),
    # E3: when B1=Strong is also specified, the score-3 rule fires
    ("E3", {"C3":"Helps Blue","B1":"Strong","E2":"Official"},  "Blue-favorable", 0.85),
    ("E3", {"C3":"Helps Red", "B2":"Strong","E2":"Official"},  "Red-favorable",  0.85),
    # E4: C4=High has one score-1 rule; C4=Low+B3=Even triggers the score-2 rule
    ("E4", {"C4":"High"},                   "High",          0.70),
    ("E4", {"C4":"Low",  "B3":"Even"},      "Low",           0.65),
    ("E4", {"C4":"Medium","B3":"Even"},     "Medium",        0.55),
    # D nodes
    ("D1", {"B1":"Strong","B2":"Weak","E3":"Blue-favorable"}, "Rising", 0.85),
    ("D1", {"B1":"Weak","B2":"Strong","E3":"Red-favorable"},  "Falling", 0.80),
    ("D2", {"B2":"Strong","B1":"Weak","E3":"Red-favorable"},  "Rising",  0.85),
    ("D3", {"A2":"Low","E2":"Rumor","E4":"High"},             "High",    0.75),
    ("D3", {"A2":"High","E2":"Official","E4":"Low"},          "Low",     0.70),
    # G nodes (F and H layers removed)
    ("G1", {"B3":"Blue advantage","D1":"Rising"}, "High",   0.65),
    ("G1", {"B3":"Red advantage", "D1":"Falling"},"Low",    0.60),
    ("G2", {"B3":"Red advantage", "D2":"Rising"}, "High",   0.65),
    ("G3", {"A3":"Toss-up","D1":"Rising","D2":"Falling"},   "Blue +5", 0.60),
    ("G3", {"A3":"Toss-up","D1":"Falling","D2":"Rising"},   "Red +5",  0.60),
    # I nodes — I1 is now fully deterministic (each row maps to exactly one outcome)
    ("I1", {"G1":"High","G2":"Low",   "G3":"Blue +5"},  "Blue landslide", 1.00),
    ("I1", {"G1":"Low", "G2":"High",  "G3":"Red +5"},   "Red landslide",  1.00),
    ("I1", {"G1":"Normal","G2":"Normal","G3":"Near-even"}, "Recount/disputed", 1.00),
    ("I2", {"I1":"Blue landslide"},   "Blue wins", 1.00),
    ("I2", {"I1":"Red landslide"},    "Red wins",  1.00),
    ("I2", {"I1":"Recount/disputed"}, "Blue wins", 0.50),
])
def test_find_dist_unit(node_id, parents, expected_state, expected_p):
    node = NODES[node_id]
    dist = _find_dist(node, parents)
    got = dist.get(expected_state, 0.0)
    assert abs(got - expected_p) < 1e-9, \
        f"_find_dist({node_id}, parents={parents}) " \
        f"returned P({expected_state!r})={got}, expected {expected_p}"


# ── 5. Every node has valid states in CPT rows ───────────────────────────────

@pytest.mark.parametrize("node_id", list(NODES.keys()))
def test_cpt_states_are_valid(node_id):
    node = NODES[node_id]
    valid = set(node["states"])
    if "prior" in node:
        assert set(node["prior"]) == valid, \
            f"{node_id}: prior keys {set(node['prior'])} != states {valid}"
    else:
        for i, rule in enumerate(node["cpt"]):
            rule_states = set(rule["dist"])
            # Deterministic rows may have only one state key (probability = 1.0).
            # We require that all keys ARE valid states (no typos) but don't
            # insist on full coverage — the sum-to-1 test (above) handles that.
            assert rule_states.issubset(valid), \
                f"{node_id} row {i}: dist has invalid keys {rule_states - valid}"


# ── 6. Every parent referenced in CPT conditions actually exists ──────────────

@pytest.mark.parametrize("node_id", list(NODES.keys()))
def test_cpt_condition_parents_exist(node_id):
    node = NODES[node_id]
    declared_parents = set(node.get("parents", []))
    if "cpt" not in node:
        return
    for i, rule in enumerate(node["cpt"]):
        for parent_id in rule["cond"]:
            assert parent_id in NODES, \
                f"{node_id} row {i}: condition references unknown node {parent_id!r}"
            assert parent_id in declared_parents, \
                f"{node_id} row {i}: condition uses {parent_id!r} which is not in declared parents {declared_parents}"


# ── 7. Tie-breaking: first max-score rule wins ────────────────────────────────
# Every configuration below has two CPT rules that both fully match and have
# the same number of conditions. The correct behaviour (score > best_score,
# not >=) is that the FIRST listed rule wins.  Each case names the winning
# rule by a distinguishing state/probability so the assertion is unambiguous.

@pytest.mark.parametrize("node_id,parents,key_state,expected_p,desc", [
    # ── D1: B1=Strong, B2=Average — C3-rule (listed first) beats E3-rule ──────
    ("D1", {"B1":"Strong","B2":"Average","C3":"Helps Blue","E3":"Blue-favorable"},
     "Rising", 0.70, "C3-rule first: Rising=0.70, not E3-rule's 0.60"),
    ("D1", {"B1":"Strong","B2":"Average","C3":"Helps Blue","E3":"Neutral/mixed"},
     "Rising", 0.70, "C3-rule first: Rising=0.70, not E3-neutral-rule's 0.48"),

    # ── D2: B2=Strong, B1=Average — symmetric C3-rule beats E3-rule ──────────
    ("D2", {"B2":"Strong","B1":"Average","C3":"Helps Red","E3":"Red-favorable"},
     "Rising", 0.70, "C3-rule first: Rising=0.70, not E3-rule's 0.60"),
    ("D2", {"B2":"Strong","B1":"Average","C3":"Helps Red","E3":"Neutral/mixed"},
     "Rising", 0.70, "C3-rule first: Rising=0.70, not E3-neutral-rule's 0.48"),

    # ── D3: A2=Low, E2=Official — A2-Low rule (listed first) beats E2-Official ─
    ("D3", {"A2":"Low","E2":"Official","E4":"Low"},
     "High", 0.45, "A2-Low rule first: High=0.45, not Official-rule's 0.08"),
    ("D3", {"A2":"Low","E2":"Official","E4":"Medium"},
     "High", 0.45, "A2-Low rule first: High=0.45, not Official-rule's 0.08"),
    ("D3", {"A2":"Low","E2":"Official","E4":"High"},
     "High", 0.45, "A2-Low rule first: High=0.45, not Official-rule's 0.08"),

    # ── G1: weather rule (listed first within same B3 group) beats momentum ────
    ("G1", {"B3":"Blue advantage","C2":"Weather event","D1":"Stable"},
     "High", 0.30, "weather rule first: High=0.30, not D1=Stable rule's 0.45"),
    ("G1", {"B3":"Blue advantage","C2":"Weather event","D1":"Falling"},
     "Low",  0.25, "weather rule first: Low=0.25, not D1=Falling rule's 0.20"),
    ("G1", {"B3":"Even","C2":"Weather event","D1":"Rising"},
     "Low",  0.35, "weather rule first: Low=0.35, not D1=Rising rule's 0.15"),
    ("G1", {"B3":"Even","C2":"Weather event","D1":"Stable"},
     "Low",  0.35, "weather rule first: Low=0.35, not D1=Stable rule's 0.20"),
    ("G1", {"B3":"Even","C2":"Weather event","D1":"Falling"},
     "Low",  0.35, "weather rule first: Low=0.35, not D1=Falling rule's 0.40"),

    # ── G2: symmetric to G1 ────────────────────────────────────────────────────
    ("G2", {"B3":"Red advantage","C2":"Weather event","D2":"Stable"},
     "High", 0.30, "weather rule first: High=0.30, not D2=Stable rule's 0.45"),
    ("G2", {"B3":"Red advantage","C2":"Weather event","D2":"Falling"},
     "Low",  0.25, "weather rule first: Low=0.25, not D2=Falling rule's 0.20"),
    ("G2", {"B3":"Even","C2":"Weather event","D2":"Rising"},
     "Low",  0.35, "weather rule first: Low=0.35, not D2=Rising rule's 0.15"),
    ("G2", {"B3":"Even","C2":"Weather event","D2":"Stable"},
     "Low",  0.35, "weather rule first: Low=0.35, not D2=Stable rule's 0.20"),
    ("G2", {"B3":"Even","C2":"Weather event","D2":"Falling"},
     "Low",  0.35, "weather rule first: Low=0.35, not D2=Falling rule's 0.40"),

    # ── G3 Toss-up: specific economy/trust rules beat generic D1/D2 rules ──────
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Rising","D2":"Rising"},
     "Blue +5", 0.55, "D1-economy rule first: Blue+5=0.55, not D2-economy rule's 0.12"),
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Rising","D2":"Falling"},
     "Blue +5", 0.60, "generic D1+D2 rule first: Blue+5=0.60, not economy-D1 rule's 0.55"),
    ("G3", {"A3":"Toss-up","D4":"Economy","D1":"Falling","D2":"Rising"},
     "Red +5",  0.60, "generic D1+D2 rule first: Red+5=0.60, not economy-D2 rule's 0.55"),
    ("G3", {"A3":"Toss-up","D4":"Trust/corruption","D1":"Rising","D2":"Rising"},
     "Blue +5", 0.55, "D1-trust rule first: Blue+5=0.55, not D2-trust rule's 0.10"),
    ("G3", {"A3":"Toss-up","D4":"Trust/corruption","D1":"Rising","D2":"Falling"},
     "Blue +5", 0.60, "generic D1+D2 rule first: Blue+5=0.60, not trust-D1 rule's 0.55"),
    ("G3", {"A3":"Toss-up","D4":"Trust/corruption","D1":"Falling","D2":"Rising"},
     "Red +5",  0.60, "generic D1+D2 rule first: Red+5=0.60, not trust-D2 rule's 0.55"),
    ("G3", {"A3":"Toss-up","D4":"Security","D1":"Falling","D2":"Rising"},
     "Red +5",  0.60, "generic D1+D2 rule first: Red+5=0.60, not security-D2 rule's 0.50"),

    # ── G3 Blue-leaning: D1 rules beat D2 rules (listed first) ─────────────────
    ("G3", {"A3":"Blue-leaning","D4":"Culture","D1":"Rising","D2":"Rising"},
     "Blue +5", 0.55, "D1-Rising rule first: Blue+5=0.55, not D2-Rising rule's 0.30"),
    ("G3", {"A3":"Blue-leaning","D4":"Culture","D1":"Stable","D2":"Rising"},
     "Blue +5", 0.45, "D1-Stable rule first: Blue+5=0.45, not D2-Rising rule's 0.30"),

    # ── G3 Red-leaning: D2 rules beat D1 rules (listed first) ──────────────────
    ("G3", {"A3":"Red-leaning","D4":"Culture","D1":"Rising","D2":"Rising"},
     "Red +5",  0.55, "D2-Rising rule first: Red+5=0.55, not D1-Rising rule's 0.30"),
    ("G3", {"A3":"Red-leaning","D4":"Culture","D1":"Rising","D2":"Stable"},
     "Red +5",  0.45, "D2-Stable rule first: Red+5=0.45, not D1-Rising rule's 0.30"),
])
def test_tie_breaking_first_rule_wins(node_id, parents, key_state, expected_p, desc):
    """
    In every tie configuration the first max-score CPT rule must win
    (score > best_score, not >=).  These cases will fail if the comparison
    is accidentally reverted to >=.
    """
    node = NODES[node_id]
    dist = _find_dist(node, parents)
    got = dist.get(key_state, 0.0)
    assert abs(got - expected_p) < 1e-9, \
        f"[{node_id}] {desc}\n  got P({key_state!r})={got}, expected {expected_p}"
