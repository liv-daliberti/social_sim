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
    # E nodes
    ("E2", {"C4":"High",   "A2":"High"},    "Official",      0.30),
    ("E2", {"C4":"Low",    "A2":"Low"},     "Rumor",         0.45),
    ("E3", {"C3":"Helps Blue","E2":"Official"},  "Blue-favorable", 0.80),
    ("E3", {"C3":"Helps Red", "E2":"Official"},  "Red-favorable",  0.80),
    ("E4", {"C4":"High"},                   "High",          0.70),
    # D nodes
    ("D1", {"B1":"Strong","B2":"Weak","E3":"Blue-favorable"}, "Rising", 0.85),
    ("D1", {"B1":"Weak","B2":"Strong","E3":"Red-favorable"},  "Falling", 0.80),
    ("D2", {"B2":"Strong","B1":"Weak","E3":"Red-favorable"},  "Rising",  0.85),
    ("D3", {"A2":"Low","E2":"Rumor","E4":"High"},             "High",    0.75),
    ("D3", {"A2":"High","E2":"Official","E4":"Low"},          "Low",     0.70),
    # F nodes
    ("F1", {"D1":"Rising","D2":"Falling","D3":"Low"}, "Blue lead", 0.80),
    ("F1", {"D1":"Falling","D2":"Rising","D3":"Low"}, "Red lead",  0.80),
    ("F2", {"E3":"Blue-favorable","E4":"High"}, "Blue surge", 0.65),
    ("F2", {"E3":"Red-favorable", "E4":"High"}, "Red surge",  0.65),
    ("F3", {"F1":"Blue lead","E3":"Blue-favorable"}, "Blue favored", 0.80),
    ("F3", {"F1":"Red lead", "E3":"Red-favorable"},  "Red favored",  0.80),
    # G nodes
    ("G1", {"B3":"Blue advantage","D1":"Rising"}, "High",   0.65),
    ("G1", {"B3":"Red advantage", "D1":"Falling"},"Low",    0.60),
    ("G2", {"B3":"Red advantage", "D2":"Rising"}, "High",   0.65),
    ("G3", {"A3":"Toss-up","D1":"Rising","D2":"Falling"},   "Blue +5", 0.60),
    ("G3", {"A3":"Toss-up","D1":"Falling","D2":"Rising"},   "Red +5",  0.60),
    # H nodes
    ("H1", {"F1":"Blue lead","F3":"Blue favored","E2":"Official"}, "Likely Blue", 0.55),
    ("H1", {"F1":"Red lead", "F3":"Red favored", "E2":"Official"}, "Likely Red",  0.54),
    # I nodes
    ("I1", {"G1":"High","G2":"Low","G3":"Blue +5"}, "Blue narrow win", 0.50),
    ("I1", {"G1":"Low", "G2":"High","G3":"Red +5"}, "Red narrow win",  0.50),
    ("I2", {"I1":"Blue landslide"},   "Blue wins", 1.00),
    ("I2", {"I1":"Red landslide"},   "Red wins",  1.00),
    ("I2", {"I1":"Recount/disputed"},"Blue wins", 0.50),
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
            assert rule_states == valid, \
                f"{node_id} row {i}: dist keys {rule_states} != states {valid}"


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
