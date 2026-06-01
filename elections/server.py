"""
Flask server for the Blue-ian vs Red-ian Election Simulator.

Routes
------
GET  /              → serve the UI
GET  /api/structure → DAG node/edge/group definitions
POST /api/simulate  → run one election, return full trace + distributions
GET  /api/batch     → run N elections; returns winner pct + per-node empirical marginals
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from engine.dag import simulate, get_structure, TOPO_ORDER, NODES

app = Flask(__name__, static_folder="ui", static_url_path="")


# ── Static UI ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/structure")
def structure():
    return jsonify(get_structure())


@app.route("/api/simulate", methods=["POST"])
def run_simulate():
    body      = request.get_json(silent=True) or {}
    seed      = body.get("seed")
    overrides = body.get("overrides", {})

    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    result = simulate(overrides=overrides, seed=seed)

    def rnd4(d): return {k: round(v, 4) for k, v in d.items()}

    return jsonify({
        "states":        result["states"],
        "probs":         rnd4(result["probs"]),
        "surprise":      {k: round(v, 3) for k, v in result["surprise"].items()},
        "distributions": {nid: rnd4(dist) for nid, dist in result["distributions"].items()},
        "narrative":     result["narrative"],
        "seed":          seed,
    })


@app.route("/api/batch")
def batch():
    """
    Run N simulations.
    Returns:
      - winner pct (for the banner)
      - per-node empirical marginal frequencies (for the DAG second bar)
    """
    try:
        n = min(int(request.args.get("n", 1000)), 10000)
    except (TypeError, ValueError):
        n = 1000

    overrides = {}
    raw = request.args.get("overrides")
    if raw:
        try:
            overrides = json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Accumulate state counts AND CPT probability sums across all runs.
    # The average CPT probability  E[P(X=s | parents)]  equals the true model
    # marginal P(X=s) by the law of total expectation.  Comparing that against
    # the empirical frequency is the correct Hoeffding test: both should converge
    # to the same value if the simulator is internally consistent.
    node_counts:  dict = {nid: {} for nid in TOPO_ORDER}
    node_cpt_sum: dict = {nid: {s: 0.0 for s in NODES[nid]["states"]}
                          for nid in TOPO_ORDER}

    for _ in range(n):
        r = simulate(overrides=overrides)
        for nid, state in r["states"].items():
            node_counts[nid][state] = node_counts[nid].get(state, 0) + 1
        for nid, dist in r["distributions"].items():
            for s, p in dist.items():
                node_cpt_sum[nid][s] += p

    total = n

    # Empirical marginal P̂(X=s) = count/n
    node_freqs = {
        nid: {s: round(node_counts[nid].get(s, 0) / total, 4)
              for s in NODES[nid]["states"]}
        for nid in TOPO_ORDER
    }

    # Model marginal  E[P(X=s|parents)] = sum_of_CPT_probs / n
    # This is what the simulator *predicts* the marginal should be.
    # A Hoeffding violation vs node_freqs means the sampler and the CPT disagree.
    node_cpt_avg = {
        nid: {s: round(node_cpt_sum[nid][s] / total, 4)
              for s in NODES[nid]["states"]}
        for nid in TOPO_ORDER
    }

    winner_counts = node_counts.get("I2", {})

    return jsonify({
        "n":            total,
        "counts":       winner_counts,
        "pct":          {k: round(v / total, 4) for k, v in winner_counts.items()},
        "nodes":        node_freqs,     # empirical marginals  (green bar)
        "nodes_cpt_avg": node_cpt_avg,  # model-predicted marginals  (blue bar in Hoeffding)
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  Election simulator running at  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
