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
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from engine.dag import simulate, get_structure, compute_exact_news_forecast, TOPO_ORDER, NODES

_ELECTIONS_ROOT   = Path(__file__).resolve().parent
_GRPO_REPORTS_DIR = _ELECTIONS_ROOT / "reports" / "tinker_elections_grpo"

# ── Sim-stats cache ───────────────────────────────────────────────────────────
_sim_stats_cache: dict = {}   # keyed by (str(path), mtime)

def _dist(rows: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        v = str(r.get(key) or "—")
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}

_HIDDEN_GROUPS = {
    "Fundamentals":     ["_hidden_economy", "_hidden_institutional_trust", "_hidden_partisan_baseline"],
    "Candidates":       ["_hidden_blue_candidate", "_hidden_red_candidate", "_hidden_ground_game"],
    "External Events":  ["_hidden_event_occurred", "_hidden_event_type", "_hidden_event_target", "_hidden_event_severity"],
    "Public Opinion":   ["_hidden_blue_momentum", "_hidden_red_momentum", "_hidden_voter_uncertainty", "_hidden_issue_salience"],
    "Mechanics":        ["_hidden_blue_turnout", "_hidden_red_turnout", "_hidden_independent_split"],
    "Outcome Dist.":    ["_hidden_vote_share_category"],
}
_HIDDEN_LABEL = {k.replace("_hidden_", "").replace("_", " ").title(): k for group in _HIDDEN_GROUPS.values() for k in group}

def _compute_sim_stats(tasks_file: str) -> dict:
    p = _ELECTIONS_ROOT / tasks_file
    if not p.exists():
        return {}
    mtime = p.stat().st_mtime
    cache_key = (str(p), mtime)
    if cache_key in _sim_stats_cache:
        return _sim_stats_cache[cache_key]
    rows: list[dict] = []
    try:
        with p.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 5000:
                    break
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return {}
    if not rows:
        return {}
    blue = sum(1 for r in rows if float(r.get("settlement_yes") or 0) > 0.5)

    # Visible (model-facing) distributions
    visible = {
        "news_tone":        _dist(rows, "news_tone"),
        "news_volume":      _dist(rows, "news_volume"),
        "news_reliability": _dist(rows, "news_reliability"),
        "news_type":        _dist(rows, "news_type"),
    }

    # Hidden (causal upstream) distributions grouped by category
    hidden_groups: dict = {}
    for group_name, fields in _HIDDEN_GROUPS.items():
        group: dict = {}
        for field in fields:
            label = field.replace("_hidden_", "").replace("_", " ").title()
            group[label] = _dist(rows, field)
        hidden_groups[group_name] = group

    stats = {
        "total_tasks":   len(rows),
        "blue_win_rate": round(blue / len(rows), 3),
        **visible,
        "hidden_groups": hidden_groups,
    }
    _sim_stats_cache[cache_key] = stats
    return stats


def _find_latest_run_dir() -> Path:
    """Return the most recently modified subdir of tinker_elections_grpo/ that has train_config.json.
    Falls back to the legacy 'run/' path if none found."""
    if env := os.environ.get("TRAINING_RUN_DIR"):
        return Path(env)
    candidates = [
        d for d in _GRPO_REPORTS_DIR.iterdir()
        if d.is_dir() and (d / "train_config.json").exists()
    ] if _GRPO_REPORTS_DIR.exists() else []
    if candidates:
        return max(candidates, key=lambda d: d.stat().st_mtime)
    return _GRPO_REPORTS_DIR / "run"


TRAINING_RUN_DIR = _find_latest_run_dir()

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
        n = min(int(request.args.get("n", 10000)), 10000)
    except (TypeError, ValueError):
        n = 10000

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
    #
    # When a node is forced via overrides its model marginal under the override is
    # a point-mass at the forced state (probability 1), not the prior CPT.  Using
    # the prior CPT would cause a spurious Hoeffding violation for every fixed node.
    node_counts:  dict = {nid: {} for nid in TOPO_ORDER}
    node_cpt_sum: dict = {nid: {s: 0.0 for s in NODES[nid]["states"]}
                          for nid in TOPO_ORDER}

    # News-forecast joint: key = "E1|E2|E3|E4", value = {I2_state: count}
    news_joint:  dict = {}
    e3_joint:    dict = {}   # E3 → {I2_state: count}  (fast single-var view)

    for _ in range(n):
        r = simulate(overrides=overrides)
        states = r["states"]
        for nid, state in states.items():
            node_counts[nid][state] = node_counts[nid].get(state, 0) + 1
        for nid, dist in r["distributions"].items():
            if nid in overrides:
                # Fixed node: conditional model marginal is a point-mass at the forced state
                forced = states[nid]
                for s in dist:
                    node_cpt_sum[nid][s] += 1.0 if s == forced else 0.0
            else:
                for s, p in dist.items():
                    node_cpt_sum[nid][s] += p

        i2  = states.get("I2", "")
        e3  = states.get("E3", "")
        e_key = (f"{states.get('E1','')}|{states.get('E2','')}|"
                 f"{states.get('E3','')}|{states.get('E4','')}")

        e3_bucket = e3_joint.setdefault(e3, {})
        e3_bucket[i2] = e3_bucket.get(i2, 0) + 1

        bucket = news_joint.setdefault(e_key, {})
        bucket[i2] = bucket.get(i2, 0) + 1

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

    # ── News-forecast stats ────────────────────────────────────────────────────
    # Bayes accuracy: for each (E1,E2,E3,E4) cell, the best prediction wins.
    # Only count cells with ≥5 samples to avoid single-sample overfitting.
    MIN_CELL = 5
    correct_news = sum(max(c.values()) for c in news_joint.values()
                       if c and sum(c.values()) >= MIN_CELL)
    n_covered    = sum(sum(c.values()) for c in news_joint.values()
                       if sum(c.values()) >= MIN_CELL)
    bayes_acc    = round(correct_news / n_covered, 4) if n_covered else 0.5

    # Base-rate accuracy: always predict the majority outcome
    base_acc = round(max(winner_counts.values()) / total, 4) if winner_counts else 0.5
    base_blue = round(winner_counts.get("Blue wins", 0) / total, 4) if winner_counts else 0.5

    # P(Blue wins | E3) for each News Tone value
    e3_blue_prob = {}
    for e3_val, counts in e3_joint.items():
        t = sum(counts.values())
        e3_blue_prob[e3_val] = round(counts.get("Blue wins", 0) / t, 3) if t else 0.5

    # Lookup table: P(Blue | E1,E2,E3,E4) — only cells with ≥5 samples
    news_conditional = {
        key: round(c.get("Blue wins", 0) / sum(c.values()), 3)
        for key, c in news_joint.items()
        if sum(c.values()) >= 5
    }

    # Empirical Bayes ceiling: irreducible variance when predicting from E signals.
    # Brier_emp = Σ_E (N_E/n) · p̂_E · (1 − p̂_E) · N_E/(N_E−1)
    # The bias correction factor N_E/(N_E−1) removes the downward bias in p̂(1−p̂)
    # (MLE estimator is biased: E[p̂(1−p̂)] = p*(1−p*) · (N_E−1)/N_E).
    # Without correction, small cells inflate the empirical reward ceiling significantly.
    brier_emp_ceiling = 0.0
    for c in news_joint.values():
        cell_n = sum(c.values())
        if cell_n < 2:
            continue
        p_bl = c.get("Blue wins", 0) / cell_n
        brier_emp_ceiling += (cell_n / total) * p_bl * (1.0 - p_bl) * cell_n / (cell_n - 1)
    reward_emp_ceiling = round(2.0 * (0.25 - brier_emp_ceiling), 4)

    news_forecast = {
        "bayes_accuracy":      bayes_acc,
        "base_accuracy":       base_acc,
        "base_blue":           base_blue,
        "info_gain_pp":        round((bayes_acc - base_acc) * 100, 1),
        "e3_blue_prob":        e3_blue_prob,           # P(Blue | News Tone)
        "news_conditional":    news_conditional,        # P(Blue | all 4 E vars)
        "brier_emp_ceiling":   round(brier_emp_ceiling, 4),
        "reward_emp_ceiling":  reward_emp_ceiling,
    }

    return jsonify({
        "n":             total,
        "counts":        winner_counts,
        "pct":           {k: round(v / total, 4) for k, v in winner_counts.items()},
        "nodes":         node_freqs,      # empirical marginals  (green bar)
        "nodes_cpt_avg": node_cpt_avg,    # model-predicted marginals  (blue bar in Hoeffding)
        "news_forecast": news_forecast,
    })


@app.route("/api/training/status")
def training_status():
    """Return training config + aggregate metrics + sparkline timeseries."""
    run_dir = _find_latest_run_dir()

    # Config — written at the start of a training run
    config: dict = {}
    for cfg_file in ("train_config.json", "run_manifest.json"):
        p = run_dir / cfg_file
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                config = raw.get("training", raw)  # run_manifest nests under "training"
                break
            except (json.JSONDecodeError, OSError):
                pass

    # Rollout log
    rollouts_path = run_dir / "train_rollouts.jsonl"
    all_rows: list[dict] = []
    if rollouts_path.exists():
        try:
            with rollouts_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            all_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    is_eval_mode = lambda r: str(r.get("mode", "")).lower() in {
        "val", "validation", "rolloutmode.val"
    }
    train_rows = [r for r in all_rows if not is_eval_mode(r)]
    eval_rows  = [r for r in all_rows if is_eval_mode(r)]
    recent     = train_rows[-50:]  # last 50 train rollouts for live metrics

    def _mean(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    metrics = {
        "total_rollouts": len(all_rows),
        "train_rollouts": len(train_rows),
        "eval_rollouts":  len(eval_rows),
        "recent_mean_reward": _mean(recent, "reward"),
        "recent_mean_brier":  _mean(recent, "brier_loss"),
        "recent_accuracy": (
            round(sum(1 for r in recent if r.get("is_correct") is True) / len(recent), 4)
            if recent else None
        ),
    }

    # Activity probe — file touched within last 30 s
    is_active = (
        rollouts_path.exists()
        and (time.time() - rollouts_path.stat().st_mtime) < 30
    )

    # Read step_metrics first so we can use eval count to align all series.
    step_metrics: list[dict] = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        try:
            with metrics_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            step_metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    # Per-step timeseries — group rollouts by optimizer step so all charts share the same x-axis.
    # Train: batch_size × group_size rollouts per step.
    # Eval:  eval_task_rows rollouts per eval pass.
    cfg_path = run_dir / "train_config.json"
    _cfg: dict = {}
    if cfg_path.exists():
        try:
            _cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    train_chunk = max(1, int(_cfg.get("batch_size", 4)) * int(_cfg.get("group_size", 4)))
    eval_chunk  = max(1, int(_cfg.get("eval_task_rows") or _cfg.get("eval_max_tasks") or 32))

    # Number of eval passes that appear in step_metrics (authoritative x-axis for eval charts).
    n_eval_steps = len([s for s in step_metrics if s.get("test/env/all/reward/total") is not None])

    def _per_step(rows: list[dict], chunk: int, key: str, as_accuracy: bool = False,
                  limit: int = 0) -> list[float]:
        """Average a metric over each chunk of `chunk` rollouts (= one optimizer step).
        If limit > 0, keep only the last `limit` chunks (aligns eval series to step_metrics count)."""
        result = []
        for i in range(0, len(rows), chunk):
            batch = rows[i : i + chunk]
            if as_accuracy:
                vals = [1.0 if r["is_correct"] else 0.0 for r in batch if r.get("is_correct") is not None]
            else:
                vals = [float(r[key]) for r in batch if r.get(key) is not None]
            if vals:
                result.append(round(sum(vals) / len(vals), 5))
        return result[-limit:] if limit and len(result) > limit else result

    n_train_steps = len(step_metrics)  # authoritative x-axis for train charts

    timeseries          = _per_step(train_rows, train_chunk, "reward",     limit=n_train_steps)
    timeseries_brier    = _per_step(train_rows, train_chunk, "brier_loss", limit=n_train_steps)
    timeseries_logloss  = _per_step(train_rows, train_chunk, "log_loss",   limit=n_train_steps)
    timeseries_accuracy = _per_step(train_rows, train_chunk, "", as_accuracy=True, limit=n_train_steps)

    # Eval rollout series aligned to n_eval_steps (drop any pre-training extra eval pass)
    timeseries_eval_brier    = _per_step(eval_rows, eval_chunk, "brier_loss",  limit=n_eval_steps)
    timeseries_eval_logloss  = _per_step(eval_rows, eval_chunk, "log_loss",    limit=n_eval_steps)
    timeseries_eval_accuracy = _per_step(eval_rows, eval_chunk, "", as_accuracy=True, limit=n_eval_steps)

    # Checkpoints
    ckpts: dict = {"count": 0, "latest": None}
    ckpt_path = run_dir / "checkpoints.jsonl"
    if ckpt_path.exists():
        ckpt_rows: list[dict] = []
        try:
            with ckpt_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            ckpt_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        ckpts = {"count": len(ckpt_rows), "latest": ckpt_rows[-1] if ckpt_rows else None}

    sim_stats = _compute_sim_stats(config.get("tasks_file", "")) if config.get("tasks_file") else {}

    # Detect label type from rollouts: if settlement_yes is a non-integer float it's a
    # probability label (p*); if it's 0.0 or 1.0 exclusively it's a binary label.
    label_type = "unknown"
    sy_vals = [r["settlement_yes"] for r in all_rows[:200] if r.get("settlement_yes") is not None]
    if sy_vals:
        non_binary = [v for v in sy_vals if float(v) not in (0.0, 1.0)]
        label_type = "float_prob" if non_binary else "binary"

    def _extract_step_series(key: str) -> list[float]:
        return [round(float(s[key]), 6) for s in step_metrics if s.get(key) is not None]

    return jsonify({
        "exists":             rollouts_path.exists() or bool(config),
        "label_type":         label_type,
        "is_active":          is_active,
        "config":             config,
        "metrics":            metrics,
        "timeseries":          timeseries,
        "timeseries_brier":    timeseries_brier,
        "timeseries_logloss":  timeseries_logloss,
        "timeseries_accuracy": timeseries_accuracy,
        "step_metrics":        step_metrics,
        "step_kl":             _extract_step_series("optim/kl_sample_train_v1"),
        "step_entropy":        _extract_step_series("optim/entropy"),
        "step_lr":             _extract_step_series("optim/lr"),
        "step_eval_reward":    [round(float(s["test/env/all/reward/total"]), 5)
                                for s in step_metrics if s.get("test/env/all/reward/total") is not None],
        "timeseries_eval_brier":    timeseries_eval_brier,
        "timeseries_eval_logloss":  timeseries_eval_logloss,
        "timeseries_eval_accuracy": timeseries_eval_accuracy,
        "checkpoints":         ckpts,
        "sim_stats":           sim_stats,
    })


def _rollout_stats(rows: list[dict]) -> dict:
    rewards  = [float(r["reward"])     for r in rows if r.get("reward")     is not None]
    briers   = [float(r["brier_loss"]) for r in rows if r.get("brier_loss") is not None]
    accurate = [r for r in rows if r.get("is_correct") is not None]
    def mn(vals): return round(sum(vals)/len(vals), 4) if vals else None
    return {
        "count":        len(rows),
        "mean_reward":  mn(rewards),
        "mean_brier":   mn(briers),
        "accuracy":     round(sum(1 for r in accurate if r["is_correct"]) / len(accurate), 3) if accurate else None,
    }


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("conversation_trace", "raw_response")}


@app.route("/api/training/rollouts")
def training_rollouts():
    """Return rollout records, flat (default) or grouped by batch (?grouped=1)."""
    run_dir = _find_latest_run_dir()
    rollouts_path = run_dir / "train_rollouts.jsonl"

    mode    = request.args.get("mode", "train")   # "train" | "val"
    grouped = request.args.get("grouped", "0") in ("1", "true")

    all_rows: list[dict] = []
    if rollouts_path.exists():
        try:
            with rollouts_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            all_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    is_eval = lambda r: str(r.get("mode", "")).lower() in {"val", "validation", "rolloutmode.val"}

    if mode == "train":
        filtered = [r for r in all_rows if not is_eval(r)]
    else:
        filtered = [r for r in all_rows if is_eval(r)]

    if not grouped:
        limit  = min(int(request.args.get("limit",  20)), 200)
        offset = max(int(request.args.get("offset",  0)),  0)
        page   = list(reversed(filtered))[offset:offset + limit]
        return jsonify({
            "rollouts": [_slim(r) for r in page],
            "total":    len(filtered),
            "mode":     mode,
            "offset":   offset,
            "limit":    limit,
        })

    # Grouped mode — chunk by inferred batch size
    cfg_path = run_dir / "train_config.json"
    cfg: dict = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if mode == "train":
        chunk_size = int(cfg.get("batch_size", 4)) * int(cfg.get("group_size", 4))
    else:
        chunk_size = int(cfg.get("eval_task_rows") or cfg.get("eval_max_tasks") or 32)

    chunk_size = max(1, chunk_size)
    batches: list[dict] = []
    for i in range(0, len(filtered), chunk_size):
        chunk = filtered[i : i + chunk_size]
        batches.append({
            "batch_idx": len(batches),
            "stats":     _rollout_stats(chunk),
            "rollouts":  [_slim(r) for r in chunk],
        })

    return jsonify({
        "batches": list(reversed(batches)),   # most-recent first
        "total":   len(filtered),
        "mode":    mode,
    })


@app.route("/api/update_cpt", methods=["POST"])
def update_cpt():
    """Receive {updates: {nid: cpt_list}} and patch NODES in memory."""
    global _exact_forecast_cache
    body = request.get_json() or {}
    changed = []
    for nid, cpt in body.get("updates", {}).items():
        if nid in NODES and isinstance(cpt, list):
            NODES[nid]["cpt"] = cpt
            changed.append(nid)
    if changed:
        _exact_forecast_cache.clear()   # invalidate all override variants on CPT edit
    return jsonify({"ok": True, "updated": changed})


# ── Exact news-forecast (analytical, non-blocking, cached) ───────────────────

import threading as _threading
import hashlib as _hashlib

# Cache keyed by (cpt_hash, overrides_json) so different override combinations
# each get their own cached result.
_exact_forecast_cache:   dict = {}    # key → result dict
_exact_forecast_running: set  = set() # keys currently being computed
_exact_forecast_lock = _threading.Lock()


def _cpt_hash() -> str:
    data = json.dumps(
        {nid: node.get("cpt", list(node.get("prior", {}).items()))
         for nid, node in NODES.items()},
        sort_keys=True
    )
    return _hashlib.md5(data.encode()).hexdigest()[:8]


def _exact_key(overrides: dict) -> tuple:
    return (_cpt_hash(), json.dumps(overrides, sort_keys=True))


def _run_exact_forecast(key: tuple, overrides: dict) -> None:
    result = compute_exact_news_forecast(overrides=overrides)
    with _exact_forecast_lock:
        _exact_forecast_cache[key] = result
        _exact_forecast_running.discard(key)


@app.route("/api/exact_forecast")
def exact_forecast():
    """
    Exact analytical P(Blue | E1,E2,E3,E4) via variable elimination.
    Accepts ?overrides=<json> so fixed variables are respected.
    Non-blocking: returns {"status":"computing"} on first call, then
    {"status":"ready",...} once the background thread finishes.
    Each (CPT snapshot, overrides) combination is cached independently.
    """
    overrides: dict = {}
    raw = request.args.get("overrides")
    if raw:
        try:
            overrides = json.loads(raw)
        except json.JSONDecodeError:
            pass

    key = _exact_key(overrides)

    with _exact_forecast_lock:
        if key in _exact_forecast_cache:
            return jsonify({"status": "ready", **_exact_forecast_cache[key]})
        if key in _exact_forecast_running:
            return jsonify({"status": "computing"})
        _exact_forecast_running.add(key)

    _threading.Thread(
        target=_run_exact_forecast, args=(key, overrides), daemon=True
    ).start()
    return jsonify({"status": "computing"})


# ── Bayes ceiling: analytical + empirical ────────────────────────────────────

_bayes_empirical_cache = None    # type: dict
_BAYES_EMPIRICAL_TTL   = 300     # seconds

def _get_analytical_bayes():
    """Return cached analytical Bayes ceiling (no-override case)."""
    key = _exact_key({})
    with _exact_forecast_lock:
        if key in _exact_forecast_cache:
            return _exact_forecast_cache[key]
    # Not yet computed — run inline (fast, ~0.1 s)
    result = compute_exact_news_forecast(overrides={})
    with _exact_forecast_lock:
        _exact_forecast_cache[key] = result
    return result


def _get_empirical_bayes(n=10000):
    """Empirical Bayes ceiling from n unconditional simulations, cached for 5 min."""
    global _bayes_empirical_cache
    now = time.time()
    if _bayes_empirical_cache and now - _bayes_empirical_cache["computed_at"] < _BAYES_EMPIRICAL_TTL:
        return _bayes_empirical_cache["result"]

    news_joint = {}
    for _ in range(n):
        r      = simulate()
        states = r["states"]
        i2     = states.get("I2", "")
        ek     = (f"{states.get('E1','')}|{states.get('E2','')}|"
                  f"{states.get('E3','')}|{states.get('E4','')}")
        bucket = news_joint.setdefault(ek, {"blue": 0, "total": 0})
        bucket["total"] += 1
        if "Blue" in i2:
            bucket["blue"] += 1

    cond = {}
    for ek, counts in news_joint.items():
        t = counts["total"]
        if t:
            p_blue = counts["blue"] / t
            cond[ek] = {"p_blue": round(p_blue, 4), "p_e": round(t / n, 6), "count": t}

    p_tot = sum(d["p_e"] for d in cond.values()) or 1.0
    # Bias-corrected Brier estimator: multiply each cell by N_e/(N_e-1) to remove
    # the downward bias in p̂(1−p̂). Without correction, empirical reward ceiling
    # is inflated because E[p̂(1−p̂)] = p*(1−p*) · (N_e−1)/N_e < p*(1−p*).
    brier_optimal = sum(
        d["p_e"] * d["p_blue"] * (1 - d["p_blue"]) * d["count"] / (d["count"] - 1)
        for d in cond.values() if d["count"] >= 2
    ) / p_tot
    reward_ceiling = round(2 * (0.25 - brier_optimal), 4)
    bayes_acc      = sum(d["p_e"] * max(d["p_blue"], 1 - d["p_blue"]) for d in cond.values()) / p_tot
    base_blue      = sum(d["p_e"] * d["p_blue"] for d in cond.values()) / p_tot

    e3_b: dict = {}; e3_t: dict = {}
    for ek, d in cond.items():
        e3 = ek.split("|")[2]
        e3_b[e3] = e3_b.get(e3, 0.0) + d["p_e"] * d["p_blue"]
        e3_t[e3] = e3_t.get(e3, 0.0) + d["p_e"]
    e3_blue_prob = {k: round(e3_b[k] / e3_t[k], 3) for k in e3_t if e3_t[k] > 0}

    result = {
        "brier_optimal":  round(brier_optimal, 4),
        "reward_ceiling": reward_ceiling,
        "bayes_accuracy": round(bayes_acc, 4),
        "base_blue":      round(base_blue, 4),
        "e3_blue_prob":   e3_blue_prob,
        "n_simulations":  n,
        "conditional":    cond,
    }
    _bayes_empirical_cache = {"result": result, "computed_at": now}
    return result


@app.route("/api/training/bayes")
def training_bayes():
    """
    Bayes-optimal learning ceiling, two ways:
      analytical — exact variable elimination over the DAG (cached forever per CPT)
      empirical  — estimated from 10 000 unconditional simulations (cached 5 min)

    Also detects current label type and provides the correct reward ceiling:
      float_prob labels  → reward_ceiling = 0.5  (model can learn p* exactly)
      binary labels      → reward_ceiling = brier-based ceiling (~0.017)
    In both cases, accuracy_ceiling = bayes_accuracy (57.3%) is meaningful:
    it is the best possible classification accuracy given only news signals.
    """
    analytical = _get_analytical_bayes()
    empirical  = _get_empirical_bayes(n=10000)

    # Detect label type from current training data
    run_dir = _find_latest_run_dir()
    rollouts_path = run_dir / "train_rollouts.jsonl"
    label_type = "unknown"
    try:
        sy_vals = []
        if rollouts_path.exists():
            with rollouts_path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 100:
                        break
                    row = json.loads(line.strip()) if line.strip() else {}
                    v = row.get("settlement_yes")
                    if v is not None:
                        sy_vals.append(float(v))
        if sy_vals:
            non_binary = [v for v in sy_vals if v not in (0.0, 1.0)]
            label_type = "float_prob" if non_binary else "binary"
    except Exception:
        pass

    # Reward ceiling depends on label type
    # Float labels: model learns p* exactly → max reward = 0.5
    # Binary labels: irreducible binary noise → max reward = brier-based analytical ceiling
    reward_ceiling_for_labels = {
        "float_prob": 0.5,
        "binary":     analytical.get("reward_ceiling"),
        "unknown":    None,
    }
    acc_ceiling = analytical.get("bayes_accuracy")   # always meaningful for classification

    return jsonify({
        "analytical":               analytical,
        "empirical":                empirical,
        "label_type":               label_type,
        "reward_ceiling":           reward_ceiling_for_labels[label_type],
        "reward_ceiling_binary":    analytical.get("reward_ceiling"),   # for reference
        "accuracy_ceiling":         acc_ceiling,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  Election simulator running at  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
