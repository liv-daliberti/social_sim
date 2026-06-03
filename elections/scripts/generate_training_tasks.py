#!/usr/bin/env python3
"""Generate election training tasks with TRUE conditional probabilities as targets.

Two modes
---------
enumerated (legacy)
    Enumerate all 216 unique (E1,E2,E3,E4) combinations.  For each combo run
    --sims simulations to estimate P(Blue wins | E).  One row per combo.

stochastic (default)
    Sample N full DAG simulations.  E values are drawn from their natural
    marginal distribution P(E1,E2,E3,E4).  The label is the EXACT analytical
    P(Blue wins | E) looked up from compute_exact_news_forecast() — zero MC
    noise.  Each row gets unique hidden-variable context even for the same
    E-tuple, giving richer training signal and proportional E-combo coverage.

Usage
-----
  # 10 K stochastic rows (recommended):
  python scripts/generate_training_tasks.py

  # Custom size:
  python scripts/generate_training_tasks.py --rows 20000

  # Legacy enumerated mode (216 rows, estimated labels):
  python scripts/generate_training_tasks.py --mode enumerated --sims 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.dag import NODES, simulate, compute_exact_news_forecast  # noqa: E402

E1_VALS = list(NODES["E1"]["states"])   # News Type   (6 values)
E2_VALS = list(NODES["E2"]["states"])   # Reliability (4 values)
E3_VALS = list(NODES["E3"]["states"])   # Tone        (3 values)
E4_VALS = list(NODES["E4"]["states"])   # Volume      (3 values)


# ── Stochastic mode ───────────────────────────────────────────────────────────

def _generate_stochastic(n: int, seed: int) -> list[dict]:
    """Sample n full-DAG simulations; label each with the exact analytical P(Blue|E)."""
    print("  Computing exact analytical conditional probabilities …", flush=True)
    forecast = compute_exact_news_forecast()
    # keyed "E1|E2|E3|E4" → p_blue
    exact: dict[str, float] = {
        k: v["p_blue"] for k, v in forecast["conditional"].items()
    }
    n_exact = len(exact)
    print(f"  Loaded {n_exact} exact E-tuple probabilities "
          f"(range {min(exact.values()):.3f}–{max(exact.values()):.3f})")

    rng = random.Random(seed)
    tasks: list[dict] = []
    unseen: int = 0

    for i in range(n):
        result  = simulate()
        s       = result["states"]
        e1, e2, e3, e4 = s["E1"], s["E2"], s["E3"], s["E4"]
        ek      = f"{e1}|{e2}|{e3}|{e4}"
        p_blue  = exact.get(ek)

        if p_blue is None:
            # Extremely rare: E-combo not in analytical table (shouldn't happen)
            unseen += 1
            p_blue = 0.5

        task_id = (f"sim_{i:06d}_{e1}_{e2}_{e3}_{e4}"
                   .lower().replace(" ", "_").replace("/", "_").replace("-", "_"))

        tasks.append({
            "task_id":          task_id,
            "question":         "Will the Blue candidate win this election?",
            "settlement_yes":   round(p_blue, 6),
            "news_type":        e1,
            "news_reliability": e2,
            "news_tone":        e3,
            "news_volume":      e4,
            # Hidden causal context (unique per row — different upstream draws)
            "_hidden_economy":             s.get("A1"),
            "_hidden_institutional_trust": s.get("A2"),
            "_hidden_partisan_baseline":   s.get("A3"),
            "_hidden_blue_candidate":      s.get("B1"),
            "_hidden_red_candidate":       s.get("B2"),
            "_hidden_ground_game":         s.get("B3"),
            "_hidden_event_occurred":      s.get("C1"),
            "_hidden_event_type":          s.get("C2"),
            "_hidden_event_target":        s.get("C3"),
            "_hidden_event_severity":      s.get("C4"),
            "_hidden_blue_momentum":       s.get("D1"),
            "_hidden_red_momentum":        s.get("D2"),
            "_hidden_voter_uncertainty":   s.get("D3"),
            "_hidden_issue_salience":      s.get("D4"),
            "_hidden_blue_turnout":        s.get("G1"),
            "_hidden_red_turnout":         s.get("G2"),
            "_hidden_independent_split":   s.get("G3"),
            "_hidden_vote_share_category": s.get("I1"),
            "_hidden_winner":              s.get("I2"),
            "_exact_label":                True,
        })

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{n} rows sampled …", flush=True)

    if unseen:
        print(f"  WARNING: {unseen} rows used fallback p=0.5 (E-tuple not in analytical table)")
    return tasks


# ── Enumerated mode (legacy) ──────────────────────────────────────────────────

def _estimate_prob(e1: str, e2: str, e3: str, e4: str, n_sims: int) -> tuple[float, dict]:
    overrides = {"E1": e1, "E2": e2, "E3": e3, "E4": e4}
    wins = 0
    last_states: dict = {}
    for _ in range(n_sims):
        result = simulate(overrides=overrides)
        s = result["states"]
        if s["I2"] == "Blue wins":
            wins += 1
        last_states = s
    return wins / n_sims, last_states


def _generate_enumerated(n_sims: int) -> list[dict]:
    all_combos = list(product(E1_VALS, E2_VALS, E3_VALS, E4_VALS))
    print(f"  {len(all_combos)} unique E-combinations.")
    print(f"  Computing exact observational P(Blue wins | E) via variable elimination …")

    forecast   = compute_exact_news_forecast()
    cond_probs = {k: v["p_blue"] for k, v in forecast["conditional"].items()}

    tasks: list[dict] = []
    for i, (e1, e2, e3, e4) in enumerate(all_combos):
        key      = f"{e1}|{e2}|{e3}|{e4}"
        true_prob = cond_probs[key]
        sample_s  = simulate(overrides={"E1": e1, "E2": e2, "E3": e3, "E4": e4})["states"]
        task_id   = (f"e_{e1}_{e2}_{e3}_{e4}"
                     .lower().replace(" ", "_").replace("/", "_").replace("-", "_"))
        tasks.append({
            "task_id":          task_id,
            "question":         "Will the Blue candidate win this election?",
            "settlement_yes":   round(true_prob, 6),
            "news_type":        e1,
            "news_reliability": e2,
            "news_tone":        e3,
            "news_volume":      e4,
            "_hidden_economy":             sample_s.get("A1"),
            "_hidden_institutional_trust": sample_s.get("A2"),
            "_hidden_partisan_baseline":   sample_s.get("A3"),
            "_hidden_blue_candidate":      sample_s.get("B1"),
            "_hidden_red_candidate":       sample_s.get("B2"),
            "_hidden_ground_game":         sample_s.get("B3"),
            "_hidden_event_occurred":      sample_s.get("C1"),
            "_hidden_event_type":          sample_s.get("C2"),
            "_hidden_event_target":        sample_s.get("C3"),
            "_hidden_event_severity":      sample_s.get("C4"),
            "_hidden_blue_momentum":       sample_s.get("D1"),
            "_hidden_red_momentum":        sample_s.get("D2"),
            "_hidden_voter_uncertainty":   sample_s.get("D3"),
            "_hidden_issue_salience":      sample_s.get("D4"),
            "_hidden_blue_turnout":        sample_s.get("G1"),
            "_hidden_red_turnout":         sample_s.get("G2"),
            "_hidden_independent_split":   sample_s.get("G3"),
            "_hidden_vote_share_category": sample_s.get("I1"),
            "_hidden_winner":              sample_s.get("I2"),
            "_exact_label":                True,
        })
        if (i + 1) % 36 == 0:
            print(f"    {i+1}/{len(all_combos)} combos done …", flush=True)
    return tasks


# ── I/O ───────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            fh.write("\n")
    print(f"  wrote {len(rows):>6} rows  →  {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", choices=["stochastic", "enumerated"], default="stochastic",
        help="Generation mode (default: stochastic)",
    )
    ap.add_argument(
        "--rows", type=int, default=10_000,
        help="[stochastic] Total training rows to generate (default: 10000)",
    )
    ap.add_argument(
        "--sims", type=int, default=2_000,
        help="[enumerated] Simulations per E-combo (default: 2000)",
    )
    ap.add_argument("--eval-frac",  type=float, default=0.20,  help="Held-out eval fraction (default 0.20)")
    ap.add_argument("--seed",       type=int,   default=42,    help="Shuffle seed for train/eval split")
    ap.add_argument("--output-dir", type=Path,  default=ROOT / "data")
    ap.add_argument("--train-file", default="elections_train.tasks.jsonl")
    ap.add_argument("--eval-file",  default="elections_eval.tasks.jsonl")
    args = ap.parse_args()

    print(f"Generating election training tasks  [mode={args.mode}] …")

    if args.mode == "stochastic":
        rows = _generate_stochastic(n=args.rows, seed=args.seed)
    else:
        rows = _generate_enumerated(n_sims=args.sims)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    eval_n     = max(1, round(len(rows) * args.eval_frac))
    train_n    = len(rows) - eval_n
    train_rows = rows[:train_n]
    eval_rows  = rows[train_n:]

    _write_jsonl(args.output_dir / args.train_file, train_rows)
    _write_jsonl(args.output_dir / args.eval_file,  eval_rows)

    probs = [r["settlement_yes"] for r in rows]

    # E-tuple coverage stats
    e_counts: dict[str, int] = {}
    for r in rows:
        ek = f"{r['news_type']}|{r['news_reliability']}|{r['news_tone']}|{r['news_volume']}"
        e_counts[ek] = e_counts.get(ek, 0) + 1
    n_unique = len(e_counts)
    avg_per_tuple = sum(e_counts.values()) / n_unique if n_unique else 0

    print(
        f"\n  mode={args.mode}  total_rows={len(rows)}\n"
        f"  p(Blue) range: {min(probs):.3f} – {max(probs):.3f}  "
        f"mean={sum(probs)/len(probs):.3f}\n"
        f"  unique E-tuples: {n_unique}/216  avg rows/tuple: {avg_per_tuple:.1f}\n"
        f"  train={train_n}  eval={eval_n}"
    )


if __name__ == "__main__":
    main()
