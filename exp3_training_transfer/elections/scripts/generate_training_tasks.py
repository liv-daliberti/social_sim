#!/usr/bin/env python3
"""Generate election training tasks with exact conditional probabilities.

Sample N full DAG simulations. E values are drawn from their natural marginal
distribution P(E1,E2,E3,E4). The label is the analytical P(Blue wins | E)
returned by compute_exact_news_forecast(), with no Monte Carlo label noise.
Each row gets unique hidden-variable context even for the same E-tuple, giving
richer training signal and proportional E-combination coverage.

Usage
-----
  # 10 K rows:
  python scripts/generate_training_tasks.py

  # Custom size:
  python scripts/generate_training_tasks.py --rows 20000

"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.dag import simulate, compute_exact_news_forecast  # noqa: E402


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
        result  = simulate(seed=rng.randrange(2**63))
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
        "--rows", type=int, default=10_000,
        help="Total training rows to generate (default: 10000)",
    )
    ap.add_argument("--eval-frac",  type=float, default=0.20,  help="Held-out eval fraction (default 0.20)")
    ap.add_argument("--seed",       type=int,   default=42,    help="Shuffle seed for train/eval split")
    ap.add_argument("--output-dir", type=Path,  default=ROOT / "data")
    ap.add_argument("--train-file", default="elections_train.tasks.jsonl")
    ap.add_argument("--eval-file",  default="elections_eval.tasks.jsonl")
    args = ap.parse_args()

    print("Generating stochastic election training tasks …")
    rows = _generate_stochastic(n=args.rows, seed=args.seed)

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
        f"\n  total_rows={len(rows)}\n"
        f"  p(Blue) range: {min(probs):.3f} – {max(probs):.3f}  "
        f"mean={sum(probs)/len(probs):.3f}\n"
        f"  unique E-tuples: {n_unique}/216  avg rows/tuple: {avg_per_tuple:.1f}\n"
        f"  train={train_n}  eval={eval_n}"
    )


if __name__ == "__main__":
    main()
