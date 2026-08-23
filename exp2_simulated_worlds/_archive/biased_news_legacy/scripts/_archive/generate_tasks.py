#!/usr/bin/env python3
"""Generate biased-news election tasks for training and evaluation.

Usage
-----
  # Default: 2000 train cities × 3 variants = 6000 train tasks
  python scripts/generate_tasks.py

  # Custom sizes:
  python scripts/generate_tasks.py --n-train 500 --n-eval 100

  # Out-of-distribution eval only (longer sequences):
  python scripts/generate_tasks.py --mode ood
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.temporal_dag import generate_tasks  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    n_biased  = sum(1 for r in rows if r["_hidden_bias"] == "biased")
    n_neutral = sum(1 for r in rows if r["_hidden_bias"] == "neutral")
    vals = [r["settlement_value"] for r in rows]
    print(
        f"  wrote {len(rows):>6} rows  →  {path}\n"
        f"           biased={n_biased}  neutral={n_neutral}  "
        f"gold_mean={sum(vals)/len(vals):.1f}  "
        f"gold_range=[{min(vals):.1f}, {max(vals):.1f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["all", "train", "eval", "ood"], default="all")
    ap.add_argument("--n-train",   type=int,  default=2000, help="Cities for training split")
    ap.add_argument("--n-eval",    type=int,  default=500,  help="Cities for in-dist eval split")
    ap.add_argument("--n-ood",     type=int,  default=200,  help="Cities for OOD eval split")
    ap.add_argument("--variants",  nargs="+", default=["V0", "V1", "V2"])
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    print(f"Generating biased-news tasks  [mode={args.mode}] …")

    if args.mode in ("all", "train"):
        rows = generate_tasks(
            n_cities=args.n_train,
            variants=args.variants,
            seed=args.seed,
            T_range=(6, 12),
        )
        _write_jsonl(args.output_dir / "biased_news_train.tasks.jsonl", rows)

    if args.mode in ("all", "eval"):
        rows = generate_tasks(
            n_cities=args.n_eval,
            variants=args.variants,
            seed=args.seed + 1,
            T_range=(6, 12),
        )
        _write_jsonl(args.output_dir / "biased_news_eval.tasks.jsonl", rows)

    if args.mode in ("all", "ood"):
        # Out-of-distribution: longer sequences, V0 only (no context hint)
        rows = generate_tasks(
            n_cities=args.n_ood,
            variants=["V0"],
            seed=args.seed + 2,
            T_range=(16, 20),
        )
        _write_jsonl(args.output_dir / "biased_news_eval_ood.tasks.jsonl", rows)


if __name__ == "__main__":
    main()
