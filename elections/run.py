#!/usr/bin/env python3
"""CLI runner — simulate one election and print the causal trace."""

import sys
import json
import argparse
import os

sys.path.insert(0, os.path.dirname(__file__))
from engine.dag import simulate, NODES

def main():
    ap = argparse.ArgumentParser(description="Blue-ian vs Red-ian Election CLI")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--override", action="append", metavar="NODE=STATE",
                    help="Fix a node (e.g. --override A1=Weak)")
    ap.add_argument("--json", action="store_true", help="Output full JSON")
    ap.add_argument("--batch", type=int, default=0, metavar="N",
                    help="Run N simulations and print winner distribution")
    args = ap.parse_args()

    overrides = {}
    for ov in (args.override or []):
        k, _, v = ov.partition("=")
        overrides[k.strip()] = v.strip()

    if args.batch > 0:
        counts = {}
        for i in range(args.batch):
            r = simulate(overrides=overrides)
            w = r["states"]["I2"]
            counts[w] = counts.get(w, 0) + 1
        total = sum(counts.values())
        print(f"\nBatch results ({total} runs):\n")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(v / total * 40)
            print(f"  {k:<25} {v:>5}  {v/total:>5.1%}  {bar}")
        print()
        return

    result = simulate(overrides=overrides, seed=args.seed)

    if args.json:
        print(json.dumps({
            "states":   result["states"],
            "probs":    {k: round(v, 4) for k, v in result["probs"].items()},
            "surprise": {k: round(v, 3) for k, v in result["surprise"].items()},
        }, indent=2))
        return

    # Pretty print
    print()
    for phase in result["narrative"]:
        print(f"  {'━'*60}")
        print(f"  {phase['phase']}")
        print(f"  {'━'*60}")
        for it in phase["items"]:
            surprise_tag = ""
            if it["surprise"] > 2.5:
                surprise_tag = "  (!)"
            elif it["surprise"] > 1.0:
                surprise_tag = "  (~)"
            print(f"  {it['label']:<28}  {it['state']:<22}  p={it['prob']:.0%}{surprise_tag}")
        print()

    winner = result["states"]["I2"]
    print(f"  WINNER: {winner}\n")


if __name__ == "__main__":
    main()
