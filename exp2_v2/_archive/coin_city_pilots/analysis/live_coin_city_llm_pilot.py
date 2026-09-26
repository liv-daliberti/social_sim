#!/usr/bin/env python3
"""Continuously refresh the matched live coin-to-city pilot figure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis.analyze_coin_city_llm_pilot import DESIGN, RESPONSES, RUN, analyze


LIVE = RUN / "live"


def draw() -> dict:
    result = analyze(
        response_dir=RESPONSES,
        design_dir=DESIGN,
        outdir=LIVE,
        output_name="live_structure.png",
    )
    (LIVE / "progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{result['updated_at']} received={result['received']}/{result['expected']} "
        f"parsed={result['parsed']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    while True:
        result = draw()
        if not args.watch or result["received"] >= result["expected"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
