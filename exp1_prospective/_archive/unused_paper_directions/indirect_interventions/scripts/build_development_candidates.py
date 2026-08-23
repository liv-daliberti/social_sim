#!/usr/bin/env python3
"""Generate three-family packets on 20 non-core development markets."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path
from typing import Any

from build_candidates import (
    CORE_PACKETS,
    DEFAULT_ENDPOINT,
    DEFAULT_KEY_ENV,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROTOCOL,
    DEFAULT_REASONING_EFFORT,
    append_jsonl,
    generate_market,
    load_local_env,
    read_jsonl,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
MARKETS = REPO / "exp1_prospective/data/selected_markets/diverse_2026-08-12.jsonl"
DEFAULT_OUTPUT = ROOT / "data/development/development_candidates_readable_v3.jsonl"
DEFAULT_RAW = ROOT / "data/development/raw_development_generations_readable_v3.jsonl"
PILOT_MARKET_IDS = ("908713", "562802", "700396")


def development_records() -> list[dict[str, Any]]:
    core_market_ids = {str(row["market_id"]) for row in read_jsonl(CORE_PACKETS)}
    eligible = [
        row for row in read_jsonl(MARKETS)
        if str(row["market_id"]) not in core_market_ids
    ]
    by_id = {str(row["market_id"]): row for row in eligible}
    chosen = [by_id[market_id] for market_id in PILOT_MARKET_IDS if market_id in by_id]
    chosen_ids = {str(row["market_id"]) for row in chosen}
    chosen.extend(
        row for row in sorted(eligible, key=lambda item: str(item["market_id"]))
        if str(row["market_id"]) not in chosen_ids
    )
    result = []
    unavailable = {
        "key_actors": ["No canonical event model is available at development stage; infer actors"],
        "key_mechanisms": ["Infer at least two causal mechanisms from the rules", "Infer a distinct second mechanism"],
        "latent_variables": ["Infer the main uncertainty"],
    }
    for row in chosen[:20]:
        result.append({
            **row,
            "_split": "development",
            "description": row.get("description") or row.get("rules") or "",
            "canonical_run": {
                "run_id": 0,
                "yes_prob": row.get("yes_price", 0.5),
                "structured_forecast": {"event_model": unavailable},
            },
        })
    if len(result) != 20:
        raise SystemExit(f"expected 20 disjoint development markets, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--protocol",
        choices=("openai_responses", "chat_completions"),
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    parser.add_argument("--api-key")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = development_records()
    completed = set()
    if args.output.exists():
        counts: dict[str, int] = {}
        for row in read_jsonl(args.output):
            task_id = row["market"]["task_id"]
            counts[task_id] = counts.get(task_id, 0) + 1
        if any(count != 3 for count in counts.values()):
            raise SystemExit("fail closed: development output has an incomplete market")
        completed = set(counts)
    selected = [
        (index, row) for index, row in enumerate(records)
        if row["task_id"] not in completed
    ]
    if args.limit:
        selected = selected[:args.limit]
    print(f"development markets=20 completed={len(completed)} queued={len(selected)}")
    if not selected:
        return
    if args.dry_run:
        from build_candidates import prompt_for
        print(prompt_for(selected[0][1], selected[0][0]))
        return
    load_local_env()
    key = args.api_key or os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"set {args.key_env} or pass --api-key")
    client = {
        "api_key": key,
        "endpoint": args.endpoint,
        "provider": "azure_openai",
        "protocol": args.protocol,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
    }
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(generate_market, client, args.model, row, index, args.retries): (index, row)
            for index, row in selected
        }
        for future in concurrent.futures.as_completed(futures):
            index, row = futures[future]
            try:
                candidates, raw = future.result()
                append_jsonl(args.output, candidates)
                append_jsonl(args.raw_output, [raw])
                print(f"PASS dev {index + 1:02d}/20 market={row['market_id']} families=3")
            except Exception as exc:
                failures.append(str(row["market_id"]))
                append_jsonl(args.raw_output, [{
                    "task_id": row["task_id"],
                    "market_id": str(row["market_id"]),
                    "error": str(exc),
                }])
                print(f"FAIL dev market={row['market_id']}: {exc}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{len(failures)} development markets failed: {failures}")


if __name__ == "__main__":
    main()
