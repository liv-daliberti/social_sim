#!/usr/bin/env python3
"""Run the post-specified v7 strong-structure positive-control arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v7_strong_hint_tasks import validate_strong_prompt
from run_batch import _strip_reasoning
from run_recovery_probe import chat_call
from run_three_city_c2_v7_confirmatory import (
    MODEL_CONFIG,
    _ORDER_SEED,
    _TASK_ID,
    _file_sha256,
    _make_client,
    _parse_reply,
    _response_state,
    _slug,
)

_DATA = _ROOT / "data" / "three_city_c2_v7_strong_hint"
_TASKS = _DATA / "tasks_c2_v7_strong_hint.jsonl"
_OUTDIR = _DATA / "confirmatory"


def _read_tasks(path: Path) -> dict[str, Dict[str, Any]]:
    tasks: dict[str, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"task_id", "prompt", "prompt_sha256"}:
            raise ValueError("strong task contains evaluator material")
        if _TASK_ID.fullmatch(record["task_id"]) is None:
            raise ValueError(f"invalid task ID: {record['task_id']!r}")
        if record["task_id"] in tasks:
            raise ValueError(f"duplicate task ID: {record['task_id']}")
        if hashlib.sha256(
            record["prompt"].encode("utf-8")
        ).hexdigest() != record["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {record['task_id']}")
        validate_strong_prompt(record["prompt"])
        tasks[record["task_id"]] = record
    return tasks


def _select_tasks(
    path: Path,
    *,
    episodes: Iterable[int],
    conditions: Iterable[int],
    prefixes: Iterable[int],
) -> list[Dict[str, Any]]:
    tasks = _read_tasks(path)
    episode_set = set(episodes)
    condition_set = set(conditions)
    prefix_set = set(prefixes)
    selected = []
    for task_id, task in tasks.items():
        match = _TASK_ID.fullmatch(task_id)
        assert match is not None
        episode, condition, prefix = map(int, match.groups())
        if (
            episode in episode_set
            and condition in condition_set
            and prefix in prefix_set
        ):
            selected.append(task)
    expected = len(episode_set) * len(condition_set) * len(prefix_set)
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} tasks, expected {expected}")
    random.Random(_ORDER_SEED).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIG))
    parser.add_argument("--tasks", type=Path, default=_TASKS)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=list(range(120)),
    )
    parser.add_argument(
        "--conditions",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4],
    )
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least one")

    tasks = _select_tasks(
        args.tasks,
        episodes=args.episodes,
        conditions=args.conditions,
        prefixes=args.prefixes,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "arm": "strong_hint",
                    "model_calls": len(tasks),
                    "episodes": len(set(args.episodes)),
                    "conditions": args.conditions,
                    "prefixes": args.prefixes,
                    "first_task_id": tasks[0]["task_id"],
                },
                indent=2,
            )
        )
        return

    client, endpoint = _make_client(
        args.model,
        endpoint_override=args.endpoint,
        timeout=args.timeout,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    slug = _slug(args.model)
    out_path = args.outdir / f"responses_{slug}.jsonl"
    manifest_path = args.outdir / f"responses_{slug}.manifest.json"
    terminal, _ = _response_state(out_path)
    pending = [
        task
        for task in tasks
        if ("strong_hint", task["task_id"]) not in terminal
    ]
    print(
        f"model={args.model} selected={len(tasks)} pending={len(pending)} "
        f"endpoint={endpoint}",
        flush=True,
    )

    parsed_count = 0
    error_count = 0
    with out_path.open("a") as handle:
        for position, task in enumerate(pending, 1):
            raw = ""
            parsed = {"predicted_poll": None, "rationale": ""}
            error = None
            attempts = 0
            response_received = False
            for attempts in range(1, args.max_attempts + 1):
                try:
                    response = chat_call(
                        client,
                        args.model,
                        [{"role": "user", "content": task["prompt"]}],
                        args.max_tokens,
                        args.temperature,
                    )
                    response_received = True
                    raw = _strip_reasoning(
                        (response.choices[0].message.content or "").strip()
                    )
                    parsed = _parse_reply(raw)
                    if parsed["predicted_poll"] is not None:
                        error = None
                        break
                    error = "unparseable response"
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raw = ""
                    if response_received:
                        break
                if attempts < args.max_attempts:
                    time.sleep(max(0.2, args.delay))

            record = {
                "task_id": task["task_id"],
                "arm": "strong_hint",
                "prompt_sha256": task["prompt_sha256"],
                "model": args.model,
                "predicted_poll": parsed["predicted_poll"],
                "rationale": parsed["rationale"],
                "raw": raw[:8000],
                "attempts": attempts,
                "response_received": response_received,
                "error": error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if record["predicted_poll"] is None:
                error_count += 1
            else:
                parsed_count += 1
            if position == 1 or position % 25 == 0 or position == len(pending):
                print(
                    f"{position}/{len(pending)} task={task['task_id']} "
                    f"prediction={record['predicted_poll']} error={error or '-'}",
                    flush=True,
                )
            if args.delay:
                time.sleep(args.delay)

    all_terminal, all_parsed = _response_state(out_path)
    selected_keys = {
        ("strong_hint", task["task_id"]) for task in tasks
    }
    manifest = {
        "experiment": "three_city_c2_v7_strong_hint",
        "status": (
            "complete"
            if selected_keys <= all_terminal
            else "partial"
        ),
        "postspecified_positive_control": True,
        "model": args.model,
        "arm": "strong_hint",
        "endpoint": endpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "order_seed": _ORDER_SEED,
        "task_file": str(args.tasks),
        "task_file_sha256": _file_sha256(args.tasks),
        "episodes": sorted(set(args.episodes)),
        "conditions": sorted(set(args.conditions)),
        "prefixes": sorted(set(args.prefixes)),
        "selected_model_calls": len(tasks),
        "responses_received": len(selected_keys & all_terminal),
        "successfully_parsed": len(selected_keys & all_parsed),
        "newly_parsed_this_run": parsed_count,
        "new_errors_this_run": error_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote responses to {out_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
