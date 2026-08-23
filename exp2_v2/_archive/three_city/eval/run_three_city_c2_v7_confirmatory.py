#!/usr/bin/env python3
"""Run paired, interleaved blind/hint v7 prompts without reading the answer key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v7_tasks import (
    HINT_SENTENCE,
    strip_hint,
    validate_prompt,
)
from run_batch import _strip_reasoning, make_client
from run_recovery_probe import chat_call

_DATA = _ROOT / "data" / "three_city_c2_v7"
_BLIND_TASKS = _DATA / "tasks_c2_v7_blind.jsonl"
_HINT_TASKS = _DATA / "tasks_c2_v7_hint.jsonl"
_OUTDIR = _DATA / "confirmatory"
_SHARED_ENDPOINT = "https://liv-forecast.services.ai.azure.com/openai/v1"
_CLAUDE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/anthropic"
_TASK_ID = re.compile(r"^c2v7_(\d{4})_c([0-2])_k(0|1|2|4)$")
_ORDER_SEED = 20260728

MODEL_CONFIG = {
    "claude-opus-4-8": {
        "endpoint": _CLAUDE_ENDPOINT,
        "keys": ("AZURE_AI_API_KEY", "CLAUDE_AZURE_API_KEY"),
        "claude": True,
    },
    "DeepSeek-V4-Pro": {
        "endpoint": _SHARED_ENDPOINT,
        "keys": ("AZURE_AI_API_KEY", "DEEPSEEK_AZURE_API_KEY"),
        "claude": False,
    },
    "gpt-5.4": {
        "endpoint": _SHARED_ENDPOINT,
        "keys": ("AZURE_AI_API_KEY",),
        "claude": False,
    },
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tasks(path: Path, *, arm: str) -> dict[str, Dict[str, Any]]:
    tasks: dict[str, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"task_id", "prompt", "prompt_sha256"}:
            raise ValueError("model task file contains evaluator material")
        if _TASK_ID.fullmatch(record["task_id"]) is None:
            raise ValueError(f"invalid task ID: {record['task_id']!r}")
        if record["task_id"] in tasks:
            raise ValueError(f"duplicate task ID: {record['task_id']}")
        if hashlib.sha256(
            record["prompt"].encode("utf-8")
        ).hexdigest() != record["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {record['task_id']}")
        validate_prompt(record["prompt"], arm=arm)
        tasks[record["task_id"]] = record
    return tasks


def _load_paired_tasks(
    blind_path: Path,
    hint_path: Path,
    *,
    episodes: Iterable[int],
    conditions: Iterable[int],
    prefixes: Iterable[int],
) -> list[tuple[str, str, Dict[str, Any]]]:
    blind = _read_tasks(blind_path, arm="blind")
    hint = _read_tasks(hint_path, arm="hint")
    if set(blind) != set(hint):
        raise ValueError("blind and hint task ID sets differ")
    episode_set = set(episodes)
    condition_set = set(conditions)
    prefix_set = set(prefixes)
    task_ids = []
    for task_id in blind:
        match = _TASK_ID.fullmatch(task_id)
        assert match is not None
        episode, condition, prefix = map(int, match.groups())
        if (
            episode in episode_set
            and condition in condition_set
            and prefix in prefix_set
        ):
            if strip_hint(hint[task_id]["prompt"]) != blind[task_id]["prompt"]:
                raise ValueError(
                    f"{task_id}: paired prompts differ beyond the hint"
                )
            task_ids.append(task_id)
    expected = len(episode_set) * len(condition_set) * len(prefix_set)
    if len(task_ids) != expected:
        raise ValueError(f"selected {len(task_ids)} base tasks, expected {expected}")

    rng = random.Random(_ORDER_SEED)
    rng.shuffle(task_ids)
    paired: list[tuple[str, str, Dict[str, Any]]] = []
    for task_id in task_ids:
        digest = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16)
        arm_order = ("blind", "hint") if digest % 2 else ("hint", "blind")
        for arm in arm_order:
            paired.append(
                (
                    arm,
                    task_id,
                    blind[task_id] if arm == "blind" else hint[task_id],
                )
            )
    return paired


def _parse_reply(raw: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed: Optional[Dict[str, Any]] = None
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or "predicted_poll" not in value:
            continue
        try:
            predicted_poll = float(value["predicted_poll"])
        except (TypeError, ValueError):
            continue
        if not 0.0 <= predicted_poll <= 100.0:
            continue
        parsed = {
            "predicted_poll": predicted_poll,
            "rationale": str(value.get("rationale", ""))[:500],
        }
    return parsed or {"predicted_poll": None, "rationale": ""}


def _make_client(
    model: str,
    *,
    endpoint_override: Optional[str],
    timeout: float,
):
    if model not in MODEL_CONFIG:
        raise ValueError(f"unsupported model: {model}")
    config = MODEL_CONFIG[model]
    key = next(
        (os.environ[name] for name in config["keys"] if os.environ.get(name)),
        "",
    )
    if not key:
        raise RuntimeError("no API key found in " + " / ".join(config["keys"]))
    endpoint = endpoint_override or config["endpoint"]
    if config["claude"]:
        from anthropic import AnthropicFoundry

        client = AnthropicFoundry(
            azure_ad_token_provider=lambda: key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=2,
        )
    else:
        client = make_client(
            "azure",
            endpoint=endpoint,
            api_key=key,
            azure_auth=True,
            timeout=timeout,
        )
    return client, endpoint


def _response_state(
    path: Path,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return terminal-response and successfully parsed task keys.

    A model response is terminal even if its JSON cannot be parsed: rerunning
    only malformed responses would selectively replace failures. Transport
    failures remain retryable because no completion was received.
    """
    if not path.exists():
        return set(), set()
    terminal = set()
    parsed = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (record["arm"], record["task_id"])
        if record.get("predicted_poll") is not None:
            terminal.add(key)
            parsed.add(key)
        elif record.get("response_received") is True:
            terminal.add(key)
    return terminal, parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIG))
    parser.add_argument("--blind-tasks", type=Path, default=_BLIND_TASKS)
    parser.add_argument("--hint-tasks", type=Path, default=_HINT_TASKS)
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
        help="0=relevant, 1=none, 2=orthogonal",
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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help=(
            "transport attempts; a received but unparseable completion is "
            "never retried"
        ),
    )
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least one")

    tasks = _load_paired_tasks(
        args.blind_tasks,
        args.hint_tasks,
        episodes=args.episodes,
        conditions=args.conditions,
        prefixes=args.prefixes,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "paired_base_tasks": len(tasks) // 2,
                    "model_calls": len(tasks),
                    "episodes": len(set(args.episodes)),
                    "conditions": args.conditions,
                    "prefixes": args.prefixes,
                    "first": {
                        "arm": tasks[0][0],
                        "task_id": tasks[0][1],
                    },
                    "hint_sentence": HINT_SENTENCE,
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
        task for task in tasks if (task[0], task[1]) not in terminal
    ]
    print(
        f"model={args.model} selected={len(tasks)} pending={len(pending)} "
        f"endpoint={endpoint}",
        flush=True,
    )

    parsed_count = 0
    error_count = 0
    with out_path.open("a") as handle:
        for position, (arm, task_id, task) in enumerate(pending, 1):
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
                "task_id": task_id,
                "arm": arm,
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
                    f"{position}/{len(pending)} arm={arm} task={task_id} "
                    f"prediction={record['predicted_poll']} error={error or '-'}",
                    flush=True,
                )
            if args.delay:
                time.sleep(args.delay)

    all_terminal, all_parsed = _response_state(out_path)
    selected_keys = {(arm, task_id) for arm, task_id, _ in tasks}
    manifest = {
        "experiment": "three_city_c2_v7_confirmatory",
        "status": (
            "complete"
            if selected_keys <= all_terminal
            else "partial"
        ),
        "model": args.model,
        "endpoint": endpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "order_seed": _ORDER_SEED,
        "paired_and_interleaved": True,
        "blind_task_file": str(args.blind_tasks),
        "blind_task_file_sha256": _file_sha256(args.blind_tasks),
        "hint_task_file": str(args.hint_tasks),
        "hint_task_file_sha256": _file_sha256(args.hint_tasks),
        "episodes": sorted(set(args.episodes)),
        "conditions": sorted(set(args.conditions)),
        "prefixes": sorted(set(args.prefixes)),
        "selected_base_tasks": len(tasks) // 2,
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
