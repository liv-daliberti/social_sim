#!/usr/bin/env python3
"""Shared runner for frozen, JSONL-backed Experiment 2 task shards.

Experiment-specific wrappers configure the module globals below before calling
``main``. Keeping the transport and append-only response logic here avoids the
old pattern where current Coin City runners imported an obsolete three-city
experiment executable and monkey-patched its private state.
"""

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
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

# Wrappers must replace these experiment-specific values before calling main.
EXPERIMENT = "unconfigured"
PROMPT_ARMS: tuple[str, ...] = ()
PREFIX_LADDER: tuple[int, ...] = ()
_RUN = _ROOT / "data"
_DESIGN = _RUN / "design"
_OUTDIR = _RUN / "responses"
_TASK_ID = re.compile(r"(?!)")
_ORDER_SEED = 20260804

MODEL_CONFIG: dict[str, dict[str, Any]] = {}


def validate_arm_prompt(prompt: str, arm: str) -> None:
    """Fail closed until an experiment wrapper installs its validator."""
    del prompt, arm
    raise RuntimeError("frozen task runner has not been configured")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_reasoning(text: str) -> str:
    """Drop reasoning-model ``<think>`` blocks before parsing the answer."""
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def _retry_openai_call(function, /, **kwargs):
    """Preserve the bounded retry policy used by the frozen original runs."""
    from openai import APIError, APITimeoutError, RateLimitError

    for attempt in range(3):
        try:
            return function(**kwargs)
        except RateLimitError:
            time.sleep(4.0 * 2**attempt)
        except (APITimeoutError, APIError):
            if attempt == 2:
                raise
            time.sleep(4.0 * 2**attempt)
    raise RuntimeError("maximum retries exceeded")


def chat_call(client, model, messages, max_tokens, temperature):
    """Call Anthropic Messages or OpenAI Chat Completions with one schema."""
    if "claude" in model.lower():
        response = client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max(max_tokens, 1_024),
        )
        text = response.content[0].text if getattr(response, "content", None) else ""
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason=getattr(response, "stop_reason", "stop"),
                )
            ]
        )
    kwargs = {"model": model, "messages": messages}
    if model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = max(max_tokens, 4_096)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    return _retry_openai_call(client.chat.completions.create, **kwargs)


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
        if (
            hashlib.sha256(record["prompt"].encode()).hexdigest()
            != record["prompt_sha256"]
        ):
            raise ValueError(f"prompt hash mismatch: {record['task_id']}")
        validate_arm_prompt(record["prompt"], arm)
        tasks[record["task_id"]] = record
    return tasks


def _select_tasks(
    path: Path,
    *,
    arm: str,
    episodes: Iterable[int],
    prefixes: Iterable[int],
) -> list[Dict[str, Any]]:
    tasks = _read_tasks(path, arm=arm)
    episode_set = set(episodes)
    prefix_set = set(prefixes)
    selected = []
    for task_id, task in tasks.items():
        match = _TASK_ID.fullmatch(task_id)
        assert match is not None
        episode, prefix = map(int, match.groups())
        if episode in episode_set and prefix in prefix_set:
            selected.append(task)
    expected = len(episode_set) * len(prefix_set)
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} tasks, expected {expected}")
    random.Random(_ORDER_SEED).shuffle(selected)
    return selected


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
        if 0.0 <= predicted_poll <= 100.0:
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
        from openai import OpenAI

        client = OpenAI(
            base_url=endpoint,
            api_key="placeholder",
            default_headers={"api-key": key},
            timeout=timeout,
            max_retries=2,
        )
    return client, endpoint


def _response_state(path: Path) -> tuple[set[str], set[str]]:
    terminal: set[str] = set()
    parsed: set[str] = set()
    if not path.exists():
        return terminal, parsed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = record["task_id"]
        if record.get("predicted_poll") is not None:
            terminal.add(task_id)
            parsed.add(task_id)
        elif record.get("response_received") is True:
            terminal.add(task_id)
    return terminal, parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIG))
    parser.add_argument("--arm", required=True, choices=PROMPT_ARMS)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(120)))
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIX_LADDER))
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least one")
    task_path = args.tasks or _DESIGN / f"tasks_c2_v9_{args.arm}.jsonl"
    tasks = _select_tasks(
        task_path,
        arm=args.arm,
        episodes=args.episodes,
        prefixes=args.prefixes,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "arm": args.arm,
                    "model_calls": len(tasks),
                    "episodes": len(set(args.episodes)),
                    "prefixes": args.prefixes,
                    "first_task_id": tasks[0]["task_id"],
                    "task_file_sha256": _file_sha256(task_path),
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
    out_path = args.outdir / f"responses_{slug}_{args.arm}.jsonl"
    manifest_path = args.outdir / f"responses_{slug}_{args.arm}.manifest.json"
    terminal, _ = _response_state(out_path)
    pending = [task for task in tasks if task["task_id"] not in terminal]
    print(
        f"model={args.model} arm={args.arm} selected={len(tasks)} "
        f"pending={len(pending)} endpoint={endpoint}",
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
                    error = (
                        None
                        if parsed["predicted_poll"] is not None
                        else "unparseable response"
                    )
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
                "arm": args.arm,
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
    selected_ids = {task["task_id"] for task in tasks}
    manifest = {
        "experiment": EXPERIMENT,
        "status": "complete" if selected_ids <= all_terminal else "partial",
        "model": args.model,
        "arm": args.arm,
        "endpoint": endpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "order_seed": _ORDER_SEED,
        "task_file": str(task_path),
        "task_file_sha256": _file_sha256(task_path),
        "episodes": sorted(set(args.episodes)),
        "prefixes": sorted(set(args.prefixes)),
        "selected_model_calls": len(tasks),
        "responses_received": len(selected_ids & all_terminal),
        "successfully_parsed": len(selected_ids & all_parsed),
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
