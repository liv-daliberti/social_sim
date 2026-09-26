#!/usr/bin/env python3
"""Run a small frontier-model pilot on frozen, structure-blind C2 prompts.

This runner deliberately reads only the model-facing task JSONL. It never opens
the evaluator answer key. Hidden conditions and gold values are joined later by
the analysis script, after model responses have been saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from build_three_city_c2_tasks import validate_structure_blind_prompt
from run_batch import _strip_reasoning, make_client
from run_recovery_probe import chat_call

_TASKS = _ROOT / "data" / "three_city_c2" / "tasks_c2_v1.jsonl"
_OUTDIR = _ROOT / "data" / "three_city_c2" / "pilot_v1"
_SHARED_ENDPOINT = "https://liv-forecast.services.ai.azure.com/openai/v1"
_CLAUDE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/anthropic"
_TASK_ID = re.compile(r"^c2_(\d{4})_v([0-3])_k([0-3])$")

MODEL_CONFIG = {
    "claude-opus-4-8": {
        "endpoint": _CLAUDE_ENDPOINT,
        # The shared Foundry resource key is authoritative for this endpoint.
        # CLAUDE_AZURE_API_KEY is retained only as a fallback for older setups.
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

# Four episodes balance target response, A/B ordering, target-match label, and
# target-news sign. This is a diagnostic sample, not an estimand.
DEFAULT_EPISODES = (0, 3, 5, 6)
DEFAULT_PREFIXES = (0, 1, 2)
DEFAULT_VARIANTS = (0, 1, 2, 3)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tasks(
    path: Path,
    *,
    episodes: Iterable[int],
    prefixes: Iterable[int],
    variants: Iterable[int],
) -> list[Dict[str, Any]]:
    episode_set = set(episodes)
    prefix_set = set(prefixes)
    variant_set = set(variants)
    selected = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"task_id", "prompt", "prompt_sha256"}:
            raise ValueError(
                "model task file contains non-prompt evaluator material"
            )
        match = _TASK_ID.fullmatch(record["task_id"])
        if match is None:
            raise ValueError(f"non-opaque task ID: {record['task_id']!r}")
        episode, variant, prefix = map(int, match.groups())
        if (
            episode not in episode_set
            or variant not in variant_set
            or prefix not in prefix_set
        ):
            continue
        validate_structure_blind_prompt(record["prompt"])
        prompt_hash = hashlib.sha256(
            record["prompt"].encode("utf-8")
        ).hexdigest()
        if prompt_hash != record["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch for {record['task_id']}")
        selected.append(record)
    selected.sort(key=lambda record: record["task_id"])
    expected = len(episode_set) * len(prefix_set) * len(variant_set)
    if len(selected) != expected:
        raise ValueError(
            f"selected {len(selected)} tasks, expected {expected}"
        )
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
        if not 0.0 <= predicted_poll <= 100.0:
            continue
        rationale = value.get("rationale", "")
        parsed = {
            "predicted_poll": predicted_poll,
            "rationale": str(rationale)[:500],
        }
    return parsed or {"predicted_poll": None, "rationale": ""}


def _make_client(
    model: str,
    *,
    endpoint_override: Optional[str],
    timeout: float,
):
    if model not in MODEL_CONFIG:
        raise ValueError(
            f"unsupported model {model!r}; choose {sorted(MODEL_CONFIG)}"
        )
    config = MODEL_CONFIG[model]
    key = next(
        (os.environ[name] for name in config["keys"] if os.environ.get(name)),
        "",
    )
    if not key:
        raise RuntimeError(
            "no API key found in " + " / ".join(config["keys"])
        )
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


def _completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("predicted_poll") is not None:
            completed.add(record["task_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIG))
    parser.add_argument("--tasks", type=Path, default=_TASKS)
    parser.add_argument("--out", type=Path, default=_OUTDIR)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPISODES),
    )
    parser.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREFIXES),
    )
    parser.add_argument(
        "--variants",
        type=int,
        nargs="+",
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = _load_tasks(
        args.tasks,
        episodes=args.episodes,
        prefixes=args.prefixes,
        variants=args.variants,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "tasks": len(tasks),
                    "episodes": args.episodes,
                    "prefixes": args.prefixes,
                    "variants": args.variants,
                    "first_task_id": tasks[0]["task_id"],
                },
                indent=2,
            )
        )
        print("\n" + tasks[0]["prompt"])
        return

    client, endpoint = _make_client(
        args.model,
        endpoint_override=args.endpoint,
        timeout=args.timeout,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    slug = _slug(args.model)
    out_path = args.out / f"responses_{slug}.jsonl"
    manifest_path = args.out / f"responses_{slug}.manifest.json"
    completed = _completed_task_ids(out_path)
    pending = [task for task in tasks if task["task_id"] not in completed]
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
            for attempts in range(1, args.max_attempts + 1):
                try:
                    # Only the prompt string enters the request. No condition,
                    # gold value, baseline, or answer-key content is in scope.
                    response = chat_call(
                        client,
                        args.model,
                        [{"role": "user", "content": task["prompt"]}],
                        args.max_tokens,
                        args.temperature,
                    )
                    raw = _strip_reasoning(
                        (response.choices[0].message.content or "").strip()
                    )
                    parsed = _parse_reply(raw)
                    if parsed["predicted_poll"] is not None:
                        error = None
                        break
                    error = "unparseable response"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raw = ""
                if attempts < args.max_attempts:
                    time.sleep(max(0.2, args.delay))

            record = {
                "task_id": task["task_id"],
                "prompt_sha256": task["prompt_sha256"],
                "model": args.model,
                "predicted_poll": parsed["predicted_poll"],
                "rationale": parsed["rationale"],
                "raw": raw[:4000],
                "attempts": attempts,
                "error": error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if record["predicted_poll"] is None:
                error_count += 1
            else:
                parsed_count += 1
            print(
                f"{position}/{len(pending)} {task['task_id']} "
                f"prediction={record['predicted_poll']} error={error or '-'}",
                flush=True,
            )
            if args.delay:
                time.sleep(args.delay)

    all_completed = _completed_task_ids(out_path)
    manifest = {
        "experiment": "three_city_c2_pilot_v1",
        "status": (
            "complete"
            if all(task["task_id"] in all_completed for task in tasks)
            else "partial"
        ),
        "model": args.model,
        "endpoint": endpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "task_file": str(args.tasks),
        "task_file_sha256": _file_sha256(args.tasks),
        "episodes": list(args.episodes),
        "prefixes": list(args.prefixes),
        "variants": list(args.variants),
        "selected_task_ids": [task["task_id"] for task in tasks],
        "selected_task_count": len(tasks),
        "parsed_this_invocation": parsed_count,
        "errors_this_invocation": error_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"done status={manifest['status']} parsed={parsed_count} "
        f"errors={error_count} out={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
