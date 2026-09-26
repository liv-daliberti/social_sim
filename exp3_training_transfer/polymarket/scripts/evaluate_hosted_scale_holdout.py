#!/usr/bin/env python3
"""Registered five-call hosted-model evaluation on the Exp4 scale holdout."""

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
from typing import Any


POLY = Path(__file__).resolve().parents[1]
REPO = POLY.parents[1]
DATA = POLY / "data"
REPORTS = POLY / "reports"
PROTOCOL = POLY / "ARCHITECTURE_PROVIDER_SWEEP_PROTOCOL.md"
REGISTRATION = (
    POLY
    / "runs"
    / "exp4_architecture_provider_sweep_registration_20260825T203304Z.json"
)
HOLDOUT = DATA / "exp4_scale_registered" / "test.tasks.jsonl"
HOLDOUT_MANIFEST = DATA / "exp4_scale_registered" / "manifest.json"
DEV = DATA / "exp3b_registered" / "dev.tasks.jsonl"
PROTOCOL_VERSION = "exp4_architecture_provider_sweep_v1"
HOLDOUT_SHA256 = "ee9bd976402a746195291b4b43e209a084a0f1c6e5bf3f27c9d4ef638f1c7f24"
DRAWS = 5
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_PRE_MODEL_RETRIES = 12
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
RETRY_AFTER = re.compile(r"retry[- ]after[\"']?[:=]\s*[\"']?([0-9.]+)", re.I)
RETRY_HINT = re.compile(r"[Pp]lease retry in ([0-9.]+)s")

PROVIDERS: dict[str, dict[str, Any]] = {
    "claude-opus-4-8": {
        "protocol": "anthropic_messages",
        "endpoint": "https://liv-forecast.services.ai.azure.com/anthropic",
        "key_env": "CLAUDE_AZURE_API_KEY",
        "temperature": None,
        "top_p": None,
        "reasoning_effort": None,
        "default_max_output_tokens": 128,
    },
    "gpt-5.6-sol": {
        "protocol": "openai_responses",
        "endpoint": (
            "https://cos-tiktok-annotation-a-resource.services.ai.azure.com/openai/v1"
        ),
        "key_env": "DEEPSEEK_AZURE_API_KEY",
        "temperature": None,
        "top_p": None,
        "reasoning_effort": "low",
        "default_max_output_tokens": 128,
    },
    "FW-Kimi-K3": {
        "protocol": "openai_chat",
        "endpoint": "https://liv.services.ai.azure.com/openai/v1",
        "key_env": "AZURE_AI_API_KEY",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "reasoning_effort": "low",
        "default_max_output_tokens": 128,
    },
    "DeepSeek-V4-Pro": {
        "protocol": "openai_chat",
        "endpoint": (
            "https://cos-tiktok-annotation-a-resource.services.ai.azure.com/openai/v1"
        ),
        "key_env": "DEEPSEEK_AZURE_API_KEY",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "reasoning_effort": None,
        "default_max_output_tokens": 128,
    },
}

sys.path.insert(0, str(POLY / "scripts"))
sys.path.insert(0, str(REPO / "exp3_training_transfer" / "biased_news"))
import evaluate_locked_test as registered  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_inputs(
    split: str, registration_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("hosted sweep registration protocol drifted")
    if registration.get("status") not in {"registered", "running", "complete"}:
        raise RuntimeError("hosted sweep is not registered")
    if sha256(PROTOCOL) != registration["protocol"]["sha256"]:
        raise RuntimeError("hosted sweep protocol hash drifted")
    if sorted(PROVIDERS) != sorted(registration["hosted_models"]):
        raise RuntimeError("hosted model roster drifted")
    if split == "holdout":
        if sha256(HOLDOUT) != HOLDOUT_SHA256:
            raise RuntimeError("hosted holdout task hash drifted")
        if (
            sha256(HOLDOUT_MANIFEST)
            != registration["frozen_data"]["holdout_manifest_sha256"]
        ):
            raise RuntimeError("hosted holdout manifest hash drifted")
        rows = read_rows(HOLDOUT)
        if len(rows) != 318:
            raise RuntimeError("hosted holdout count drifted")
    else:
        rows = read_rows(DEV)
        if len(rows) != 512:
            raise RuntimeError("hosted development count drifted")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise RuntimeError("hosted task IDs are not unique")
    return rows, registration


def make_client(config: dict[str, Any], timeout: float):
    key = os.environ.get(config["key_env"], "")
    if not key:
        raise RuntimeError(f"no API key found in {config['key_env']}")
    if config["protocol"] == "anthropic_messages":
        from anthropic import AnthropicFoundry

        return AnthropicFoundry(
            api_key=key,
            base_url=config["endpoint"],
            timeout=timeout,
            max_retries=0,
        )
    from openai import OpenAI

    return OpenAI(
        base_url=config["endpoint"],
        api_key="placeholder",
        default_headers={"api-key": key},
        timeout=timeout,
        max_retries=0,
    )


def anthropic_text(message: Any) -> str:
    return "".join(
        block.text or ""
        for block in (getattr(message, "content", None) or [])
        if getattr(block, "type", None) == "text"
    )


def issue(
    client: Any,
    model: str,
    config: dict[str, Any],
    prompt: str,
    max_output_tokens: int,
) -> tuple[str, str | None, str | None, dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    protocol = config["protocol"]
    if protocol == "anthropic_messages":
        response = client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
        )
        return (
            anthropic_text(response),
            getattr(response, "id", None),
            getattr(response, "stop_reason", None),
            {"temperature": config["temperature"], "top_p": config["top_p"]},
        )
    if protocol == "openai_responses":
        response = client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": config["reasoning_effort"]},
        )
        return (
            response.output_text or "",
            getattr(response, "id", None),
            getattr(response, "status", None),
            {
                "temperature": None,
                "top_p": None,
                "reasoning_effort": config["reasoning_effort"],
            },
        )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
    }
    if config["reasoning_effort"] is not None:
        kwargs["reasoning_effort"] = config["reasoning_effort"]
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    return (
        choice.message.content or "",
        getattr(response, "id", None),
        getattr(choice, "finish_reason", None),
        {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "reasoning_effort": config["reasoning_effort"],
        },
    )


def issue_with_transport_retries(
    *args: Any,
) -> tuple[str, str | None, str | None, dict[str, Any], int]:
    retries = 0
    for attempt in range(1, MAX_PRE_MODEL_RETRIES + 2):
        try:
            raw, response_id, finish, controls = issue(*args)
            return raw, response_id, finish, controls, retries
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status not in RETRY_STATUS or attempt > MAX_PRE_MODEL_RETRIES:
                raise
            retries += 1
            hint = RETRY_AFTER.search(str(exc)) or RETRY_HINT.search(str(exc))
            delay = (
                min(120.0, float(hint.group(1)) + 1.0)
                if hint
                else min(120.0, 4.0 * 2 ** (attempt - 1))
            )
            print(
                f"pre-model retry {attempt} after HTTP {status}; sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay + random.uniform(0.0, 2.0))
    raise RuntimeError("unreachable")


def output_paths(
    output_dir: Path, model: str, split: str, draw_index: int
) -> tuple[Path, Path]:
    stem = f"exp4_hosted_{slug(model)}_{split}_draw{draw_index}"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}.manifest.json"


def completed_task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        result.add(record["task_id"])
    return result


def run_shard(args: argparse.Namespace) -> None:
    if args.draw_index is None or not 0 <= args.draw_index < DRAWS:
        raise SystemExit("--draw-index must be one of 0, 1, 2, 3, 4")
    rows, registration = validate_inputs(args.split, args.registration)
    if args.split == "holdout" and args.limit is not None:
        raise SystemExit("holdout shards cannot use --limit")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]
    config = PROVIDERS[args.model]
    max_tokens = args.max_output_tokens or config["default_max_output_tokens"]
    raw_path, manifest_path = output_paths(
        args.output_dir, args.model, args.split, args.draw_index
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "split": args.split,
                    "draw_index": args.draw_index,
                    "tasks": len(rows),
                    "task_file_sha256": sha256(
                        HOLDOUT if args.split == "holdout" else DEV
                    ),
                    "registration_sha256": sha256(args.registration),
                    "config": config,
                    "max_output_tokens": max_tokens,
                    "raw_path": str(raw_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    client = make_client(config, args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = completed_task_ids(raw_path)
    selected_ids = {row["task_id"] for row in rows}
    if not done <= selected_ids:
        raise RuntimeError("existing hosted shard contains unselected task IDs")
    pending = [row for row in rows if row["task_id"] not in done]
    random.Random(20_260_825 + args.draw_index).shuffle(pending)
    print(
        f"model={args.model} split={args.split} draw={args.draw_index} "
        f"selected={len(rows)} pending={len(pending)}",
        flush=True,
    )
    new_parsed = 0
    new_received = 0
    transport_retries = 0
    with raw_path.open("a", encoding="utf-8") as handle:
        for position, row in enumerate(pending, 1):
            raw = ""
            response_id = None
            finish = None
            controls: dict[str, Any] = {}
            error = None
            received = False
            try:
                (
                    raw,
                    response_id,
                    finish,
                    controls,
                    retries,
                ) = issue_with_transport_retries(
                    client, args.model, config, row["input"], max_tokens
                )
                transport_retries += retries
                received = True
                new_received += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            probability = registered.parse_yes_probability(raw) if received else None
            if received and probability is None:
                error = "unparseable response"
            if probability is not None:
                new_parsed += 1
            record = {
                "task_id": row["task_id"],
                "event_id": row["event_id"],
                "series_id": row["series_id"],
                "template_key": row["template_key"],
                "prompt_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
                "model": args.model,
                "split": args.split,
                "draw_index": args.draw_index,
                "yes_prob": probability,
                "raw": raw[:8000],
                "response_id": response_id,
                "finish_reason": finish,
                "response_received": received,
                "error": error,
                "effective_controls": controls,
                "max_output_tokens": max_tokens,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            if position == 1 or position % 25 == 0 or position == len(pending):
                print(
                    f"{position}/{len(pending)} task={row['task_id']} "
                    f"p={probability} error={error or '-'}",
                    flush=True,
                )
            if args.delay:
                time.sleep(args.delay)
    all_records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    record_ids = {record["task_id"] for record in all_records}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": "off_the_shelf_api_reference",
        "status": "complete" if record_ids == selected_ids else "partial",
        "model": args.model,
        "provider_protocol": config["protocol"],
        "endpoint": config["endpoint"],
        "split": args.split,
        "draw_index": args.draw_index,
        "selected_tasks": len(rows),
        "responses_received": sum(
            record["response_received"] for record in all_records
        ),
        "successfully_parsed": sum(
            record["yes_prob"] is not None for record in all_records
        ),
        "new_responses_received": new_received,
        "new_successfully_parsed": new_parsed,
        "transport_retries_this_run": transport_retries,
        "requested_controls": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": None,
        },
        "effective_controls": {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": None,
            "reasoning_effort": config["reasoning_effort"],
            "max_output_tokens": max_tokens,
        },
        "task_file": str(HOLDOUT if args.split == "holdout" else DEV),
        "task_file_sha256": sha256(HOLDOUT if args.split == "holdout" else DEV),
        "registration": str(args.registration),
        "registration_sha256": sha256(args.registration),
        "protocol_sha256": sha256(PROTOCOL),
        "analysis_code_sha256": sha256(Path(__file__)),
        "raw": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"raw -> {raw_path}")
    print(f"manifest -> {manifest_path}")


def finalize(args: argparse.Namespace) -> None:
    if args.split != "holdout":
        raise SystemExit("--finalize currently serves only the complete holdout")
    rows, _ = validate_inputs(args.split, args.registration)
    by_draw: list[dict[str, dict[str, Any]]] = []
    manifests = []
    for draw_index in range(DRAWS):
        raw_path, manifest_path = output_paths(
            args.output_dir, args.model, args.split, draw_index
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("selected_tasks") != 318
        ):
            raise RuntimeError(f"hosted draw {draw_index} is incomplete")
        if manifest.get("analysis_code_sha256") != sha256(Path(__file__)):
            raise RuntimeError(f"hosted draw {draw_index} code hash drifted")
        if sha256(raw_path) != manifest.get("raw_sha256"):
            raise RuntimeError(f"hosted draw {draw_index} raw hash drifted")
        records = {record["task_id"]: record for record in read_rows(raw_path)}
        if len(records) != 318:
            raise RuntimeError(f"hosted draw {draw_index} task roster drifted")
        by_draw.append(records)
        manifests.append(manifest)
    task_ids = [row["task_id"] for row in rows]
    aggregates: list[float | None] = []
    raw_rows = []
    for row, task_id in zip(rows, task_ids, strict=True):
        records = [draw[task_id] for draw in by_draw]
        expected_prompt_hash = hashlib.sha256(row["input"].encode()).hexdigest()
        if any(record["prompt_sha256"] != expected_prompt_hash for record in records):
            raise RuntimeError(f"hosted prompt hash drifted for {task_id}")
        values = [record["yes_prob"] for record in records]
        aggregate = (
            None if any(value is None for value in values) else sum(values) / DRAWS
        )
        aggregates.append(aggregate)
        raw_rows.append(
            {
                "task_id": task_id,
                "event_id": row["event_id"],
                "series_id": row["series_id"],
                "template_key": row["template_key"],
                "settlement_yes": row["settlement_yes"],
                "market_yes_prob": row["market_yes_prob"],
                "draws": values,
                "aggregate_yes_prob": aggregate,
                "response_ids": [record["response_id"] for record in records],
            }
        )
    model_losses = [
        registered.losses(probability, bool(row["settlement_yes"]))[0]
        for probability, row in zip(aggregates, rows, strict=True)
    ]
    market = [float(row["market_yes_prob"]) for row in rows]
    market_losses = [
        registered.losses(probability, bool(row["settlement_yes"]))[0]
        for probability, row in zip(market, rows, strict=True)
    ]
    prefix = args.output_dir / f"exp4_hosted_{slug(args.model)}_stochastic"
    raw_output = prefix.with_suffix(".jsonl")
    with raw_output.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": "off_the_shelf_api_reference",
        "model": args.model,
        "provider_protocol": PROVIDERS[args.model]["protocol"],
        "endpoint": PROVIDERS[args.model]["endpoint"],
        "test_was_used_for_configuration": False,
        "holdout": {"n": 318, "test_tasks_sha256": HOLDOUT_SHA256},
        "draws": DRAWS,
        "requested_controls": manifests[0]["requested_controls"],
        "effective_controls": manifests[0]["effective_controls"],
        "endpoint_summary": {
            **registered.summarize(aggregates, rows),
            "draw_parse_coverage": sum(
                record[task_id]["yes_prob"] is not None
                for record in by_draw
                for task_id in task_ids
            )
            / (len(rows) * DRAWS),
            "complete_task_parse_coverage": sum(
                value is not None for value in aggregates
            )
            / len(aggregates),
        },
        "market": registered.summarize(market, rows),
        "hosted_minus_market_brier": registered.cluster_bootstrap_delta(
            rows, model_losses, market_losses
        ),
        "draw_manifests": [
            {
                "path": str(
                    output_paths(args.output_dir, args.model, args.split, index)[1]
                ),
                "sha256": sha256(
                    output_paths(args.output_dir, args.model, args.split, index)[1]
                ),
            }
            for index in range(DRAWS)
        ],
        "raw": str(raw_output),
        "raw_sha256": sha256(raw_output),
        "analysis_code_sha256": sha256(Path(__file__)),
        "registration_sha256": sha256(args.registration),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = prefix.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"raw -> {raw_output}")
    print(f"summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(PROVIDERS))
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--draw-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--output-dir", type=Path, default=REPORTS / "exp4_hosted")
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--timeout", type=float, default=1_200.0)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.finalize:
        finalize(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
