#!/usr/bin/env python3
"""Run frozen indirect packets; refuses all execution before a valid freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
DEFAULT_INSTRUMENT = ROOT / "data/frozen/instrument.jsonl"
DEFAULT_MANIFEST = ROOT / "data/frozen/manifest.json"
DEFAULT_RUNS = ROOT / "runs"
INITIAL_DIR = REPO / "exp1_prospective/data/initial_forecasts"
PROMPT_PATH = ROOT / "prompts/update.txt"
MODEL_DEFAULTS = {
    "gpt-5.4": {
        "backend": "openai",
        "endpoint": "https://liv-forecast.services.ai.azure.com/openai/v1",
        "key_env": "AZURE_AI_API_KEY",
    },
    "DeepSeek-V4-Pro": {
        "backend": "openai",
        "endpoint": "https://cos-tiktok-annotation-a-resource.services.ai.azure.com/openai/v1",
        "key_env": "DEEPSEEK_AZURE_API_KEY",
    },
    "claude-opus-4-8": {
        "backend": "anthropic",
        "endpoint": "https://liv-forecast.services.ai.azure.com/anthropic",
        "key_env": "CLAUDE_AZURE_API_KEY",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_local_env() -> None:
    path = REPO / "exp1_prospective/agent/.env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def verify_freeze(instrument: Path, manifest_path: Path) -> dict[str, Any]:
    if not instrument.exists() or not manifest_path.exists():
        raise SystemExit("REFUSED: instrument is not frozen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(instrument.read_bytes()).hexdigest()
    if (
        manifest.get("status") != "FROZEN_FOR_TARGET_EVALUATION"
        or manifest.get("instrument_sha256") != actual
        or manifest.get("target_outputs_opened_before_freeze") is not False
    ):
        raise SystemExit("REFUSED: freeze manifest is absent, inconsistent, or contaminated")
    return manifest


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", model).strip("-")


def find_initials(model: str) -> Path:
    slug = model.replace("/", "-").replace(":", "-").replace(".", "-")
    candidates = sorted(INITIAL_DIR.glob(f"forecasts_{slug}_*.jsonl"), reverse=True)
    if not candidates:
        for path in sorted(INITIAL_DIR.glob("forecasts_*.jsonl"), reverse=True):
            manifest = path.with_suffix("").with_suffix(".manifest.json")
            if manifest.exists() and model in manifest.read_text(encoding="utf-8", errors="ignore"):
                candidates.append(path)
                break
    if not candidates:
        raise SystemExit(f"no initial forecasts found for {model}")
    return candidates[0]


def canonical_initials(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[row["task_id"]] = row
    result = {}
    for task_id, row in latest.items():
        runs = [
            run for run in row.get("k_runs", [])
            if run.get("structured_forecast") and run.get("yes_prob") is not None
        ]
        if not runs and row.get("structured_forecast") and row.get("yes_prob") is not None:
            runs = [{
                "run_id": 0,
                "structured_forecast": row["structured_forecast"],
                "yes_prob": row["yes_prob"],
            }]
        if runs:
            result[task_id] = {
                **row,
                "canonical_run": min(runs, key=lambda run: int(run.get("run_id", 10**9))),
            }
    return result


def render_prompt(packet: dict[str, Any], initial: dict[str, Any]) -> str:
    bridge = packet.get("bridge_context")
    if bridge:
        context = "HYPOTHETICAL CAUSAL BRIDGE (accept as true for this case):\n" + bridge
    elif packet["condition"] == "masked":
        context = "No causal bridge is provided. Do not invent an unstated mapping."
    else:
        context = "No additional causal bridge is supplied for this calibration condition."
    evidence = packet["evidence"]["headline"] + "\n" + packet["evidence"]["text"]
    replacements = {
        "{question}": packet["question"],
        "{resolution_criteria}": packet["resolution_criteria"],
        "{initial_forecast}": json.dumps(
            initial["canonical_run"]["structured_forecast"],
            ensure_ascii=False,
            indent=2,
        ),
        "{context_block}": context,
        "{evidence}": evidence,
    }
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    for source, target in replacements.items():
        prompt = prompt.replace(source, target)
    return prompt


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def normalize_response(text: str) -> tuple[float | None, dict[str, Any] | None, str | None]:
    try:
        value = extract_json(text)
        yes_prob = value.get("yes_prob")
        h1 = value.get("h1_posterior")
        if isinstance(yes_prob, bool) or not isinstance(yes_prob, (int, float)):
            raise ValueError("yes_prob is not numeric")
        if isinstance(h1, bool) or not isinstance(h1, (int, float)):
            raise ValueError("h1_posterior is not numeric")
        yes_prob, h1 = float(yes_prob), float(h1)
        if not 0 <= yes_prob <= 1 or not 0 <= h1 <= 1:
            raise ValueError("probability is outside [0,1]")
        if abs(yes_prob - h1) > 1e-8:
            raise ValueError("yes_prob and h1_posterior differ")
        return yes_prob, value, None
    except Exception as exc:
        return None, None, str(exc)


def call_openai(endpoint: str, key: str, model: str, prompt: str) -> tuple[str, str | None]:
    import requests
    response = requests.post(
        endpoint.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "api-key": key,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1200,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"].get("content") or "", body.get("id")


def call_anthropic(endpoint: str, key: str, model: str, prompt: str) -> tuple[str, str | None]:
    import requests
    response = requests.post(
        endpoint.rstrip("/") + "/v1/messages",
        headers={
            "Authorization": f"Bearer {key}",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1200,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    text = "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
    return text, body.get("id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instrument", type=Path, default=DEFAULT_INSTRUMENT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--initial-forecasts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("openai", "anthropic"))
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key")
    parser.add_argument("--key-env")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    freeze = verify_freeze(args.instrument, args.manifest)
    config = MODEL_DEFAULTS.get(args.model, {})
    backend = args.backend or config.get("backend")
    endpoint = args.endpoint or config.get("endpoint")
    if not backend or not endpoint:
        raise SystemExit("unknown model: pass --backend and --endpoint")
    initial_path = args.initial_forecasts or find_initials(args.model)
    initials = canonical_initials(initial_path)
    packets = read_jsonl(args.instrument)
    missing = sorted({packet["task_id"] for packet in packets} - set(initials))
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} markets lack canonical parseable initial forecasts")
    if freeze["packet_count"] != len(packets):
        raise SystemExit("REFUSED: manifest packet count differs from instrument")
    if args.limit:
        packets = packets[:args.limit]
    if args.dry_run:
        print(render_prompt(packets[0], initials[packets[0]["task_id"]]))
        return

    load_local_env()
    key_env = args.key_env or config.get("key_env", "")
    key = args.api_key or os.environ.get(key_env, "")
    if not key:
        raise SystemExit(f"set {key_env} or pass --api-key")
    output = args.output or DEFAULT_RUNS / model_slug(args.model) / "responses.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {
            row["evaluation_id"] for row in read_jsonl(output)
            if row.get("model") == args.model
        }
    pending = [packet for packet in packets if f"{args.model}|{packet['packet_id']}" not in done]
    print(f"model={args.model} packets={len(packets)} completed={len(done)} pending={len(pending)}")
    with output.open("a", encoding="utf-8") as handle:
        for index, packet in enumerate(pending, 1):
            initial = initials[packet["task_id"]]
            initial_prob = float(initial["canonical_run"]["yes_prob"])
            evaluation_id = f"{args.model}|{packet['packet_id']}"
            prompt = render_prompt(packet, initial)
            raw, response_id, call_error = "", None, None
            try:
                if backend == "anthropic":
                    raw, response_id = call_anthropic(endpoint, key, args.model, prompt)
                else:
                    raw, response_id = call_openai(endpoint, key, args.model, prompt)
                updated, parsed, parse_error = normalize_response(raw)
            except Exception as exc:
                updated, parsed, parse_error, call_error = None, None, None, str(exc)
            record = {
                "protocol_version": "indirect-core-v1",
                "evaluation_id": evaluation_id,
                "model": args.model,
                "deployment_endpoint": endpoint,
                "instrument_sha256": freeze["instrument_sha256"],
                "packet_id": packet["packet_id"],
                "market_id": packet["market_id"],
                "task_id": packet["task_id"],
                "candidate_id": packet["candidate_id"],
                "condition": packet["condition"],
                "initial_run_id": initial["canonical_run"].get("run_id", 0),
                "initial_yes_prob": initial_prob,
                "updated_yes_prob": updated,
                "delta_yes_prob": (
                    round(updated - initial_prob, 8) if updated is not None else None
                ),
                "parse_valid": updated is not None,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "call_error": call_error,
                "raw_response": raw,
                "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "response_id": response_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index:04d}/{len(pending)} {packet['condition']:<12} "
                f"{'valid' if record['parse_valid'] else 'INVALID'}",
                flush=True,
            )
            if index < len(pending):
                time.sleep(args.delay)
    print(f"wrote {len(pending)} records -> {output}")


if __name__ == "__main__":
    main()
