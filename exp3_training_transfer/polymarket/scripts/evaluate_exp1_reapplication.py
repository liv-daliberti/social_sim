#!/usr/bin/env python3
"""Apply the frozen Experiment 1 evidence packets to base or one 3B adapter.

The evaluation holds the initial structured forecast and evidence packet fixed
across model conditions.  It therefore isolates conditional updating from web
retrieval and initial-world-model construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP1 = REPO / "exp1_prospective"
CONTEXTS = EXP1 / "data/initial_forecasts/forecasts_qwen2.5-7b_2026-06-10.jsonl"
PACKETS = EXP1 / "data/counterfactuals/counterfactuals_2026-06-10.jsonl"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
EXPECTED_MARKETS = 100
EXPECTED_PACKETS = 900
DIRECTION_ORDER = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}
DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")


# Byte-identical to exp1_prospective/agent/local_updated_forecast.py::_UPDATE_PROMPT.
UPDATE_PROMPT = """\
You previously produced this structured forecast for a binary prediction market:

QUESTION: {question}

INITIAL FORECAST:
{initial_sf_json}

---
NEW EVIDENCE (received after your initial research):

{evidence_text}
---

Using ONLY the information above — your prior structured forecast plus this new evidence — update your world model and produce a revised forecast.

Rules:
- DO NOT search the web, use any tools, or look up any external information.
  Reason entirely from your existing forecast and the new evidence above.
- Update H1 and H2 posterior_probability to reflect what the new evidence implies.
- Update the supporting_evidence or contradicting_evidence list for the affected hypothesis to include a brief note about the new evidence.
- Update yes_prob (must equal H1 posterior_probability).
- Update rationale to explain concisely what changed and why.
- Keep all other fields (key_actors, key_mechanisms, etc.) unchanged.
- Output ONLY a valid JSON object in the exact same structure as the initial forecast above.  No markdown fences, no commentary before or after."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_task_id(task_id: str) -> str:
    return DATE_SUFFIX_RE.sub("", task_id)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def canonical_forecast(record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    for run in record.get("k_runs", []):
        forecast = run.get("structured_forecast")
        if forecast and forecast.get("hypotheses"):
            return int(run.get("run_id", 0)), forecast
    forecast = record.get("structured_forecast")
    if forecast and forecast.get("hypotheses"):
        return 0, forecast
    raise AssertionError(f"no structured forecast for {record.get('task_id')}")


def probability(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None


def h1_probability(forecast: dict[str, Any]) -> float | None:
    hypotheses = forecast.get("hypotheses", [])
    h1 = next(
        (item for item in hypotheses if isinstance(item, dict) and item.get("id") == "H1"),
        {},
    )
    return probability(h1.get("posterior_probability"))


def parse_forecast(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError:
            pass
    return None, "no valid JSON object"


def load_instrument(limit: int = 0) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(CONTEXTS):
        latest[normalize_task_id(str(record["task_id"]))] = record

    packets_by_task: dict[str, list[dict[str, Any]]] = {}
    seen_packet_ids: set[str] = set()
    for packet in read_jsonl(PACKETS):
        task_id = normalize_task_id(str(packet["task_id"]))
        packet_id = str(packet["cf_id"])
        if packet_id in seen_packet_ids:
            raise AssertionError(f"duplicate packet id: {packet_id}")
        seen_packet_ids.add(packet_id)
        packets_by_task.setdefault(task_id, []).append(packet)

    task_ids = sorted(packets_by_task)
    if limit:
        task_ids = task_ids[:limit]
    examples: list[dict[str, Any]] = []
    for task_id in task_ids:
        record = latest.get(task_id)
        if record is None:
            raise AssertionError(f"packet task lacks initial context: {task_id}")
        run_id, forecast = canonical_forecast(record)
        initial_yes = probability(forecast.get("yes_prob"))
        initial_h1 = h1_probability(forecast)
        if initial_yes is None:
            initial_yes = initial_h1
        if initial_yes is None or initial_h1 is None:
            raise AssertionError(f"invalid initial probability: {task_id}")
        if abs(initial_yes - initial_h1) >= 0.02:
            raise AssertionError(f"incoherent initial forecast: {task_id}")

        packets = sorted(
            packets_by_task[task_id],
            key=lambda item: (DIRECTION_ORDER.get(str(item.get("direction")), 99), int(item["cf_index"])),
        )
        counts = {direction: 0 for direction in DIRECTION_ORDER}
        for packet in packets:
            direction = str(packet.get("direction"))
            if direction not in counts:
                raise AssertionError(f"unknown direction for {packet.get('cf_id')}: {direction}")
            counts[direction] += 1
        if counts != {"pro_H1": 3, "anti_H1": 3, "orthogonal": 3}:
            raise AssertionError(f"unbalanced packets for {task_id}: {counts}")

        initial_json = json.dumps(forecast, indent=2, ensure_ascii=True)
        for packet in packets:
            update_prompt = UPDATE_PROMPT.format(
                question=record.get("question", ""),
                initial_sf_json=initial_json,
                evidence_text=packet.get("evidence_text", ""),
            )
            rendered_prompt = (
                "<|im_start|>user\n" + update_prompt + "\n/no_think<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            examples.append(
                {
                    "task_id": task_id,
                    "source_task_id": record["task_id"],
                    "market_id": record.get("market_id"),
                    "question": record.get("question", ""),
                    "initial_run_id": run_id,
                    "initial_yes_prob": initial_yes,
                    "initial_h1_prob": initial_h1,
                    "initial_forecast_sha256": hashlib.sha256(initial_json.encode()).hexdigest(),
                    "cf_id": packet["cf_id"],
                    "direction": packet["direction"],
                    "cf_index": packet["cf_index"],
                    "evidence_text": packet.get("evidence_text", ""),
                    "prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
                    "rendered_prompt": rendered_prompt,
                }
            )

    if not limit:
        if len(task_ids) != EXPECTED_MARKETS or len(examples) != EXPECTED_PACKETS:
            raise AssertionError(
                f"instrument drift: markets={len(task_ids)}, packets={len(examples)}"
            )
    return examples


def resolve_updated_fields(
    text: str,
) -> tuple[dict[str, Any] | None, float | None, float | None, str | None]:
    parsed, error = parse_forecast(text)
    if parsed is None:
        return None, None, None, error
    updated_h1 = h1_probability(parsed)
    updated_yes = probability(parsed.get("yes_prob"))
    if updated_yes is None:
        updated_yes = updated_h1
    problems = []
    if updated_h1 is None:
        problems.append("missing/invalid H1 posterior_probability")
    if updated_yes is None:
        problems.append("missing/invalid yes_prob")
    if updated_h1 is not None and updated_yes is not None and abs(updated_h1 - updated_yes) >= 0.02:
        problems.append("yes_prob differs from H1 posterior by >= .02")
    return parsed, updated_yes, updated_h1, "; ".join(problems) or None


def write_outputs(
    examples: list[dict[str, Any]],
    texts: list[str],
    *,
    name: str,
    adapter: Path | None,
    output: Path,
    max_tokens: int,
    max_model_len: int,
    max_prompt_tokens: int,
) -> None:
    if len(examples) != len(texts):
        raise AssertionError("generation count does not match instrument")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    valid = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for example, text in zip(examples, texts):
            parsed, updated_yes, updated_h1, parse_error = resolve_updated_fields(text)
            if updated_yes is not None and updated_h1 is not None:
                valid += 1
            record = {key: value for key, value in example.items() if key != "rendered_prompt"}
            record.update(
                {
                    "evaluation_name": name,
                    "forecast_model": MODEL,
                    "adapter_path": str(adapter) if adapter else None,
                    "updated_yes_prob": updated_yes,
                    "updated_h1_prob": updated_h1,
                    "delta_yes_prob": (
                        updated_yes - float(example["initial_yes_prob"])
                        if updated_yes is not None
                        else None
                    ),
                    "delta_h1_prob": (
                        updated_h1 - float(example["initial_yes_prob"])
                        if updated_h1 is not None
                        else None
                    ),
                    "updated_structured_forecast": parsed,
                    "parse_error": parse_error,
                    "response": text,
                }
            )
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(output)

    manifest = {
        "protocol_version": "exp3b_exp1_reapplication_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": name,
        "model": MODEL,
        "adapter_path": str(adapter) if adapter else None,
        "adapter_config_sha256": (
            sha256_file(adapter / "adapter_config.json") if adapter else None
        ),
        "adapter_model_sha256": (
            sha256_file(adapter / "adapter_model.safetensors") if adapter else None
        ),
        "contexts": str(CONTEXTS),
        "contexts_sha256": sha256_file(CONTEXTS),
        "packets": str(PACKETS),
        "packets_sha256": sha256_file(PACKETS),
        "update_prompt_sha256": hashlib.sha256(UPDATE_PROMPT.encode()).hexdigest(),
        "n_markets": len({item["task_id"] for item in examples}),
        "n_packets": len(examples),
        "n_valid": valid,
        "parse_coverage": valid / len(examples),
        "decoding": {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_prompt_tokens": max_prompt_tokens,
            "thinking_disabled": True,
        },
        "design": (
            "Paired fixed-context reapplication: every model receives the same canonical "
            "Qwen2.5-7B initial structured forecast and the same frozen Experiment 1 packet."
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    examples = load_instrument(args.limit)
    audit = {
        "markets": len({item["task_id"] for item in examples}),
        "packets": len(examples),
        "directions": {
            direction: sum(item["direction"] == direction for item in examples)
            for direction in DIRECTION_ORDER
        },
        "max_prompt_characters": max(len(item["rendered_prompt"]) for item in examples),
        "contexts_sha256": sha256_file(CONTEXTS),
        "packets_sha256": sha256_file(PACKETS),
    }
    if args.dry_run:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    if args.adapter is not None and not (args.adapter / "adapter_config.json").is_file():
        raise AssertionError(f"not a PEFT adapter: {args.adapter}")

    import vllm
    from vllm.lora.request import LoRARequest

    llm = vllm.LLM(
        model=MODEL,
        enable_lora=args.adapter is not None,
        max_lora_rank=32,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    token_lengths = [len(tokenizer.encode(item["rendered_prompt"])) for item in examples]
    max_prompt_tokens = max(token_lengths)
    if max_prompt_tokens + args.max_tokens > args.max_model_len:
        raise AssertionError(
            f"prompt+generation exceeds model length: {max_prompt_tokens}+{args.max_tokens}"
        )

    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    request = (
        LoRARequest(args.name, 1, str(args.adapter)) if args.adapter is not None else None
    )
    outputs = llm.generate(
        [item["rendered_prompt"] for item in examples],
        sampling,
        lora_request=request,
    )
    texts = [output.outputs[0].text for output in outputs]
    write_outputs(
        examples,
        texts,
        name=args.name,
        adapter=args.adapter,
        output=args.output,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_prompt_tokens=max_prompt_tokens,
    )
    print(json.dumps({**audit, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
