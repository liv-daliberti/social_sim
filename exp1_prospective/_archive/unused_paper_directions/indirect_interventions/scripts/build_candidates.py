#!/usr/bin/env python3
"""Generate the three-family indirect instrument on the frozen Exp. 1 core.

This script may inspect initial forecasts and frozen direct packets, but never
target-model updates. It is append-only and resumable at the market level.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
PROMPT_PATH = ROOT / "prompts/generate_family.txt"
SCHEMA_PATH = ROOT / "schema/candidate.schema.json"
CORE_PACKETS = REPO / "exp1_prospective/data/counterfactuals/counterfactuals_2026-06-10.jsonl"
CANONICAL_FORECASTS = REPO / "exp1_prospective/data/initial_forecasts/forecasts_DeepSeek-V4-Pro_2026-06-10.jsonl"
DEFAULT_OUTPUT = ROOT / "data/candidates/core_candidates.jsonl"
DEFAULT_RAW = ROOT / "data/candidates/raw_core_generations.jsonl"
DEFAULT_CONTROLS = ROOT / "data/candidates/core_controls.jsonl"
DEFAULT_DEVELOPMENT_GATE = ROOT / "reports/development_gate.json"
PROTOCOL_VERSION = "indirect-core-v1"
DEFAULT_ENDPOINT = "https://liv.services.ai.azure.com/openai/v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_KEY_ENV = "GPT56_AZURE_API_KEY"
DEFAULT_PROTOCOL = "openai_responses"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 12000
EVIDENCE_WORDS = (20, 45)
DIRECTIONAL_BRIDGE_WORDS = (20, 40)
BROKEN_BRIDGE_WORDS = (15, 28)
DECISIVE_CLAUSE_WORDS = (3, 12)
BRIDGE_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
MAX_BRIDGE_SENTENCES = 2
MAX_BRIDGE_SENTENCE_WORDS = 24

FORBIDDEN_EVIDENCE = re.compile(
    r"\b(?:yes|no|more likely|less likely|odds|probability|forecast|on-track|"
    r"setback|boost|hurt|win|lose|approve|reject)\b",
    re.IGNORECASE,
)
EXPECTED = {
    "positive": ("increase_yes", 1),
    "negative": ("decrease_yes", -1),
    "broken": ("no_material_effect", 0),
}
LATIN = (
    ("positive", "negative", "neutral"),
    ("negative", "neutral", "positive"),
    ("neutral", "positive", "negative"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_local_env() -> None:
    env_path = REPO / "exp1_prospective/agent/.env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def canonical_records(path: Path) -> dict[str, dict[str, Any]]:
    """Select final checkpoint record, then run_id=0, for every market."""
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if task_id:
            latest[task_id] = row
    selected: dict[str, dict[str, Any]] = {}
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
        if not runs:
            continue
        run = min(runs, key=lambda item: int(item.get("run_id", 10**9)))
        selected[task_id] = {**row, "canonical_run": run}
    return selected


def frozen_core() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(CORE_PACKETS):
        by_task.setdefault(row["task_id"], []).append(row)
    forecasts = canonical_records(CANONICAL_FORECASTS)
    missing = sorted(set(by_task) - set(forecasts))
    if missing:
        raise SystemExit(f"missing parseable canonical forecasts for {len(missing)} core markets")
    records = [forecasts[task_id] for task_id in sorted(by_task)]
    if len(records) != 100 or any(len(by_task[row["task_id"]]) != 9 for row in records):
        raise SystemExit("frozen core is not exactly 100 markets x 9 direct packets")
    return records, by_task


def prompt_for(record: dict[str, Any], market_index: int) -> str:
    sf = record["canonical_run"]["structured_forecast"]
    event_model = sf.get("event_model", {})
    replacements = {
        "{question}": record["question"],
        "{resolution_criteria}": record.get("description", ""),
        "{initial_event_model}": json.dumps(event_model, ensure_ascii=False, indent=2),
        "{surface_valences}": ", ".join(LATIN[market_index % 3]),
    }
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    for source, target in replacements.items():
        prompt = prompt.replace(source, target)
    return prompt


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("generator response contains no JSON object")
        parsed = json.loads(stripped[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("generator response must be a JSON object")
    return parsed


def signed_product(edges: list[str]) -> int:
    product = 1
    for edge in edges:
        if edge == "0":
            return 0
        if edge == "-":
            product *= -1
        elif edge != "+":
            raise ValueError(f"invalid signed edge {edge!r}")
    return product


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned[:48] or "family"


def normalize_family(
    family: dict[str, Any],
    record: dict[str, Any],
    market_index: int,
    family_index: int,
    generator_id: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    expected_valence = LATIN[market_index % 3][family_index]
    evidence = family.get("evidence", {})
    if evidence.get("surface_valence") != expected_valence:
        raise ValueError(
            f"family {family_index}: expected {expected_valence} surface valence, "
            f"got {evidence.get('surface_valence')!r}"
        )
    evidence_text = f"{evidence.get('headline', '')} {evidence.get('text', '')}".strip()
    word_count = len(evidence_text.split())
    if not EVIDENCE_WORDS[0] <= word_count <= EVIDENCE_WORDS[1]:
        raise ValueError(f"family {family_index}: evidence has {word_count} words")
    forbidden = FORBIDDEN_EVIDENCE.search(evidence_text)
    if forbidden:
        raise ValueError(f"family {family_index}: forbidden evidence cue {forbidden.group(0)!r}")

    frame = family.get("bridge_frame", {})
    prefix = frame.get("prefix", "")
    suffix = frame.get("suffix", "")
    if len(prefix) < 20 or len(suffix) < 20:
        raise ValueError(f"family {family_index}: bridge frame is too short")
    raw_contexts = family.get("contexts", {})
    contexts: dict[str, dict[str, Any]] = {}
    for role, (label, product) in EXPECTED.items():
        raw = raw_contexts.get(role, {})
        if raw.get("intended_label") != label:
            raise ValueError(f"family {family_index}/{role}: incorrect intended label")
        edges = raw.get("chain_edges", [])
        nodes = raw.get("chain_nodes", [])
        if len(edges) < 2 or len(nodes) != len(edges) + 1:
            raise ValueError(f"family {family_index}/{role}: malformed causal chain")
        if signed_product(edges) != product:
            raise ValueError(f"family {family_index}/{role}: signed chain has wrong product")
        decisive = raw.get("decisive_clause", "")
        if role in {"positive", "negative"}:
            if len(decisive.strip()) < 3:
                raise ValueError(f"family {family_index}/{role}: decisive clause is missing")
            bridge_text = prefix + decisive + suffix
            bridge_words = len(bridge_text.split())
            decisive_words = len(decisive.split())
            if not DIRECTIONAL_BRIDGE_WORDS[0] <= bridge_words <= DIRECTIONAL_BRIDGE_WORDS[1]:
                raise ValueError(
                    f"family {family_index}/{role}: bridge has {bridge_words} words"
                )
            if not DECISIVE_CLAUSE_WORDS[0] <= decisive_words <= DECISIVE_CLAUSE_WORDS[1]:
                raise ValueError(
                    f"family {family_index}/{role}: decisive clause has "
                    f"{decisive_words} words"
                )
        else:
            bridge_text = raw.get("bridge_text", "")
            if len(bridge_text) < 30:
                raise ValueError(f"family {family_index}/broken: bridge is too short")
            bridge_words = len(bridge_text.split())
            if not BROKEN_BRIDGE_WORDS[0] <= bridge_words <= BROKEN_BRIDGE_WORDS[1]:
                raise ValueError(
                    f"family {family_index}/broken: bridge has {bridge_words} words"
                )
        acronyms = sorted(set(BRIDGE_ACRONYM.findall(bridge_text)) - {"YES", "NO"})
        if acronyms:
            raise ValueError(
                f"family {family_index}/{role}: replace all-caps abbreviation(s): "
                f"{', '.join(acronyms)}"
            )
        sentences = [
            sentence.strip() for sentence in re.split(r"[.!?]+", bridge_text)
            if sentence.strip()
        ]
        if len(sentences) > MAX_BRIDGE_SENTENCES:
            raise ValueError(
                f"family {family_index}/{role}: bridge has {len(sentences)} sentences"
            )
        if any(len(sentence.split()) > MAX_BRIDGE_SENTENCE_WORDS for sentence in sentences):
            raise ValueError(
                f"family {family_index}/{role}: bridge sentence exceeds "
                f"{MAX_BRIDGE_SENTENCE_WORDS} words"
            )
        if ";" in bridge_text:
            raise ValueError(f"family {family_index}/{role}: bridge contains a semicolon")
        contexts[role] = {
            "bridge_text": bridge_text,
            **({"decisive_clause": decisive} if decisive else {}),
            "intended_label": label,
            "chain_nodes": nodes,
            "chain_edges": edges,
        }
    if contexts["positive"]["decisive_clause"] == contexts["negative"]["decisive_clause"]:
        raise ValueError(f"family {family_index}: directional clauses are identical")

    market_id = str(record["market_id"])
    split = record.get("_split", "core")
    event_model = (
        family.get("initial_event_model")
        if split == "development"
        else record["canonical_run"]["structured_forecast"]["event_model"]
    )
    if not isinstance(event_model, dict):
        raise ValueError(f"family {family_index}: initial event model is missing")
    return {
        "protocol_version": (
            "indirect-development-v1" if split == "development" else PROTOCOL_VERSION
        ),
        "candidate_id": f"{split}_{market_id}_f{family_index}_{slug(family.get('family_slug', 'family'))}",
        "development_only": split == "development",
        "split": split,
        "family_index": family_index,
        "market": {
            "task_id": record["task_id"],
            "market_id": market_id,
            "question": record["question"],
            "resolution_criteria": record.get("description", ""),
            "category": record.get("category", ""),
            "end_time": None,
        },
        "initial_event_model": event_model,
        "evidence": {
            "headline": evidence["headline"].strip(),
            "text": evidence["text"].strip(),
            "surface_valence": expected_valence,
        },
        "bridge_frame": {"prefix": prefix, "suffix": suffix},
        "contexts": contexts,
        "generation": {
            "generator_id": generator_id,
            "prompt_sha256": prompt_sha256,
            "status": "draft_for_human_validation",
            "notes": family.get("private_rationale", ""),
        },
    }


def validate_schema(rows: list[dict[str, Any]]) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for row in rows:
        jsonschema.validate(row, schema)


def _responses_output_text(body: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API object."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for block in item.get("content", []):
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


def call_generator(client: dict[str, Any], model: str, prompt: str, retries: int) -> tuple[str, dict[str, Any]]:
    import requests

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            protocol = client.get("protocol", DEFAULT_PROTOCOL)
            if protocol == "openai_responses":
                path = "/responses"
                payload = {
                    "model": model,
                    "input": [{"role": "user", "content": prompt}],
                    "reasoning": {"effort": client["reasoning_effort"]},
                    "max_output_tokens": client["max_output_tokens"],
                    "text": {"format": {"type": "json_object"}},
                    "store": False,
                }
            elif protocol == "chat_completions":
                path = "/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": client["max_output_tokens"],
                }
            else:
                raise ValueError(f"unsupported generator protocol: {protocol}")
            response = requests.post(
                client["endpoint"].rstrip("/") + path,
                headers={
                    "Authorization": f"Bearer {client['api_key']}",
                    "api-key": client["api_key"],
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            body = response.json()
            if protocol == "openai_responses":
                if body.get("status") not in {None, "completed"}:
                    raise RuntimeError(
                        f"incomplete response status={body.get('status')!r}: "
                        f"{body.get('incomplete_details')!r}"
                    )
                text = _responses_output_text(body)
            else:
                text = body["choices"][0]["message"].get("content") or ""
            if not text.strip():
                raise RuntimeError("generator returned no output text")
            return text, body
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"generator failed after {retries} attempts: {last_error}")


def generate_market(
    client: dict[str, str],
    model: str,
    record: dict[str, Any],
    market_index: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = prompt_for(record, market_index)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    text, response = call_generator(client, model, prompt, retries)
    payload = extract_json(text)
    families = payload.get("families")
    if not isinstance(families, list) or len(families) != 3:
        raise ValueError("generator did not return exactly three families")
    rows = [
        normalize_family(family, record, market_index, index, model, prompt_sha)
        for index, family in enumerate(families)
    ]
    if len({row["evidence"]["headline"] for row in rows}) != 3:
        raise ValueError("three families do not have distinct evidence")
    validate_schema(rows)
    raw = {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": record["task_id"],
        "market_id": str(record["market_id"]),
        "model": model,
        "prompt_sha256": prompt_sha,
        "provider": client.get("provider", "azure_openai"),
        "endpoint": client["endpoint"],
        "api_protocol": client.get("protocol", DEFAULT_PROTOCOL),
        "reasoning_effort": client.get("reasoning_effort"),
        "max_output_tokens": client.get("max_output_tokens"),
        "response_id": response.get("id"),
        "response": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return rows, raw


def build_controls(
    records: list[dict[str, Any]],
    packets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result = []
    mapping = {
        "pro_H1": ("direct_pro", "increase_yes"),
        "anti_H1": ("direct_anti", "decrease_yes"),
        "orthogonal": ("orthogonal", "no_material_effect"),
    }
    for record in records:
        market = {
            "task_id": record["task_id"],
            "market_id": str(record["market_id"]),
            "question": record["question"],
            "resolution_criteria": record.get("description", ""),
            "category": record.get("category", ""),
        }
        event_model = record["canonical_run"]["structured_forecast"]["event_model"]
        for direction, (condition, label) in mapping.items():
            choices = sorted(
                (row for row in packets[record["task_id"]] if row["direction"] == direction),
                key=lambda row: (int(row.get("cf_index", 99)), row["cf_id"]),
            )
            source = choices[0]
            result.append({
                "protocol_version": PROTOCOL_VERSION,
                "control_id": f"control_{record['market_id']}_{condition}",
                "condition": condition,
                "market": market,
                "initial_event_model": event_model,
                "evidence": {
                    "headline": source["evidence_headline"],
                    "text": source["evidence_text"],
                },
                "expected_label": label,
                "source_cf_id": source["cf_id"],
            })
        result.append({
            "protocol_version": PROTOCOL_VERSION,
            "control_id": f"control_{record['market_id']}_literal_null",
            "condition": "literal_null",
            "market": market,
            "initial_event_model": event_model,
            "evidence": {
                "headline": "No additional event information",
                "text": "No new evidence is available beyond the initial structured forecast shown above.",
            },
            "expected_label": "no_material_effect",
            "source_cf_id": None,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--development-gate", type=Path, default=DEFAULT_DEVELOPMENT_GATE)
    args = parser.parse_args()

    records, packets = frozen_core()
    controls = build_controls(records, packets)
    args.controls_output.parent.mkdir(parents=True, exist_ok=True)
    args.controls_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in controls),
        encoding="utf-8",
    )
    completed: set[str] = set()
    if args.output.exists():
        counts: dict[str, int] = {}
        for row in read_jsonl(args.output):
            task_id = row["market"]["task_id"]
            counts[task_id] = counts.get(task_id, 0) + 1
        completed = {task_id for task_id, count in counts.items() if count == 3}
        incomplete = {task_id: count for task_id, count in counts.items() if count != 3}
        if incomplete:
            raise SystemExit(f"fail closed: incomplete market families in output: {incomplete}")

    selected = [
        (index, row) for index, row in enumerate(records)
        if index >= args.offset and row["task_id"] not in completed
    ]
    if args.limit:
        selected = selected[:args.limit]
    print(f"frozen core: {len(records)} markets; completed: {len(completed)}; queued: {len(selected)}")
    print(f"controls: {len(controls)} -> {args.controls_output}")
    if not selected:
        return
    if args.dry_run:
        print(prompt_for(selected[0][1], selected[0][0]))
        return

    if not args.development_gate.exists():
        raise SystemExit(
            f"refusing core generation before the 20-market development gate: "
            f"{args.development_gate}"
        )
    development_gate = json.loads(args.development_gate.read_text(encoding="utf-8"))
    if not development_gate.get("ready_for_core_generation"):
        raise SystemExit("refusing core generation: development human gate did not pass")

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

    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(generate_market, client, args.model, record, index, args.retries): (index, record)
            for index, record in selected
        }
        for future in concurrent.futures.as_completed(futures):
            index, record = futures[future]
            try:
                rows, raw = future.result()
                append_jsonl(args.output, rows)
                append_jsonl(args.raw_output, [raw])
                print(f"PASS {index + 1:03d}/100 market={record['market_id']} families=3", flush=True)
            except Exception as exc:
                failure = {
                    "task_id": record["task_id"],
                    "market_id": str(record["market_id"]),
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                failures.append(failure)
                append_jsonl(args.raw_output, [failure])
                print(f"FAIL {index + 1:03d}/100 market={record['market_id']}: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} markets failed validation; rerun after revising/regenerating")
    print(f"complete: {len(selected)} markets ({3 * len(selected)} families)")


if __name__ == "__main__":
    main()
