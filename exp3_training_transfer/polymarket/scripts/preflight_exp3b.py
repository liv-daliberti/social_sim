#!/usr/bin/env python3
"""Fail-closed audit for the frozen Experiment 3B data and split contract."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer
from build_exp3b_dataset import build_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "exp3b_registered"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT_MAX_TOKENS = 1_792
MODEL_CACHE = ROOT.parents[1] / ".runtime" / "hf_home" / "hub"
EXPECTED_SHA256 = "5697f7672bcb8dcc0af73272ff9ff1ec662241e92b0ac84f6e2bd4ba8f638e7d"
EXPECTED_COUNTS = {"train": 1_736, "dev": 512, "test": 1_024}
FORBIDDEN_PROMPT_TOKENS = (
    "winner_yes",
    "winner_label",
    "resolution_method",
    "latest_prices",
    "strict_settlement_yes",
    "final volume",
    "carry_forward",
    "category:",
    "scheduled market close",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def keys(row: dict[str, Any]) -> set[str]:
    result = {f"event:{row['event_id']}", f"template:{row['template_key']}"}
    if row["series_id"]:
        result.add(f"series:{row['series_id']}")
    return result


def audit(data: Path) -> dict[str, Any]:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise AssertionError("dataset build did not pass")
    if manifest.get("protocol_version") != "exp3b_registered_v1":
        raise AssertionError("unexpected protocol version")
    source = manifest.get("source") or {}
    if not source.get("fully_scanned") or source.get("sha256") != EXPECTED_SHA256:
        raise AssertionError("source archive was not fully scanned at the registered checksum")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, cache_dir=MODEL_CACHE, trust_remote_code=True, local_files_only=True
    )
    rows = {split: read_jsonl(data / f"{split}.tasks.jsonl") for split in EXPECTED_COUNTS}
    for split, expected in EXPECTED_COUNTS.items():
        if len(rows[split]) != expected:
            raise AssertionError(f"{split}: expected {expected} rows, got {len(rows[split])}")
        expected_hash = manifest["output_sha256"][f"{split}.tasks.jsonl"]
        actual_hash = sha256(data / f"{split}.tasks.jsonl")
        if actual_hash != expected_hash:
            raise AssertionError(f"{split}: JSONL hash drift")

    ranges = {
        "train": (None, datetime(2025, 7, 1, tzinfo=timezone.utc)),
        "dev": (datetime(2025, 7, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)),
        "test": (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc)),
    }
    key_sets: dict[str, set[str]] = {}
    max_prompt_chars = 0
    max_prompt_tokens = 0
    for split, split_rows in rows.items():
        task_ids = [row["task_id"] for row in split_rows]
        if len(task_ids) != len(set(task_ids)):
            raise AssertionError(f"{split}: duplicate task ids")
        key_sets[split] = set().union(*(keys(row) for row in split_rows))
        lower, upper = ranges[split]
        for row in split_rows:
            decision = parse_ts(row["decision_ts"])
            if lower is not None and decision < lower or decision >= upper:
                raise AssertionError(f"{split}: decision timestamp outside frozen range")
            prompt = row["input"]
            if prompt != build_prompt(row):
                raise AssertionError(
                    f"{split}: prompt does not match its serialized question, rules, cutoff, and prices"
                )
            max_prompt_chars = max(max_prompt_chars, len(prompt))
            wrapped = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            token_count = len(tokenizer(wrapped)["input_ids"])
            max_prompt_tokens = max(max_prompt_tokens, token_count)
            if token_count > PROMPT_MAX_TOKENS:
                raise AssertionError(
                    f"{split}: prompt has {token_count} tokens; limit is {PROMPT_MAX_TOKENS}"
                )
            lowered = prompt.lower()
            for token in FORBIDDEN_PROMPT_TOKENS:
                if token in lowered:
                    raise AssertionError(f"{split}: forbidden prompt field {token!r}")
            history = row["price_history"]
            if len(history) < 4 or len(history) > 14:
                raise AssertionError(f"{split}: invalid price history length")
            history_dates = [parse_ts(point["ts"]).date() for point in history]
            if len(history_dates) != len(set(history_dates)):
                raise AssertionError(f"{split}: repeated UTC date in price history")
            if any(parse_ts(point["ts"]) > decision for point in history):
                raise AssertionError(f"{split}: post-cutoff candle")
            if abs(float(history[-1]["yes_close"]) - float(row["market_yes_prob"])) > 1e-12:
                raise AssertionError(f"{split}: market baseline does not match latest real candle")
            reference = json.loads(row["reference"])
            for field in ("task_id", "market_id", "event_id", "decision_ts", "market_yes_prob", "settlement_yes"):
                if reference[field] != row[field]:
                    raise AssertionError(f"{split}: reference mismatch for {field}")

        hf_rows = load_from_disk(str(data / f"hf_{split}"))["train"]
        if len(hf_rows) != len(split_rows):
            raise AssertionError(f"{split}: Hugging Face row count mismatch")
        if hf_rows["input"] != [row["input"] for row in split_rows]:
            raise AssertionError(f"{split}: Hugging Face prompts drifted")
        if hf_rows["reference"] != [row["reference"] for row in split_rows]:
            raise AssertionError(f"{split}: Hugging Face references drifted")

    schedule = load_from_disk(str(data / "hf_train_schedule"))["train"]
    if len(schedule) != 4_800:
        raise AssertionError(f"training schedule: expected 4800 rows, got {len(schedule)}")
    train_pairs = {(row["input"], row["reference"]) for row in rows["train"]}
    schedule_pairs = list(zip(schedule["input"], schedule["reference"]))
    if any(pair not in train_pairs for pair in schedule_pairs):
        raise AssertionError("training schedule contains a row outside the frozen train split")
    schedule_ids = [json.loads(reference)["task_id"] for reference in schedule["reference"]]
    repetitions = Counter(schedule_ids)
    if set(repetitions) != {row["task_id"] for row in rows["train"]}:
        raise AssertionError("training schedule does not cover every frozen train market")
    if max(repetitions.values()) - min(repetitions.values()) > 1:
        raise AssertionError("training schedule repetitions are not balanced")
    order_hash = hashlib.sha256("\n".join(schedule_ids).encode()).hexdigest()
    schedule_manifest = manifest.get("training_schedule") or {}
    if order_hash != schedule_manifest.get("task_order_sha256"):
        raise AssertionError("training schedule order hash drift")

    overlap = {
        "train_dev": len(key_sets["train"] & key_sets["dev"]),
        "train_test": len(key_sets["train"] & key_sets["test"]),
        "dev_test": len(key_sets["dev"] & key_sets["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"cross-split family leakage: {overlap}")
    task_sets = {split: {row["task_id"] for row in values} for split, values in rows.items()}
    if task_sets["train"] & task_sets["dev"] or task_sets["train"] & task_sets["test"] or task_sets["dev"] & task_sets["test"]:
        raise AssertionError("task ids overlap across splits")

    return {
        "status": "pass",
        "protocol_version": "exp3b_registered_v1",
        "source_sha256": EXPECTED_SHA256,
        "counts": {split: len(values) for split, values in rows.items()},
        "yes_rates": {
            split: sum(bool(row["settlement_yes"]) for row in values) / len(values)
            for split, values in rows.items()
        },
        "cross_split_key_overlap": overlap,
        "training_schedule": {
            "prompt_presentations": len(schedule),
            "unique_markets": len(repetitions),
            "minimum_repetitions": min(repetitions.values()),
            "maximum_repetitions": max(repetitions.values()),
            "task_order_sha256": order_hash,
        },
        "maximum_prompt_characters": max_prompt_chars,
        "maximum_prompt_tokens": max_prompt_tokens,
        "prompt_token_limit": PROMPT_MAX_TOKENS,
        "training_evaluation_split": "dev",
        "locked_final_split": "test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--write", type=Path, default=ROOT / "protocol" / "exp3b_preflight.json")
    args = parser.parse_args()
    report = audit(args.data)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
