#!/usr/bin/env python3
"""Run all frozen Experiment 2 symbol-control arms on a local open model.

The three arms are evaluated together so that the arbitrary-symbol result has
configuration-matched no-context and semantic-context comparisons.  Prompts are
read verbatim from the frozen task files, decoded greedily once, and parsed with
the same permissive JSON parser used by the hosted-model runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_frozen_task_shard import _parse_reply, _strip_reasoning  # noqa: E402


EXPERIMENT = "coin_city_stable_relationship_claude_n250_v4"
RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
DEFAULT_OUTDIR = RUN / "responses" / "symbol_control_open_qwen3_4b_20260824"
ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")
EXPECTED_TASKS = 1_250
ORDER_SEED = 20260804
DECODE_SEED = 20260824


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_tasks(arm: str) -> tuple[Path, list[dict]]:
    path = DESIGN / f"tasks_{arm}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_TASKS:
        raise ValueError(f"{arm}: found {len(rows)} tasks, expected {EXPECTED_TASKS}")
    task_ids: set[str] = set()
    for row in rows:
        if set(row) != {"task_id", "prompt", "prompt_sha256"}:
            raise ValueError(f"{arm}: task file exposes fields beyond the model contract")
        task_id = row["task_id"]
        if task_id in task_ids:
            raise ValueError(f"{arm}: duplicate task ID {task_id}")
        task_ids.add(task_id)
        actual = hashlib.sha256(row["prompt"].encode()).hexdigest()
        if actual != row["prompt_sha256"]:
            raise ValueError(f"{arm}: prompt hash mismatch for {task_id}")
    random.Random(ORDER_SEED).shuffle(rows)
    return path, rows


def complete_output(path: Path, tasks: list[dict]) -> bool:
    if not path.exists():
        return False
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(tasks):
        return False
    expected = {row["task_id"]: row["prompt_sha256"] for row in tasks}
    observed = {row.get("task_id"): row.get("prompt_sha256") for row in rows}
    return observed == expected


def format_prompts(tokenizer, tasks: list[dict], chat_template_mode: str) -> list[str]:
    template_kwargs = {}
    if chat_template_mode == "qwen-no-thinking":
        template_kwargs["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for task in tasks
    ]


def write_arm(
    *,
    arm: str,
    tasks: list[dict],
    generated,
    model_label: str,
    output_path: Path,
) -> tuple[int, int]:
    if len(generated) != len(tasks):
        raise RuntimeError(f"{arm}: generated {len(generated)} outputs for {len(tasks)} tasks")
    now = datetime.now(timezone.utc).isoformat()
    records = []
    parsed_count = 0
    for task, result in zip(tasks, generated):
        raw = _strip_reasoning((result.outputs[0].text or "").strip())
        parsed = _parse_reply(raw)
        parsed_count += parsed["predicted_poll"] is not None
        records.append(
            {
                "task_id": task["task_id"],
                "arm": arm,
                "prompt_sha256": task["prompt_sha256"],
                "model": model_label,
                "predicted_poll": parsed["predicted_poll"],
                "rationale": parsed["rationale"],
                "raw": raw[:8000],
                "attempts": 1,
                "response_received": True,
                "error": None if parsed["predicted_poll"] is not None else "unparseable response",
                "created_at": now,
            }
        )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return len(records), parsed_count


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model-label", default="Qwen3-4B-Instruct-2507")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--chat-template-mode",
        choices=("qwen-no-thinking", "auto"),
        default="qwen-no-thinking",
    )
    parser.add_argument(
        "--model-license-family",
        default="Apache-2.0 Qwen open-weight release",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    loaded = {arm: read_tasks(arm) for arm in ARMS}
    task_hashes = {arm: file_sha256(path) for arm, (path, _) in loaded.items()}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "model_label": args.model_label,
                    "arms": {arm: len(tasks) for arm, (_, tasks) in loaded.items()},
                    "task_file_sha256": task_hashes,
                    "model_calls": sum(len(tasks) for _, tasks in loaded.values()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    from vllm import LLM, SamplingParams

    args.outdir.mkdir(parents=True, exist_ok=True)
    engine = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=DECODE_SEED,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = engine.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=DECODE_SEED,
    )

    counts: dict[str, dict[str, int]] = {}
    for arm, (_, tasks) in loaded.items():
        output_path = args.outdir / f"responses_{args.model_label}_{arm}.jsonl"
        if complete_output(output_path, tasks):
            rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
            counts[arm] = {
                "responses_received": len(rows),
                "successfully_parsed": sum(row.get("predicted_poll") is not None for row in rows),
            }
            print(f"{arm}: verified complete existing output; skipping", flush=True)
            continue
        prompts = format_prompts(tokenizer, tasks, args.chat_template_mode)
        longest = max(len(tokenizer.encode(prompt)) for prompt in prompts)
        if longest + args.max_tokens > args.max_model_len:
            raise ValueError(
                f"{arm}: longest prompt ({longest}) plus output ceiling "
                f"({args.max_tokens}) exceeds max model length ({args.max_model_len})"
            )
        print(f"{arm}: generating {len(prompts)} prompts; longest={longest} tokens", flush=True)
        generated = engine.generate(prompts, sampling)
        received, parsed = write_arm(
            arm=arm,
            tasks=tasks,
            generated=generated,
            model_label=args.model_label,
            output_path=output_path,
        )
        counts[arm] = {"responses_received": received, "successfully_parsed": parsed}
        print(f"{arm}: wrote {received} responses ({parsed} parsed)", flush=True)

    manifest = {
        "experiment": EXPERIMENT,
        "control": "configuration-matched open-model arbitrary-symbol replication",
        "status": "complete" if all(counts[a]["responses_received"] == EXPECTED_TASKS for a in ARMS) else "partial",
        "model": args.model,
        "model_label": args.model_label,
        "model_license_family": args.model_license_family,
        "arms": list(ARMS),
        "task_file_sha256": task_hashes,
        "order_seed": ORDER_SEED,
        "decode_seed": DECODE_SEED,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "chat_template": (
            "tokenizer.apply_chat_template(enable_thinking=False)"
            if args.chat_template_mode == "qwen-no-thinking"
            else "tokenizer.apply_chat_template(default model template)"
        ),
        "parser": "eval/run_frozen_task_shard.py::_parse_reply",
        "counts": counts,
        "software": {
            "python": platform.python_version(),
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.outdir / f"responses_{args.model_label}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
