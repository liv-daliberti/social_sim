#!/usr/bin/env python3
"""Fail-closed checks for the 100%-aligned context-only rerun."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from build_coin_city_context100_rerun import (
    ARM,
    CUMULATIVE_CALLS,
    DESIGN,
    EXPERIMENT,
    NEW_CALLS,
    PARENT,
    RUN,
    aligned_episode,
    validate_context100_prompt,
)
from engine.coin_city_llm_pilot import (
    EPISODES,
    HARD_CALL_CAP,
    MODELS,
    ROUNDS,
    TARGET_CONTEXT_NONE,
    make_prompt,
    prompt_sha256,
    target_background,
    task_id,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    parent_episodes = _read_jsonl(PARENT / "design" / "episodes.jsonl")
    aligned_episodes = _read_jsonl(DESIGN / "episodes_aligned_context.jsonl")
    tasks = _read_jsonl(DESIGN / f"tasks_{ARM}.jsonl")
    keys = _read_jsonl(DESIGN / "answer_key.jsonl")
    old_no_context = {
        row["task_id"]: row
        for row in _read_jsonl(PARENT / "design" / "tasks_abc_no_context.jsonl")
    }
    task_by_id = {row["task_id"]: row for row in tasks}
    aligned_by_id = {row["episode"]: row for row in aligned_episodes}
    expected_ids = {
        task_id(episode, k) for episode in range(EPISODES) for k in ROUNDS
    }
    checks = {
        "experiment": manifest.get("experiment") == EXPERIMENT,
        "new_calls_exactly_600": NEW_CALLS == manifest.get("new_calls") == 600,
        "cumulative_calls_2400_below_cap": (
            CUMULATIVE_CALLS == manifest.get("cumulative_calls") == 2400
            and CUMULATIVE_CALLS <= HARD_CALL_CAP == 5000
        ),
        "one_attempt": manifest.get("max_attempts_per_task") == 1,
        "three_models": len(MODELS) == 3,
        "episode_count": len(aligned_episodes) == EPISODES,
        "task_count_and_ids": (
            len(tasks) == EPISODES * len(ROUNDS)
            and len(task_by_id) == len(expected_ids)
            and set(task_by_id) == expected_ids
        ),
        "answer_key_count": len(keys) == len(expected_ids),
    }
    all_aligned = True
    parent_data_unchanged = True
    prompt_hashes = True
    prompt_contract = True
    exact_prompt = True
    changed = 0
    for parent, aligned in zip(parent_episodes, aligned_episodes):
        expected_episode = aligned_episode(parent)
        if aligned != expected_episode or aligned["cue_high"] != aligned["target_high"]:
            all_aligned = False
        if parent["cue_high"] != aligned["cue_high"]:
            changed += 1
        for k in ROUNDS:
            identifier = task_id(aligned["episode"], k)
            task = task_by_id[identifier]
            if prompt_sha256(task["prompt"]) != task["prompt_sha256"]:
                prompt_hashes = False
            try:
                validate_context100_prompt(task["prompt"], ARM)
            except ValueError:
                prompt_contract = False
            if task["prompt"] != make_prompt(aligned, k, "abc_context"):
                exact_prompt = False
            context_sentence = target_background(aligned, "abc_context")
            if (
                task["prompt"].replace(context_sentence, TARGET_CONTEXT_NONE)
                != old_no_context[identifier]["prompt"]
            ):
                parent_data_unchanged = False
    checks.update(
        {
            "all_40_contexts_aligned": all_aligned,
            "exactly_8_misleading_prompts_corrected": changed == 8,
            "numeric_and_reference_data_unchanged": parent_data_unchanged,
            "prompt_hashes": prompt_hashes,
            "prompt_contract": prompt_contract,
            "exact_prompt_regeneration": exact_prompt,
        }
    )
    old_responses = []
    for path in (PARENT / "responses").glob("responses_*.jsonl"):
        old_responses.extend(_read_jsonl(path))
    checks["parent_1800_responses_complete"] = (
        len(old_responses) == 1800
        and all(row.get("predicted_poll") is not None for row in old_responses)
    )
    checks["frozen_file_hashes"] = (
        _sha256(DESIGN / f"tasks_{ARM}.jsonl") == manifest["task_sha256"]
        and _sha256(DESIGN / "answer_key.jsonl") == manifest["answer_key_sha256"]
        and _sha256(DESIGN / "episodes_aligned_context.jsonl")
        == manifest["episodes_sha256"]
        and _sha256(PARENT / "design" / "manifest.json")
        == manifest["parent_manifest_sha256"]
    )
    source_paths = {
        "builder": HERE / "build_coin_city_context100_rerun.py",
        "validator": HERE / "validate_coin_city_context100_rerun.py",
        "runner": HERE / "run_coin_city_context100_rerun.py",
        "launcher": HERE / "run_coin_city_context100_local.sh",
        "renderer": ROOT / "analysis" / "render_coin_city_context100.py",
    }
    checks["frozen_source_hashes"] = all(
        _sha256(path) == manifest["source_sha256"].get(name)
        for name, path in source_paths.items()
    )
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "prior_calls": 1800,
        "new_calls": NEW_CALLS,
        "cumulative_calls": CUMULATIVE_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "prompts_corrected": changed,
    }
    (DESIGN / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("aligned-context preflight failed: " + ", ".join(failed))
    print(f"Aligned-context preflight passed: {NEW_CALLS} new; {CUMULATIVE_CALLS}/5000 cumulative")


if __name__ == "__main__":
    main()
