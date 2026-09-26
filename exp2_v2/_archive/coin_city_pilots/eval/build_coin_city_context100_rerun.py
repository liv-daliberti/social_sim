#!/usr/bin/env python3
"""Build a context-only rerun whose structural clue is always aligned."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_llm_pilot import (
    EPISODES,
    HARD_CALL_CAP,
    MODELS,
    ROUNDS,
    TARGET_CONTEXT_HIGH,
    TARGET_CONTEXT_LOW,
    baseline_estimates,
    make_prompt,
    prompt_sections,
    prompt_sha256,
    task_id,
    validate_arm_prompt,
)


EXPERIMENT = "coin_city_context100_rerun_v1"
ARM = "abc_context_100pct"
PRIOR_CALLS = 1_800
NEW_CALLS = EPISODES * len(ROUNDS) * len(MODELS)
CUMULATIVE_CALLS = PRIOR_CALLS + NEW_CALLS
PARENT = ROOT / "data" / "coin_city_llm_pilot_v1"
RUN = PARENT / "context100_rerun"
DESIGN = RUN / "design"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_episode(episode: dict) -> dict:
    result = dict(episode)
    result["cue_high"] = bool(result["target_high"])
    result["cue_correct"] = True
    return result


def validate_context100_prompt(prompt_text: str, arm: str) -> None:
    if arm != ARM:
        raise ValueError(f"unexpected arm: {arm}")
    validate_arm_prompt(prompt_text, "abc_context")


def main() -> None:
    if NEW_CALLS != 600 or CUMULATIVE_CALLS != 2400:
        raise SystemExit("call-count invariant failed")
    if CUMULATIVE_CALLS > HARD_CALL_CAP:
        raise SystemExit("cumulative test calls exceed the 5,000-call cap")
    if (RUN / "responses").exists() and any((RUN / "responses").glob("*.jsonl")):
        raise SystemExit("refusing to rebuild after aligned-context responses exist")
    episodes = _read_jsonl(PARENT / "design" / "episodes.jsonl")
    aligned = [aligned_episode(episode) for episode in episodes]
    DESIGN.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DESIGN / "episodes_aligned_context.jsonl", aligned)

    tasks = []
    keys = []
    for episode in aligned:
        for k in ROUNDS:
            identifier = task_id(episode["episode"], k)
            prompt_text = make_prompt(episode, k, "abc_context")
            validate_context100_prompt(prompt_text, ARM)
            tasks.append(
                {
                    "task_id": identifier,
                    "prompt": prompt_text,
                    "prompt_sha256": prompt_sha256(prompt_text),
                }
            )
            estimates = baseline_estimates(episode, k)
            keys.append(
                {
                    "task_id": identifier,
                    "episode": episode["episode"],
                    "k": k,
                    "gold_expected_poll": episode["target_truth"],
                    "target_high": episode["target_high"],
                    "cue_high": episode["cue_high"],
                    "cue_correct": True,
                    "baselines": {
                        "city_c_evidence_only": estimates["baseline"],
                        "abc_no_context": estimates["abc_no_context"],
                        "abc_aligned_context": estimates["abc_context"],
                    },
                }
            )
    task_path = DESIGN / f"tasks_{ARM}.jsonl"
    _write_jsonl(task_path, tasks)
    _write_jsonl(DESIGN / "answer_key.jsonl", keys)
    example = aligned[0]
    (DESIGN / "example_prompt.txt").write_text(
        make_prompt(example, 2, "abc_context") + "\n"
    )
    (DESIGN / "example_prompt_sections.json").write_text(
        json.dumps(
            {ARM: prompt_sections(example, 2, "abc_context")},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    source_paths = {
        "builder": HERE / "build_coin_city_context100_rerun.py",
        "validator": HERE / "validate_coin_city_context100_rerun.py",
        "runner": HERE / "run_coin_city_context100_rerun.py",
        "launcher": HERE / "run_coin_city_context100_local.sh",
        "renderer": ROOT / "analysis" / "render_coin_city_context100.py",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing aligned-context source: " + ", ".join(missing))
    manifest = {
        "experiment": EXPERIMENT,
        "status": "frozen_no_new_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": ARM,
        "episodes": EPISODES,
        "rounds": list(ROUNDS),
        "models": list(MODELS),
        "tasks_per_model": EPISODES * len(ROUNDS),
        "prior_calls": PRIOR_CALLS,
        "new_calls": NEW_CALLS,
        "cumulative_calls": CUMULATIVE_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "max_attempts_per_task": 1,
        "task_sha256": _sha256(task_path),
        "answer_key_sha256": _sha256(DESIGN / "answer_key.jsonl"),
        "episodes_sha256": _sha256(DESIGN / "episodes_aligned_context.jsonl"),
        "parent_manifest_sha256": _sha256(PARENT / "design" / "manifest.json"),
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "context_sentences": {
            "high_response_type": TARGET_CONTEXT_HIGH,
            "low_response_type": TARGET_CONTEXT_LOW,
        },
    }
    (DESIGN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Built {NEW_CALLS}-call aligned-context rerun at {RUN}")


if __name__ == "__main__":
    main()
