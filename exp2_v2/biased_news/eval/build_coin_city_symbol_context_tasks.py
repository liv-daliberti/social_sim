#!/usr/bin/env python3
"""Build the arbitrary-symbol control for the frozen Coin City design."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    task_id,
)
from engine.coin_city_symbol_context_arm import (  # noqa: E402
    ARM,
    CONTEXT_ALIGNMENT,
    SYMBOLS,
    city_symbol,
    make_prompt,
    strong_symbol,
    validate_arm_prompt,
    weak_symbol,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    episode_rows = _read_jsonl(DESIGN / "episodes.jsonl")
    episodes = {int(row["episode"]): row for row in episode_rows}
    semantic_rows = _read_jsonl(DESIGN / "tasks_abc_context.jsonl")
    semantic = {row["task_id"]: row for row in semantic_rows}
    if len(episodes) != EPISODES:
        raise SystemExit(f"expected {EPISODES} episodes, found {len(episodes)}")
    if len(semantic) != EPISODES * len(C_CASE_LEVELS):
        raise SystemExit("truthful semantic task file is incomplete")

    records: list[dict] = []
    target_cross = Counter()
    mapping_counts = Counter()
    for index in range(EPISODES):
        episode = episodes[index]
        if strong_symbol(episode) == weak_symbol(episode):
            raise SystemExit(f"symbol mapping collapsed in episode {index}")
        mapping_counts[f"higher_response={strong_symbol(episode)}"] += 1
        target_regime = "higher_response" if episode["cue_strong"] else "lower_response"
        target_cross[f"{target_regime}:{city_symbol(episode, 'C')}"] += 1

        for c_cases in C_CASE_LEVELS:
            identifier = task_id(index, c_cases)
            prompt = make_prompt(episode, c_cases)
            validate_arm_prompt(prompt, ARM)
            if NUMBER.findall(prompt) != NUMBER.findall(semantic[identifier]["prompt"]):
                raise SystemExit(f"numeric content changed: {identifier}")
            records.append(
                {
                    "task_id": identifier,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )

    expected_cross = {
        "higher_response:KIV",
        "higher_response:ZOR",
        "lower_response:KIV",
        "lower_response:ZOR",
    }
    if set(target_cross) != expected_cross or max(target_cross.values()) - min(
        target_cross.values()
    ) > 1:
        raise SystemExit(f"target regime x symbol is imbalanced: {dict(target_cross)}")
    if mapping_counts != Counter({"higher_response=KIV": 125, "higher_response=ZOR": 125}):
        raise SystemExit(f"episode-local mappings are imbalanced: {dict(mapping_counts)}")

    out_path = DESIGN / f"tasks_{ARM}.jsonl"
    out_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    example_path = DESIGN / f"example_prompt_{ARM}.txt"
    example_path.write_text(records[0]["prompt"] + "\n")

    source_files = [
        ROOT / "engine" / "coin_city_symbol_context_arm.py",
        HERE / "build_coin_city_symbol_context_tasks.py",
        HERE / "run_coin_city_symbol_context.py",
        ROOT / "analysis" / "analyze_coin_city_symbol_context.py",
    ]
    manifest = {
        "arm": ARM,
        "experiment": EXPERIMENT,
        "context_alignment": CONTEXT_ALIGNMENT,
        "design": (
            "KIV/ZOR replace every national/local cue. Their higher/lower-response "
            "mapping reverses by episode parity and must be induced from City A/B "
            "examples before applying the matching label to City C."
        ),
        "symbols": list(SYMBOLS),
        "episodes": EPISODES,
        "city_c_case_levels": list(C_CASE_LEVELS),
        "tasks": len(records),
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "target_regime_by_symbol": dict(sorted(target_cross.items())),
        "numeric_tokens_match_truthful_semantic_arm": True,
        "task_file_sha256": _sha256(out_path),
        "frozen_episode_file_sha256": _sha256(DESIGN / "episodes.jsonl"),
        "frozen_truthful_task_file_sha256": _sha256(
            DESIGN / "tasks_abc_context.jsonl"
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_files
        },
    }
    manifest_path = DESIGN / f"tasks_{ARM}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
