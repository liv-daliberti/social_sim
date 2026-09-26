#!/usr/bin/env python3
"""Read-only frozen-design audit for the stable-response LLM run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_n100_fair import (
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    FUTURE_CALLS,
    HARD_CALL_CAP,
    PROMPT_ARMS,
    make_prompt,
    prompt_sha256,
    task_id,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"


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
    validation = json.loads((DESIGN / "validation.json").read_text())
    episodes = _read_jsonl(DESIGN / "episodes.jsonl")
    expected_ids = {
        task_id(episode, c_cases)
        for episode in range(EPISODES)
        for c_cases in C_CASE_LEVELS
    }
    checks = {
        "experiment": manifest.get("experiment") == EXPERIMENT,
        "frozen_builder_validation": validation.get("status") == "passed",
        "call_cap": FUTURE_CALLS == manifest.get("planned_calls") == 4_500 <= HARD_CALL_CAP,
        "city_c_levels": tuple(manifest.get("city_c_case_levels", ())) == C_CASE_LEVELS,
        "episode_count": len(episodes) == EPISODES,
    }

    task_contracts = True
    shared_query_everywhere = True
    exact_regeneration = True
    task_hashes = True
    for arm in PROMPT_ARMS:
        path = DESIGN / f"tasks_{arm}.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != len(expected_ids) or {row["task_id"] for row in rows} != expected_ids:
            task_contracts = False
        by_id = {row["task_id"]: row for row in rows}
        for episode in episodes:
            for c_cases in C_CASE_LEVELS:
                identifier = task_id(episode["episode"], c_cases)
                row = by_id.get(identifier)
                if row is None:
                    exact_regeneration = False
                    continue
                if set(row) != {"task_id", "prompt", "prompt_sha256"}:
                    task_contracts = False
                if prompt_sha256(row["prompt"]) != row["prompt_sha256"]:
                    task_contracts = False
                try:
                    validate_arm_prompt(row["prompt"], arm)
                except ValueError:
                    task_contracts = False
                if row["prompt"] != make_prompt(episode, c_cases, arm):
                    exact_regeneration = False
                if row["prompt"].count("A new City C case begins") != 1:
                    shared_query_everywhere = False
        if _sha256(path) != manifest["task_sha256"].get(arm):
            task_hashes = False
    checks.update(
        {
            "task_counts_ids_and_prompt_contracts": task_contracts,
            "exact_prompt_regeneration": exact_regeneration,
            "shared_new_city_c_query_in_every_prompt": shared_query_everywhere,
            "task_file_hashes": task_hashes,
            "scoring_key_count": len(_read_jsonl(DESIGN / "scoring_key.jsonl")) == len(expected_ids),
        }
    )

    source_paths = {
        "engine": ROOT / "engine" / "coin_city_stable_relationship_n100_fair.py",
        "builder_validator": HERE / "build_coin_city_stable_relationship_n100_fair.py",
        "renderer": ROOT / "analysis" / "render_coin_city_stable_relationship_n100_fair.py",
        "runner": HERE / "run_coin_city_stable_relationship_n100_fair.py",
        "frozen_validator": HERE / "validate_coin_city_stable_relationship_n100_fair.py",
        "local_launcher": HERE / "run_coin_city_stable_relationship_n100_fair_local.sh",
    }
    checks["frozen_source_hashes"] = all(
        path.exists() and _sha256(path) == manifest.get("source_sha256", {}).get(name)
        for name, path in source_paths.items()
    )
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "experiment": EXPERIMENT,
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed": failed,
        "frozen_tasks": len(expected_ids) * len(PROMPT_ARMS),
        "planned_calls": FUTURE_CALLS,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("frozen stable-response audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
