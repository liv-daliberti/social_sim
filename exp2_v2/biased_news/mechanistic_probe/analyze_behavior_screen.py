#!/usr/bin/env python3
"""Score one or more sealed Qwen behavior screens against the frozen tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_probe import behavior_results, clean_json, regression_metrics
from extract_qwen_hidden_states import file_sha256, load_tasks, read_jsonl


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_TASKS_DIR = (
    ROOT
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "qwen3_8b_toy_v1"
)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260824


def baselines(tasks: list[dict]) -> dict:
    test = [
        row
        for row in tasks
        if row["split"] == "test"
        and row["c_cases"] == 0
        and row["arm"] == "abc_context"
    ]
    gold = np.asarray([row["gold_expected_poll"] for row in test], dtype=float)
    start = np.asarray([row["query_starting_poll"] for row in test], dtype=float)
    news = np.asarray([row["query_net_news"] for row in test], dtype=float)
    regime = np.asarray([row["regime_mean_slope"] for row in test], dtype=float)
    return {
        "no_change": regression_metrics(start, gold),
        "unit_slope": regression_metrics(start + news, gold),
        "regime_mean": regression_metrics(start + regime * news, gold),
    }


def paired_summary(differences: list[float], rng: np.random.Generator) -> dict:
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci95": [None, None]}
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def paired_behavior_contrasts(
    behavior_path: Path,
    task_by_id: dict[str, dict],
) -> dict:
    rows = read_jsonl(behavior_path)
    usable = {
        (int(row["episode"]), int(row["c_cases"]), row["arm"]): row
        for row in rows
        if row["implied_slope"] is not None
    }
    episodes = sorted({int(row["episode"]) for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def value(row: dict, metric: str) -> float:
        task = task_by_id[row["sample_id"]]
        if metric == "poll_absolute_error":
            return abs(float(row["predicted_poll"]) - float(task["gold_expected_poll"]))
        if metric == "slope_absolute_error":
            return abs(float(row["implied_slope"]) - float(task["target_slope"]))
        if metric == "regime_aligned_slope":
            sign = 1.0 if task["target_strong"] else -1.0
            return sign * float(row["implied_slope"])
        raise ValueError(metric)

    def compare(
        left: tuple[int, str],
        right: tuple[int, str],
    ) -> dict:
        common = [
            episode
            for episode in episodes
            if (episode, left[0], left[1]) in usable
            and (episode, right[0], right[1]) in usable
        ]
        result = {"difference_order": "left_minus_right"}
        for metric in (
            "poll_absolute_error",
            "slope_absolute_error",
            "regime_aligned_slope",
        ):
            differences = [
                value(usable[(episode, left[0], left[1])], metric)
                - value(usable[(episode, right[0], right[1])], metric)
                for episode in common
            ]
            result[metric] = paired_summary(differences, rng)
        return result

    result = {
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "by_prefix": {},
        "evidence_k4_minus_k0": {},
    }
    for c_cases in (0, 4):
        result["by_prefix"][f"k{c_cases}"] = {
            "context_minus_no_context": compare(
                (c_cases, "abc_context"),
                (c_cases, "abc_no_context"),
            ),
            "wrong_context_minus_correct_context": compare(
                (c_cases, "abc_wrong_context"),
                (c_cases, "abc_context"),
            ),
        }
    for arm in ("abc_context", "abc_no_context", "abc_wrong_context"):
        result["evidence_k4_minus_k0"][arm] = compare((4, arm), (0, arm))
    return result


def validate_screen(screen_dir: Path, expected_ids: set[str], tasks_sha: str) -> dict:
    manifest_path = screen_dir / "behavior_manifest.json"
    if not manifest_path.exists():
        manifest_path = screen_dir / "extraction_manifest.json"
    behavior_path = screen_dir / "behavior_test.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = read_jsonl(behavior_path)
    observed_ids = [row["sample_id"] for row in rows]
    if manifest["status"] != "complete":
        raise RuntimeError(f"incomplete screen: {screen_dir}")
    if manifest["tasks_sha256"] != tasks_sha:
        raise RuntimeError(f"task hash drift: {screen_dir}")
    if manifest["behavior_sha256"] != file_sha256(behavior_path):
        raise RuntimeError(f"behavior hash drift: {screen_dir}")
    if len(rows) != 96 or set(observed_ids) != expected_ids:
        raise RuntimeError(f"sealed prompt coverage drift: {screen_dir}")
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError(f"duplicate behavior records: {screen_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks, task_manifest = load_tasks(args.tasks_dir)
    task_by_id = {row["sample_id"]: row for row in tasks}
    test_ids = {row["sample_id"] for row in tasks if row["split"] == "test"}
    result = {
        "study": "qwen_behavioral_size_screen_v1",
        "status": "complete",
        "tasks_sha256": task_manifest["tasks_sha256"],
        "sealed_test_episodes": 16,
        "sealed_test_prompts_per_model": 96,
        "baselines": baselines(tasks),
        "models": {},
    }
    for screen_dir in args.screen:
        manifest = validate_screen(
            screen_dir, test_ids, task_manifest["tasks_sha256"]
        )
        label = manifest["model_label"]
        if label in result["models"]:
            raise RuntimeError(f"duplicate model label: {label}")
        behavior_path = screen_dir / "behavior_test.jsonl"
        result["models"][label] = {
            "manifest": manifest,
            "behavior": behavior_results(behavior_path, task_by_id),
            "paired_contrasts": paired_behavior_contrasts(
                behavior_path,
                task_by_id,
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(result), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
