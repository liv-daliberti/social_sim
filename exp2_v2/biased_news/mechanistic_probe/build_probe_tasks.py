#!/usr/bin/env python3
"""Build a deterministic, episode-disjoint toy task set for hidden-state probing."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    EXPERIMENT,
    STRONG_SLOPE_MEAN,
    WEAK_SLOPE_MEAN,
    fit_change_slope,
    task_id,
)


DESIGN = ROOT / "data" / EXPERIMENT / "design"
DEFAULT_OUTDIR = ROOT / "data" / EXPERIMENT / "mechanistic_probe" / "qwen3_8b_toy_v1"
ARMS = ("abc_context", "abc_no_context", "abc_wrong_context")
DEPTHS = (0, 4)
CELL_ORDER = (("A", False), ("A", True), ("B", False), ("B", True))
SELECTION_SEED = 20260824
CV_FOLDS = 4
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prompt_maps() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for arm in ARMS:
        path = DESIGN / f"tasks_{arm}.jsonl"
        rows = read_jsonl(path)
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if set(row) != {"task_id", "prompt", "prompt_sha256"}:
                raise ValueError(f"{path}: unexpected task schema")
            if text_sha256(row["prompt"]) != row["prompt_sha256"]:
                raise ValueError(f"{path}: bad prompt hash for {row['task_id']}")
            if row["task_id"] in by_id:
                raise ValueError(f"{path}: duplicate task {row['task_id']}")
            by_id[row["task_id"]] = row
        maps[arm] = by_id
        hashes[arm] = file_sha256(path)
    if len({len(rows) for rows in maps.values()}) != 1:
        raise ValueError("prompt arms have different task counts")
    return maps, hashes


def select_episodes(
    episodes: list[dict[str, Any]],
    *,
    episodes_per_cell: int,
    test_per_cell: int,
) -> tuple[list[dict[str, Any]], dict[int, tuple[str, int | None]]]:
    if not 0 < test_per_cell < episodes_per_cell:
        raise ValueError("test_per_cell must be between zero and episodes_per_cell")
    dev_per_cell = episodes_per_cell - test_per_cell
    if dev_per_cell % CV_FOLDS:
        raise ValueError("development episodes per cell must divide evenly across CV folds")

    by_cell: dict[tuple[str, bool], list[dict[str, Any]]] = {
        cell: [] for cell in CELL_ORDER
    }
    for episode in episodes:
        cell = (episode["strong_reference_city"], bool(episode["target_strong"]))
        if cell not in by_cell:
            raise ValueError(f"unexpected factorial cell {cell}")
        by_cell[cell].append(episode)

    selected: list[dict[str, Any]] = []
    assignment: dict[int, tuple[str, int | None]] = {}
    for cell_index, cell in enumerate(CELL_ORDER):
        candidates = sorted(by_cell[cell], key=lambda row: int(row["episode"]))
        random.Random(SELECTION_SEED + cell_index).shuffle(candidates)
        chosen = candidates[:episodes_per_cell]
        if len(chosen) != episodes_per_cell:
            raise ValueError(f"cell {cell} has only {len(chosen)} episodes")

        test_rows = chosen[:test_per_cell]
        dev_rows = chosen[test_per_cell:]
        random.Random(SELECTION_SEED + 100 + cell_index).shuffle(dev_rows)
        for index, episode in enumerate(dev_rows):
            assignment[int(episode["episode"])] = ("dev", index % CV_FOLDS)
        for episode in test_rows:
            assignment[int(episode["episode"])] = ("test", None)
        selected.extend(chosen)

    selected.sort(key=lambda row: int(row["episode"]))
    return selected, assignment


def visible_target_ols(episode: dict[str, Any], c_cases: int) -> float | None:
    rows = episode["target"][:c_cases]
    return None if not rows else float(fit_change_slope(rows))


def make_records(
    selected: list[dict[str, Any]],
    assignment: dict[int, tuple[str, int | None]],
    prompts: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for episode in selected:
        episode_id = int(episode["episode"])
        split, cv_fold = assignment[episode_id]
        target_strong = bool(episode["target_strong"])
        regime_mean = STRONG_SLOPE_MEAN if target_strong else WEAK_SLOPE_MEAN
        ref_slopes = {
            "A": float(fit_change_slope(episode["reference_a"])),
            "B": float(fit_change_slope(episode["reference_b"])),
        }
        for c_cases in DEPTHS:
            frozen_id = task_id(episode_id, c_cases)
            numeric_signatures = {
                arm: NUMBER.findall(prompts[arm][frozen_id]["prompt"]) for arm in ARMS
            }
            if len({tuple(values) for values in numeric_signatures.values()}) != 1:
                raise ValueError(
                    f"matched arms change numeric content for episode={episode_id}, k={c_cases}"
                )
            for arm in ARMS:
                source = prompts[arm][frozen_id]
                cue_strong: bool | None
                if arm == "abc_context":
                    cue_strong = target_strong
                elif arm == "abc_wrong_context":
                    cue_strong = not target_strong
                else:
                    cue_strong = None
                records.append(
                    {
                        "sample_id": f"{frozen_id}__{arm}",
                        "task_id": frozen_id,
                        "episode": episode_id,
                        "split": split,
                        "cv_fold": cv_fold,
                        "arm": arm,
                        "c_cases": c_cases,
                        "prompt": source["prompt"],
                        "prompt_sha256": source["prompt_sha256"],
                        "strong_reference_city": episode["strong_reference_city"],
                        "target_strong": target_strong,
                        "cue_strong": cue_strong,
                        "target_slope": float(episode["target_slope"]),
                        "regime_mean_slope": float(regime_mean),
                        "residual_slope": float(episode["target_slope"] - regime_mean),
                        "visible_target_ols": visible_target_ols(episode, c_cases),
                        "reference_a_ols": ref_slopes["A"],
                        "reference_b_ols": ref_slopes["B"],
                        "query_starting_poll": float(episode["query_starting_poll"]),
                        "query_net_news": int(episode["query_net_news"]),
                        "gold_expected_poll": float(episode["gold_expected_poll"]),
                    }
                )
    records.sort(
        key=lambda row: (
            0 if row["split"] == "dev" else 1,
            row["episode"],
            row["c_cases"],
            ARMS.index(row["arm"]),
        )
    )
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("duplicate sample IDs")
    return records


def write_exact(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} exists with different content; pass --force")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--episodes-per-cell", type=int, default=16)
    parser.add_argument("--test-per-cell", type=int, default=4)
    parser.add_argument("--study", default="qwen3_8b_toy_v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    episodes_path = DESIGN / "episodes.jsonl"
    episodes = read_jsonl(episodes_path)
    prompts, prompt_hashes = prompt_maps()
    selected, assignment = select_episodes(
        episodes,
        episodes_per_cell=args.episodes_per_cell,
        test_per_cell=args.test_per_cell,
    )
    records = make_records(selected, assignment, prompts)
    tasks_content = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    tasks_path = args.outdir / "tasks.jsonl"
    write_exact(tasks_path, tasks_content, force=args.force)

    cell_counts: dict[str, dict[str, int]] = {}
    for city, target_strong in CELL_ORDER:
        label = f"{city}_{'strong' if target_strong else 'weak'}"
        cell_counts[label] = {"dev": 0, "test": 0}
    for episode in selected:
        label = (
            f"{episode['strong_reference_city']}_"
            f"{'strong' if episode['target_strong'] else 'weak'}"
        )
        split, _ = assignment[int(episode["episode"])]
        cell_counts[label][split] += 1

    manifest = {
        "study": args.study,
        "status": "tasks_built",
        "experiment": EXPERIMENT,
        "source_episodes": str(episodes_path.relative_to(ROOT)),
        "source_episodes_sha256": file_sha256(episodes_path),
        "source_prompt_sha256": prompt_hashes,
        "builder_sha256": file_sha256(Path(__file__)),
        "selection_seed": SELECTION_SEED,
        "episodes_per_cell": args.episodes_per_cell,
        "test_per_cell": args.test_per_cell,
        "cv_folds": CV_FOLDS,
        "arms": list(ARMS),
        "evidence_depths": list(DEPTHS),
        "cell_counts": cell_counts,
        "selected_episode_ids": [int(row["episode"]) for row in selected],
        "record_count": len(records),
        "tasks_sha256": text_sha256(tasks_content),
    }
    manifest_content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    write_exact(args.outdir / "task_manifest.json", manifest_content, force=args.force)

    print(
        json.dumps(
            {
                "outdir": str(args.outdir),
                "records": len(records),
                "episodes": len(selected),
                "cell_counts": cell_counts,
                "tasks_sha256": manifest["tasks_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

