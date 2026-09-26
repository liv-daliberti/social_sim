#!/usr/bin/env python3
"""Freeze paired disclosed/undisclosed tasks for the Experiment 3 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
DEFAULT_OUTDIR = HERE / "data"
DISCLOSURES = ("disclosed", "undisclosed")
K_VALUES = (3, 6, 9)
EXPECTED_WORLDS = (
    "mediated_feedback_linear",
    "mediated_saturating",
    "persistent_threshold",
    "feedback_saturating_extrap",
)
STUDY = "exp3_mechanism_probe_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_files(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(FAMILY)): file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def load_source(disclosure: str) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    path = FAMILY / "data" / f"{disclosure}_causal_family" / "heldout"
    dataset = load_from_disk(str(path))["train"]
    rows = []
    for item in dataset:
        reference = json.loads(item["reference"])
        if reference["disclosure"] != disclosure:
            raise ValueError(f"bad disclosure for {reference['task_id']}")
        rows.append({"prompt": item["input"], "reference": reference})
    return rows


def episode_assignments(rows: list[dict[str, Any]]) -> dict[tuple[str, int], tuple[str, int | None]]:
    seeds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        ref = row["reference"]
        seeds[ref["world"]].add(int(ref["seed"]))
    if tuple(sorted(seeds)) != tuple(sorted(EXPECTED_WORLDS)):
        raise ValueError(f"unexpected held-out worlds: {sorted(seeds)}")

    assignments = {}
    for world in EXPECTED_WORLDS:
        ordered = sorted(seeds[world])
        if len(ordered) != 60:
            raise ValueError(f"{world} has {len(ordered)} episodes, expected 60")
        for rank, seed in enumerate(ordered):
            if rank % 4 == 0:
                assignments[(world, seed)] = ("test", None)
            else:
                assignments[(world, seed)] = ("dev", (rank // 4) % 5)
    return assignments


def make_records(source: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    disclosed_by_id = {
        row["reference"]["task_id"]: row for row in source["disclosed"]
    }
    undisclosed_by_id = {
        row["reference"]["task_id"]: row for row in source["undisclosed"]
    }
    if set(disclosed_by_id) != set(undisclosed_by_id):
        raise ValueError("disclosed and undisclosed task IDs differ")
    assignments = episode_assignments(source["disclosed"])

    records = []
    for task_id in sorted(disclosed_by_id):
        left = disclosed_by_id[task_id]
        right = undisclosed_by_id[task_id]
        left_ref = left["reference"]
        right_ref = right["reference"]
        paired_fields = (
            "world",
            "block",
            "seed",
            "k",
            "g",
            "numeric_hash",
            "truth_response",
            "prior_response",
            "truth_targets",
        )
        for field in paired_fields:
            if left_ref[field] != right_ref[field]:
                raise ValueError(f"pair mismatch for {task_id}: {field}")
        world = left_ref["world"]
        seed = int(left_ref["seed"])
        split, fold = assignments[(world, seed)]
        episode_id = f"{world}:{seed}"
        for disclosure, item in (("disclosed", left), ("undisclosed", right)):
            ref = item["reference"]
            prompt = item["prompt"]
            records.append(
                {
                    "sample_id": f"{task_id}:{disclosure}",
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "world": world,
                    "block": ref["block"],
                    "seed": seed,
                    "k": int(ref["k"]),
                    "disclosure": disclosure,
                    "split": split,
                    "dev_fold": fold,
                    "target_gain": float(ref["g"]),
                    "truth_response": [float(value) for value in ref["truth_response"]],
                    "prior_response": [float(value) for value in ref["prior_response"]],
                    "numeric_hash": ref["numeric_hash"],
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    records.sort(key=lambda row: row["sample_id"])
    return records


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1_440:
        raise ValueError(f"expected 1,440 records, observed {len(records)}")
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("duplicate sample IDs")
    pairs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pairs[(row["task_id"], row["k"])].append(row)
    for key, pair in pairs.items():
        if len(pair) != 2 or {row["disclosure"] for row in pair} != set(DISCLOSURES):
            raise ValueError(f"bad disclosure pair: {key}")
        if len({row["numeric_hash"] for row in pair}) != 1:
            raise ValueError(f"numeric mismatch: {key}")

    split_counts = Counter(
        (row["world"], row["split"], row["disclosure"], row["k"])
        for row in records
    )
    for world in EXPECTED_WORLDS:
        for disclosure in DISCLOSURES:
            for k in K_VALUES:
                if split_counts[(world, "dev", disclosure, k)] != 45:
                    raise ValueError("development split is not balanced")
                if split_counts[(world, "test", disclosure, k)] != 15:
                    raise ValueError("test split is not balanced")
    fold_counts = Counter(
        (row["world"], row["disclosure"], row["k"], row["dev_fold"])
        for row in records
        if row["split"] == "dev"
    )
    for world in EXPECTED_WORLDS:
        for disclosure in DISCLOSURES:
            for k in K_VALUES:
                for fold in range(5):
                    if fold_counts[(world, disclosure, k, fold)] != 9:
                        raise ValueError("development folds are not balanced")
    return {
        "record_count": len(records),
        "episode_count": len({row["episode_id"] for row in records}),
        "development_episodes": len(
            {row["episode_id"] for row in records if row["split"] == "dev"}
        ),
        "test_episodes": len(
            {row["episode_id"] for row in records if row["split"] == "test"}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    source = {disclosure: load_source(disclosure) for disclosure in DISCLOSURES}
    records = make_records(source)
    summary = validate_records(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.outdir / "tasks.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    temporary = tasks_path.with_suffix(".jsonl.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(tasks_path)

    source_hashes = {}
    for disclosure in DISCLOSURES:
        path = FAMILY / "data" / f"{disclosure}_causal_family" / "heldout"
        source_hashes.update(dataset_files(path))
    manifest = {
        "study": STUDY,
        "status": "tasks_frozen",
        "source_protocol": "c3_mechanism_disclosure_v3",
        "disclosures": list(DISCLOSURES),
        "k_values": list(K_VALUES),
        "worlds": list(EXPECTED_WORLDS),
        "split_rule": (
            "within each world, sorted episode rank modulo 4 equals 0 is test; "
            "remaining ranks are development with fold=(rank//4) modulo 5"
        ),
        "source_file_sha256": source_hashes,
        "tasks_sha256": file_sha256(tasks_path),
        **summary,
    }
    manifest_path = args.outdir / "task_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
