#!/usr/bin/env python3
"""Freeze balanced arbitrary-symbol tasks for the Qwen3-14B probe.

The analyzer retains its legacy arm identifiers for compatibility, but every
prompt in this task set uses episode-randomized KIV/ZOR labels rather than the
semantic national/local cues used by the original mechanistic probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    EXPERIMENT,
    SHARED_PROMPT_INTRO,
    STRONG_SLOPE_MEAN,
    WEAK_SLOPE_MEAN,
    _table,
    fit_change_slope,
    shared_prompt_closing,
    task_id,
)
from engine.coin_city_symbol_context_arm import (  # noqa: E402
    SYMBOL_INSTRUCTIONS,
    SYMBOLS,
    make_prompt as make_symbol_prompt,
    strong_symbol,
    symbol_background,
    validate_arm_prompt as validate_symbol_prompt,
    weak_symbol,
)


DESIGN = ROOT / "data" / EXPERIMENT / "design"
DEFAULT_OUTDIR = (
    ROOT / "data" / EXPERIMENT / "mechanistic_probe" / "qwen3_14b_symbol_probe_v1"
)
ARMS = ("abc_context", "abc_no_context", "abc_wrong_context")
ARM_SEMANTICS = {
    "abc_context": "correct episode-randomized arbitrary City C label",
    "abc_no_context": "City A/B arbitrary labels present; City C label absent",
    "abc_wrong_context": "opposite episode-randomized arbitrary City C label",
}
DEPTHS = (0, 4)
SELECTION_SEED = 20260825
CV_FOLDS = 4
EPISODES_PER_CELL = 10
TEST_PER_CELL = 2
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
FORBIDDEN_SEMANTICS = (
    "national news",
    "local news",
    "larger immediate",
    "smaller immediate",
    "strong regime",
    "weak regime",
    "strongly responsive",
    "weakly responsive",
    "regression",
    "coefficient",
    "weighted average",
    "calculate a slope",
    "closer to city",
)


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


def factorial_cell(episode: dict[str, Any]) -> tuple[str, bool, str]:
    return (
        str(episode["strong_reference_city"]),
        bool(episode["target_strong"]),
        strong_symbol(episode),
    )


CELL_ORDER = tuple(
    (city, target_strong, mapping)
    for city in ("A", "B")
    for target_strong in (False, True)
    for mapping in SYMBOLS
)


def select_episodes(
    episodes: list[dict[str, Any]],
    *,
    episodes_per_cell: int = EPISODES_PER_CELL,
    test_per_cell: int = TEST_PER_CELL,
) -> tuple[list[dict[str, Any]], dict[int, tuple[str, int | None]]]:
    if not 0 < test_per_cell < episodes_per_cell:
        raise ValueError("test_per_cell must be between zero and episodes_per_cell")
    dev_per_cell = episodes_per_cell - test_per_cell
    if dev_per_cell % CV_FOLDS:
        raise ValueError("development episodes must divide evenly across folds")
    by_cell: dict[tuple[str, bool, str], list[dict[str, Any]]] = {
        cell: [] for cell in CELL_ORDER
    }
    for episode in episodes:
        if bool(episode["cue_strong"]) != bool(episode["target_strong"]):
            raise ValueError(f"episode {episode['episode']} has a non-truthful source cue")
        cell = factorial_cell(episode)
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


def make_no_target_label_prompt(episode: dict[str, Any], c_cases: int) -> str:
    sections = [
        "CITY A\n"
        + symbol_background(episode, "A")
        + "\n"
        + _table(episode["reference_a"]),
        "CITY B\n"
        + symbol_background(episode, "B")
        + "\n"
        + _table(episode["reference_b"]),
        "CITY C\n" + _table(episode["target"][:c_cases]),
    ]
    return (
        SHARED_PROMPT_INTRO
        + "\n\n"
        + SYMBOL_INSTRUCTIONS
        + "\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + shared_prompt_closing(episode)
    )


def prompts_for_episode(
    episode: dict[str, Any], c_cases: int
) -> dict[str, tuple[str, bool | None, str | None]]:
    correct = make_symbol_prompt(episode, c_cases)
    inverted_episode = dict(episode)
    inverted_episode["cue_strong"] = not bool(episode["target_strong"])
    wrong = make_symbol_prompt(inverted_episode, c_cases)
    absent = make_no_target_label_prompt(episode, c_cases)
    return {
        "abc_context": (
            correct,
            bool(episode["target_strong"]),
            strong_symbol(episode)
            if episode["target_strong"]
            else weak_symbol(episode),
        ),
        "abc_no_context": (absent, None, None),
        "abc_wrong_context": (
            wrong,
            not bool(episode["target_strong"]),
            weak_symbol(episode)
            if episode["target_strong"]
            else strong_symbol(episode),
        ),
    }


def validate_prompt_set(
    episode: dict[str, Any],
    prompts: dict[str, tuple[str, bool | None, str | None]],
) -> None:
    if set(prompts) != set(ARMS):
        raise ValueError("prompt arms differ from the frozen protocol")
    numeric = {arm: NUMBER.findall(item[0]) for arm, item in prompts.items()}
    if len({tuple(values) for values in numeric.values()}) != 1:
        raise ValueError(f"numeric content changed in episode {episode['episode']}")

    for arm, (prompt, cue_strong, target_label) in prompts.items():
        lowered = prompt.lower()
        for forbidden in FORBIDDEN_SEMANTICS:
            if forbidden in lowered:
                raise ValueError(f"{arm} leaks semantic cue: {forbidden}")
        if not prompt.startswith(SHARED_PROMPT_INTRO + "\n\n" + SYMBOL_INSTRUCTIONS):
            raise ValueError(f"{arm} changed the shared symbolic instructions")
        if prompt.count("Background label: KIV.") + prompt.count(
            "Background label: ZOR."
        ) != (2 if arm == "abc_no_context" else 3):
            raise ValueError(f"{arm} has the wrong number of labels")
        c_section = prompt.split("CITY C\n", 1)[1].split("\n\n", 1)[0]
        if ("Background label:" in c_section) != (target_label is not None):
            raise ValueError(f"{arm} City C label presence mismatch")
        if target_label is not None and f"Background label: {target_label}." not in c_section:
            raise ValueError(f"{arm} City C label identity mismatch")
        if cue_strong is None and arm != "abc_no_context":
            raise ValueError(f"{arm} cue metadata mismatch")

    validate_symbol_prompt(prompts["abc_context"][0], "abc_symbol_context")
    validate_symbol_prompt(prompts["abc_wrong_context"][0], "abc_symbol_context")


def visible_target_ols(episode: dict[str, Any], c_cases: int) -> float | None:
    rows = episode["target"][:c_cases]
    return None if not rows else float(fit_change_slope(rows))


def make_records(
    selected: list[dict[str, Any]],
    assignment: dict[int, tuple[str, int | None]],
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
            prompt_set = prompts_for_episode(episode, c_cases)
            validate_prompt_set(episode, prompt_set)
            for arm in ARMS:
                prompt, cue_strong, target_label = prompt_set[arm]
                records.append(
                    {
                        "sample_id": f"{frozen_id}__{arm}",
                        "task_id": frozen_id,
                        "episode": episode_id,
                        "split": split,
                        "cv_fold": cv_fold,
                        "arm": arm,
                        "arm_semantics": ARM_SEMANTICS[arm],
                        "c_cases": c_cases,
                        "prompt": prompt,
                        "prompt_sha256": text_sha256(prompt),
                        "strong_reference_city": episode["strong_reference_city"],
                        "strong_symbol": strong_symbol(episode),
                        "weak_symbol": weak_symbol(episode),
                        "target_symbol": target_label,
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
    expected = len(selected) * len(DEPTHS) * len(ARMS)
    if len(records) != expected or len({row["sample_id"] for row in records}) != expected:
        raise ValueError(f"expected exactly {expected} unique prompt records")
    return records


def protocol_counts(
    selected: list[dict[str, Any]],
    assignment: dict[int, tuple[str, int | None]],
    *,
    episodes_per_cell: int = EPISODES_PER_CELL,
    test_per_cell: int = TEST_PER_CELL,
) -> dict[str, Any]:
    dev_per_cell = episodes_per_cell - test_per_cell
    factorial = Counter()
    target_symbols = Counter()
    folds = Counter()
    for episode in selected:
        episode_id = int(episode["episode"])
        split, fold = assignment[episode_id]
        city, target_strong, mapping = factorial_cell(episode)
        factorial[f"{split}:{city}:{int(target_strong)}:{mapping}"] += 1
        label = strong_symbol(episode) if target_strong else weak_symbol(episode)
        target_symbols[f"{split}:{int(target_strong)}:{label}"] += 1
        if split == "dev":
            folds[f"fold_{fold}:{city}:{int(target_strong)}:{mapping}"] += 1
    if set(factorial.values()) != {dev_per_cell, test_per_cell}:
        raise ValueError(f"factorial balance failed: {dict(factorial)}")
    if set(target_symbols.values()) != {2 * dev_per_cell, 2 * test_per_cell}:
        raise ValueError(f"target-label balance failed: {dict(target_symbols)}")
    if set(folds.values()) != {dev_per_cell // CV_FOLDS}:
        raise ValueError(f"cross-validation balance failed: {dict(folds)}")
    return {
        "factorial_cells": dict(sorted(factorial.items())),
        "target_regime_by_symbol": dict(sorted(target_symbols.items())),
        "development_fold_cells": dict(sorted(folds.items())),
    }


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
    parser.add_argument("--episodes-per-cell", type=int, default=EPISODES_PER_CELL)
    parser.add_argument("--test-per-cell", type=int, default=TEST_PER_CELL)
    parser.add_argument("--study", default="qwen3_14b_symbol_probe_v1")
    parser.add_argument("--registration-date", default="2026-08-25")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    episodes_path = DESIGN / "episodes.jsonl"
    episodes = read_jsonl(episodes_path)
    selected, assignment = select_episodes(
        episodes,
        episodes_per_cell=args.episodes_per_cell,
        test_per_cell=args.test_per_cell,
    )
    records = make_records(selected, assignment)
    counts = protocol_counts(
        selected,
        assignment,
        episodes_per_cell=args.episodes_per_cell,
        test_per_cell=args.test_per_cell,
    )
    n_test = sum(1 for split, _ in assignment.values() if split == "test")
    tasks_content = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    tasks_path = args.outdir / "tasks.jsonl"
    write_exact(tasks_path, tasks_content, force=args.force)

    frozen_sources = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (
            episodes_path,
            ROOT / "engine" / "coin_city_stable_relationship_claude_n250.py",
            ROOT / "engine" / "coin_city_symbol_context_arm.py",
            HERE / "extract_qwen_hidden_states.py",
            HERE / "analyze_probe.py",
            Path(__file__),
        )
    }
    manifest = {
        "study": args.study,
        "status": "registered_tasks_frozen_before_activation_extraction",
        "registration_date": args.registration_date,
        "experiment": EXPERIMENT,
        "model": "Qwen/Qwen3-14B",
        "model_gate": "passed episode-randomized arbitrary-label behavioral control",
        "source_files_sha256": frozen_sources,
        "selection_seed": SELECTION_SEED,
        "selection_factors": [
            "strong_reference_city",
            "target_strong",
            "episode_local_strong_symbol",
        ],
        "episodes_per_factorial_cell": args.episodes_per_cell,
        "test_per_factorial_cell": args.test_per_cell,
        "episode_counts": {
            "dev": len(selected) - n_test,
            "test": n_test,
            "total": len(selected),
        },
        "cv_folds": CV_FOLDS,
        "arms": list(ARMS),
        "arm_semantics": ARM_SEMANTICS,
        "evidence_depths": list(DEPTHS),
        "representations": {
            "primary": "final input token at embedding output and every transformer layer",
            "controls": ["mean-pooled embedding output", "mean-pooled final layer"],
        },
        "selection_scope": "development abc_context only; layer and ridge selected separately by target and k",
        "targets": ["regime", "slope", "within-regime residual_slope"],
        "primary_interpretive_test": "held-out within-regime residual_slope at k=4",
        "permutation_test": {
            "repeats": 10000,
            "scope": "held-out labels with fitted development probe, layer, and ridge fixed",
        },
        "numeric_tokens_identical_across_arms": True,
        "semantic_regime_words_absent": True,
        "balance": counts,
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
                "tasks_sha256": manifest["tasks_sha256"],
                "factorial_cells": len(counts["factorial_cells"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
