#!/usr/bin/env python3
"""Shared validation and token-span helpers for the symbolic relational probe."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENT = "coin_city_stable_relationship_claude_n250_v4"
SOURCE_RUN_DIR = (
    ROOT / "data" / EXPERIMENT / "mechanistic_probe" / "qwen3_14b_symbol_probe_v1"
)
DEFAULT_RUN_DIR = (
    ROOT
    / "data"
    / EXPERIMENT
    / "mechanistic_probe"
    / "qwen3_14b_symbol_relational_v1"
)
MODEL_ID = "Qwen/Qwen3-14B"
MODEL_COMMIT = "40c069824f4251a91eefaf281ebe4c544efd3e18"
STUDY = "qwen3_14b_symbol_relational_v1"
# Each registered run pins the frozen task file it may read and the probe run it
# draws its source activations from. A run directory outside this registry is
# refused rather than silently trusted.
RUN_REGISTRY = {
    "qwen3_14b_symbol_relational_v1": {
        "study": "qwen3_14b_symbol_relational_v1",
        "source_probe_run": "qwen3_14b_symbol_probe_v1",
        "tasks_sha256": "6d4dd81bb031d7dc00320e97d2c1896a132c3deab54a6642b6b0105a954014a9",
        "task_manifest_sha256": "35cf4745a92fa10def497662156edcc9f16f98c8f36bfac19e72a986c9020a2c",
        "source_features_sha256": "72a2f3efc27739b5bdb96b544e015841df47b2cc55becf8518b62fe545bd2551",
        "source_results_sha256": "029cb6cf5754d659a9c4f3788f607df88b4b69e80355687e0fe0d2f29744e1db",
    },
    "qwen3_14b_symbol_relational_v2": {
        "study": "qwen3_14b_symbol_relational_v2",
        "source_probe_run": "qwen3_14b_symbol_probe_v2",
        "tasks_sha256": "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01",
        "task_manifest_sha256": "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56",
        "source_features_sha256": "72635a04043467c245b4b3972e6f9193baaa1baf004f4c8e0862c57160e98a57",
        "source_results_sha256": "b02aafe3a5d2012e1bafade81f2622b077bc9f9e74cd8209b30e471902ca45a2",
    },
    # The symbol probe runs are the source of each relational run's task file, and
    # are read directly when auditing the frozen design, so they carry the same
    # task hashes under their own names.
    "qwen3_14b_symbol_probe_v1": {
        "study": "qwen3_14b_symbol_probe_v1",
        "source_probe_run": "qwen3_14b_symbol_probe_v1",
        "tasks_sha256": "6d4dd81bb031d7dc00320e97d2c1896a132c3deab54a6642b6b0105a954014a9",
        "task_manifest_sha256": "35cf4745a92fa10def497662156edcc9f16f98c8f36bfac19e72a986c9020a2c",
        "source_features_sha256": "72a2f3efc27739b5bdb96b544e015841df47b2cc55becf8518b62fe545bd2551",
        "source_results_sha256": "029cb6cf5754d659a9c4f3788f607df88b4b69e80355687e0fe0d2f29744e1db",
    },
    "qwen3_14b_symbol_probe_v2": {
        "study": "qwen3_14b_symbol_probe_v2",
        "source_probe_run": "qwen3_14b_symbol_probe_v2",
        "tasks_sha256": "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01",
        "task_manifest_sha256": "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56",
        "source_features_sha256": "72635a04043467c245b4b3972e6f9193baaa1baf004f4c8e0862c57160e98a57",
        "source_results_sha256": "b02aafe3a5d2012e1bafade81f2622b077bc9f9e74cd8209b30e471902ca45a2",
    },
}


def run_spec(run_dir) -> dict:
    name = Path(run_dir).name
    if name not in RUN_REGISTRY:
        raise ValueError(f"unregistered relational run directory: {name}")
    return RUN_REGISTRY[name]


TASKS_SHA256 = RUN_REGISTRY["qwen3_14b_symbol_relational_v1"]["tasks_sha256"]
TASK_MANIFEST_SHA256 = RUN_REGISTRY["qwen3_14b_symbol_relational_v1"][
    "task_manifest_sha256"
]
SOURCE_FEATURES_SHA256 = RUN_REGISTRY["qwen3_14b_symbol_relational_v1"][
    "source_features_sha256"
]
SOURCE_RESULTS_SHA256 = RUN_REGISTRY["qwen3_14b_symbol_relational_v1"][
    "source_results_sha256"
]
# Balance factors crossed by the builder, and the development folds within them.
FACTORIAL_CELLS = 8
CV_FOLDS = 4
# Completeness floors, expressed as the fractions the v1 protocol froze (14/16
# patch episodes and 58/64 self-patch pairs) so they scale with the sealed test.
MIN_PATCH_EPISODE_FRACTION = 14 / 16
MIN_SELF_PAIR_FRACTION = 58 / 64


def design_counts(manifest: dict) -> dict:
    """Sizes the protocol depends on, read from the frozen task manifest."""
    episodes = manifest["episode_counts"]
    test = int(episodes["test"])
    return {
        "episodes": int(episodes["total"]),
        "development_episodes": int(episodes["dev"]),
        "sealed_test_episodes": test,
        "all_records": int(manifest["record_count"]),
        "label_extraction_records": int(episodes["total"]) * len(ARMS),
        # Per sealed episode: two evidence depths x two recipient arms x five
        # patch conditions.
        "patch_records": test * 2 * len(ARMS) * 5,
        "self_patch_pairs": test * 2 * len(ARMS),
        "min_patch_episodes": int(test * MIN_PATCH_EPISODE_FRACTION),
        "min_self_patch_pairs": int(test * 2 * len(ARMS) * MIN_SELF_PAIR_FRACTION),
    }
SYMBOLS = ("KIV", "ZOR")
ANCHORS = ("city_a", "city_b", "city_c")
ARMS = ("abc_context", "abc_wrong_context")
TARGETS = ("regime", "slope", "residual_slope")
REPRESENTATIONS = (
    "city_c",
    "matching_reference",
    "city_c_minus_matching_reference",
    "city_c_minus_nonmatching_reference",
    "matching_minus_nonmatching_reference",
)
PRIMARY_REPRESENTATION = "city_c_minus_matching_reference"
PRIMARY_TARGET = "regime"
ALPHAS = (1e-4, 3e-3, 1e-1, 3.0, 100.0)
PERMUTATION_SEED = 20260825
PATCH_SEED = 20260825
PERMUTATION_REPEATS = 10_000
TRANSFORMER_LAYERS = 40
# A state saved after the final block has no downstream block through which it
# can change later-token activations, so layer 40 remains extractable/reportable
# but is excluded from causal selection and patching.
LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER = TRANSFORMER_LAYERS - 1
LABEL_PATTERN = re.compile(r"Background label: (KIV|ZOR)\.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
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


def load_tasks(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_path = run_dir / "tasks.jsonl"
    manifest_path = run_dir / "task_manifest.json"
    spec = run_spec(run_dir)
    if file_sha256(tasks_path) != spec["tasks_sha256"]:
        raise ValueError("tasks.jsonl does not match the frozen symbolic task hash")
    if file_sha256(manifest_path) != spec["task_manifest_sha256"]:
        raise ValueError("task_manifest.json does not match the frozen hash")
    tasks = read_jsonl(tasks_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_records = design_counts(manifest)["all_records"]
    if len(tasks) != expected_records:
        raise ValueError(f"expected {expected_records} frozen symbolic task records")
    if [row["sample_id"] for row in tasks] != list(
        dict.fromkeys(row["sample_id"] for row in tasks)
    ):
        raise ValueError("duplicate or reordered sample identifiers")
    for row in tasks:
        if text_sha256(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['sample_id']}")
    return tasks, manifest


def label_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the nonduplicated causal prefix used for label-state extraction."""
    selected = [
        row
        for row in tasks
        if row["c_cases"] == 0 and row["arm"] in ARMS
    ]
    expected = len({int(row["episode"]) for row in tasks}) * len(ARMS)
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} k=0 correct/swapped records, found {len(selected)}"
        )
    return selected


def formatted_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def find_label_anchors(tokenizer, rendered: str) -> dict[str, dict[str, Any]]:
    """Resolve each city label to its exact two-token span in rendered chat text."""
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    input_ids = list(encoded["input_ids"])
    result: dict[str, dict[str, Any]] = {}
    for city_index, city in enumerate("ABC"):
        marker = f"CITY {city}\n"
        section_start = rendered.index(marker) + len(marker)
        later = [
            rendered.index(f"CITY {other}\n")
            for other in "ABC"[city_index + 1 :]
            if f"CITY {other}\n" in rendered[section_start:]
        ]
        section_end = min(later) if later else len(rendered)
        match = LABEL_PATTERN.search(rendered, section_start, section_end)
        if match is None:
            raise ValueError(f"CITY {city} has no symbolic label")
        symbol = match.group(1)
        char_start, char_end = match.span(1)
        positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if len(positions) != 2 or positions[1] != positions[0] + 1:
            raise ValueError(
                f"CITY {city} {symbol} does not occupy two contiguous tokens: "
                f"{positions}"
            )
        decoded = tokenizer.decode(
            [input_ids[index] for index in positions],
            skip_special_tokens=False,
        ).strip()
        if decoded != symbol:
            raise ValueError(
                f"CITY {city} span decodes to {decoded!r}, expected {symbol!r}"
            )
        result[f"city_{city.lower()}"] = {
            "symbol": symbol,
            "char_span": [char_start, char_end],
            "token_positions": positions,
            "token_ids": [input_ids[index] for index in positions],
        }
    if set(result) != set(ANCHORS):
        raise ValueError("label anchor set is incomplete")
    return result


def reference_symbols(row: dict[str, Any]) -> dict[str, str]:
    strong = str(row["strong_symbol"])
    weak = str(row["weak_symbol"])
    if strong not in SYMBOLS or weak not in SYMBOLS or strong == weak:
        raise ValueError(f"invalid episode-local symbol mapping: {row['sample_id']}")
    strong_city = str(row["strong_reference_city"])
    return {
        "city_a": strong if strong_city == "A" else weak,
        "city_b": strong if strong_city == "B" else weak,
    }


def reference_roles(row: dict[str, Any]) -> tuple[str, str]:
    symbols = reference_symbols(row)
    target_symbol = row.get("target_symbol")
    matches = [anchor for anchor, symbol in symbols.items() if symbol == target_symbol]
    if len(matches) != 1:
        raise ValueError(f"target symbol has no unique reference: {row['sample_id']}")
    matching = matches[0]
    nonmatching = "city_b" if matching == "city_a" else "city_a"
    return matching, nonmatching


def validate_prefix_equivalence(
    tokenizer,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that k=0 and k=4 are identical through the City C label span."""
    by_key = {
        (int(row["episode"]), str(row["arm"]), int(row["c_cases"])): row
        for row in tasks
        if row["arm"] in ARMS
    }
    checked = 0
    span_lengths: set[int] = set()
    token_id_pairs: set[tuple[int, int]] = set()
    for episode in sorted({key[0] for key in by_key}):
        for arm in ARMS:
            zero = by_key[(episode, arm, 0)]
            four = by_key[(episode, arm, 4)]
            rendered_zero = formatted_prompt(tokenizer, zero["prompt"])
            rendered_four = formatted_prompt(tokenizer, four["prompt"])
            anchors_zero = find_label_anchors(tokenizer, rendered_zero)
            anchors_four = find_label_anchors(tokenizer, rendered_four)
            end_zero = anchors_zero["city_c"]["token_positions"][-1]
            end_four = anchors_four["city_c"]["token_positions"][-1]
            ids_zero = tokenizer.encode(rendered_zero, add_special_tokens=False)
            ids_four = tokenizer.encode(rendered_four, add_special_tokens=False)
            if end_zero != end_four or ids_zero[: end_zero + 1] != ids_four[: end_four + 1]:
                raise ValueError(
                    f"k=0/k=4 prefix mismatch through City C label: episode={episode}, "
                    f"arm={arm}"
                )
            for anchor in ANCHORS:
                if anchors_zero[anchor]["symbol"] != anchors_four[anchor]["symbol"]:
                    raise ValueError("k=0/k=4 label identity mismatch")
                positions = anchors_zero[anchor]["token_positions"]
                span_lengths.add(len(positions))
                token_id_pairs.add(tuple(anchors_zero[anchor]["token_ids"]))
            checked += 1
    expected_pairs = len({key[0] for key in by_key}) * len(ARMS)
    if (
        checked != expected_pairs
        or span_lengths != {2}
        or len(token_id_pairs) != 2
    ):
        raise ValueError(
            f"token preflight failed: checked={checked}, spans={span_lengths}, "
            f"label tokenizations={token_id_pairs}"
        )
    return {
        "episode_arm_prefix_pairs_checked": checked,
        "label_span_token_count": 2,
        "distinct_label_token_id_pairs": [list(pair) for pair in sorted(token_id_pairs)],
        "k0_k4_identical_through_city_c_label": True,
    }
