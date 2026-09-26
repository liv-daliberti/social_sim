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
        "model_id": MODEL_ID,
        "model_commit": MODEL_COMMIT,
        "transformer_layers": 40,
        "replication_role": "discovery_gate",
        "source_probe_run": "qwen3_14b_symbol_probe_v1",
        "tasks_sha256": "6d4dd81bb031d7dc00320e97d2c1896a132c3deab54a6642b6b0105a954014a9",
        "task_manifest_sha256": "35cf4745a92fa10def497662156edcc9f16f98c8f36bfac19e72a986c9020a2c",
        "source_features_sha256": "72a2f3efc27739b5bdb96b544e015841df47b2cc55becf8518b62fe545bd2551",
        "source_results_sha256": "029cb6cf5754d659a9c4f3788f607df88b4b69e80355687e0fe0d2f29744e1db",
    },
    "qwen3_14b_symbol_relational_v2": {
        "study": "qwen3_14b_symbol_relational_v2",
        "model_id": MODEL_ID,
        "model_commit": MODEL_COMMIT,
        "transformer_layers": 40,
        "replication_role": "discovery_gate",
        "source_probe_run": "qwen3_14b_symbol_probe_v2",
        "tasks_sha256": "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01",
        "task_manifest_sha256": "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56",
        "source_features_sha256": "72635a04043467c245b4b3972e6f9193baaa1baf004f4c8e0862c57160e98a57",
        "source_results_sha256": "b02aafe3a5d2012e1bafade81f2622b077bc9f9e74cd8209b30e471902ca45a2",
    },
    # The Qwen3-14B v2 gate prospectively authorized one cross-family
    # replication. Llama-3.1-70B is fixed here before any label-token state is
    # extracted or any activation-patch generation is produced. It is the only
    # evaluated non-Qwen checkpoint that had both a significant held-out
    # relational representation and cue-sensitive zero-shot behavior.
    "llama3_1_70b_symbol_relational_v1": {
        "study": "llama3_1_70b_symbol_relational_v1",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "model_commit": "1605565b47bb9346c5515c34102e054115b4f98b",
        "transformer_layers": 80,
        "replication_role": "cross_family_confirmation",
        "parent_study": "qwen3_14b_symbol_relational_v2",
        "parent_results_sha256": "249a07a54ddd1fe8478eea40a127cf421c8d92926d6b9ec8e8ad997f96bbba78",
        "source_probe_run": "llama3_1_70b_symbol_probe_v2",
        "tasks_sha256": "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01",
        "task_manifest_sha256": "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56",
        # The large source tensor was removed after the checkpoint-comparison
        # analysis; its extraction manifest retains and validates this digest.
        # The relational replication extracts its own label-token states and
        # does not consume the old last-token tensor.
        "source_features_sha256": "3523095ec15434b7878dce9ec7b6f46bd50991703e6a7566376d8bece263c207",
        "source_features_required": False,
        "source_extraction_manifest_sha256": "daf41a11b1f20191dfd7edf8bbaa4f61339a921f564db1579b371012bf8a2f86",
        "source_results_sha256": "b59ac310960620169f5e94d9c619156df59b40f7d4adf1b8b1c92a4e64499b94",
    },
    # These two checkpoint replications were explicitly requested before any
    # new label-token activation or patch outcome was generated. They reuse the
    # powered v2 task split, estimand, controls, and pass/fail gate unchanged.
    "qwen3_32b_symbol_relational_v1": {
        "study": "qwen3_32b_symbol_relational_v1",
        "model_id": "Qwen/Qwen3-32B",
        "model_commit": "9216db5781bf21249d130ec9da846c4624c16137",
        "transformer_layers": 64,
        "replication_role": "same_family_scale_confirmation",
        "parent_study": "qwen3_14b_symbol_relational_v2",
        "parent_results_sha256": "249a07a54ddd1fe8478eea40a127cf421c8d92926d6b9ec8e8ad997f96bbba78",
        "source_probe_run": "qwen3_32b_symbol_probe_v2",
        "source_receipt_required": True,
        "tasks_sha256": "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01",
        "task_manifest_sha256": "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56",
        "target_selection_rule": (
            "the dense same-release Qwen3 scale checkpoint with significant "
            "arbitrary-label forecast discrimination; fixed before its new "
            "arbitrary-label hidden-state or patch result was available"
        ),
    },
    "qwen2_5_72b_symbol_relational_v1": {
        "study": "qwen2_5_72b_symbol_relational_v1",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "model_commit": "495f39366efef23836d0cfae4fbe635880d2be31",
        "transformer_layers": 80,
        "replication_role": "cross_generation_scale_confirmation",
        "parent_study": "qwen3_14b_symbol_relational_v2",
        "parent_results_sha256": "249a07a54ddd1fe8478eea40a127cf421c8d92926d6b9ec8e8ad997f96bbba78",
        "source_probe_run": "qwen2_5_72b_symbol_probe_v2",
        "tasks_sha256": "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01",
        "task_manifest_sha256": "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56",
        "source_features_sha256": "7d0347cf7a8ce8f06156bbaaa808e17048e06da082bfcfb632c576144a1b17f2",
        "source_features_required": False,
        "source_extraction_manifest_sha256": "db9e859ce67c22f6c363ba6dc76d662814c6552fb7f0092338f1b9141cb820fd",
        "source_results_sha256": "b5b003a0a3af99cbe27891b02ad26093e37df1db5952946512b85d48ad2fdad9",
        "target_selection_rule": (
            "the largest completed Qwen checkpoint with significant "
            "arbitrary-label forecast discrimination and held-out decoding; "
            "fixed before any label-token activation or patch result"
        ),
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


def checkpoint_spec(run_dir) -> dict:
    """Return the checkpoint-specific frozen settings for a relational run."""
    spec = run_spec(run_dir)
    required = ("model_id", "model_commit", "transformer_layers")
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError(
            f"relational run {Path(run_dir).name} lacks checkpoint settings: {missing}"
        )
    return spec


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
