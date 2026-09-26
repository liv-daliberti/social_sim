#!/usr/bin/env python3
"""Freeze a registered symbolic relational/causal protocol before extraction."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from symbol_relational_common import (
    CV_FOLDS,
    checkpoint_spec,
    run_spec,
    FACTORIAL_CELLS,
    design_counts,
    ALPHAS,
    ARMS,
    DEFAULT_RUN_DIR,
    PERMUTATION_REPEATS,
    PERMUTATION_SEED,
    PRIMARY_REPRESENTATION,
    PRIMARY_TARGET,
    REPRESENTATIONS,
    SOURCE_RUN_DIR,
    TARGETS,
    file_sha256,
    find_label_anchors,
    formatted_prompt,
    label_tasks,
    load_tasks,
    validate_prefix_equivalence,
)


HERE = Path(__file__).resolve().parent
CODE_FILES = (
    "symbol_relational_common.py",
    "extract_symbol_label_states.py",
    "analyze_symbol_relational_probe.py",
    "patch_symbol_label_activations.py",
    "freeze_symbol_relational_protocol.py",
)
FORBIDDEN_EXISTING = (
    "label_token_states.npz",
    "label_extraction_manifest.json",
    "development_selection.json",
    "activation_patch_generations.jsonl",
    "activation_patch_manifest.json",
    "relational_probe_results.json",
)


def write_exact(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} exists with different content")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def validate_correct_swapped_token_intervention(
    tokenizer,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (int(row["episode"]), int(row["c_cases"]), str(row["arm"])): row
        for row in tasks
        if row["arm"] in ARMS
    }
    difference_counts = Counter()
    checked = 0
    for episode in sorted({key[0] for key in by_key}):
        for k in (0, 4):
            correct = by_key[(episode, k, "abc_context")]
            wrong = by_key[(episode, k, "abc_wrong_context")]
            correct_text = formatted_prompt(tokenizer, correct["prompt"])
            wrong_text = formatted_prompt(tokenizer, wrong["prompt"])
            correct_anchors = find_label_anchors(tokenizer, correct_text)
            wrong_anchors = find_label_anchors(tokenizer, wrong_text)
            correct_ids = tokenizer.encode(correct_text, add_special_tokens=False)
            wrong_ids = tokenizer.encode(wrong_text, add_special_tokens=False)
            if len(correct_ids) != len(wrong_ids):
                raise ValueError("correct/swapped token lengths differ")
            differences = [
                index
                for index, (left, right) in enumerate(zip(correct_ids, wrong_ids))
                if left != right
            ]
            correct_span = correct_anchors["city_c"]["token_positions"]
            wrong_span = wrong_anchors["city_c"]["token_positions"]
            if differences != correct_span or correct_span != wrong_span:
                raise ValueError(
                    f"intervention differs outside City C label: episode={episode}, k={k}"
                )
            if any(
                correct_anchors[anchor]["token_ids"]
                != wrong_anchors[anchor]["token_ids"]
                for anchor in ("city_a", "city_b")
            ):
                raise ValueError("reference-city labels changed under target swap")
            difference_counts[len(differences)] += 1
            checked += 1
    expected_pairs = len({key[0] for key in by_key}) * 2
    if checked != expected_pairs or difference_counts != Counter({2: expected_pairs}):
        raise ValueError("correct/swapped intervention audit is incomplete")
    return {
        "matched_correct_swapped_pairs_checked": checked,
        "differing_token_count": 2,
        "differences_exactly_city_c_symbol_span": True,
        "reference_labels_unchanged": True,
    }


def design_balance(
    tasks: list[dict[str, Any]], counts: dict[str, Any]
) -> dict[str, Any]:
    episode_rows: dict[int, dict[str, Any]] = {}
    for row in tasks:
        episode_rows.setdefault(int(row["episode"]), row)
    if len(episode_rows) != counts["episodes"]:
        raise ValueError(f"expected {counts['episodes']} unique episodes")
    factorial = Counter()
    folds = Counter()
    for episode, row in episode_rows.items():
        split = str(row["split"])
        cell = (
            str(row["strong_reference_city"]),
            bool(row["target_strong"]),
            str(row["strong_symbol"]),
        )
        factorial[(split, *cell)] += 1
        if split == "dev":
            folds[(int(row["cv_fold"]), *cell)] += 1
    dev_per_cell = counts["development_episodes"] // FACTORIAL_CELLS
    test_per_cell = counts["sealed_test_episodes"] // FACTORIAL_CELLS
    if set(factorial.values()) != {dev_per_cell, test_per_cell}:
        raise ValueError(f"factorial balance drifted: {factorial}")
    if set(folds.values()) != {dev_per_cell // CV_FOLDS}:
        raise ValueError(f"fold balance drifted: {folds}")
    return {
        "episodes": counts["episodes"],
        "development_episodes": counts["development_episodes"],
        "sealed_test_episodes": counts["sealed_test_episodes"],
        "factorial_cell_counts": {
            ":".join(map(str, key)): value
            for key, value in sorted(factorial.items())
        },
        "development_fold_cell_counts": {
            ":".join(map(str, key)): value
            for key, value in sorted(folds.items())
        },
    }


def source_artifacts(spec: dict) -> dict[str, str]:
    source_dir = SOURCE_RUN_DIR.with_name(spec["source_probe_run"])
    if spec.get("source_receipt_required"):
        receipt_path = source_dir / "causal_source_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "sealed_for_relational_replication":
            raise ValueError("source probe receipt is not sealed")
        if receipt.get("study") != spec["source_probe_run"]:
            raise ValueError("source probe receipt study drifted")
        checkpoint = receipt.get("model") or {}
        if checkpoint != {
            "id": spec["model_id"],
            "commit": spec["model_commit"],
        }:
            raise ValueError("source probe receipt checkpoint drifted")
        sealed = receipt.get("artifacts") or {}
        expected = {
            "tasks.jsonl": sealed.get("tasks.jsonl"),
            "task_manifest.json": sealed.get("task_manifest.json"),
            "hidden_states.npz": sealed.get("hidden_states.npz"),
            "probe_results.json": sealed.get("probe_results.json"),
        }
        if expected["tasks.jsonl"] != spec["tasks_sha256"]:
            raise ValueError("source receipt task hash drifted")
        if expected["task_manifest.json"] != spec["task_manifest_sha256"]:
            raise ValueError("source receipt task-manifest hash drifted")
        extraction_digest = sealed.get("extraction_manifest.json")
        if file_sha256(source_dir / "extraction_manifest.json") != extraction_digest:
            raise ValueError("source extraction manifest changed after sealing")
        if file_sha256(source_dir / "probe_results.json") != expected["probe_results.json"]:
            raise ValueError("source probe result changed after sealing")
        manifest = json.loads(
            (source_dir / "extraction_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("features_sha256") != expected["hidden_states.npz"]:
            raise ValueError("sealed source feature digest drifted")
        return {
            **expected,
            "extraction_manifest.json": extraction_digest,
            "causal_source_receipt.json": file_sha256(receipt_path),
        }
    expected = {
        "tasks.jsonl": spec["tasks_sha256"],
        "task_manifest.json": spec["task_manifest_sha256"],
        "hidden_states.npz": spec["source_features_sha256"],
        "probe_results.json": spec["source_results_sha256"],
    }
    for name, digest in expected.items():
        if name == "hidden_states.npz" and not spec.get(
            "source_features_required", True
        ):
            manifest_path = source_dir / "extraction_manifest.json"
            manifest_digest = spec["source_extraction_manifest_sha256"]
            if file_sha256(manifest_path) != manifest_digest:
                raise ValueError("source extraction manifest drifted")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("features_sha256") != digest:
                raise ValueError("historical source feature hash drifted")
            continue
        observed = file_sha256(source_dir / name)
        if observed != digest:
            raise ValueError(
                f"source artifact drifted: {name}: {observed} != {digest}"
            )
    return expected


def copy_exact(source: Path, destination: Path, digest: str) -> None:
    if destination.exists():
        if file_sha256(destination) != digest:
            raise FileExistsError(f"{destination} exists with a different hash")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    if file_sha256(destination) != digest:
        raise RuntimeError(f"copy hash mismatch: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    for name in FORBIDDEN_EXISTING:
        if (args.run_dir / name).exists():
            raise FileExistsError(
                f"{name} already exists; protocol cannot be frozen after result access"
            )
    spec = run_spec(args.run_dir)
    checkpoint = checkpoint_spec(args.run_dir)
    model_id = str(checkpoint["model_id"])
    model_commit = str(checkpoint["model_commit"])
    transformer_layers = int(checkpoint["transformer_layers"])
    last_effective_layer = transformer_layers - 1
    source_dir = SOURCE_RUN_DIR.with_name(spec["source_probe_run"])
    artifacts = source_artifacts(spec)
    replication = None
    role = spec.get("replication_role", "discovery_gate")
    if role.endswith("_confirmation"):
        parent_dir = SOURCE_RUN_DIR.with_name(spec["parent_study"])
        parent_path = parent_dir / "relational_probe_results.json"
        if file_sha256(parent_path) != spec["parent_results_sha256"]:
            raise ValueError("authorizing parent result drifted")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if not parent.get("stability_gate", {}).get("passed"):
            raise ValueError("authorizing parent stability gate did not pass")
        if role == "cross_family_confirmation" and (
            parent.get("stability_gate", {}).get("cross_family_replication")
            != "authorized_by_frozen_gate"
        ):
            raise ValueError("parent did not authorize the cross-family replication")
        replication = {
            "role": role,
            "parent_study": spec["parent_study"],
            "parent_results_sha256": spec["parent_results_sha256"],
            "target_fixed_before_label_state_extraction": True,
            "target_selection_rule": spec.get(
                "target_selection_rule",
                "the evaluated non-Qwen checkpoint with both significant held-out "
                "relational decoding and cue-sensitive zero-shot behavior",
            ),
            "confirmatory_claim": (
                "the development-selected three-layer City C label patch has a "
                "positive bidirectional regime-aligned effect at k=0"
            ),
            "all_original_gate_checks_reused": True,
        }
    copy_exact(
        source_dir / "tasks.jsonl",
        args.run_dir / "tasks.jsonl",
        spec["tasks_sha256"],
    )
    copy_exact(
        source_dir / "task_manifest.json",
        args.run_dir / "task_manifest.json",
        spec["task_manifest_sha256"],
    )
    tasks, task_manifest = load_tasks(args.run_dir)
    counts = design_counts(task_manifest)
    if len(label_tasks(tasks)) != counts["label_extraction_records"]:
        raise ValueError("label extraction subset is incomplete")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    prefix = validate_prefix_equivalence(tokenizer, tasks)
    intervention = validate_correct_swapped_token_intervention(tokenizer, tasks)
    balance = design_balance(tasks, counts)
    preflight = {
        "study": spec["study"],
        "status": "passed_before_label_activation_extraction",
        "model": model_id,
        "model_commit": model_commit,
        "tokenizer_class": tokenizer.__class__.__name__,
        "prefix": prefix,
        "intervention": intervention,
        "balance": balance,
    }
    preflight_content = json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    preflight_path = args.run_dir / "tokenization_preflight.json"
    write_exact(preflight_path, preflight_content)

    code_hashes = {
        name: file_sha256(HERE / name)
        for name in CODE_FILES
    }
    protocol = {
        "study": spec["study"],
        "status": "frozen_before_label_activation_extraction",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "paper_integration": (
            "prospective checkpoint replication; integrate only after frozen analysis"
            if replication
            else "none; internal mechanistic diagnostic only"
        ),
        "model": {"id": model_id, "commit": model_commit},
        "replication": replication,
        "source_artifacts_sha256": artifacts,
        "task_manifest_study": task_manifest["study"],
        "task_design": {
            "all_records": counts["all_records"],
            "label_extraction_records": counts["label_extraction_records"],
            "label_extraction_scope": (
                "k=0 correct and swapped arms only; k=4 is causally identical "
                "through the City C label and is not duplicated"
            ),
            "development_episodes": counts["development_episodes"],
            "sealed_test_episodes": counts["sealed_test_episodes"],
            "arms": list(ARMS),
            "balance_factors": [
                "strong_reference_city",
                "target_strong",
                "episode_local_strong_symbol",
            ],
        },
        "anchor": {
            "site": "both exact symbol subtokens in each Background label line",
            "cities": ["A", "B", "C"],
            "subtoken_pooling_for_probes": "arithmetic mean of the two subtokens",
            "saved_tensor": (
                "both subtokens separately at embedding output and transformer "
                f"outputs 1--{transformer_layers}"
            ),
        },
        "representations": {
            "primary": PRIMARY_REPRESENTATION,
            "diagnostic": [
                name for name in REPRESENTATIONS if name != PRIMARY_REPRESENTATION
            ],
            "definitions": {
                "city_c": "City C label state",
                "matching_reference": "A/B label state whose symbol matches City C",
                "city_c_minus_matching_reference": "City C minus matching A/B label state",
                "city_c_minus_nonmatching_reference": "City C minus other A/B label state",
                "matching_minus_nonmatching_reference": "matching A/B minus other A/B label state",
            },
        },
        "probe": {
            "primary_target": PRIMARY_TARGET,
            "secondary_targets": [
                target for target in TARGETS if target != PRIMARY_TARGET
            ],
            "training": "correct-label development episodes only",
            "cv": "four episode-disjoint, fully factorially balanced folds",
            "ridge_alphas": list(ALPHAS),
            "layer_selection": (
                "maximum development CV among causally effective transformer "
                f"outputs 1--{last_effective_layer}; "
                "earlier layer breaks ties"
            ),
            "final_output_layer": (
                f"layer {transformer_layers} is saved as a "
                "diagnostic but is ineligible for selection because it has no "
                "downstream transformer block"
            ),
            "embedding_output": "control only; cannot be selected",
            "sealed_test": (
                f"{counts['sealed_test_episodes']} correct-label and "
                f"{counts['sealed_test_episodes']} swapped-label k=0 prompts"
            ),
            "permutation": {
                "repeats": PERMUTATION_REPEATS,
                "seed": PERMUTATION_SEED,
                "scheme": (
                    "held-out labels permuted with the development-selected "
                    "representation, layer, ridge, and fitted probe fixed"
                ),
            },
        },
        "causal_patching": {
            "site": "both City C symbol subtokens only",
            "directions": [
                "correct_into_swapped",
                "swapped_into_correct",
            ],
            "primary_layer_rule": (
                "three contiguous hidden-state layers centered on the "
                "development-selected primary relational-probe layer; clamp "
                f"to 1--3 or {last_effective_layer - 2}--"
                f"{last_effective_layer} at boundaries"
            ),
            "primary_evidence_depth": 0,
            "secondary_evidence_depth": 4,
            "controls": [
                "unpatched",
                "self patch at selected window",
                "cross patch at embedding output only",
                "cross patch at all causally effective hidden-state layers "
                f"0--{last_effective_layer}",
            ],
            "primary_endpoint": (
                "episode-level mean of the regime-aligned correct-into-swapped "
                "shift and the sign-reversed swapped-into-correct shift"
            ),
            "primary_test": "exact one-sided episode sign-flip test",
        },
        "stability_gate_before_cross_family_replication": {
            "all_required": True,
            "checks": [
                "primary correct-label relational regime AUC > .5",
                "primary held-out permutation p < .05",
                f"at least {counts['min_patch_episodes']} complete causal-patch "
                "episodes at k=0",
                "primary causal-patch exact sign-flip p < .05",
                "positive symmetric patch mean",
                "positive mean in both patch directions",
                f"at least {counts['min_self_patch_pairs']} parsed self-patch pairs",
                "self-patch exact poll match rate >= .95",
            ],
            "failure_action": (
                "stop; report the checkpoint replication as failed"
                if replication
                else "stop; do not launch a cross-family replication"
            ),
            "pass_action": (
                "report the prospectively frozen checkpoint replication"
                if replication
                else (
                    "freeze and run one cross-family replication with the identical "
                    "anchor, representation, patch rule, and gate"
                )
            ),
        },
        "tokenization_preflight_sha256": file_sha256(preflight_path),
        "code_sha256": code_hashes,
    }
    protocol_content = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = args.run_dir / "relational_protocol_manifest.json"
    write_exact(protocol_path, protocol_content)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "protocol": str(protocol_path),
                "protocol_sha256": file_sha256(protocol_path),
                "tokenization_preflight_sha256": file_sha256(preflight_path),
                "code_sha256": code_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
