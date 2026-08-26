#!/usr/bin/env python3
"""Development-select and sealed-test the symbolic label-token representations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import analyze_probe as base
from symbol_relational_common import (
    TRANSFORMER_LAYERS,
    design_counts,
    run_spec,
    ALPHAS,
    ANCHORS,
    ARMS,
    DEFAULT_RUN_DIR,
    LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER,
    PERMUTATION_REPEATS,
    PERMUTATION_SEED,
    PRIMARY_REPRESENTATION,
    PRIMARY_TARGET,
    REPRESENTATIONS,
    TARGETS,
    file_sha256,
    label_tasks,
    load_tasks,
    read_jsonl,
    reference_roles,
)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_features(
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    path = run_dir / "label_token_states.npz"
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        anchor_names = payload["anchor_names"].astype(str).tolist()
        token_states = payload["label_token_states"].astype(np.float32)
        symbols = payload["symbols"].astype(str)
    if sample_ids != [row["sample_id"] for row in rows]:
        raise ValueError("label-state sample order differs from the frozen tasks")
    if anchor_names != list(ANCHORS):
        raise ValueError("label-state anchor order drifted")
    expected_shape = (len(rows), TRANSFORMER_LAYERS + 1, len(ANCHORS), 2)
    if token_states.shape[:4] != expected_shape:
        raise ValueError(
            f"unexpected label-state shape: {token_states.shape}, "
            f"expected {expected_shape} + (hidden,)"
        )
    if not np.isfinite(token_states).all():
        raise ValueError("label-state tensor contains nonfinite values")
    for index, row in enumerate(rows):
        expected_a = row["strong_symbol"] if row["strong_reference_city"] == "A" else row["weak_symbol"]
        expected_b = row["strong_symbol"] if row["strong_reference_city"] == "B" else row["weak_symbol"]
        if symbols[index].tolist() != [expected_a, expected_b, row["target_symbol"]]:
            raise ValueError(f"saved symbol metadata mismatch: {row['sample_id']}")

    anchor_mean = token_states.mean(axis=3)
    by_anchor = {
        anchor: anchor_mean[:, :, anchor_index, :]
        for anchor_index, anchor in enumerate(ANCHORS)
    }
    derived = {
        name: np.empty((len(rows), token_states.shape[1], token_states.shape[-1]), dtype=np.float32)
        for name in REPRESENTATIONS
    }
    for index, row in enumerate(rows):
        matching, nonmatching = reference_roles(row)
        c_state = by_anchor["city_c"][index]
        match_state = by_anchor[matching][index]
        nonmatch_state = by_anchor[nonmatching][index]
        derived["city_c"][index] = c_state
        derived["matching_reference"][index] = match_state
        derived["city_c_minus_matching_reference"][index] = c_state - match_state
        derived["city_c_minus_nonmatching_reference"][index] = c_state - nonmatch_state
        derived["matching_minus_nonmatching_reference"][index] = (
            match_state - nonmatch_state
        )
    return derived


def development_indices(
    rows: list[dict[str, Any]], expected_development: int
) -> np.ndarray:
    indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == "dev" and row["arm"] == "abc_context"
        ],
        dtype=int,
    )
    if len(indices) != expected_development:
        raise ValueError(
            f"expected {expected_development} correct-label development episodes, "
            f"got {len(indices)}"
        )
    return indices


def three_layer_window(
    center: int,
    last_effective_layer: int = LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER,
) -> list[int]:
    if not 1 <= center <= last_effective_layer:
        raise ValueError("patch center must have a downstream transformer block")
    start = min(max(1, center - 1), last_effective_layer - 2)
    return [start, start + 1, start + 2]


def select_on_development(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    counts: dict[str, Any],
) -> dict[str, Any]:
    dev = development_indices(rows, counts["development_episodes"])
    dev_rows = [rows[index] for index in dev]
    folds = np.asarray([row["cv_fold"] for row in dev_rows], dtype=int)
    if set(folds.tolist()) != {0, 1, 2, 3}:
        raise ValueError("development folds are incomplete")
    selection: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        selection[representation] = {}
        for target in TARGETS:
            target_values = base.target_values(dev_rows, target)
            layers: list[dict[str, Any]] = []
            # Layer 0 is reported as a literal-token control, never selected.
            embedding_alpha, embedding_cv, embedding_scores = base.cross_validated_alpha(
                train=features[representation][dev, 0, :],
                target=target_values,
                folds=folds,
                target_name=target,
                alphas=np.asarray(ALPHAS, dtype=float),
            )
            for layer in range(1, LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER + 1):
                alpha, score, alpha_scores = base.cross_validated_alpha(
                    train=features[representation][dev, layer, :],
                    target=target_values,
                    folds=folds,
                    target_name=target,
                    alphas=np.asarray(ALPHAS, dtype=float),
                )
                layers.append(
                    {
                        "layer": layer,
                        "alpha": float(alpha),
                        "cv_score": float(score),
                        "alpha_cv_scores": alpha_scores,
                    }
                )
            best = max(
                layers,
                key=lambda item: (
                    -float("inf")
                    if not np.isfinite(item["cv_score"])
                    else item["cv_score"],
                    -item["layer"],
                ),
            )
            selection[representation][target] = {
                "selected_by": (
                    "maximum four-fold development CV across transformer outputs "
                    "1--39; ties choose the earlier layer"
                ),
                "selected_layer": int(best["layer"]),
                "selected_alpha": float(best["alpha"]),
                "development_cv_score": float(best["cv_score"]),
                "embedding_control": {
                    "layer": 0,
                    "alpha": float(embedding_alpha),
                    "development_cv_score": float(embedding_cv),
                    "alpha_cv_scores": embedding_scores,
                },
                "development_layer_curve": layers,
            }

    patch_center = selection[PRIMARY_REPRESENTATION][PRIMARY_TARGET]["selected_layer"]
    result = {
        "study": run_spec(run_dir)["study"],
        "status": "development_selection_complete_test_unread",
        "selection_scope": (
            "64 correct-label development episodes only; no sealed-test metric "
            "is computed or written by this mode"
        ),
        "representations": list(REPRESENTATIONS),
        "targets": list(TARGETS),
        "alphas": list(ALPHAS),
        "selectable_hidden_state_layers": list(
            range(1, LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER + 1)
        ),
        "final_hidden_state_layer_role": (
            "layer 40 is extracted as a diagnostic but cannot be selected because "
            "it has no downstream transformer block"
        ),
        "embedding_layer_role": "reported control only; excluded from layer selection",
        "selection": selection,
        "primary": {
            "representation": PRIMARY_REPRESENTATION,
            "target": PRIMARY_TARGET,
            "selected_layer": patch_center,
            "selected_alpha": selection[PRIMARY_REPRESENTATION][PRIMARY_TARGET][
                "selected_alpha"
            ],
            "development_cv_score": selection[PRIMARY_REPRESENTATION][PRIMARY_TARGET][
                "development_cv_score"
            ],
        },
        "causal_patch_window_hidden_state_layers": three_layer_window(patch_center),
        "tasks_sha256": file_sha256(run_dir / "tasks.jsonl"),
        "protocol_sha256": file_sha256(
            run_dir / "relational_protocol_manifest.json"
        ),
        "label_states_sha256": file_sha256(run_dir / "label_token_states.npz"),
    }
    return result


def evaluate_fixed_selection(
    *,
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    selection: dict[str, Any],
    permutations: int,
    counts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev = development_indices(rows, counts["development_episodes"])
    dev_rows = [rows[index] for index in dev]
    test_by_arm = {
        arm: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["split"] == "test" and row["arm"] == arm
            ],
            dtype=int,
        )
        for arm in ARMS
    }
    sealed = counts["sealed_test_episodes"]
    if any(len(indices) != sealed for indices in test_by_arm.values()):
        raise ValueError(
            f"sealed correct/swapped arms must each contain {sealed} episodes"
        )

    results: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    base.PERMUTATION_SEED = PERMUTATION_SEED
    for representation in REPRESENTATIONS:
        results[representation] = {}
        for target in TARGETS:
            chosen = selection["selection"][representation][target]
            layer = int(chosen["selected_layer"])
            alpha = float(chosen["selected_alpha"])
            train_x = features[representation][dev, layer, :]
            train_y = base.target_values(dev_rows, target)
            arm_results: dict[str, Any] = {}
            for arm, indices in test_by_arm.items():
                arm_rows = [rows[index] for index in indices]
                prediction = base.dual_predict(
                    base.prepare_dual(train_x, features[representation][indices, layer, :]),
                    train_y,
                    alpha,
                )
                arm_results[arm] = base.evaluate_predictions(prediction, arm_rows, target)
                for row, value in zip(arm_rows, prediction):
                    predictions.append(
                        {
                            "representation": representation,
                            "target": target,
                            "selected_layer": layer,
                            "selected_alpha": alpha,
                            "sample_id": row["sample_id"],
                            "episode": row["episode"],
                            "arm": row["arm"],
                            "prediction": float(value),
                            "true_target": float(base.target_values([row], target)[0]),
                            "cue_strong": row["cue_strong"],
                        }
                    )
            correct_indices = test_by_arm["abc_context"]
            null = base.permutation_null(
                train=train_x,
                test=features[representation][correct_indices, layer, :],
                train_target=train_y,
                test_target=base.target_values(
                    [rows[index] for index in correct_indices], target
                ),
                alpha=alpha,
                target_name=target,
                repeats=permutations,
            )

            embedding = chosen["embedding_control"]
            embedding_layer = int(embedding["layer"])
            embedding_alpha = float(embedding["alpha"])
            embedding_prediction = base.dual_predict(
                base.prepare_dual(
                    features[representation][dev, embedding_layer, :],
                    features[representation][correct_indices, embedding_layer, :],
                ),
                train_y,
                embedding_alpha,
            )
            embedding_rows = [rows[index] for index in correct_indices]
            results[representation][target] = {
                "selected_layer": layer,
                "selected_alpha": alpha,
                "development_cv_score": chosen["development_cv_score"],
                "test": arm_results,
                "correct_test_permutation": null,
                "embedding_control_correct_test": base.evaluate_predictions(
                    embedding_prediction,
                    embedding_rows,
                    target,
                ),
            }
    return results, predictions


# Complete enumeration of the sign-flip null costs 2**n means, which is tractable
# for the 16-episode design but not for larger sealed tests (2**88 is ~3e26). Above
# the threshold the same null is sampled instead, with a fixed seed.
EXACT_SIGN_FLIP_MAX_N = 20
SIGN_FLIP_DRAWS = 200_000


def exact_sign_flip(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": float("nan"), "one_sided_p": float("nan")}
    observed = float(values.mean())
    if len(values) <= EXACT_SIGN_FLIP_MAX_N:
        null = np.empty(1 << len(values), dtype=float)
        for index, signs in enumerate(
            itertools.product((-1.0, 1.0), repeat=len(values))
        ):
            null[index] = float(np.mean(values * np.asarray(signs)))
        scheme = "exact episode-level sign flip"
        draws = int(null.size)
    else:
        generator = np.random.default_rng(PERMUTATION_SEED)
        signs = generator.choice(
            np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))
        )
        null = (signs * values).mean(axis=1)
        scheme = (
            f"sampled episode-level sign flip, {SIGN_FLIP_DRAWS} draws, "
            f"seed {PERMUTATION_SEED}"
        )
        draws = SIGN_FLIP_DRAWS
    return {
        "n": int(len(values)),
        "mean": observed,
        "one_sided_p": float(np.mean(null >= observed - 1e-15)),
        "null_mean": float(null.mean()),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "scheme": scheme,
        "null_draws": draws,
    }


def bootstrap_mean(values: np.ndarray, repeats: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(PERMUTATION_SEED)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def analyze_patching(
    run_dir: Path, rows: list[dict[str, Any]], counts: dict[str, Any]
) -> dict[str, Any]:
    path = run_dir / "activation_patch_generations.jsonl"
    records = read_jsonl(path)
    task_by_episode = {
        int(row["episode"]): row
        for row in rows
        if row["arm"] == "abc_context"
    }
    expected = counts["patch_records"]
    if len(records) != expected:
        raise ValueError(f"expected {expected} patch-generation records, got {len(records)}")
    by_key = {
        (
            int(row["episode"]),
            int(row["c_cases"]),
            str(row["recipient_arm"]),
            str(row["condition"]),
        ): row
        for row in records
    }
    if len(by_key) != expected:
        raise ValueError("patch-generation keys are not unique")

    self_pairs = []
    for episode in sorted({key[0] for key in by_key}):
        for k in (0, 4):
            for arm in ARMS:
                baseline = by_key[(episode, k, arm, "unpatched")]
                self_patch = by_key[(episode, k, arm, "self_selected_window")]
                if baseline["parsed"] and self_patch["parsed"]:
                    self_pairs.append(
                        float(baseline["predicted_poll"])
                        == float(self_patch["predicted_poll"])
                    )
    self_control = {
        "expected_pairs": counts["self_patch_pairs"],
        "parsed_pairs": len(self_pairs),
        "exact_poll_match_count": int(sum(self_pairs)),
        "exact_poll_match_rate": (
            float(np.mean(self_pairs)) if self_pairs else float("nan")
        ),
    }

    conditions = (
        "cross_selected_window",
        "cross_embedding_only",
        "cross_all_layers",
    )
    result: dict[str, Any] = {
        "record_count": len(records),
        "self_patch_control": self_control,
        "conditions": {},
    }
    for condition in conditions:
        result["conditions"][condition] = {}
        for k in (0, 4):
            c_to_w = []
            w_to_c = []
            episode_values = []
            for episode in sorted({key[0] for key in by_key}):
                correct_base = by_key[(episode, k, "abc_context", "unpatched")]
                wrong_base = by_key[(episode, k, "abc_wrong_context", "unpatched")]
                wrong_patched = by_key[(episode, k, "abc_wrong_context", condition)]
                correct_patched = by_key[(episode, k, "abc_context", condition)]
                needed = (correct_base, wrong_base, wrong_patched, correct_patched)
                if not all(row["parsed"] for row in needed):
                    continue
                task = task_by_episode[episode]
                sign = 1.0 if task["target_strong"] else -1.0
                forward = sign * (
                    float(wrong_patched["implied_slope"])
                    - float(wrong_base["implied_slope"])
                )
                reverse = sign * (
                    float(correct_base["implied_slope"])
                    - float(correct_patched["implied_slope"])
                )
                c_to_w.append(forward)
                w_to_c.append(reverse)
                episode_values.append(0.5 * (forward + reverse))
            symmetric = np.asarray(episode_values, dtype=float)
            test = exact_sign_flip(symmetric)
            test["bootstrap_ci95"] = bootstrap_mean(symmetric)
            test["correct_into_swapped_mean"] = (
                float(np.mean(c_to_w)) if c_to_w else float("nan")
            )
            test["swapped_into_correct_mean"] = (
                float(np.mean(w_to_c)) if w_to_c else float("nan")
            )
            result["conditions"][condition][f"k{k}"] = test
    return result


def final_analysis(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    selection: dict[str, Any],
    permutations: int,
    counts: dict[str, Any],
) -> dict[str, Any]:
    probe, predictions = evaluate_fixed_selection(
        rows=rows,
        features=features,
        selection=selection,
        permutations=permutations,
        counts=counts,
    )
    patching = analyze_patching(run_dir, rows, counts)
    primary = probe[PRIMARY_REPRESENTATION][PRIMARY_TARGET]
    representation_score = primary["test"]["abc_context"]["true_target"]["roc_auc"]
    representation_p = primary["correct_test_permutation"]["one_sided_p"]
    causal = patching["conditions"]["cross_selected_window"]["k0"]
    self_control = patching["self_patch_control"]
    gate_checks = {
        "primary_relational_regime_auc_above_chance": representation_score > 0.5,
        "primary_relational_permutation_p_below_0_05": representation_p < 0.05,
        f"primary_patch_complete_episode_count_at_least_"
        f"{counts['min_patch_episodes']}": causal["n"] >= counts["min_patch_episodes"],
        "primary_patch_exact_sign_flip_p_below_0_05": causal["one_sided_p"] < 0.05,
        "primary_patch_symmetric_mean_positive": causal["mean"] > 0,
        "both_patch_directions_positive": (
            causal["correct_into_swapped_mean"] > 0
            and causal["swapped_into_correct_mean"] > 0
        ),
        f"self_patch_parsed_pairs_at_least_{counts['min_self_patch_pairs']}": (
            self_control["parsed_pairs"] >= counts["min_self_patch_pairs"]
        ),
        "self_patch_exact_poll_match_rate_at_least_0_95": (
            self_control["exact_poll_match_rate"] >= 0.95
        ),
    }
    gate_passed = all(gate_checks.values())
    result = {
        "study": run_spec(run_dir)["study"],
        "status": "complete",
        "scope": (
            "paper-external diagnostic; linear label-token decodability and "
            "causal residual-stream patching at a development-selected layer window"
        ),
        "selection_sha256": file_sha256(run_dir / "development_selection.json"),
        "tasks_sha256": file_sha256(run_dir / "tasks.jsonl"),
        "protocol_sha256": file_sha256(
            run_dir / "relational_protocol_manifest.json"
        ),
        "label_states_sha256": file_sha256(run_dir / "label_token_states.npz"),
        "patch_generations_sha256": file_sha256(
            run_dir / "activation_patch_generations.jsonl"
        ),
        "permutation_repeats": permutations,
        "permutation_seed": PERMUTATION_SEED,
        "probe": probe,
        "patching": patching,
        "stability_gate": {
            "passed": gate_passed,
            "checks": gate_checks,
            "cross_family_replication": (
                "authorized_by_frozen_gate" if gate_passed else "not_authorized"
            ),
        },
    }
    write_json(run_dir / "relational_probe_results.json", result)
    content = "".join(
        json.dumps(clean_json(row), sort_keys=True) + "\n" for row in predictions
    )
    temporary = run_dir / "relational_selected_predictions.jsonl.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(run_dir / "relational_selected_predictions.jsonl")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("selection-only", "final"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--permutations", type=int, default=PERMUTATION_REPEATS)
    args = parser.parse_args()

    tasks, task_manifest = load_tasks(args.run_dir)
    counts = design_counts(task_manifest)
    rows = label_tasks(tasks)
    features = load_features(args.run_dir, rows)
    selection_path = args.run_dir / "development_selection.json"
    if args.mode == "selection-only":
        if selection_path.exists():
            raise FileExistsError(
                f"{selection_path} already exists; refusing to reselection after test access"
            )
        selection = select_on_development(
            run_dir=args.run_dir,
            rows=rows,
            features=features,
            counts=counts,
        )
        write_json(selection_path, selection)
        print(
            json.dumps(
                {
                    "selection": str(selection_path),
                    "primary": selection["primary"],
                    "patch_window": selection[
                        "causal_patch_window_hidden_state_layers"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "development_selection_complete_test_unread":
        raise ValueError("development selection status drifted")
    result = final_analysis(
        run_dir=args.run_dir,
        rows=rows,
        features=features,
        counts=counts,
        selection=selection,
        permutations=args.permutations,
    )
    primary = result["probe"][PRIMARY_REPRESENTATION][PRIMARY_TARGET]
    print(
        json.dumps(
            {
                "primary": primary,
                "primary_patch": result["patching"]["conditions"][
                    "cross_selected_window"
                ]["k0"],
                "stability_gate": result["stability_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
