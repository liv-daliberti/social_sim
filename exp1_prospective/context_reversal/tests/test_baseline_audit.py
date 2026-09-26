"""Feature leakage and pairing invariants for the frozen-material lexical audit."""
import copy

import pytest

from exp1_prospective.context_reversal.baseline_audit import FEATURES, audit, extract_examples


def fixture_plan():
    rows = []
    for family in ("one", "two", "three"):
        for context in ("positive", "negative", "broken", "masked"):
            for repeat in range(2):
                baseline = f"Instructions\n\nQUESTION\nWill the target win?\n\nRESOLUTION RULE\nCertified result.\n\nADDITIONAL CONTEXT\nThe resource serves {context} constituency."
                news = f"The {family} resource increased."
                rows.append({
                    "family_id": f"PRIVATE_FAMILY_{family}", "context_id": context,
                    "repeat": repeat, "material_status": "authored_development_unvalidated",
                    "baseline_prompt": baseline,
                    "update_templates": {"new_news": baseline + f"\n\nMESSAGE\n{news}\n\nRevise your previous probability as appropriate."},
                    "private_metadata": {"gold": "DO_NOT_READ_PRIVATE_GOLD"},
                })
    return rows


def test_features_are_visible_only_masked_removed_and_repeats_deduplicated():
    examples = extract_examples(fixture_plan())
    assert len(examples) == 9
    assert all(row["context_id"] != "masked" for row in examples)
    for row in examples:
        for feature in FEATURES:
            assert "DO_NOT_READ_PRIVATE_GOLD" not in row[feature]
            assert "PRIVATE_FAMILY" not in row[feature]


def test_identical_news_cannot_discriminate_within_held_out_families():
    report = audit(extract_examples(fixture_plan()))
    for task in report["tasks"].values():
        assert task["n_folds"] == 3
        assert task["training_majority_baseline"]["balanced_accuracy"] == .5
        for row in task["results"]:
            if row["feature_set"] in FEATURES[:2]:
                assert row["balanced_accuracy"] == .5
                assert row["family_all_contexts_correct_rate"] == 0


def test_unmatched_news_and_changed_repeat_prompts_are_rejected():
    rows = fixture_plan()
    changed = copy.deepcopy(rows)
    changed[0]["baseline_prompt"] += "changed repeat"
    with pytest.raises(ValueError, match="Repeat prompts differ"):
        extract_examples(changed)
    for row in rows:
        if row["family_id"] == "PRIVATE_FAMILY_one" and row["context_id"] == "broken":
            row["update_templates"]["new_news"] = row["update_templates"]["new_news"].replace("resource increased.", "resource decreased.")
    with pytest.raises(ValueError, match="Expected matched news_only"):
        extract_examples(rows)


def test_mechanism_class_holdout_uses_grouped_folds_without_feature_changes():
    examples = extract_examples(fixture_plan())
    original = copy.deepcopy(examples)
    groups = {"PRIVATE_FAMILY_one": "class_a", "PRIVATE_FAMILY_two": "class_a", "PRIVATE_FAMILY_three": "class_b"}
    report = audit(examples, groups)
    assert examples == original
    assert report["private_metadata_used_as_features"] is False
    assert report["mechanism_class_cv"]["n_mechanism_classes"] == 2
    assert report["mechanism_class_cv"]["class_family_counts"] == {"class_a": 2, "class_b": 1}
    for task in report["mechanism_class_cv"]["tasks"].values():
        assert task["n_folds"] == 2
        assert task["n_families"] == 3
        for row in task["results"]:
            if row["feature_set"] in FEATURES[:2]:
                assert row["balanced_accuracy"] == .5
