"""Scientific invariants of the context-reversal development analysis."""
import copy
import hashlib
import json

import numpy as np
import pytest

from exp1_prospective.context_reversal.analyze import (
    CONDITIONS,
    CONTEXTS,
    MATERIAL_STATUS,
    FamilyBootstrap,
    Sample,
    analyze,
    main,
)


def fixture_records(families=("a", "b", "c"), repeats=2):
    design, responses = [], []
    for family in families:
        for repeat in range(repeats):
            for context in CONTEXTS:
                unit = {
                    "trial_id": f"{family}-{context}-{repeat}",
                    "family_id": family,
                    "domain": "synthetic",
                    "context_id": context,
                    "repeat": repeat,
                    "material_status": MATERIAL_STATUS,
                    "baseline_prompt": "fixture baseline",
                    "update_templates": {arm: "fixture update" for arm in CONDITIONS[1:]},
                }
                design.append(unit)
                updates = {"positive": .6, "negative": .4, "broken": .5, "masked": .53}
                for condition in CONDITIONS:
                    value = {
                        "baseline": .5,
                        "new_news": updates[context],
                        "no_news": .51,
                        "repeated_news": .505,
                    }[condition]
                    responses.append({
                        **{key: unit[key] for key in ("trial_id", "family_id", "domain", "context_id", "repeat")},
                        "model_key": "fixture-model",
                        "stage": "baseline" if condition == "baseline" else "update",
                        "condition": condition,
                        "probability": value,
                        "status": "ok",
                    })
    return design, responses


def report_for(design, responses):
    return analyze(design, responses, "fixture-model", bootstrap_draws=400, seed=20260921)


def test_directions_pairs_and_shared_baseline_drift_adjustment():
    design, responses = fixture_records()
    report = report_for(design, responses)
    primary = report["primary"]
    assert primary["direction_correct"]["pooled"]["estimate"] == 1
    assert primary["direction_correct"]["pooled"]["n_planned"] == 12
    assert primary["paired_reversal"]["both_directions_correct"]["estimate"] == 1
    assert primary["paired_reversal"]["both_directions_correct"]["n_planned"] == 6
    assert primary["paired_reversal"]["positive_minus_negative_pp"]["estimate"] == pytest.approx(20)
    assert report["drift_adjusted"]["by_context"]["positive"]["signed_mean_pp"]["estimate"] == pytest.approx(9)
    assert report["drift_adjusted"]["by_context"]["negative"]["signed_mean_pp"]["estimate"] == pytest.approx(-11)
    assert report["raw_updates"]["positive"]["no_news"]["signed_mean_pp"]["estimate"] == pytest.approx(1)
    assert report["raw_updates"]["positive"]["repeated_news"]["signed_mean_pp"]["estimate"] == pytest.approx(.5)
    assert report["baseline_probability_percent"]["positive"]["estimate"] == 50
    assert report["broken_stability"]["raw_new_news"]["status"] == "criterion_met"
    assert report["broken_stability"]["drift_adjusted"]["status"] == "criterion_met"
    assert "masked" not in report["primary"]["direction_correct"]
    assert report["raw_updates"]["masked"]["new_news"]["signed_mean_pp"]["estimate"] == pytest.approx(3)


def test_missing_whole_family_keeps_all_planned_trial_and_pair_denominators():
    design, responses = fixture_records()
    responses = [row for row in responses if row["family_id"] != "c"]
    report = report_for(design, responses)
    trial = report["primary"]["direction_correct"]["pooled"]
    pair = report["primary"]["paired_reversal"]["both_directions_correct"]
    assert (trial["n_success"], trial["n_planned"], trial["n_missing"]) == (8, 12, 4)
    assert trial["estimate"] == pytest.approx(2 / 3)
    assert (pair["n_success"], pair["n_planned"], pair["n_missing"]) == (4, 6, 2)
    assert report["coverage"]["totals"]["missing_record"] == 32
    observed_only = report["raw_updates"]["positive"]["new_news"]["signed_mean_pp"]
    assert observed_only["estimate"] == pytest.approx(10)  # Missing forecasts are not zeros.
    assert observed_only["n_observed"] == 4
    assert report["broken_stability"]["raw_new_news"]["status"] == "indeterminate"


def test_zero_wrong_sign_and_invalid_trials_fail_without_silent_dropping():
    design, responses = fixture_records()
    for row in responses:
        if row["condition"] != "new_news" or row["context_id"] != "positive":
            continue
        if row["family_id"] == "a":
            row["probability"] = .5  # Exact zero is a failure, not an excluded abstention.
        elif row["family_id"] == "b":
            row["probability"] = .4  # Wrong direction.
        else:
            row["probability"] = None
            row["status"] = "parse_error"
    report = report_for(design, responses)
    pooled = report["primary"]["direction_correct"]["pooled"]
    assert (pooled["n_success"], pooled["n_planned"], pooled["n_missing"]) == (6, 12, 2)
    assert pooled["estimate"] == .5
    assert report["primary"]["paired_reversal"]["both_directions_correct"]["estimate"] == 0
    assert report["coverage"]["by_context_condition"]["positive"]["new_news"]["reason:parse_error"] == 2


def test_missing_baseline_blocks_movement_even_if_update_probability_present():
    design, responses = fixture_records()
    responses = [row for row in responses if not (row["trial_id"] == "a-positive-0" and row["condition"] == "baseline")]
    report = report_for(design, responses)
    assert report["primary"]["direction_correct"]["positive"]["n_missing"] == 1
    assert report["drift_adjusted"]["direction_correct"]["positive"]["n_missing"] == 1
    assert report["coverage"]["by_context_condition"]["positive"]["new_news"]["valid"] == 6
    assert report["raw_updates"]["positive"]["new_news"]["signed_mean_pp"]["n_observed"] == 5


def test_bootstrap_resamples_whole_families_not_individual_repeats():
    engine = FamilyBootstrap(["a", "b", "c"], draws=1000, seed=123)
    once = [Sample("a", -10), Sample("b", 2), Sample("c", 20)]
    repeated = [item for item in once for _ in range(25)]
    first = engine.summarize(once)
    second = engine.summarize(repeated)
    assert first["estimate"] == second["estimate"]
    assert first["ci95"] == second["ci95"]
    expected = np.asarray([-10, 2, 20])[engine.indices].mean(axis=1)
    assert first["ci95"] == pytest.approx(np.quantile(expected, [.025, .975]))


def test_opposing_large_broken_movements_do_not_establish_stability():
    design, responses = fixture_records(repeats=2)
    for row in responses:
        if row["context_id"] == "broken" and row["condition"] == "new_news":
            row["probability"] = .7 if row["repeat"] == 0 else .3
    report = report_for(design, responses)
    diagnostic = report["broken_stability"]["raw_new_news"]
    assert diagnostic["signed_mean_equivalent"] is True
    assert diagnostic["absolute_movement_bounded"] is False
    assert diagnostic["status"] == "criterion_not_met"
    assert report["raw_updates"]["broken"]["new_news"]["mean_absolute_pp"]["estimate"] == pytest.approx(20)


def test_duplicate_responses_and_design_metadata_mismatches_are_rejected():
    design, responses = fixture_records()
    with pytest.raises(ValueError, match="Duplicate response"):
        report_for(design, responses + [responses[0]])
    changed = copy.deepcopy(responses)
    changed[0]["context_id"] = "masked"
    with pytest.raises(ValueError, match="Response/design mismatch"):
        report_for(design, changed)
    with pytest.raises(ValueError, match="all four contexts"):
        report_for(design[1:], responses)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_invalid_probability_is_explicit_missing_failure(value):
    design, responses = fixture_records()
    for row in responses:
        if row["trial_id"] == "a-positive-0" and row["condition"] == "new_news":
            row["probability"] = value
    report = report_for(design, responses)
    assert report["primary"]["direction_correct"]["pooled"]["n_success"] == 11
    assert report["primary"]["direction_correct"]["pooled"]["n_missing"] == 1
    assert report["coverage"]["totals"]["reason:invalid_probability"] == 1


def test_no_responses_still_reports_the_plan_and_indeterminate_equivalence():
    design, _ = fixture_records()
    report = report_for(design, [])
    assert report["primary"]["direction_correct"]["pooled"]["estimate"] == 0
    assert report["primary"]["paired_reversal"]["both_directions_correct"]["n_planned"] == 6
    assert report["raw_updates"]["broken"]["new_news"]["signed_mean_pp"]["estimate"] is None
    assert report["broken_stability"]["raw_new_news"]["status"] == "indeterminate"
    json.dumps(report, allow_nan=False)


def test_cli_writes_auditable_json_and_markdown(tmp_path):
    design, responses = fixture_records()
    design_path, response_path = tmp_path / "design.jsonl", tmp_path / "responses.jsonl"
    design_path.write_text("".join(json.dumps(row) + "\n" for row in design))
    for row in responses:
        row["input_sha256"] = hashlib.sha256(design_path.read_bytes()).hexdigest()
    response_path.write_text("".join(json.dumps(row) + "\n" for row in responses))
    output = tmp_path / "analysis"
    assert main(["--design", str(design_path), "--responses", str(response_path), "--model-key", "fixture-model", "--output", str(output), "--bootstrap-draws", "100"]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["provenance"]["design"]["sha256"]
    markdown = (output / "summary.md").read_text()
    assert "Missing record" in markdown
    assert "Nonsignificance does not establish equivalence" in markdown
    assert "Masked has no expected direction" in markdown


def test_cli_rejects_responses_from_another_design(tmp_path):
    design, responses = fixture_records()
    design_path, response_path = tmp_path / "design.jsonl", tmp_path / "responses.jsonl"
    design_path.write_text("".join(json.dumps(row) + "\n" for row in design))
    for row in responses:
        row["input_sha256"] = "wrong-design-hash"
    response_path.write_text("".join(json.dumps(row) + "\n" for row in responses))
    with pytest.raises(SystemExit) as error:
        main(["--design", str(design_path), "--responses", str(response_path), "--model-key", "fixture-model", "--output", str(tmp_path / "analysis")])
    assert error.value.code == 2
    assert not (tmp_path / "analysis").exists()
