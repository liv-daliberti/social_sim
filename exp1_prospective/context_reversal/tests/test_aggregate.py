"""Aggregation must retain the whole prescribed roster, including failed trials."""
import json

import pytest

from exp1_prospective.context_reversal.aggregate import MODEL_KEYS, aggregate, main
from exp1_prospective.context_reversal.analyze import CONTEXTS, MATERIAL_STATUS, analyze


def write_summaries(root):
    design = [
        {
            "trial_id": f"{family}-{context}", "family_id": family,
            "domain": "fixture", "context_id": context, "repeat": 0,
            "material_status": MATERIAL_STATUS,
        }
        for family in ("a", "b") for context in CONTEXTS
    ]
    for model in MODEL_KEYS:
        summary = analyze(design, [], model, bootstrap_draws=20)
        summary["provenance"] = {"design": {"sha256": "a" * 64}}
        path = root / model / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(summary))


def test_full_roster_is_preserved_even_when_every_response_is_missing(tmp_path):
    write_summaries(tmp_path)
    report = aggregate(tmp_path)
    assert report["prescribed_models"] == list(MODEL_KEYS)
    assert list(report["models"]) == list(MODEL_KEYS)
    for model in MODEL_KEYS:
        summary = report["models"][model]
        assert summary["coverage"]["totals"]["valid"] == 0
        assert summary["coverage"]["totals"]["expected"] == 32
        assert summary["primary"]["direction_correct"]["pooled"]["n_planned"] == 4
        assert summary["primary"]["paired_reversal"]["both_directions_correct"]["n_planned"] == 2
        assert summary["broken_stability"]["raw_new_news"]["status"] == "indeterminate"


def test_absent_model_refuses_aggregation_and_does_not_write_partial_summary(tmp_path):
    write_summaries(tmp_path)
    (tmp_path / MODEL_KEYS[1] / "summary.json").unlink()
    with pytest.raises(ValueError, match=MODEL_KEYS[1]):
        aggregate(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(["--results-root", str(tmp_path)])
    assert error.value.code == 2
    assert not (tmp_path / "development_summary.json").exists()
    assert not (tmp_path / "development_summary.md").exists()


@pytest.mark.parametrize("change", ["design", "model_key"])
def test_mismatched_designs_or_model_keys_fail_closed(tmp_path, change):
    write_summaries(tmp_path)
    path = tmp_path / MODEL_KEYS[2] / "summary.json"
    summary = json.loads(path.read_text())
    if change == "design":
        summary["provenance"]["design"]["sha256"] = "b" * 64
    else:
        summary["model_key"] = "unexpected_model"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        aggregate(tmp_path)


def test_cli_uses_requested_root_and_emits_development_caveat_and_controls(tmp_path):
    write_summaries(tmp_path)
    assert main(["--results-root", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "development_summary.json").read_text())
    assert report["analysis_status"] == "development_unvalidated_descriptive_only"
    markdown = (tmp_path / "development_summary.md").read_text()
    assert "unvalidated development materials" in markdown
    assert "No-news and repeated-news drift" in markdown
    assert "Drift-adjusted paired reversal" in markdown
    assert "Missing / invalid records" in markdown
    assert "Nonsignificance is not equivalence" in markdown
    assert markdown.index(MODEL_KEYS[0]) < markdown.index(MODEL_KEYS[1]) < markdown.index(MODEL_KEYS[2])
