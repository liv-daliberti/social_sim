from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "exp1_prospective" / "stage3_materials_annotation"


def test_canonical_final_summary_regenerates(tmp_path: Path) -> None:
    output = tmp_path / "final_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(STUDY / "make_final_summary.py"),
            "--responses",
            str(STUDY / "data/exports/registered_20260827T204456Z.csv"),
            "--status",
            str(STUDY / "data/exports/status_20260827T204456Z.json"),
            "--private-key",
            str(STUDY / "generated_v6/private_key.jsonl"),
            "--policy",
            str(STUDY / "generated_v6/posthoc_exclusions.json"),
            "--output",
            str(output),
            "--protocol-version",
            "stage3_materials_annotation_v8",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(output.read_text())
    assert summary["analysis"] == "final_descriptive_materials_review"
    assert summary["analysis_status"] == "complete"
    assert summary["collection"]["closed"] is True
    assert summary["collection"]["completed_reviewer_count"] == 9
    assert summary["collection"]["included_reviewer_count"] == 8
    assert summary["collection"]["descriptive_target_met"] is True
    assert summary["excluded_reviewers"] == ["annotator_02"]
    assert summary["pooled_included"]["direction_correct_n"] == 108
    assert summary["pooled_included"]["direction_denom"] == 144
    assert summary["all_completed_sensitivity"]["direction_correct_n"] == 115
    assert summary["all_completed_sensitivity"]["direction_denom"] == 162
    assert summary["partial_reviewers"] == {}
    for obsolete in ("final_gate_available", "reason_final_gate_unavailable", "status"):
        assert obsolete not in summary
