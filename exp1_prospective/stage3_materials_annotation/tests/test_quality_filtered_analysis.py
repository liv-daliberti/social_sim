from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from exp1_prospective.stage3_materials_annotation import analyze_annotations as base
from exp1_prospective.stage3_materials_annotation import (
    analyze_annotations_quality_filtered as quality,
)


HERE = Path(__file__).resolve().parent.parent
GENERATED = HERE / "generated_v6"


class QualityFilteredAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assignments = json.loads(
            (GENERATED / "assignments.json").read_text(encoding="utf-8")
        )
        self.private_rows = base.read_jsonl(GENERATED / "private_key.jsonl")
        self.policy = json.loads(
            (GENERATED / "posthoc_exclusions.json").read_text(encoding="utf-8")
        )
        with (GENERATED / "responses_template.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            self.responses = list(csv.DictReader(handle))
        expected = {
            row["item_id"]: row["expected_conditional_direction"]
            for row in self.private_rows
        }
        for row in self.responses:
            row.update(
                {
                    "consent_confirmed": "yes",
                    "conditional_direction": expected[row["item_id"]],
                    "direction_confidence": "5",
                    "clarity": "5",
                    "plausibility": "5",
                    "usable_premise": "yes",
                }
            )
        for row in self.responses:
            if row["reviewer_id"] == "annotator_02":
                row["conditional_direction"] = "more_likely"

    def test_rule_detects_and_filters_trigger_without_deleting_raw(self) -> None:
        all_joined = base.validate_and_join(
            self.responses, self.assignments, self.private_rows
        )
        excluded, _patterns = quality.detect_exclusions(
            self.responses, self.policy
        )
        self.assertEqual(excluded, {"annotator_02"})
        filtered_rows, filtered_assignments = quality.filtered_inputs(
            self.responses, self.assignments, excluded
        )
        quality_joined = base.validate_and_join(
            filtered_rows, filtered_assignments, self.private_rows
        )
        self.assertEqual(len(all_joined), 126)
        self.assertEqual(len(quality_joined), 108)
        self.assertEqual(
            {row["reviewer_id"] for row in quality_joined},
            {
                "annotator_01",
                "annotator_03",
                "annotator_04",
                "annotator_05",
                "annotator_06",
                "annotator_07",
            },
        )

    def test_incomplete_reviewer_cannot_trigger_rule(self) -> None:
        partial = [
            row
            for row in self.responses
            if not (
                row["reviewer_id"] == "annotator_02"
                and row["item_id"]
                != next(
                    item["item_id"]
                    for item in self.responses
                    if item["reviewer_id"] == "annotator_02"
                )
            )
        ]
        excluded, _patterns = quality.detect_exclusions(partial, self.policy)
        self.assertNotIn("annotator_02", excluded)


if __name__ == "__main__":
    unittest.main()
