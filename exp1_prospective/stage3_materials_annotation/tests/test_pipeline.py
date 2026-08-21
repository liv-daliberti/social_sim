from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from exp1_prospective.stage3_materials_annotation import analyze_annotations


HERE = Path(__file__).resolve().parent.parent
GENERATED = HERE / "generated_v8"


class AnnotationPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assignments = json.loads(
            (GENERATED / "assignments.json").read_text(encoding="utf-8")
        )
        self.private_rows = analyze_annotations.read_jsonl(
            GENERATED / "private_key.jsonl"
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

    def test_perfect_registered_responses_pass(self) -> None:
        joined = analyze_annotations.validate_and_join(
            self.responses, self.assignments, self.private_rows
        )
        summary = analyze_annotations.summarize(joined)
        self.assertEqual(summary["rating_count"], 162)
        self.assertEqual(summary["passing_item_count"], 18)
        self.assertAlmostEqual(summary["fleiss_kappa_conditional_direction"], 1.0)
        self.assertTrue(summary["global_gate_pass"])

    def test_missing_rating_fails_closed(self) -> None:
        with self.assertRaises(SystemExit):
            analyze_annotations.validate_and_join(
                self.responses[:-1], self.assignments, self.private_rows
            )

    def test_ambiguous_rating_remains_in_denominator(self) -> None:
        self.responses[0]["conditional_direction"] = "ambiguous"
        joined = analyze_annotations.validate_and_join(
            self.responses, self.assignments, self.private_rows
        )
        summary = analyze_annotations.summarize(joined)
        self.assertEqual(summary["rating_count"], 162)
        self.assertEqual(summary["direction_correct_n"], 161)

    def test_every_item_has_frozen_june_10_context(self) -> None:
        public_rows = analyze_annotations.read_jsonl(
            GENERATED / "public_items.jsonl"
        )
        self.assertEqual(len(public_rows), 18)
        self.assertTrue(all(row["forecast_context"].strip() for row in public_rows))
        china = next(
            row for row in public_rows if "China GDP growth in Q2 2026" in row["question"]
        )
        self.assertIn("Q1 2026 at 5.0%", china["forecast_context"])


if __name__ == "__main__":
    unittest.main()
