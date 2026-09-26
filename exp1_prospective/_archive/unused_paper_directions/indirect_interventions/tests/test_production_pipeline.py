from __future__ import annotations

import unittest

import numpy as np

from exp1_prospective.indirect_interventions.scripts.analyze import (
    cluster_bootstrap,
    condition_correct,
    holm_adjust,
    market_metrics,
)
from exp1_prospective.indirect_interventions.scripts.build_review_packet import (
    assignments,
    candidate_readability_errors,
)
from exp1_prospective.indirect_interventions.scripts.build_candidates import (
    DEFAULT_KEY_ENV,
    DEFAULT_MODEL,
    DEFAULT_PROTOCOL,
    _responses_output_text,
)
from exp1_prospective.indirect_interventions.scripts.run_updates import normalize_response
from exp1_prospective.indirect_interventions.scripts.validate_core_annotations import (
    fleiss_kappa,
)


class ProbabilityValidationTests(unittest.TestCase):
    def test_valid_equal_probabilities(self) -> None:
        value, parsed, error = normalize_response(
            '{"yes_prob": 0.42, "h1_posterior": 0.42, "rationale": "x"}'
        )
        self.assertEqual(value, 0.42)
        self.assertIsNotNone(parsed)
        self.assertIsNone(error)

    def test_mixed_zero_to_hundred_scale_fails_closed(self) -> None:
        value, parsed, error = normalize_response(
            '{"yes_prob": 42, "h1_posterior": 42, "rationale": "x"}'
        )
        self.assertIsNone(value)
        self.assertIsNone(parsed)
        self.assertIn("outside", error or "")

    def test_structural_probability_mismatch_fails_closed(self) -> None:
        value, _, error = normalize_response(
            '{"yes_prob": 0.42, "h1_posterior": 0.43, "rationale": "x"}'
        )
        self.assertIsNone(value)
        self.assertIn("differ", error or "")


class GeneratorConfigurationTests(unittest.TestCase):
    def test_gpt56_is_the_frozen_generator(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "gpt-5.6-sol")
        self.assertEqual(DEFAULT_PROTOCOL, "openai_responses")
        self.assertEqual(DEFAULT_KEY_ENV, "GPT56_AZURE_API_KEY")

    def test_responses_output_text_extraction(self) -> None:
        body = {
            "output": [{
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"families": []}'}
                ],
            }]
        }
        self.assertEqual(_responses_output_text(body), '{"families": []}')


class AssignmentTests(unittest.TestCase):
    def test_every_task_gets_three_distinct_raters(self) -> None:
        task_ids = [f"task_{index}" for index in range(130)]
        reviewers = assignments(
            task_ids,
            panel="full_context",
            reviewer_count=12,
            ratings_per_task=3,
            seed=7,
        )
        owners = {task_id: [] for task_id in task_ids}
        for reviewer in reviewers:
            for task_id in reviewer["task_ids"]:
                owners[task_id].append(reviewer["reviewer_id"])
        self.assertTrue(all(len(value) == len(set(value)) == 3 for value in owners.values()))
        loads = [len(reviewer["task_ids"]) for reviewer in reviewers]
        self.assertLessEqual(max(loads) - min(loads), 1)

    def test_development_assignments_never_repeat_a_market(self) -> None:
        group_ids = {
            f"market_{market}_task_{task}": f"market_{market}"
            for market in range(20)
            for task in range(9)
        }
        reviewers = assignments(
            list(group_ids),
            panel="full_context",
            reviewer_count=27,
            ratings_per_task=3,
            seed=20260813,
            group_ids=group_ids,
        )
        for reviewer in reviewers:
            groups = [group_ids[task_id] for task_id in reviewer["task_ids"]]
            self.assertEqual(len(groups), 20)
            self.assertEqual(len(groups), len(set(groups)))

    def test_undersized_no_repeat_panel_fails_closed(self) -> None:
        task_ids = [f"same_market_task_{index}" for index in range(9)]
        with self.assertRaisesRegex(ValueError, "requires at least 27"):
            assignments(
                task_ids,
                panel="full_context",
                reviewer_count=15,
                ratings_per_task=3,
                seed=7,
                group_ids={task_id: "same_market" for task_id in task_ids},
            )


class ReadabilityGateTests(unittest.TestCase):
    @staticmethod
    def candidate(*, evidence_words: int = 20, directional_words: int = 20) -> dict:
        words = lambda prefix, count: " ".join(f"{prefix}{index}" for index in range(count))
        return {
            "candidate_id": "readability_fixture",
            "evidence": {"headline": "headline", "text": words("e", evidence_words - 1)},
            "contexts": {
                "positive": {
                    "bridge_text": words("p", directional_words),
                    "decisive_clause": words("pc", 3),
                },
                "negative": {
                    "bridge_text": words("n", directional_words),
                    "decisive_clause": words("nc", 3),
                },
                "broken": {"bridge_text": words("b", 15)},
            },
        }

    def test_compact_candidate_passes(self) -> None:
        self.assertEqual(candidate_readability_errors(self.candidate()), [])

    def test_overlong_directional_bridge_fails(self) -> None:
        errors = candidate_readability_errors(self.candidate(directional_words=41))
        bridge_errors = [error for error in errors if "bridge=41 words" in error]
        self.assertEqual(len(bridge_errors), 2)

    def test_all_caps_bridge_abbreviation_fails(self) -> None:
        candidate = self.candidate()
        candidate["contexts"]["positive"]["bridge_text"] += " FOMC"
        errors = candidate_readability_errors(candidate)
        self.assertTrue(any("all-caps abbreviation(s)=FOMC" in error for error in errors))


class MetricTests(unittest.TestCase):
    def test_registered_thresholds_include_small_updates_as_wrong(self) -> None:
        self.assertTrue(condition_correct("positive", 0.03, 0.03))
        self.assertFalse(condition_correct("positive", 0.029, 0.03))
        self.assertTrue(condition_correct("negative", -0.03, 0.03))
        self.assertTrue(condition_correct("broken", 0.029, 0.03))
        self.assertFalse(condition_correct("broken", 0.03, 0.03))
        self.assertFalse(condition_correct("positive", None, 0.03))

    def test_invalid_pair_is_zero_contrast_and_zero_accuracy(self) -> None:
        rows = []
        for condition, delta in (
            ("positive", None),
            ("negative", -0.05),
            ("broken", 0.0),
            ("masked", 0.0),
        ):
            rows.append({
                "task_id": "market",
                "candidate_id": "family",
                "condition": condition,
                "delta_yes_prob": delta,
            })
        for condition, delta in (
            ("direct_pro", 0.05),
            ("direct_anti", -0.05),
            ("orthogonal", 0.0),
            ("literal_null", 0.0),
        ):
            rows.append({
                "task_id": "market",
                "candidate_id": f"control_{condition}",
                "condition": condition,
                "delta_yes_prob": delta,
            })
        metrics = market_metrics(rows, 0.03)["market"]
        self.assertEqual(metrics["reversal_contrast"], 0.0)
        self.assertAlmostEqual(metrics["triplet_accuracy"], 2 / 3)
        self.assertEqual(metrics["complete_triplet"], 0.0)

    def test_cluster_bootstrap_reproducible(self) -> None:
        values = {"a": 0.1, "b": 0.2, "c": 0.3}
        first = cluster_bootstrap(values, replicates=100, seed=99)
        second = cluster_bootstrap(values, replicates=100, seed=99)
        np.testing.assert_array_equal(first, second)

    def test_holm_adjustment_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.04})
        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])


class AgreementTests(unittest.TestCase):
    def test_perfect_fleiss_agreement(self) -> None:
        groups = [
            ["increase", "increase", "increase"],
            ["decrease", "decrease", "decrease"],
            ["no_effect", "no_effect", "no_effect"],
        ]
        self.assertEqual(fleiss_kappa(groups), 1.0)


if __name__ == "__main__":
    unittest.main()
