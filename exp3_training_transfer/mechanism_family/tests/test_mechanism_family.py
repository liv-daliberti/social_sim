from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contrasts import (  # noqa: E402
    hierarchical_base_contrasts,
    hierarchical_disclosure_benefit_contrasts,
    hierarchical_model_benefit_contrasts,
    hierarchical_structureless_contrasts,
)
from audit_canaries import (  # noqa: E402
    audit_base_evaluations,
    is_base_canary_ledger,
)
from make_paper_outputs import (  # noqa: E402
    add_overall_rows,
    render_secondary_table,
    validate_score_rows,
    validate_full_ledger,
)
from make_dataset import make_bundle, make_row  # noqa: E402
from output_contract import (  # noqa: E402
    FORECAST_ARRAY_GBNF,
    forecast_array_grammar,
    parse_forecast_array,
)
from prompt import MECHANISM_BEGIN, OUTPUT_PREFIX, OUTPUT_SUFFIX, parse_forecasts  # noqa: E402
from report import hierarchical_arm_contrasts, score_records  # noqa: E402
from worlds import (  # noqa: E402
    BY_NAME,
    TEST,
    TRAIN,
    oracle_forecast,
    response_vector,
)


class MechanismFamilyTests(unittest.TestCase):
    def test_base_canary_auditor_resolves_and_checks_six_replacements(self):
        models = ("qwen3_4b", "qwen3_8b", "llama3_1_8b")
        blocks = (
            "mixed_composition",
            "nonlinear_composition",
            "parameter_extrapolation",
            "topology_composition",
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            reports = temporary / "reports"
            reports.mkdir()
            submissions, job_id_map, states = [], {}, {}
            for index, (disclosure, model) in enumerate(
                    (d, m) for d in ("disclosed", "undisclosed") for m in models):
                old_job_id, new_job_id = str(100 + index), str(200 + index)
                job_id_map[old_job_id] = new_job_id
                states[new_job_id] = "COMPLETED"
                submissions.append({
                    "kind": "base_evaluation",
                    "job_id": old_job_id,
                    "disclosure": disclosure,
                    "model": model,
                    "environment": {
                        "MAX_MODEL_LEN": "3072",
                        "MAX_TOKENS": "192",
                        "STOCHASTIC_N": "5",
                        "STOCHASTIC_TEMP": "0.7",
                        "STRUCTURED_OUTPUT": "forecast_array",
                    },
                })
                root = reports / f"base_{disclosure}_{model}_j{new_job_id}"
                root.mkdir()
                (root / "greedy.json").write_text("[]\n")
                (root / "stochastic_n5.json").write_text("[]\n")
                for decode, draws_per_task, filename in (
                    ("greedy", 1, "greedy.scores.summary.json"),
                    ("stochastic", 5, "stochastic_n5.scores.summary.json"),
                ):
                    rows = [{
                        "model": model,
                        "disclosure": disclosure,
                        "arm": "base",
                        "seed": 0,
                        "decode": decode,
                        "block": block,
                        "k": k,
                        "n_tasks": 60,
                        "n_draws": 60 * draws_per_task,
                        "parse_rate": 1.0,
                    } for block in blocks for k in (3, 6, 9)]
                    (root / filename).write_text(json.dumps(rows))
            ledger = {
                "protocol": "c3_mechanism_disclosure_v4",
                "structured_decoding": {
                    "kind": "gbnf", "backend": "xgrammar", "value_constraints": "none"
                },
                "submissions": submissions,
                "base_evaluation_replacement": {"job_id_map": job_id_map},
            }
            ledger_path = temporary / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            result = audit_base_evaluations(
                ledger_path, reports=reports, states=states
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(result["jobs"]), 6)
            self.assertEqual(
                {item["job_id"] for item in result["jobs"]}, set(job_id_map.values())
            )
            self.assertTrue(all(item["greedy_scored_draws"] == 720
                                for item in result["jobs"]))
            self.assertTrue(all(item["stochastic_scored_draws"] == 3600
                                for item in result["jobs"]))

    def test_full_grid_cannot_be_selected_as_base_canary_ledger(self):
        canary = {
            "submissions": [
                {
                    "kind": "training",
                    "arm": "causal_family",
                    "seed": 42,
                }
                for _ in range(6)
            ] + [
                {"kind": "base_evaluation"}
                for _ in range(6)
            ]
        }
        self.assertTrue(is_base_canary_ledger(canary))
        full_grid = json.loads(json.dumps(canary))
        full_grid["submissions"].extend(
            {"kind": "training", "arm": "population_prior", "seed": 43}
            for _ in range(36)
        )
        self.assertFalse(is_base_canary_ledger(full_grid))

    def test_full_ledger_validator_requires_exact_registered_roster(self):
        disclosures = ("disclosed", "undisclosed")
        models = ("qwen3_4b", "qwen3_8b", "llama3_1_8b")
        submissions = []
        job_id = 1000
        for disclosure in disclosures:
            for model in models:
                for arm in ("causal_family", "population_prior"):
                    for seed in (42, 43, 44):
                        submissions.append({
                            "kind": "training",
                            "job_id": str(job_id),
                            "disclosure": disclosure,
                            "model": model,
                            "arm": arm,
                            "seed": seed,
                        })
                        job_id += 1
            for seed in (42, 43, 44):
                submissions.append({
                    "kind": "training",
                    "job_id": str(job_id),
                    "disclosure": disclosure,
                    "model": "qwen3_4b",
                    "arm": "structureless",
                    "seed": seed,
                })
                job_id += 1
            for model in models:
                submissions.append({
                    "kind": "base_evaluation",
                    "job_id": str(job_id),
                    "disclosure": disclosure,
                    "model": model,
                })
                job_id += 1
        ledger = {
            "protocol": "c3_mechanism_disclosure_v4",
            "structured_decoding": {
                "kind": "gbnf", "backend": "xgrammar", "value_constraints": "none"
            },
            "submissions": submissions,
        }
        validate_full_ledger(ledger)
        duplicate = json.loads(json.dumps(ledger))
        first = duplicate["submissions"][0]
        duplicate["submissions"][1].update({
            key: first[key] for key in ("disclosure", "model", "arm", "seed")
        })
        with self.assertRaises(AssertionError):
            validate_full_ledger(duplicate)

    def test_endpoint_score_validator_requires_exact_paired_grid(self):
        item = {
            "kind": "training",
            "job_id": "1234",
            "disclosure": "undisclosed",
            "model": "qwen3_8b",
            "arm": "causal_family",
            "seed": 43,
        }
        worlds = {
            "topology_composition": "mediated_feedback_linear",
            "nonlinear_composition": "mediated_saturating",
            "mixed_composition": "persistent_threshold",
            "parameter_extrapolation": "feedback_saturating_extrap",
        }
        rows = []
        for block, world in worlds.items():
            for horizon in (3, 6, 9):
                for episode in range(60):
                    task_id = f"{world}:{73000000 + episode}:{horizon}"
                    for draw in range(5):
                        rows.append({
                            "model": "qwen3_8b",
                            "disclosure": "undisclosed",
                            "arm": "causal_family",
                            "seed": 43,
                            "decode": "stochastic",
                            "block": block,
                            "world": world,
                            "k": horizon,
                            "task_id": task_id,
                            "draw": draw,
                        })
        contract = validate_score_rows(rows, item, "stochastic")
        self.assertEqual(len(contract), 720)

        missing = rows[:-1]
        with self.assertRaises(AssertionError):
            validate_score_rows(missing, item, "stochastic")

        drifted = [dict(row) for row in rows]
        drifted[0]["model"] = "qwen3_4b"
        with self.assertRaises(AssertionError):
            validate_score_rows(drifted, item, "stochastic")

    def test_structured_output_grammar_has_exact_forecast_arity(self):
        self.assertEqual(FORECAST_ARRAY_GBNF, forecast_array_grammar(10))
        root = FORECAST_ARRAY_GBNF.splitlines()[0]
        self.assertEqual(root.count("number"), 10)
        self.assertIn('"\\\"forecasts\\\""', root)
        self.assertNotIn("minimum", FORECAST_ARRAY_GBNF)
        self.assertNotIn("maximum", FORECAST_ARRAY_GBNF)
        with self.assertRaises(ValueError):
            forecast_array_grammar(0)

    def test_strict_compact_output_contract(self):
        self.assertEqual(
            parse_forecast_array('{"forecasts":[1,2,3]}', 3, (0, 10)),
            [1.0, 2.0, 3.0],
        )
        invalid = (
            '{"forecast_A":1,"forecast_B":2,"forecast_C":3}',
            'answer: {"forecasts":[1,2,3]}',
            '{"forecasts":[1,2]}',
            '{"forecasts":[1,2,3],"note":"x"}',
            '{"forecasts":[1,2,3],"forecasts":[4,5,6]}',
            '{"forecasts":[true,2,3]}',
            '{"forecasts":[1,NaN,3]}',
        )
        for response in invalid:
            with self.subTest(response=response):
                self.assertIsNone(parse_forecast_array(response, 3, (0, 10)))

    def test_prompt_contract_uses_no_non_json_placeholders(self):
        row = make_row(TEST[0], 73_000_000, 9, "disclosed", "causal_family",
                       evaluation=True)
        prompt = row["input"]
        self.assertNotIn("<A>", prompt)
        self.assertIn(f"Start your response with {OUTPUT_PREFIX}", prompt)
        self.assertIn(f"After the tenth number, write {OUTPUT_SUFFIX} and stop.", prompt)

    def test_train_test_structures_are_disjoint(self):
        train = {world.mechanism_key for world in TRAIN}
        test = {world.mechanism_key for world in TEST}
        self.assertFalse(train & test)
        extrap = BY_NAME["feedback_saturating_extrap"]
        self.assertGreater(extrap.gain_lo, max(world.gain_hi for world in TRAIN))
        self.assertGreater(extrap.noise, max(world.noise for world in TRAIN))

    def test_disclosure_changes_only_structural_text(self):
        world = TEST[0]
        disclosed = make_row(world, 73_000_000, 9, "disclosed", "causal_family",
                             evaluation=True)
        blind = make_row(world, 73_000_000, 9, "undisclosed", "causal_family",
                         evaluation=True)
        d_ref, u_ref = json.loads(disclosed["reference"]), json.loads(blind["reference"])
        self.assertEqual(d_ref["numeric_hash"], u_ref["numeric_hash"])
        d_ref.pop("disclosure")
        u_ref.pop("disclosure")
        self.assertEqual(d_ref, u_ref)
        self.assertIn(MECHANISM_BEGIN, disclosed["input"])
        self.assertNotIn(MECHANISM_BEGIN, blind["input"])

    def test_confirmatory_arms_have_identical_prompts(self):
        world = TRAIN[2]
        causal = make_row(world, 31_000_010, 6, "undisclosed", "causal_family")
        prior = make_row(world, 31_000_010, 6, "undisclosed", "population_prior")
        self.assertEqual(causal["input"], prior["input"])
        causal_ref = json.loads(causal["reference"])
        prior_ref = json.loads(prior["reference"])
        self.assertNotEqual(causal_ref["response_targets"], prior_ref["response_targets"])

    def test_structureless_removes_driver_signal_and_intervention_target(self):
        world = TRAIN[1]
        causal_bundle, _ = make_bundle(world, 31_000_100, 9, structureless=False)
        control_bundle, _ = make_bundle(world, 31_000_100, 9, structureless=True)
        self.assertEqual(causal_bundle["target_observed"], control_bundle["target_observed"])
        self.assertNotEqual(causal_bundle["target_inputs"], control_bundle["target_inputs"])
        row = make_row(world, 31_000_100, 9, "undisclosed", "structureless")
        ref = json.loads(row["reference"])
        self.assertTrue(np.allclose(ref["response_targets"], 0.0))

    def test_gold_round_trip_and_scorer(self):
        row = make_row(TEST[1], 74_000_000, 9, "undisclosed", "causal_family",
                       evaluation=True)
        ref = json.loads(row["reference"])
        parsed = parse_forecasts(ref["gold"], ref["scenario_labels"], ref["clip"])
        self.assertTrue(np.allclose(parsed, ref["targets"], atol=1e-3))
        records = [{"output": [ref["gold"], ref["gold"]],
                    "reference": row["reference"], "temperature": 0.7}]
        scored = score_records(records, {"model": "test", "disclosure": "undisclosed",
                                         "arm": "causal_family", "seed": 42})
        self.assertEqual(len(scored), 2)
        self.assertTrue(all(item["parsed"] for item in scored))
        self.assertLess(max(item["response_mae"] for item in scored), 1e-3)
        self.assertTrue(all(item["mechanism_correct"] for item in scored))
        invalid = score_records(
            [{"output": ["not json"], "reference": row["reference"], "temperature": 0.0}],
            {"model": "test", "disclosure": "undisclosed",
             "arm": "causal_family", "seed": 42},
        )[0]
        self.assertFalse(invalid["parsed"])
        self.assertEqual(invalid["response_mae"], ref["clip"][1] - ref["clip"][0])

    def test_blind_oracle_beats_population_floor(self):
        world = TEST[2]
        row = make_row(world, 75_000_000, 9, "undisclosed", "causal_family",
                       evaluation=True)
        ref = json.loads(row["reference"])
        oracle, _ = oracle_forecast(
            world, ref["calibrations"], np.asarray(ref["target_inputs"]),
            np.asarray(ref["target_observed"]), disclosed=False,
        )
        truth = np.asarray(ref["truth_targets"])
        prior = np.asarray(ref["prior_targets"])
        oracle_error = np.mean(np.abs(response_vector(oracle) - response_vector(truth)))
        prior_error = np.mean(np.abs(response_vector(prior) - response_vector(truth)))
        self.assertLess(oracle_error, prior_error)

    def test_hierarchical_contrast_clusters_draws_inside_seed(self):
        rows = []
        for seed in (42, 43, 44):
            for arm, error in (("causal_family", 1.0), ("population_prior", 3.0)):
                for task in ("a", "b", "c"):
                    for draw in range(5):
                        rows.append({
                            "model": "m", "disclosure": "undisclosed", "decode": "stochastic",
                            "block": "b", "seed": seed, "arm": arm, "task_id": task,
                            "parsed": True, "response_mae": error + 0.01 * draw,
                        })
        result = hierarchical_arm_contrasts(rows, repetitions=200, bootstrap_seed=7)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["estimate"], 2.0)
        self.assertEqual(result[0]["training_seeds"], [42, 43, 44])

    def test_overall_estimand_pools_blocks_without_task_id_collisions(self):
        rows = []
        for seed in (42, 43, 44):
            for block, causal_error in (("topology_composition", 1.0),
                                        ("nonlinear_composition", 3.0)):
                for arm, error in (("causal_family", causal_error),
                                   ("population_prior", 5.0)):
                    rows.append({
                        "model": "m", "disclosure": "undisclosed",
                        "decode": "greedy", "block": block, "seed": seed,
                        "arm": arm, "task_id": "same-local-id",
                        "parsed": True, "response_mae": error,
                    })
        analysis_rows = add_overall_rows(rows)
        self.assertEqual(len(analysis_rows), 2 * len(rows))
        pooled_ids = {
            row["task_id"] for row in analysis_rows if row["block"] == "overall"
        }
        self.assertEqual(
            pooled_ids,
            {
                "topology_composition::same-local-id",
                "nonlinear_composition::same-local-id",
            },
        )
        results = hierarchical_arm_contrasts(
            analysis_rows, repetitions=200, bootstrap_seed=7
        )
        overall = next(row for row in results if row["block"] == "overall")
        self.assertAlmostEqual(overall["estimate"], 3.0)

    def test_registered_model_disclosure_base_and_structureless_contrasts(self):
        rows = []
        errors = {
            "disclosed": {
                "qwen3_4b": {"causal_family": 2.0, "population_prior": 4.0},
                "qwen3_8b": {"causal_family": 1.0, "population_prior": 5.0},
                "llama3_1_8b": {"causal_family": 1.0, "population_prior": 4.0},
            },
            "undisclosed": {
                "qwen3_4b": {"causal_family": 3.0, "population_prior": 4.0},
                "qwen3_8b": {"causal_family": 2.0, "population_prior": 5.0},
                "llama3_1_8b": {"causal_family": 2.0, "population_prior": 4.0},
            },
        }
        for disclosure, models in errors.items():
            for model in models:
                for task in ("a", "b", "c"):
                    for draw in range(5):
                        rows.append({
                            "model": model, "disclosure": disclosure,
                            "decode": "stochastic", "block": "b", "seed": 0,
                            "arm": "base", "task_id": task,
                            "response_mae": 6.0 + 0.01 * draw,
                        })
            for seed in (42, 43, 44):
                for model, arms in models.items():
                    for arm, error in arms.items():
                        for task in ("a", "b", "c"):
                            for draw in range(5):
                                rows.append({
                                    "model": model, "disclosure": disclosure,
                                    "decode": "stochastic", "block": "b", "seed": seed,
                                    "arm": arm, "task_id": task,
                                    "response_mae": error + 0.01 * draw,
                                })
                for task in ("a", "b", "c"):
                    for draw in range(5):
                        rows.append({
                            "model": "qwen3_4b", "disclosure": disclosure,
                            "decode": "stochastic", "block": "b", "seed": seed,
                            "arm": "structureless", "task_id": task,
                            "response_mae": 5.0 + 0.01 * draw,
                        })

        args = (rows, 200, 7)
        base = hierarchical_base_contrasts(*args)
        q8_disclosed = next(
            row for row in base
            if row["model"] == "qwen3_8b" and row["disclosure"] == "disclosed"
        )
        self.assertAlmostEqual(q8_disclosed["estimate"], 5.0)
        structureless = hierarchical_structureless_contrasts(*args)
        self.assertAlmostEqual(structureless[0]["estimate"], 3.0)
        model = hierarchical_model_benefit_contrasts(*args)
        scale = next(row for row in model if "minus_qwen3_4b" in row["contrast"]
                     and row["disclosure"] == "disclosed")
        architecture = next(row for row in model if "minus_llama3_1_8b" in row["contrast"]
                            and row["disclosure"] == "disclosed")
        self.assertAlmostEqual(scale["estimate"], 2.0)
        self.assertAlmostEqual(architecture["estimate"], 1.0)
        disclosure = hierarchical_disclosure_benefit_contrasts(*args)
        q8 = next(row for row in disclosure if row["model"] == "qwen3_8b")
        self.assertAlmostEqual(q8["estimate"], 1.0)

    def test_secondary_table_exposes_registered_overall_contrasts(self):
        def effect(**fields):
            return {
                "decode": "greedy",
                "block": "overall",
                "estimate": 1.25,
                "ci95_low": 0.5,
                "ci95_high": 2.0,
                **fields,
            }

        result = {
            "hierarchical_base_contrasts": [
                effect(model="qwen3_8b", disclosure="disclosed")
            ],
            "hierarchical_structureless_contrasts": [
                effect(model="qwen3_4b", disclosure="undisclosed")
            ],
            "hierarchical_model_benefit_contrasts": [
                effect(
                    contrast="qwen3_8b_minus_qwen3_4b_causal_benefit",
                    disclosure="disclosed",
                )
            ],
            "hierarchical_disclosure_benefit_contrasts": [
                effect(
                    model="llama3_1_8b",
                    contrast="disclosed_minus_undisclosed_causal_benefit",
                )
            ],
        }
        rendered = render_secondary_table(result)
        self.assertIn("Base $-$ causal & Disclosed / Qwen3-8B", rendered)
        self.assertIn(
            "Structureless $-$ causal & Undisclosed / Qwen3-4B", rendered
        )
        self.assertIn("Qwen3-8B $-$ Qwen3-4B causal benefit", rendered)
        self.assertIn(
            "Disclosed $-$ undisclosed causal benefit & Llama-3.1-8B", rendered
        )
        self.assertIn("1.25 [0.50,2.00]", rendered)


if __name__ == "__main__":
    unittest.main()
