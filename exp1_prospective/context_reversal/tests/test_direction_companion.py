"""Offline tests of frozen text, independent generation, safe resume and scoring."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import analyze_direction as analysis
from exp1_prospective.context_reversal import design as parent_design
from exp1_prospective.context_reversal import direction_design as design
from exp1_prospective.context_reversal import materials
from exp1_prospective.context_reversal import run_direction_local as runner
from exp1_prospective.context_reversal import run_local as common


def fixtures():
    parents = parent_design.compile_plan(materials.load_families()[:2], repeats=1)
    return parents, design.compile_plan(parents, "parent-hash")


def config():
    return {"input_sha256": "input-hash", "model_key": "fake-local",
            "decode": {"seed": 42, "chat_template_mode": "qwen-no-thinking"},
            "local_model": {"path": "/fake/local/model"}}


def response(unit, condition, label):
    row = runner.base_record(unit, condition, config())
    row.update(status="ok", direction=label, raw=json.dumps({"direction": label}))
    return row


class DirectionDesignTests(unittest.TestCase):
    def test_exact_visible_text_and_no_forecast_or_private_metadata(self):
        parents, units = fixtures()
        self.assertEqual(len(units), 8)
        for parent, unit in zip(parents, units):
            context, messages = design.extract_visible_text(parent)
            self.assertEqual(context, unit["scenario_text"])
            for condition in design.CONDITIONS:
                prompt = unit["direction_prompts"][condition]
                self.assertTrue(prompt.startswith(context + "\n\nMESSAGE\n" + messages[condition]))
                self.assertNotIn(parent_design.PRIOR_TOKEN, prompt)
                self.assertNotIn("YOUR PREVIOUS FORECAST", prompt)
                self.assertNotIn(parent["family_id"], prompt)
                self.assertNotIn('"probability"', prompt)
                self.assertNotIn("expected_direction", prompt)

    def test_changed_parent_context_or_prompt_hash_fails_closed(self):
        parents, units = fixtures()
        parents[0]["update_templates"]["new_news"] = "changed" + parents[0]["update_templates"]["new_news"]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            design.compile_plan(parents, "parent-hash")
        units[0]["direction_prompt_sha256"]["new_news"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            design.validate_units(units)

    def test_strict_enum_parser(self):
        for label in design.DIRECTIONS:
            self.assertEqual(runner.parse_direction(json.dumps({"direction": label})), label)
        for raw in ['{"direction":"increase"}', '{"direction": "UP"}', '{"direction": null}',
                    '{"direction": "increase", "extra": 1}', '{"direction": "increase"}\n',
                    '{"probability": 0.6}', 'unclear']:
            with self.assertRaises(ValueError):
                runner.parse_direction(raw)

    def test_frozen_output_is_idempotent_and_rejects_changes(self):
        parents, _ = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            parent, output = Path(tmp) / "parent.jsonl", Path(tmp) / "direction.jsonl"
            parent.write_text("".join(common.canonical_json(row) + "\n" for row in parents))
            first = design.write_design(parent, output)
            self.assertEqual(design.write_design(parent, output), first)
            self.assertEqual(first["total_calls_per_model"], 24)
            output.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                design.write_design(parent, output)


class DirectionRunnerTests(unittest.TestCase):
    def run_fake(self, directory, interrupt_after=None, bad_first=False):
        _, units = fixtures()
        args = argparse.Namespace(output=Path(directory) / "responses.jsonl", tensor_parallel_size=1,
                                  max_model_len=8192, gpu_memory_utilization=.9, dtype="bfloat16", seed=42,
                                  enforce_eager=True, batch_size=1, max_tokens=128, temperature=.7)
        calls, sampled = [], []
        outer = self

        class FakeTokenizer:
            chat_template = "fake template"
            def apply_chat_template(self, messages, **kwargs):
                outer.assertEqual(len(messages), 1)
                outer.assertEqual(messages[0]["role"], "user")
                outer.assertFalse(kwargs["enable_thinking"])
                return messages[0]["content"]
            def encode(self, text, **kwargs):
                return list(text.encode())

        class FakeLLM:
            def __init__(self, **kwargs):
                outer.assertFalse(kwargs["trust_remote_code"])
            def get_tokenizer(self):
                return FakeTokenizer()
            def generate(self, prompts, sampling_params, **kwargs):
                if interrupt_after is not None and len(calls) >= interrupt_after:
                    raise KeyboardInterrupt("simulated stop")
                text = "invalid" if bad_first and not calls else '{"direction": "unchanged"}'
                calls.extend(bytes(p["prompt_token_ids"]).decode() for p in prompts)
                sampled.extend(sampling_params)
                completion = SimpleNamespace(text=text, finish_reason="stop", stop_reason=None, token_ids=[1, 2])
                return [SimpleNamespace(outputs=[completion]) for _ in prompts]

        fake_vllm, fake_sampling = ModuleType("vllm"), ModuleType("vllm.sampling_params")
        fake_vllm.LLM, fake_vllm.SamplingParams = FakeLLM, lambda **kwargs: kwargs
        fake_sampling.GuidedDecodingParams = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.sampling_params": fake_sampling}), patch.object(common, "describe_hardware", return_value={}):
            if interrupt_after is not None:
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute(args, units, config())
            else:
                runner.execute(args, units, config())
        rows = [json.loads(line) for line in args.output.read_text().splitlines()]
        return calls, rows, sampled, args, units

    def test_three_independent_chats_and_complete_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, rows, sampled, _, units = self.run_fake(tmp)
            self.assertEqual(len(calls), 24)
            self.assertEqual(set(calls), {p for unit in units for p in unit["direction_prompts"].values()})
            self.assertTrue(all("probability" not in row and "prior_probability" not in row for row in rows))
            self.assertEqual(len({row["seed"] for row in rows}), 24)
            self.assertTrue(all(p["top_p"] == 1 and p["temperature"] == .7 and p["max_tokens"] == 128 for p in sampled))
            self.assertEqual(self.run_fake(tmp)[0], [])

    def test_resume_retains_parse_failure_without_blocking_other_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, first, _, _, _ = self.run_fake(tmp, interrupt_after=2, bad_first=True)
            self.assertEqual(len(first), 2)
            calls, rows, _, _, _ = self.run_fake(tmp)
            self.assertEqual(len(calls), 22)
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(r["status"] == "parse_error" for r in rows), 1)
            self.assertEqual(sum(r["status"] == "ok" for r in rows), 23)

    def test_resume_rejects_tampered_seed_prompt_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows, _, args, units = self.run_fake(tmp)
            manifest = Path(str(args.output) + ".manifest.json")
            with self.assertRaisesRegex(ValueError, "differs"):
                runner.load_existing(args.output, manifest, {**config(), "model_key": "changed"}, units)
            for field, value in (("seed", -1), ("prompt", "tampered"), ("direction", "increase")):
                changed = copy.deepcopy(rows)
                changed[0][field] = value
                args.output.write_text("".join(json.dumps(r) + "\n" for r in changed))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    runner.load_existing(args.output, manifest, config(), units)


class DirectionAnalysisTests(unittest.TestCase):
    def test_missing_unclear_and_zero_direction_fail_planned_denominators(self):
        _, units = fixtures()
        responses = [response(u, c, analysis.expected_direction(u["context_id"], c) or "unclear")
                     for u in units for c in design.CONDITIONS]
        positive = next(r for r in responses if r["context_id"] == "positive" and r["condition"] == "new_news")
        positive.update(direction="unchanged", raw='{"direction": "unchanged"}')
        negative = next(r for r in responses if r["context_id"] == "negative" and r["condition"] == "new_news")
        negative.update(direction="unclear", raw='{"direction": "unclear"}')
        responses.pop()
        report = analysis.analyze(units, responses, "fake-local", bootstrap_draws=100)
        pooled = report["companion"]["direction_correct"]["pooled"]
        self.assertEqual((pooled["n_success"], pooled["n_planned"]), (2, 4))
        self.assertEqual(report["companion"]["paired_reversal"]["both_directions_correct"]["n_success"], 1)
        self.assertIsNone(report["by_context_condition"]["masked"]["new_news"]["accuracy"])
        self.assertEqual(report["by_context_condition"]["masked"]["no_news"]["expected_direction"], "unchanged")
        self.assertEqual(report["coverage"]["totals"]["missing_record"], 1)
        self.assertEqual(report["confusion_counts"]["decrease"]["unclear"], 1)

    def test_all_missing_is_zero_accuracy_with_full_denominators(self):
        _, units = fixtures()
        report = analysis.analyze(units, [], "fake-local", bootstrap_draws=100)
        self.assertEqual(report["coverage"]["totals"]["expected"], 24)
        self.assertEqual(report["coverage"]["totals"]["missing_record"], 24)
        self.assertEqual(report["companion"]["direction_correct"]["pooled"]["estimate"], 0)
        self.assertEqual(report["companion"]["direction_correct"]["pooled"]["n_missing"], 4)

    def test_numeric_join_uses_original_repeat_zero_and_detects_endpoint_block(self):
        parents, units = fixtures()
        directions = [response(u, c, analysis.expected_direction(u["context_id"], c) or "unclear")
                      for u in units for c in design.CONDITIONS]
        numbers = []
        cfg = {**config(), "input_sha256": "parent-hash"}
        for unit in parents:
            prior = 1.0 if unit["context_id"] == "positive" else .5
            baseline = common.base_record(unit, "baseline", "baseline", unit["baseline_prompt"], cfg)
            baseline.update(status="ok", probability=prior, raw=json.dumps({"probability": prior}))
            numbers.append(baseline)
            for condition in design.CONDITIONS:
                value = .4 if unit["context_id"] == "negative" and condition == "new_news" else prior
                update = common.base_record(unit, "update", condition, common.render_update(unit, condition, prior), cfg, prior)
                update.update(status="ok", probability=value, raw=json.dumps({"probability": value}))
                numbers.append(update)
        numbers.append({"model_key": "fake-local", "repeat": 1})
        report = analysis.analyze(units, directions, "fake-local", probability_design=parents,
                                  probability_responses=numbers, bootstrap_draws=100)
        joined = report["exploratory_probability_join"]
        self.assertEqual(joined["excluded_probability_records"], 1)
        self.assertEqual(joined["signed_new_news_pooled"]["both_correct"], 2)
        self.assertEqual(joined["signed_new_news_pooled"]["classification_correct_numeric_incorrect"], 2)
        self.assertEqual(joined["endpoint_flags_new_news_units"]["expected_direction_blocked_by_baseline_endpoint"]["count"], 2)
        numbers[1]["prior_probability"] = .4
        with self.assertRaisesRegex(ValueError, "prior"):
            analysis.analyze(units, directions, "fake-local", probability_design=parents,
                             probability_responses=numbers, bootstrap_draws=100)


if __name__ == "__main__":
    unittest.main()
