"""CPU tests of outcome preservation, independent updates, and safe resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import run_local as runner


def make_unit(trial_id="f1:positive:r0", context_id="positive"):
    return {"trial_id": trial_id, "family_id": "f1", "domain": "test", "context_id": context_id,
            "repeat": 0, "material_status": "authored_development_unvalidated",
            "baseline_prompt": "baseline question", "update_templates": {
                condition: f"{condition}: prior {runner.PRIOR_TOKEN}" for condition in runner.CONDITIONS}}


class RunnerTests(unittest.TestCase):
    def test_strict_probability_contract(self):
        for text, expected in [('{"probability": 0}', 0), ('{"probability": 1.000000}', 1), ('{"probability": 0.125}', .125)]:
            self.assertEqual(runner.parse_probability(text), expected)
        for text in ['{"probability": -0.1}', '{"probability": 1.1}', '{"probability": true}',
                     '{"probability": NaN}', '{"probability": 0.4, "extra": 1}', '0.4',
                     '{"probability": 0.1234567}', '{"probability": 0.4} extra']:
            with self.assertRaises(ValueError, msg=text):
                runner.parse_probability(text)

    def test_unit_contract_rejects_missing_prior_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "units.jsonl"
            unit = make_unit()
            path.write_text(json.dumps(unit) + "\n")
            self.assertEqual(runner.read_units(path), [unit])
            path.write_text((json.dumps(unit) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                runner.read_units(path)
            unit["update_templates"]["no_news"] = "missing"
            path.write_text(json.dumps(unit) + "\n")
            with self.assertRaisesRegex(ValueError, "placeholder"):
                runner.read_units(path)

    def test_seed_differs_by_repeat_and_condition(self):
        seeds = {runner.request_seed(42, f"f1:positive:r{repeat}", "update", condition)
                 for repeat in range(3) for condition in runner.CONDITIONS}
        self.assertEqual(len(seeds), 9)
        self.assertEqual(runner.request_seed(42, "a", "baseline", "baseline"), runner.request_seed(42, "a", "baseline", "baseline"))

    def run_fake(self, directory, bad_baseline=False, interrupt_after=None):
        output = Path(directory) / "responses.jsonl"
        units = [make_unit()]
        config = {"input_sha256": "input-hash", "model_key": "fake-local", "decode": {"seed": 42, "chat_template_mode": "auto"},
                  "local_model": {"path": "/fake/local/model"}}
        args = argparse.Namespace(output=output, tensor_parallel_size=1, max_model_len=4096,
                                  gpu_memory_utilization=.9, dtype="bfloat16", seed=42,
                                  enforce_eager=False, batch_size=1, max_tokens=128, temperature=.7)
        calls = []

        class FakeTokenizer:
            chat_template = "fake template"
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]
            def encode(self, text, **kwargs):
                return list(text.encode())

        class FakeLLM:
            def __init__(self, **kwargs):
                pass
            def get_tokenizer(self):
                return FakeTokenizer()
            def generate(self, prompts, sampling_params, **kwargs):
                if interrupt_after is not None and len(calls) >= interrupt_after:
                    raise KeyboardInterrupt("simulated allocation interruption")
                decoded = [bytes(p["prompt_token_ids"]).decode() for p in prompts]
                calls.extend(decoded)
                text = "invalid" if bad_baseline and decoded[0] == "baseline question" else '{"probability": 0.4}'
                completion = SimpleNamespace(text=text, finish_reason="stop", stop_reason=None, token_ids=[1, 2])
                return [SimpleNamespace(outputs=[completion]) for _ in prompts]

        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = lambda **kwargs: kwargs
        fake_sampling = ModuleType("vllm.sampling_params")
        fake_sampling.GuidedDecodingParams = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.sampling_params": fake_sampling}), patch.object(runner, "describe_hardware", return_value={"visible_gpus": []}):
            if interrupt_after is not None:
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute(args, units, config)
            else:
                runner.execute(args, units, config)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        return calls, records, config, units

    def test_updates_fork_from_same_baseline_and_complete_resume_skips_model(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, records, config, units = self.run_fake(directory)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(records), 4)
            self.assertEqual({r["condition"] for r in records}, {"baseline", *runner.CONDITIONS})
            for row in records:
                if row["stage"] == "update":
                    self.assertEqual(row["prior_probability"], .4)
                    self.assertEqual(row["prompt"], f'{row["condition"]}: prior 0.4')
            calls, _, _, _ = self.run_fake(directory)
            self.assertEqual(calls, [])

    def test_invalid_baseline_preserves_all_three_blocked_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, records, _, _ = self.run_fake(directory, bad_baseline=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(sum(r["status"] == "blocked_baseline" for r in records), 3)
            self.assertTrue(all(r["probability"] is None for r in records))
            self.assertTrue(all(r["prompt"] is None for r in records if r["stage"] == "update"))

    def test_interrupted_resume_never_reelicits_completed_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, records, _, _ = self.run_fake(directory, interrupt_after=1)
            self.assertEqual(len(records), 1)
            calls, records, _, _ = self.run_fake(directory)
            self.assertEqual(len(calls), 3)
            self.assertNotIn("baseline question", calls)
            self.assertEqual(len(records), 4)

    def test_resume_rejects_changed_config_or_tampered_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records, config, units = self.run_fake(directory)
            output = Path(directory) / "responses.jsonl"
            manifest = Path(str(output) + ".manifest.json")
            with self.assertRaisesRegex(ValueError, "differs"):
                runner.load_existing(output, manifest, {**config, "model_key": "another"}, units)
            records[-1]["prompt"] = "tampered"
            output.write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaisesRegex(ValueError, "prompt mismatch"):
                runner.load_existing(output, manifest, config, units)


if __name__ == "__main__":
    unittest.main()
