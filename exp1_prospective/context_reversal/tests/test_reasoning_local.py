"""CPU checks for final-only parsing, unconstrained decoding, and exact resume."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import run_reasoning_local as runner


def make_unit(context):
    return {"trial_id": f"f1:{context}:r0", "family_id": "f1", "domain": "test",
            "context_id": context, "repeat": 0, "material_status": "authored_development_unvalidated",
            "baseline_prompt": f"baseline {context}", "update_templates": {
                arm: f"{context} {arm}: prior {runner.common.PRIOR_TOKEN}" for arm in runner.CONDITIONS}}


class ReasoningParserTests(unittest.TestCase):
    def test_final_json_is_taken_only_after_reasoning_close(self):
        raw = '<think>Candidate {"probability": 0.1}; revise it.</think>\n{"probability": 0.7}'
        row = runner.inspect_completion(raw, "enabled", "stop")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["probability"], .7)
        self.assertTrue(row["reasoning_present"])
        self.assertTrue(row["reasoning_closed"])
        self.assertEqual(row["final_output"], '{"probability": 0.7}')
        # An open marker may instead be supplied in the chat template.
        no_open = runner.inspect_completion(raw[len('<think>'):], "enabled", "stop", reasoning_open_in_prompt=True)
        self.assertEqual(no_open["probability"], .7)
        missing_open = runner.inspect_completion(raw[len('<think>'):], "enabled", "stop")
        self.assertEqual(missing_open["failure_kind"], "missing_reasoning_open")

    def test_json_in_unclosed_reasoning_never_becomes_a_prediction(self):
        row = runner.inspect_completion('<think>{"probability": 0.1}', "enabled", "stop")
        self.assertIsNone(row["probability"])
        self.assertEqual(row["failure_kind"], "missing_reasoning_close")
        self.assertIsNone(row["final_output"])

    def test_empty_reasoning_is_valid_but_flagged(self):
        row = runner.inspect_completion('</think> {"probability": 0.6}', "enabled", "stop", reasoning_open_in_prompt=True)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["probability"], .6)
        self.assertTrue(row["reasoning_empty"])
        self.assertFalse(row["reasoning_present"])

    def test_truncation_dominates_even_parseable_json(self):
        for raw, mode in [('analysis still running', 'enabled'),
                          ('<think>analysis</think>{"probability": 0.6}', 'enabled'),
                          ('{"probability": 0.6}', 'disabled')]:
            with self.subTest(mode=mode, raw=raw):
                row = runner.inspect_completion(raw, mode, "length", reasoning_open_in_prompt=mode == "enabled")
                self.assertEqual(row["failure_kind"], "truncated")
                self.assertEqual(row["status"], "parse_error")
                self.assertIsNone(row["probability"])

    def test_strict_json_rejects_markdown_extra_fields_duplicates_and_nonfinite(self):
        for text in ['```json\n{"probability": 0.4}\n```', '{"probability": true}',
                     '{"probability": NaN}', '{"probability": Infinity}', '{"probability": 1e999}',
                     '{"probability": -0.1}', '{"probability": 1.01}', '{"probability": "0.4"}',
                     '{"probability": 0.4, "extra": 1}', '{"probability": 0.4, "probability": 0.5}',
                     '{"probability": 0.4} more', '{"probability": 0.4}{"probability": 0.6}', '0.4']:
            with self.subTest(text=text):
                row = runner.inspect_completion(text, 'disabled', 'stop')
                self.assertEqual(row["failure_kind"], "invalid_final_json")
                self.assertIsNone(row["probability"])
        for text, expected in [(' {"probability": 0.123456789} \n', .123456789), ('{"probability": 1e-1}', .1)]:
            self.assertEqual(runner.inspect_completion(text, 'disabled', 'stop')["probability"], expected)

    def test_disabled_mode_does_not_extract_thinking_output(self):
        row = runner.inspect_completion('<think>text</think>{"probability": 0.2}', 'disabled', 'stop')
        self.assertEqual(row["failure_kind"], "invalid_final_json")

    def test_repeated_thinking_markers_are_invalid(self):
        for raw in ['<think>a<think>b</think>{"probability": 0.2}',
                    '<think>a</think></think>{"probability": 0.2}']:
            self.assertEqual(runner.inspect_completion(raw, 'enabled', 'stop')["failure_kind"], "invalid_reasoning_frame")


class FakeTokenizer:
    chat_template = "fake Qwen3 template"
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        prefix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return f'<|im_start|>user\n{messages[0]["content"]}<|im_end|>\n<|im_start|>assistant\n{prefix}'
    def encode(self, text, **kwargs):
        return list(text.encode())


class ReasoningRunnerTests(unittest.TestCase):
    def run_fake(self, directory, *, thinking='enabled', bad_baseline=None, truncate_baseline=None,
                 interrupt_after=None, fail_update=False):
        output = Path(directory) / 'responses.jsonl'
        units = [make_unit('positive'), make_unit('negative')]
        config = {'input_sha256': 'input-hash', 'model_key': f'fake-{thinking}',
                  'decode': {'seed': 42, 'thinking': thinking, 'enable_thinking': thinking == 'enabled', 'max_tokens': 4096},
                  'engine': {'max_model_len': 8192}, 'local_model': {'path': '/fake/local/model'}}
        args = argparse.Namespace(output=output, tensor_parallel_size=1, max_model_len=8192,
                                  gpu_memory_utilization=.9, dtype='bfloat16', seed=42, enforce_eager=True,
                                  batch_size=1, max_tokens=4096, temperature=.7, thinking=thinking)
        calls, sampling_calls, engines = [], [], []

        class FakeLLM:
            def __init__(self, **kwargs):
                engines.append(kwargs)
            def get_tokenizer(self):
                return FakeTokenizer()
            def generate(self, prompts, sampling_params, **kwargs):
                if interrupt_after is not None and len(calls) >= interrupt_after:
                    raise KeyboardInterrupt('simulated allocation interruption')
                sampling_calls.extend(sampling_params)
                results = []
                for prompt in prompts:
                    chat = bytes(prompt['prompt_token_ids']).decode()
                    text = chat.split('<|im_start|>user\n', 1)[1].split('<|im_end|>', 1)[0]
                    calls.append(text)
                    context = 'positive' if 'positive' in text else 'negative'
                    p = .2 if context == 'positive' else .8
                    final = json.dumps({'probability': p if text.startswith('baseline') else .5})
                    raw = 'Work through evidence.</think>\n' + final if thinking == 'enabled' else final
                    finish = 'stop'
                    if text == f'baseline {bad_baseline}' or (fail_update and 'new_news' in text):
                        raw = 'Work through evidence.</think> invalid' if thinking == 'enabled' else 'invalid'
                    if text == f'baseline {truncate_baseline}':
                        finish = 'length'
                    results.append(SimpleNamespace(outputs=[SimpleNamespace(
                        text=raw, finish_reason=finish, stop_reason=None, token_ids=[1, 2, 3])]))
                return results

        module = ModuleType('vllm')
        module.LLM = FakeLLM
        module.SamplingParams = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {'vllm': module}), patch.object(runner.common, 'describe_hardware', return_value={'visible_gpus': []}), contextlib.redirect_stdout(io.StringIO()):
            if interrupt_after is not None:
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute(args, units, config)
            else:
                runner.execute(args, units, config)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        return calls, records, config, units, sampling_calls, engines

    def test_two_modes_both_unconstrained_with_own_context_priors(self):
        for thinking in ('enabled', 'disabled'):
            with self.subTest(thinking=thinking), tempfile.TemporaryDirectory() as directory:
                calls, records, config, units, sampling, engines = self.run_fake(directory, thinking=thinking)
                self.assertEqual(len(calls), 8)
                self.assertEqual(len(records), 8)
                self.assertTrue(all(r['status'] == 'ok' for r in records))
                self.assertTrue(all('guided_decoding' not in params for params in sampling))
                self.assertTrue(all(params['temperature'] == .7 and params['top_p'] == 1.0 and params['skip_special_tokens'] is False for params in sampling))
                self.assertNotIn('guided_decoding_backend', engines[0])
                for row in records:
                    self.assertEqual(row['seed'], runner.common.request_seed(42, row['trial_id'], row['stage'], row['condition']))
                    self.assertEqual(row['reasoning_present'], thinking == 'enabled')
                    self.assertEqual(row['total_token_count'], row['prompt_token_count'] + 3)
                    if row['stage'] == 'update':
                        prior = .2 if row['context_id'] == 'positive' else .8
                        self.assertEqual(row['prior_probability'], prior)
                        self.assertEqual(row['prompt'], f"{row['context_id']} {row['condition']}: prior {prior}")
                calls, _, _, _, _, engines = self.run_fake(directory, thinking=thinking)
                self.assertEqual(calls, [])
                self.assertEqual(engines, [])

    def test_bad_or_truncated_baseline_blocks_only_its_three_branches(self):
        for kwargs, failure in [({'bad_baseline': 'positive'}, 'invalid_final_json'),
                                ({'truncate_baseline': 'positive'}, 'truncated')]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                calls, records, _, _, _, _ = self.run_fake(directory, **kwargs)
                self.assertEqual(len(calls), 5)
                self.assertEqual(len(records), 8)
                rows = [r for r in records if r['context_id'] == 'positive']
                self.assertTrue(all(r['probability'] is None for r in rows))
                self.assertEqual(rows[0]['failure_kind'], failure)
                self.assertEqual(sum(r['status'] == 'blocked_baseline' for r in rows), 3)
                calls, _, _, _, _, _ = self.run_fake(directory)
                self.assertEqual(calls, [], 'No automatic retries of an invalid completed record')

    def test_interruption_resumes_only_missing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, records, _, _, _, _ = self.run_fake(directory, interrupt_after=3)
            original = {runner.common.record_key(r): r for r in records}
            calls, records, _, _, _, _ = self.run_fake(directory)
            self.assertEqual(len(calls), 5)
            self.assertTrue(all(not call.startswith('baseline') for call in calls))
            for row in records:
                if runner.common.record_key(row) in original:
                    self.assertEqual(row, original[runner.common.record_key(row)])

    def test_invalid_updates_are_preserved_and_do_not_block_other_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, records, _, _, _, _ = self.run_fake(directory, fail_update=True)
            self.assertEqual(len(calls), 8)
            self.assertEqual(sum(r['status'] == 'parse_error' for r in records), 2)
            self.assertEqual(sum(r['status'] == 'ok' for r in records), 6)
            self.assertEqual(self.run_fake(directory)[0], [])

    def test_resume_rejects_code_mode_raw_diagnostics_and_cross_context_prior_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records, config, units, _, _ = self.run_fake(directory)
            output = Path(directory) / 'responses.jsonl'
            manifest = Path(str(output) + '.manifest.json')
            for changed in [{**config, 'runner_sha256': 'different'},
                            {**config, 'decode': {**config['decode'], 'thinking': 'disabled'}}]:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    runner.load_existing(output, manifest, changed, units)
            mutations = [('raw', '{"probability": 0.4}', 'completion mismatch'),
                         ('reasoning_char_count', 999, 'completion mismatch'),
                         ('prior_probability', .99, 'own context baseline'),
                         ('prompt', 'different', 'prompt mismatch')]
            for field, value, error in mutations:
                changed = json.loads(json.dumps(records))
                update = next(row for row in changed if row['stage'] == 'update')
                update[field] = value
                output.write_text(''.join(json.dumps(row) + '\n' for row in changed))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                    runner.load_existing(output, manifest, config, units)

    def test_chat_template_must_honor_mode(self):
        for thinking in ('enabled', 'disabled'):
            chat, opened = runner.render_chat(FakeTokenizer(), 'same prompt', thinking)
            self.assertEqual(opened, thinking == 'enabled')
            self.assertIn('same prompt', chat)
        tokenizer = FakeTokenizer()
        tokenizer.apply_chat_template = lambda *args, **kwargs: '<|im_start|>assistant\n<think>\n</think>\n'
        with self.assertRaisesRegex(ValueError, 'suppresses'):
            runner.render_chat(tokenizer, 'prompt', 'enabled')


if __name__ == '__main__':
    unittest.main()
