"""Format-only parsing and no-resampling recovery checks; no model imports."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import run_reasoning_local as v1
from exp1_prospective.context_reversal import run_reasoning_local_v2 as v2
from exp1_prospective.context_reversal import seed_reasoning_recovery_v2 as recovery
from exp1_prospective.context_reversal.tests.test_reasoning_local import FakeTokenizer, make_unit


class FormatRecoveryTests(unittest.TestCase):
    def test_one_optional_fence_only_in_final_segment(self):
        for fence in ('```json\n', '```\n'):
            for mode in ('enabled', 'disabled'):
                final = fence + '{"probability": 0.6}\n```'
                raw = '<think>Candidate {"probability": 0.1}</think>\n' + final if mode == 'enabled' else final
                row = v2.inspect_completion(raw, mode, 'stop')
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(row['probability'], .6)
                self.assertEqual(row['final_format'], 'fenced_json')
                self.assertEqual(row['final_output'], '{"probability": 0.6}')
                self.assertIn(final, row['final_output_raw'])
        bare = v2.inspect_completion('{"probability": 0.6}', 'disabled', 'stop')
        self.assertEqual(bare['final_format'], 'bare_json')

    def test_malformed_prose_duplicate_objects_and_nonjson_stay_invalid(self):
        invalid = ['Answer: ```json\n{"probability": 0.6}\n```',
                   '```json\n{"probability": 0.6}\n``` extra',
                   '```JSON\n{"probability": 0.6}\n```',
                   '```json\n{"probability": 0.6}\n```\n```json\n{"probability": 0.7}\n```',
                   '```json\n{"probability": 0.6} {"probability": 0.7}\n```',
                   '```json\n{"probability": true}\n```',
                   '```json\n{"probability": 1.2}\n```',
                   '```json\n{"probability": 0.6, "probability": 0.7}\n```',
                   '```json\n{"probability": 0.6, "extra": 2}\n```']
        for raw in invalid:
            self.assertEqual(v2.inspect_completion(raw, 'disabled', 'stop')['status'], 'parse_error', raw)

    def test_truncation_and_unclosed_thought_are_never_repaired(self):
        raw = '<think>reasoning</think>```json\n{"probability": 0.6}\n```'
        row = v2.inspect_completion(raw, 'enabled', 'length')
        self.assertEqual(row['failure_kind'], 'truncated')
        self.assertIsNone(row['probability'])
        row = v2.inspect_completion('<think>```json\n{"probability": 0.6}\n```', 'enabled', 'stop')
        self.assertEqual(row['failure_kind'], 'missing_reasoning_close')
        self.assertIsNone(row['probability'])

    def fixture(self, *, malformed=False, truncated=False):
        units = [make_unit('positive'), make_unit('negative')]
        config = {'input_sha256': 'input-hash', 'model_key': 'fake-disabled',
                  'decode': {'seed': 42, 'thinking': 'disabled', 'enable_thinking': False, 'max_tokens': 4096},
                  'engine': {'max_model_len': 8192}, 'local_model': {'path': '/fake/local/model'}}
        records = {}
        for unit in units:
            positive = unit['context_id'] == 'positive'
            base = v1.base_record(unit, 'baseline', 'baseline', unit['baseline_prompt'], config)
            raw = '```json\n{"probability": 0.2}\n```' if positive else '{"probability": 0.8}'
            if positive and malformed:
                raw += ' extra'
            base.update(raw=raw, response_received=True, finish_reason='length' if positive and truncated else 'stop',
                        prompt_token_count=10, output_token_count=2, total_token_count=12)
            base.update(v1.inspect_completion(raw, 'disabled', base['finish_reason']))
            records[v1.common.record_key(base)] = base
            for condition in v1.CONDITIONS:
                prompt = None if positive else v1.common.render_update(unit, condition, base['probability'])
                row = v1.base_record(unit, 'update', condition, prompt, config, base['probability'])
                if positive:
                    row.update(status='blocked_baseline', failure_kind='blocked_baseline',
                               error='Context-specific baseline has no valid probability',
                               prompt_template=unit['update_templates'][condition],
                               prompt_template_sha256=v1.common.text_sha256(unit['update_templates'][condition]))
                else:
                    raw = '{"probability": 0.5}'
                    row.update(raw=raw, response_received=True, finish_reason='stop', prompt_token_count=10,
                               output_token_count=2, total_token_count=12)
                    row.update(v1.inspect_completion(raw, 'disabled', 'stop'))
                records[v1.common.record_key(row)] = row
        source = {'source_response': {'path': '/source-v1.jsonl', 'sha256': 'source-hash'},
                  'source_manifest': {'sha256': 'manifest-hash'}, 'source_run_signature': 'source-signature',
                  'created_at': 'test-time'}
        return units, config, records, source

    def test_reparse_preserves_every_received_output_and_omits_only_newly_unblocked(self):
        units, config, records, source = self.fixture()
        original = copy.deepcopy(records)
        reparsed, omitted = recovery.reparse_records(records, config, source)
        self.assertEqual(records, original)
        self.assertEqual(len(reparsed), 5)
        self.assertEqual(len(omitted), 3)
        self.assertTrue(all(r['status'] == 'ok' for r in reparsed.values()))
        self.assertTrue(all(r['key'][1] == 'update' for r in omitted))
        for key, row in reparsed.items():
            self.assertEqual(row['raw'], original[key]['raw'])
            self.assertEqual(row['seed'], original[key]['seed'])
            self.assertEqual(row['prompt'], original[key]['prompt'])
            self.assertEqual(row['source_record_sha256'], v1.common.text_sha256(v1.common.canonical_json(original[key])))

    def test_unrecoverable_baselines_and_their_blocked_updates_are_retained(self):
        for kwargs, kind in [({'malformed': True}, 'invalid_final_json'), ({'truncated': True}, 'truncated')]:
            _, config, records, source = self.fixture(**kwargs)
            reparsed, omitted = recovery.reparse_records(records, config, source)
            self.assertEqual(len(reparsed), 8)
            self.assertEqual(omitted, [])
            self.assertEqual(reparsed[('f1:positive:r0', 'baseline', 'baseline')]['failure_kind'], kind)
            self.assertEqual(sum(r['status'] == 'blocked_baseline' for r in reparsed.values()), 3)

    def test_recovered_execution_generates_only_three_previously_unattempted_updates(self):
        units, config, records, source = self.fixture()
        reparsed, omitted = recovery.reparse_records(records, config, source)
        source['previously_blocked_unattempted_updates'] = omitted
        calls = []
        class LLM:
            def __init__(self, **kwargs): pass
            def get_tokenizer(self): return FakeTokenizer()
            def generate(self, prompts, sampling_params, **kwargs):
                for prompt in prompts:
                    calls.append(bytes(prompt['prompt_token_ids']).decode())
                return [SimpleNamespace(outputs=[SimpleNamespace(text='```json\n{"probability": 0.4}\n```',
                         finish_reason='stop', stop_reason=None, token_ids=[1, 2])]) for _ in prompts]
        fake = ModuleType('vllm')
        fake.LLM, fake.SamplingParams = LLM, lambda **kwargs: kwargs
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'responses.jsonl'
            manifest = Path(str(output) + '.manifest.json')
            output.write_text(''.join(v1.common.canonical_json(r) + '\n' for r in reparsed.values()))
            manifest.write_text(json.dumps({'run_signature': v1.common.text_sha256(v1.common.canonical_json(config)),
                                           'config': config, 'recovery': source}))
            args = argparse.Namespace(output=output, tensor_parallel_size=1, max_model_len=8192,
                                      gpu_memory_utilization=.9, dtype='bfloat16', seed=42,
                                      enforce_eager=True, batch_size=16, max_tokens=4096, temperature=.7, thinking='disabled')
            with patch.dict(sys.modules, {'vllm': fake}), patch.object(v2.common, 'describe_hardware', return_value={}), contextlib.redirect_stdout(io.StringIO()):
                v2.execute(args, units, config)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all('baseline ' not in prompt for prompt in calls))
            result = v2.load_existing(output, manifest, config, units)
            self.assertEqual(len(result), 8)
            self.assertTrue(all(r['status'] == 'ok' for r in result.values()))
            for key, row in reparsed.items(): self.assertEqual(result[key], row)
            for row in result.values():
                if row.get('recovery_origin') == 'previously_blocked_unattempted_update':
                    self.assertEqual(row['prior_probability'], .2)
            with patch.dict(sys.modules, {'vllm': None}), contextlib.redirect_stdout(io.StringIO()):
                v2.execute(args, units, config)
            self.assertEqual(len(calls), 3)

    def test_source_must_be_terminal_before_any_raw_read(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'unfinished.jsonl'
            Path(str(source)+'.manifest.json').write_text(json.dumps({'status': 'running'}))
            with self.assertRaisesRegex(ValueError, 'terminal'):
                recovery.seed(source, Path(directory)/'new.jsonl')


if __name__ == '__main__':
    unittest.main()
