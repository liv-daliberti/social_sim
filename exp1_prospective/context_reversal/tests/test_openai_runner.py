"""Offline behavioral tests; fixture clients never access credentials or network."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import run_openai as runner
from exp1_prospective.context_reversal.tests.test_runner import make_unit


def response(raw='{"probability":0.4}', status="completed"):
    return SimpleNamespace(output_text=raw, status=status, model="gpt-5.6-sol", id="resp_fixture",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=30, total_tokens=130,
                                                 input_tokens_details=SimpleNamespace(cached_tokens=50),
                                                 output_tokens_details=SimpleNamespace(reasoning_tokens=20)))


class FixtureClient:
    def __init__(self, result=None):
        self.responses = self
        self.calls = []
        self.result = result
        self.closed = False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if callable(self.result):
            return self.result(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result or response()

    def close(self):
        self.closed = True


class OpenAIRunnerTests(unittest.TestCase):
    def setup_run(self, directory, units=None, **options):
        units = units or [make_unit()]
        input_path, output = Path(directory) / "plan.jsonl", Path(directory) / "responses.jsonl"
        input_path.write_text("".join(json.dumps(u) + "\n" for u in units))
        args = argparse.Namespace(input=input_path, output=output, model_key="gpt_5_6_frontier",
                                  model="gpt-5.6-sol", concurrency=1, max_calls=320, budget_usd=15.0)
        for key, value in options.items():
            setattr(args, key, value)
        return args, units, runner.make_config(args, units)

    def run_fixture(self, args, units, config, client):
        capture = io.StringIO()
        with redirect_stdout(capture), redirect_stderr(capture):
            manifest = runner.execute(args, units, config, client_factory=lambda: client)
        records = [json.loads(line) for line in args.output.read_text().splitlines()]
        return manifest, records, capture.getvalue()

    def test_json_whitespace_and_strict_numeric_contract(self):
        for raw in ['{"probability":0.4}', ' { "probability" : 4e-1 }\n', '{"probability":1}', '{"probability":0}']:
            self.assertIn(runner.parse_probability(raw), (0, .4, 1))
        for raw in ['{"probability":true}', '{"probability":"0.4"}', '{"probability":NaN}',
                    '{"probability":Infinity}', '{"probability":1.01}', '{"probability":-0.1}',
                    '{"probability":0.4,"extra":1}', '{"probability":0.4,"probability":0.5}',
                    '0.4', '[]', '```json\n{"probability":0.4}\n```', '{"probability":0.4} trailing']:
            with self.assertRaises(ValueError, msg=raw):
                runner.parse_probability(raw)

    def test_independent_branches_exact_request_and_usage_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            args, units, config = self.setup_run(directory)
            client = FixtureClient(lambda call: response('{"probability":0.4}' if call['input'][0]['content'] == 'baseline question' else '{"probability":0.8}'))
            manifest, records, _ = self.run_fixture(args, units, config, client)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(client.calls), 4)
            self.assertEqual(manifest['attempted_calls'], 4)
            self.assertAlmostEqual(manifest['billed_cost_estimate_usd'], 4 * .00082)
            self.assertEqual(manifest['usage_totals']['reasoning_tokens'], 80)
            self.assertTrue(client.closed)
            for row in records:
                if row['stage'] == 'update':
                    self.assertEqual(row['prior_probability'], .4)
                    self.assertEqual(row['prompt'], f"{row['condition']}: prior 0.4")
            for call in client.calls:
                self.assertEqual(call['model'], 'gpt-5.6-sol')
                self.assertEqual(call['reasoning'], {'effort': 'low'})
                self.assertEqual(call['max_output_tokens'], 2048)
                self.assertFalse(call['store'])
                self.assertEqual(call['tools'], [])
                self.assertEqual(len(call['input']), 1)
                self.assertNotIn('temperature', call)
                self.assertNotIn('seed', call)
                self.assertTrue(call['text']['format']['strict'])
            with patch.object(runner, 'create_client', side_effect=AssertionError('completed resume must be offline')):
                self.assertEqual(runner.execute(args, units, config)['attempted_calls'], 4)

    def test_failed_baseline_preserves_all_outcomes_without_updates(self):
        for result, expected in [(response('garbage'), 'parse_error'), (response('', status='incomplete'), 'generation_error')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                args, units, config = self.setup_run(directory)
                client = FixtureClient(result)
                manifest, records, _ = self.run_fixture(args, units, config, client)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(len(records), 4)
                self.assertEqual(records[0]['status'], expected)
                self.assertEqual(sum(r['status'] == 'blocked_baseline' for r in records), 3)
                self.assertTrue(all(r['probability'] is None for r in records))
                self.assertEqual(manifest['status'], 'complete_with_errors')

    def test_authentication_error_aborts_and_never_logs_exception_secret(self):
        class UnsafeAuthenticationError(Exception):
            status_code = 401
            code = 'invalid_api_key'
        secret = 'TEST_ONLY_SECRET_DO_NOT_LOG'
        with tempfile.TemporaryDirectory() as directory:
            units = [make_unit(), make_unit('f1:negative:r0', 'negative')]
            args, units, config = self.setup_run(directory, units)
            client = FixtureClient(UnsafeAuthenticationError(secret))
            manifest, records, logs = self.run_fixture(args, units, config, client)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(records), 8)
            self.assertNotIn(secret, logs + args.output.read_text() + Path(str(args.output) + '.manifest.json').read_text())
            self.assertEqual(sum(r['request_attempted'] for r in records), 1)
            runner.load_existing(args.output, Path(str(args.output) + '.manifest.json'), config, units)
            with patch.object(runner, 'create_client', side_effect=AssertionError('terminal failure must not retry')):
                self.assertEqual(runner.execute(args, units, config)['status'], 'failed')

    def test_budget_guard_and_call_ceiling_account_for_unattempted_rows(self):
        for options, expected_calls, reason in [({'budget_usd': .001}, 0, 'budget_ceiling'), ({'max_calls': 2}, 2, 'call_ceiling')]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                args, units, config = self.setup_run(directory, **options)
                client = FixtureClient()
                manifest, records, _ = self.run_fixture(args, units, config, client)
                self.assertEqual(len(client.calls), expected_calls)
                self.assertEqual(len(records), 4)
                self.assertEqual(manifest['error']['kind'], reason)
                self.assertLessEqual(manifest['conservative_spend_usd'], args.budget_usd)
                runner.load_existing(args.output, Path(str(args.output) + '.manifest.json'), config, units)

    def test_resume_uses_saved_baseline_and_charges_uncertain_attempt_once(self):
        with tempfile.TemporaryDirectory() as directory:
            args, units, config = self.setup_run(directory)
            client = FixtureClient()
            _, records, _ = self.run_fixture(args, units, config, client)
            baseline = next(r for r in records if r['stage'] == 'baseline')
            manifest_path = Path(str(args.output) + '.manifest.json')
            manifest = json.loads(manifest_path.read_text())
            # Simulate a crash after reserving one update, before saving its response.
            manifest['attempt_ledger'] = manifest['attempt_ledger'][:2]
            manifest['status'] = 'interrupted'
            manifest_path.write_text(json.dumps(manifest))
            args.output.write_text(json.dumps(baseline) + '\n')
            resumed = FixtureClient()
            final, records, _ = self.run_fixture(args, units, config, resumed)
            self.assertEqual(len(resumed.calls), 2)
            self.assertTrue(all(c['input'][0]['content'] != 'baseline question' for c in resumed.calls))
            self.assertEqual(final['attempted_calls'], 4)
            uncertain = [r for r in records if r['error'] == {'kind': 'interrupted_request_result_unavailable'}]
            self.assertEqual(len(uncertain), 1)
            self.assertIsNone(uncertain[0]['probability'])
            self.assertGreater(final['conservative_spend_usd'], final['billed_cost_estimate_usd'])

    def test_resume_rejects_changed_configuration_and_tampered_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            args, units, config = self.setup_run(directory)
            _, records, _ = self.run_fixture(args, units, config, FixtureClient())
            manifest_path = Path(str(args.output) + '.manifest.json')
            with self.assertRaisesRegex(ValueError, 'differs'):
                runner.load_existing(args.output, manifest_path, {**config, 'model': 'other'}, units)
            records[-1]['prompt'] = 'tampered'
            args.output.write_text(''.join(json.dumps(r) + '\n' for r in records))
            with self.assertRaisesRegex(ValueError, 'prompt mismatch'):
                runner.load_existing(args.output, manifest_path, config, units)

    def test_dry_run_never_constructs_client_or_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.setup_run(directory)
            argv = ['runner', '--input', str(args.input), '--output', str(args.output), '--model', 'gpt-5.6-sol', '--dry-run']
            with patch('sys.argv', argv), patch.object(runner, 'create_client', side_effect=AssertionError('no client')), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            self.assertFalse(args.output.exists())
            self.assertFalse(Path(str(args.output) + '.manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
