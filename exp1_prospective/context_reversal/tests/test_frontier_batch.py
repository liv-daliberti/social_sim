"""Offline transport tests: no SDK import, credentials or network required."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import collect_frontier_token_counts as counter
from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_token_certificate as cert
from exp1_prospective.context_reversal import run_frozen_frontier as frozen
from exp1_prospective.context_reversal import run_frozen_frontier_batch as batch
from exp1_prospective.context_reversal.tests.test_frontier_budget import artifact, frozen_fixture


class FakeClient:
    def __init__(self):
        self.count_calls, self.upload_calls, self.create_calls, self.retrieve_calls = [], [], [], []
        self.file_data, self.batch_data = {}, {}
        self.fail_count = self.fail_create = self.fail_upload = self.fail_retrieve = False
        self.count_value = 100
        self.responses = SimpleNamespace(input_tokens=SimpleNamespace(with_raw_response=SimpleNamespace(count=self.count)))
        self.files = SimpleNamespace(create=self.upload, content=self.content)
        self.batches = SimpleNamespace(create=self.create, retrieve=self.retrieve)

    def close(self): pass

    def count(self, **body):
        self.count_calls.append(body)
        if self.fail_count:
            self.fail_count = False
            raise OSError('simulated count result lost')
        response = {'object': 'response.input_tokens', 'input_tokens': self.count_value}
        return SimpleNamespace(status_code=200, headers={'x-request-id': f'count_{len(self.count_calls)}'},
                               parse=lambda: SimpleNamespace(**response))

    def upload(self, *, file, purpose):
        assert purpose == 'batch'
        text = file.read().decode()
        self.upload_calls.append(text)
        file_id = f'file_{len(self.file_data)}'
        self.file_data[file_id] = text
        if self.fail_upload:
            self.fail_upload = False
            raise OSError('simulated upload receipt lost')
        return {'id': file_id}

    def content(self, file_id):
        return SimpleNamespace(text=self.file_data[file_id])

    def create(self, **body):
        self.create_calls.append(body)
        identifier = f'batch_{len(self.batch_data)}'
        result = {'id': identifier, 'status': 'in_progress', 'output_file_id': None, 'error_file_id': None, **body}
        self.batch_data[identifier] = result
        if self.fail_create:
            self.fail_create = False
            raise OSError('simulated Batch receipt lost after creation')
        return copy.deepcopy(result)

    def retrieve(self, identifier):
        self.retrieve_calls.append(identifier)
        if self.fail_retrieve:
            self.fail_retrieve = False
            raise OSError('simulated retrieve interrupted')
        return copy.deepcopy(self.batch_data[identifier])

    def complete(self, identifier, transform=None, terminal='completed'):
        remote = self.batch_data[identifier]
        requests = [json.loads(line) for line in self.file_data[remote['input_file_id']].splitlines()]
        records = []
        for request in requests:
            prompt = request['body']['input'][0]['content']
            p = .2 if 'positive' in prompt else .8 if 'negative' in prompt else .5
            body = {'id': 'response_' + request['custom_id'], 'model': frozen.MODEL, 'status': 'completed',
                    'service_tier': 'default', 'usage': {'input_tokens': 100, 'output_tokens': 8, 'total_tokens': 108},
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'probability': p})}]}]}
            record = {'custom_id': request['custom_id'], 'response': {'status_code': 200, 'request_id': 'r_' + request['custom_id'], 'body': body}, 'error': None}
            record = transform(request, record) if transform else record
            if record is not None:
                records.append(record)
        # Returned order must not determine baseline/prior association.
        file_id = identifier + '_results'
        self.file_data[file_id] = ''.join(json.dumps(r) + '\n' for r in reversed(records))
        remote.update(status=terminal, output_file_id=file_id)


def precount_fixture(directory, **changes):
    path, freeze = frozen_fixture(directory, pricing='batch', prices_nusd_per_token=budget.PRICES_NUSD_PER_TOKEN['batch'], **changes)
    freeze['token_certificate_output'] = str(Path(directory) / 'token_certificate.json')
    for name in ('collect_frontier_token_counts.py', 'run_frozen_frontier_batch.py'):
        freeze['code_sha256'][name] = frozen.common.file_sha256(frozen.SOURCE / name)
    path.write_text(json.dumps(freeze))
    return path, freeze


def batch_fixture(directory, client=None, **changes):
    client = client or FakeClient()
    path, freeze = precount_fixture(directory, **changes)
    report = counter.collect(path, live=True, client_factory=lambda: client, authorization_path=Path(directory) / 'count_auth.json')
    freeze['artifacts']['token_certificate'] = artifact(Path(report['certificate_path']))
    freeze['artifacts']['token_count_collection'] = artifact(Path(report['collection_path']))
    final = Path(directory) / 'generation_freeze.json'
    final.write_text(json.dumps(freeze))
    return final, freeze, client


def run(path, directory, client, action, **kwargs):
    return batch.execute(path, action, live=True, client_factory=lambda: client, ledger_path=Path(directory) / 'ledger.json', **kwargs)


class CounterTests(unittest.TestCase):
    def test_default_offline_and_review_gate_do_not_construct_client_or_write(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze = precount_fixture(d)
            before = set(Path(d).iterdir())
            with patch.object(counter.pilot, 'create_client', side_effect=AssertionError('no key read')):
                result = counter.collect(path)
                self.assertEqual(result['api_calls'], 0)
                self.assertEqual(set(Path(d).iterdir()), before)
                freeze['target_inference_started_before_freeze'] = True
                path.write_text(json.dumps(freeze))
                with self.assertRaisesRegex(ValueError, 'target-model inference'):
                    counter.collect(path, live=True)

    def test_received_counter_receipts_resume_without_recount_and_bind_exact_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze = precount_fixture(d); client = FakeClient(); auth = Path(d) / 'counts_auth.json'
            first = counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth)
            self.assertEqual(len(client.count_calls), 4)
            self.assertEqual(client.count_calls[0]['text']['format']['schema'], frozen.pilot.PROBABILITY_SCHEMA)
            second = counter.collect(path, live=True, client_factory=lambda: (_ for _ in ()).throw(AssertionError('no recount')), authorization_path=auth)
            self.assertEqual(first, second)
            certificate = json.loads(Path(first['certificate_path']).read_text())
            cert.certified_bounds(frozen.common.read_units(Path(freeze['artifacts']['plan']['path'])), freeze['artifacts']['plan']['sha256'], 2048, certificate)
            certificate['receipts'][0]['count_body_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'body changed'):
                cert.certified_bounds(frozen.common.read_units(Path(freeze['artifacts']['plan']['path'])), freeze['artifacts']['plan']['sha256'], 2048, certificate)

    def test_certificate_destination_cannot_reset_count_authorization_and_final_freeze_resumes_offline(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient(); path, freeze, client = batch_fixture(d, client)
            before = len(client.count_calls)
            result = counter.collect(path, live=True, client_factory=lambda: (_ for _ in ()).throw(AssertionError('no final-freeze client')), authorization_path=Path(d) / 'count_auth.json')
            self.assertEqual(result['output_writes'], 0)
            original = Path(d) / 'freeze.json'; original_freeze = json.loads(original.read_text())
            original_freeze['token_certificate_output'] = str(Path(d) / 'different_counts.json')
            original.write_text(json.dumps(original_freeze))
            with self.assertRaisesRegex(ValueError, 'already bound'):
                counter.collect(original, live=True, client_factory=lambda: client, authorization_path=Path(d) / 'count_auth.json')
            self.assertEqual(len(client.count_calls), before)

    def test_certificate_symlink_alias_reuses_state_and_missing_collection_never_recounts(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze = precount_fixture(d); client = FakeClient(); auth = Path(d) / 'count_auth.json'
            result = counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth)
            alias = Path(d) / 'alias.json'; alias.symlink_to(freeze['token_certificate_output'])
            freeze['token_certificate_output'] = str(alias); path.write_text(json.dumps(freeze))
            counter.collect(path, live=True, client_factory=lambda: (_ for _ in ()).throw(AssertionError('no alias recount')), authorization_path=auth)
            self.assertEqual(len(client.count_calls), 4)
            collection = Path(result['collection_path']); saved = collection.read_text()
            damaged = json.loads(saved); damaged['counts'] = {}; collection.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, 'missing received counts'):
                counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth)
            self.assertEqual(len(client.count_calls), 4)
            collection.write_text(saved)
            collection.unlink()
            with self.assertRaisesRegex(ValueError, 'without its receipt collection'):
                counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth)
            self.assertEqual(len(client.count_calls), 4)

    def test_unknown_count_never_reissued_explicit_receipt_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze = precount_fixture(d); client = FakeClient(); client.fail_count = True; auth = Path(d) / 'count_auth.json'
            with self.assertRaises(OSError):
                counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth)
            with self.assertRaisesRegex(ValueError, 'unknown result'):
                counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth)
            self.assertEqual(len(client.count_calls), 1)
            state = json.loads(Path(freeze['token_certificate_output'] + '.collection.json').read_text())
            request = state['requests']['requests'][0]
            receipt = {'request_id': request['request_id'], 'endpoint': cert.ENDPOINT, 'http_status': 200,
                       'count_body_sha256': request['count_body_sha256'], 'provider_request_id': 'recovered_trace',
                       'received_at': frozen.common.utc_now(), 'response': {'object': 'response.input_tokens', 'input_tokens': 100}}
            receipts = Path(d) / 'recovered_receipts.json'; receipts.write_text(json.dumps([receipt]))
            result = counter.collect(path, live=True, client_factory=lambda: client, authorization_path=auth, receipt_file=receipts)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(client.count_calls), 4)


class BatchTests(unittest.TestCase):
    def test_offline_default_no_client_no_writes_and_entire_cap_precedes_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze, client = batch_fixture(d)
            before = set(Path(d).iterdir())
            result = batch.execute(path, 'submit-baselines', client_factory=lambda: (_ for _ in ()).throw(AssertionError('no client')), ledger_path=Path(d) / 'ledger.json')
            self.assertEqual(result['status'], 'offline_valid')
            self.assertEqual(set(Path(d).iterdir()), before)
            self.assertFalse(client.create_calls)
        with tempfile.TemporaryDirectory() as d:
            path, _, client = batch_fixture(d, families=80, budget_usd='1')
            with self.assertRaisesRegex(ValueError, 'Entire frozen cohort'):
                run(path, d, client, 'submit-baselines')
            self.assertFalse(client.upload_calls)
            self.assertFalse((Path(d) / 'ledger.json').exists())

    def test_two_phase_batch_own_context_priors_and_complete_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze, client = batch_fixture(d)
            submitted = run(path, d, client, 'submit-baselines')
            self.assertEqual(submitted['attempted_calls'], 3)
            self.assertEqual(submitted['inflight_calls'], 3)
            state = json.loads((Path(d) / 'ledger.json').read_text())
            self.assertEqual(len(state['entries']), 12)
            self.assertEqual(state['exposure_nusd'], sum(e['reservation_nusd'] for e in state['entries'].values()))
            repeated = run(path, d, client, 'submit-baselines')
            self.assertEqual(repeated['new_submissions'], 0)
            self.assertEqual(len(client.create_calls), 1)
            self.assertEqual(run(path, d, client, 'collect-baselines')['phase_status'], 'waiting')
            client.complete('batch_0')
            run(path, d, client, 'collect-baselines')
            with self.assertRaisesRegex(ValueError, 'Exact materialized'):
                run(path, d, client, 'submit-updates')
            run(path, d, client, 'count-updates')
            updates = run(path, d, client, 'submit-updates')
            self.assertEqual(updates['attempted_calls'], 12)
            requests = [json.loads(line) for line in client.file_data[client.batch_data['batch_1']['input_file_id']].splitlines()]
            self.assertEqual(len(requests), 9)
            for request in requests:
                prompt = request['body']['input'][0]['content']
                expected = .2 if 'positive' in prompt else .8 if 'negative' in prompt else .5
                self.assertIn('prior ' + str(expected), prompt)
                self.assertEqual(request['body']['max_output_tokens'], 2048)
                self.assertEqual(request['body']['reasoning'], {'effort': 'low'})
            client.complete('batch_1')
            result = run(path, d, client, 'collect-updates')
            self.assertEqual((result['status'], result['records']), ('complete', 12))
            self.assertLess(float(result['exposure_usd']), 24)
            rows = [json.loads(line) for line in Path(freeze['output_path']).read_text().splitlines()]
            by_trial = {r['trial_id']: r for r in rows if r['stage'] == 'baseline'}
            self.assertTrue(all(r['prior_probability'] == by_trial[r['trial_id']]['probability'] for r in rows if r['stage'] == 'update'))
            creates = len(client.create_calls); retrieves = len(client.retrieve_calls)
            run(path, d, client, 'collect-updates'); run(path, d, client, 'submit-updates')
            self.assertEqual((len(client.create_calls), len(client.retrieve_calls)), (creates, retrieves))

    def test_ambiguous_creation_is_fully_reserved_never_resubmitted_and_adopts_existing_id(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, client = batch_fixture(d); client.fail_create = True
            with self.assertRaises(OSError): run(path, d, client, 'submit-baselines')
            state = json.loads((Path(d) / 'ledger.json').read_text())
            self.assertEqual(state['batch_phases']['baselines']['status'], 'creation_unknown')
            self.assertEqual(sum(e['state'] == 'inflight' for e in state['entries'].values()), 3)
            with self.assertRaisesRegex(ValueError, 'Ambiguous prior'):
                run(path, d, client, 'submit-baselines')
            self.assertEqual(len(client.create_calls), 1)
            run(path, d, client, 'adopt-baseline-batch', remote_id='batch_0')
            client.fail_retrieve = True
            with self.assertRaises(OSError): run(path, d, client, 'collect-baselines')
            self.assertEqual(json.loads((Path(d) / 'ledger.json').read_text())['batch_phases']['baselines']['batch_id'], 'batch_0')
            client.complete('batch_0'); run(path, d, client, 'collect-baselines')
            self.assertEqual(len(client.create_calls), 1)

    def test_unknown_upload_requires_matching_existing_file_then_only_one_batch(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, client = batch_fixture(d); client.fail_upload = True
            with self.assertRaises(OSError): run(path, d, client, 'submit-baselines')
            with self.assertRaisesRegex(ValueError, 'Ambiguous prior'):
                run(path, d, client, 'submit-baselines')
            self.assertFalse(client.create_calls)
            client.file_data['wrong'] = 'different request'
            with self.assertRaisesRegex(ValueError, 'remote file differs'):
                run(path, d, client, 'adopt-baseline-file', remote_id='wrong')
            run(path, d, client, 'adopt-baseline-file', remote_id='file_0')
            run(path, d, client, 'submit-baselines')
            self.assertEqual((len(client.upload_calls), len(client.create_calls)), (1, 1))

    def test_exact_update_count_overrun_blocks_all_update_generation(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, client = batch_fixture(d)
            run(path, d, client, 'submit-baselines'); client.complete('batch_0'); run(path, d, client, 'collect-baselines')
            client.count_value = 200000
            with self.assertRaisesRegex(ValueError, 'exceeds the frozen'):
                run(path, d, client, 'count-updates')
            with self.assertRaisesRegex(ValueError, 'Ledger halted'):
                run(path, d, client, 'submit-updates')
            self.assertEqual(len(client.create_calls), 1)
            self.assertTrue(json.loads((Path(d) / 'ledger.json').read_text())['halted'])
            self.assertEqual(len(Path(json.loads(path.read_text())['output_path']).read_text().splitlines()), 12)

    def test_truncation_missing_response_and_blocked_branches_are_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path, freeze, client = batch_fixture(d)
            run(path, d, client, 'submit-baselines')
            def transform(request, record):
                prompt = request['body']['input'][0]['content']
                if 'positive' in prompt:
                    record['response']['body']['status'] = 'incomplete'
                if 'negative' in prompt:
                    return None
                return record
            client.complete('batch_0', transform, terminal='expired')
            result = run(path, d, client, 'collect-baselines')
            self.assertEqual(result['status_counts']['blocked_baseline'], 6)
            self.assertEqual(result['unknown_billed_calls'], 1)
            run(path, d, client, 'count-updates'); run(path, d, client, 'submit-updates')
            client.complete('batch_1')
            result = run(path, d, client, 'collect-updates')
            self.assertEqual((result['status'], result['records'], result['attempted_calls']), ('complete_with_errors', 12, 6))
            self.assertEqual(len(client.create_calls), 2)
            state = json.loads((Path(d) / 'ledger.json').read_text())
            unknown = [e for e in state['entries'].values() if e['state'] == 'unknown_billed']
            self.assertEqual(unknown[0]['charge_nusd'], unknown[0]['reservation_nusd'])
            self.assertEqual(len(Path(freeze['output_path']).read_text().splitlines()), 12)

    def test_wrong_model_retains_usage_halts_and_never_becomes_complete_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, client = batch_fixture(d); run(path, d, client, 'submit-baselines')
            def wrong(request, record):
                record['response']['body']['model'] = 'different_model'
                return record
            client.complete('batch_0', wrong)
            result = run(path, d, client, 'collect-baselines')
            self.assertEqual((result['status'], result['records']), ('failed', 12))
            self.assertLess(float(result['exposure_usd']), .01)
            self.assertEqual(run(path, d, client, 'collect-baselines')['status'], 'failed')
            with self.assertRaisesRegex(ValueError, 'Ledger halted'): run(path, d, client, 'submit-updates')
            self.assertEqual(len(client.create_calls), 1)

    def test_malformed_provider_structures_settle_and_keep_remaining_results(self):
        for malformed in ('usage_detail', 'output', 'content'):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as d:
                path, _, client = batch_fixture(d); run(path, d, client, 'submit-baselines')
                def transform(request, record):
                    body = record['response']['body']
                    if 'positive' in request['body']['input'][0]['content']:
                        if malformed == 'usage_detail': body['usage']['input_tokens_details'] = 1
                        elif malformed == 'output': body['output'] = 1
                        else: body['output'][0]['content'] = 1
                    return record
                client.complete('batch_0', transform)
                result = run(path, d, client, 'collect-baselines')
                self.assertEqual(result['phases']['baselines']['status'], 'collected')
                self.assertEqual(result['inflight_calls'], 0)
                self.assertEqual(result['status_counts']['ok'], 3 if malformed == 'usage_detail' else 2)
                self.assertEqual(result['unknown_billed_calls'], 1 if malformed == 'usage_detail' else 0)

    def test_crash_after_terminal_phase_write_finishes_blocked_branches_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path, _, client = batch_fixture(d); run(path, d, client, 'submit-baselines')
            def wrong(request, record):
                record['response']['body']['model'] = 'wrong'
                return record
            client.complete('batch_0', wrong)
            original = budget.Ledger.persist
            tripped = [False]
            def crash(ledger):
                original(ledger)
                if not tripped[0] and ledger.state.get('batch_phases', {}).get('baselines', {}).get('status') == 'collected':
                    tripped[0] = True
                    raise KeyboardInterrupt()
            with patch.object(budget.Ledger, 'persist', crash), self.assertRaises(KeyboardInterrupt):
                run(path, d, client, 'collect-baselines')
            result = run(path, d, client, 'collect-baselines')
            self.assertEqual((result['status'], result['records']), ('failed', 12))
            self.assertEqual(len(client.create_calls), 1)

    def test_tampered_prior_or_count_body_rejected_before_upload(self):
        for tamper in ('prior', 'consistent_raw_prior', 'count'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as d:
                path, _, client = batch_fixture(d)
                run(path, d, client, 'submit-baselines'); client.complete('batch_0'); run(path, d, client, 'collect-baselines')
                run(path, d, client, 'count-updates')
                if tamper in ('prior', 'consistent_raw_prior'):
                    ledger = Path(d) / 'ledger.json'; state = json.loads(ledger.read_text())
                    row = next(e['row'] for e in state['entries'].values() if e['state'] == 'settled')
                    row['probability'] = .99
                    if tamper == 'consistent_raw_prior': row['raw'] = '{"probability":0.99}'
                    ledger.write_text(json.dumps(state))
                else:
                    counts = Path(json.loads(path.read_text())['output_path'] + '.batch_updates.counts.json')
                    state = json.loads(counts.read_text()); next(iter(state['counts'].values()))['receipt']['count_body_sha256'] = 'tampered'
                    counts.write_text(json.dumps(state))
                with self.assertRaises(ValueError): run(path, d, client, 'submit-updates')
                self.assertEqual(len(client.upload_calls), 1)


if __name__ == '__main__': unittest.main()
