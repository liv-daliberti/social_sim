#!/usr/bin/env python3
"""Explicit, resumable Batch Responses evaluation; offline without --live.

Reserve the entire cohort before any generation. Submit baselines, collect them,
count exact materialized updates, then submit/collect updates. Never retry a
received completion. Ambiguous submission stops until its existing batch/file
ID is reconciled; known batches are retrieved, never re-created.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from exp1_prospective.context_reversal import collect_frontier_token_counts as counter
from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_token_certificate as cert
from exp1_prospective.context_reversal import run_frozen_frontier as frozen
from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal import run_openai as pilot

TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}
ENDPOINT = '/v1/responses'
ACTIONS = ('status', 'submit-baselines', 'collect-baselines', 'count-updates',
           'submit-updates', 'collect-updates', 'adopt-baseline-file',
           'adopt-update-file', 'adopt-baseline-batch', 'adopt-update-batch')


def as_dict(value):
    return value if isinstance(value, dict) else value.model_dump(mode='json')


def custom_id(key):
    return 'request_' + cert.fingerprint(list(key))


def file_text(client, file_id):
    content = client.files.content(file_id)
    text = content.text
    frozen.require(isinstance(text, str), 'Batch file content is not UTF-8 text')
    return text


def validate_certificate_source(freeze, units):
    artifacts = freeze['artifacts']
    frozen.require({'token_certificate', 'token_count_collection'} <= set(artifacts),
                   'Batch execution requires a frozen certificate and official receipt collection')
    certificate = json.loads(frozen.checked_artifact(artifacts['token_certificate']).read_text())
    state = json.loads(frozen.checked_artifact(artifacts['token_count_collection']).read_text())
    requests = cert.request_manifest(units, artifacts['plan']['sha256'], freeze['max_output_tokens'])
    frozen.require(state.get('schema_version') == 'frontier_token_collection_v1' and state.get('status') == 'complete'
                   and state.get('binding') == counter.count_binding(freeze) and state.get('requests') == requests
                   and state.get('request_manifest_sha256') == cert.fingerprint(requests), 'Token-count collection provenance mismatch')
    frozen.require(set(state['counts']) == {r['request_id'] for r in requests['requests']}
                   and all(c.get('status') == 'received' for c in state['counts'].values()), 'Incomplete official count collection')
    imported = cert.import_receipts(requests, [state['counts'][r['request_id']]['receipt'] for r in requests['requests']])
    frozen.require(imported == certificate, 'Certificate does not match the archived official count receipts')


def parse_response(row, result):
    """Retain billing even for provider failures, malformed JSON or wrong models."""
    row = copy.deepcopy(row)
    if result is None:
        row['error'] = {'kind': 'terminal_batch_missing_result_no_retry'}
        return row, False
    response = result.get('response') or {}
    if not isinstance(response, dict) or not isinstance(response.get('body') or {}, dict):
        row.update(batch_result_sha256=cert.fingerprint(result), error={'kind': 'malformed_batch_response'})
        return row, False
    body = response.get('body') or {}
    code = response.get('status_code')
    row.update(batch_result_sha256=cert.fingerprint(result), batch_request_id=response.get('request_id'),
               response_received=bool(response), response_id=body.get('id'), returned_model=body.get('model'),
               response_status=body.get('status'), returned_service_tier=body.get('service_tier'), created_at=common.utc_now())
    usage = body.get('usage')
    if isinstance(usage, dict) and all(isinstance(usage.get(k) or {}, dict) for k in ('input_tokens_details', 'output_tokens_details')):
        usage = dict(usage)
        for field in ('input_tokens_details', 'output_tokens_details'):
            usage[field] = SimpleNamespace(**(usage.get(field) or {}))
        row['usage'] = pilot.response_usage(SimpleNamespace(usage=SimpleNamespace(**usage)))
    else:
        row['usage'] = None
    output = body.get('output') if isinstance(body.get('output'), list) else []
    row['raw'] = ''.join(c.get('text', '') for item in output if isinstance(item, dict) and item.get('type') == 'message'
                         for c in (item.get('content') if isinstance(item.get('content'), list) else []) if isinstance(c, dict) and c.get('type') == 'output_text' and isinstance(c.get('text'), str))
    if code != 200 or result.get('error'):
        row['error'] = {'kind': 'batch_request_failed', 'http_status': code if type(code) is int else None}
        return row, code in (400, 401, 403, 404, 429)
    if row['returned_model'] != frozen.MODEL:
        row['error'] = {'kind': 'unexpected_returned_model'}
        return row, True
    if row['returned_service_tier'] not in (None, 'default'):
        row['error'] = {'kind': 'unexpected_billing_service_tier'}
        return row, True
    if row['response_status'] != 'completed':
        row['error'] = {'kind': 'response_not_completed'}
        return row, False
    try:
        row.update(probability=frozen.parse_probability(row['raw']), status='ok', error=None)
    except (ValueError, TypeError, OverflowError):
        row.update(status='parse_error', error={'kind': 'invalid_probability_json'})
    return row, False


class BatchRun:
    def __init__(self, freeze_path, freeze, units, specs, preflight, ledger, client_factory):
        self.freeze_path, self.freeze, self.units, self.specs = freeze_path, freeze, units, specs
        self.preflight, self.ledger = preflight, ledger
        self.output = Path(freeze['output_path'])
        self.factory, self.client = client_factory or pilot.create_client, None
        self.config = {'input_sha256': freeze['artifacts']['plan']['sha256'], 'model_key': freeze['model_key'],
                       'model': frozen.MODEL, 'freeze_sha256': preflight['binding']['freeze_sha256'],
                       'transport': 'batch', 'decode': {'max_output_tokens': freeze['max_output_tokens']}}
        self.signature = cert.fingerprint(self.config)
        self.phases = ledger.state.setdefault('batch_phases', {})

    def path(self, phase, suffix):
        return Path(str(self.output) + f'.batch_{phase}.{suffix}')

    def get_client(self):
        if self.client is None:
            self.client = self.factory()
        return self.client

    def row(self, unit, condition):
        stage = 'baseline' if condition == 'baseline' else 'update'
        prior, prompt = None, unit['baseline_prompt']
        if stage == 'update':
            baseline = self.ledger.rows().get((unit['trial_id'], 'baseline', 'baseline'))
            frozen.require(baseline is not None, 'Update cannot precede its own context baseline')
            prior = baseline['probability']
            frozen.require(prior is None or len(json.dumps(prior).encode()) <= budget.PRIOR_SERIALIZED_BYTE_LIMIT, 'Prior exceeds frozen serialization bound')
            prompt = common.render_update(unit, condition, prior) if baseline['status'] == 'ok' else None
        row = pilot.base_record(unit, stage, condition, prompt, self.config, prior)
        row['reserved_cost_usd'] = float(budget.dollars(self.specs[budget.record_id(common.record_key(row))]['reservation_nusd']))
        if prompt is None:
            row.update(status='blocked_baseline', error={'kind': 'baseline_has_no_valid_probability'})
        return row

    def validate_rows(self):
        saved = self.ledger.rows()
        by_id = {u['trial_id']: u for u in self.units}
        for key, row in saved.items():
            expected = self.row(by_id[key[0]], key[2])
            for field in ('trial_id', 'family_id', 'domain', 'context_id', 'repeat', 'material_status', 'stage', 'condition',
                          'model_key', 'requested_model', 'input_sha256', 'run_signature', 'prompt', 'prompt_sha256', 'prior_probability', 'reserved_cost_usd'):
                frozen.require(row.get(field) == expected[field], f'Persisted Batch response changed {field}')
            frozen.require(row.get('status') in {'ok', 'parse_error', 'generation_error', 'blocked_baseline'}, 'Invalid saved Batch status')
            frozen.require((row['status'] == 'blocked_baseline') == (expected['status'] == 'blocked_baseline'), 'Saved baseline blocking changed')
            state = self.ledger.state['entries'][budget.record_id(key)]['state']
            frozen.require(row.get('request_attempted') is (state != 'unattempted_terminal'), 'Saved attempt flag differs from ledger')
            if row['status'] == 'ok':
                frozen.require(row.get('returned_model') == frozen.MODEL and row.get('response_status') == 'completed'
                               and row.get('returned_service_tier') in (None, 'default') and row.get('response_received') is True
                               and type(row.get('probability')) in (int, float) and frozen.parse_probability(row['raw']) == row['probability'], 'Saved probability/model differs from the received completion')
            else:
                frozen.require(row.get('probability') is None, 'Failed Batch response has a probability')
        for name, phase in self.phases.items():
            frozen.require(name in ('baselines', 'updates'), 'Unknown saved Batch phase')
            frozen.require(phase.get('status') in {'prepared', 'upload_intent', 'upload_unknown', 'uploaded', 'creation_intent', 'creation_unknown', 'submitted', 'collected'}, 'Unknown saved Batch status')
            keys = [tuple(k) for k in phase.get('keys', [])]
            frozen.require(len(keys) == len(set(keys)) and all(budget.record_id(k) in self.specs and k[1] == ('baseline' if name == 'baselines' else 'update') for k in keys), 'Saved Batch request coverage changed')
            if name == 'baselines':
                frozen.require(set(keys) == {(u['trial_id'], 'baseline', 'baseline') for u in self.units}, 'Saved baseline Batch does not cover every planned baseline')
            if phase.get('batch_id') or phase['status'] in ('creation_intent', 'creation_unknown'):
                frozen.require(all(self.ledger.state['entries'][budget.record_id(k)]['state'] in ('inflight', 'settled', 'unknown_billed') for k in keys), 'Submitted Batch lost a full request reservation')
            if phase['status'] == 'collected':
                frozen.require(all(self.ledger.state['entries'][budget.record_id(k)]['state'] in ('settled', 'unknown_billed') for k in keys), 'Collected Batch has nonterminal requests')
            if phase.get('results_sha256'):
                frozen.require(self.path(name, 'results.json').is_file() and common.file_sha256(self.path(name, 'results.json')) == phase['results_sha256'], 'Archived Batch results changed')
                archive = json.loads(self.path(name, 'results.json').read_text())
                self.validate_batch(archive['batch'], phase)
                source = {}
                expected_ids = {custom_id(k) for k in keys}
                for file in archive['files'].values():
                    for line in file['text'].splitlines():
                        if not line.strip():
                            continue
                        item = json.loads(line); identifier = item.get('custom_id')
                        frozen.require(identifier in expected_ids and identifier not in source, 'Archived result coverage changed')
                        source[identifier] = item
                for key in keys:
                    entry = self.ledger.state['entries'][budget.record_id(key)]
                    if entry['state'] not in ('settled', 'unknown_billed'):
                        continue
                    expected, _ = parse_response(self.row(by_id[key[0]], key[2]), source.get(custom_id(key)))
                    for field in ('raw', 'probability', 'status', 'error', 'returned_model', 'response_status', 'response_received', 'response_id', 'usage', 'returned_service_tier', 'batch_result_sha256', 'batch_request_id'):
                        frozen.require(entry['row'].get(field) == expected.get(field), 'Saved completion differs from its archived provider result')
                    frozen.require(entry['row'].get('batch_id') == phase['batch_id'] and entry['row'].get('batch_custom_id') == custom_id(key), 'Saved completion Batch identity changed')
            if phase.get('keys'):
                rows = [saved.get(tuple(k)) or self.row(by_id[k[0]], k[2]) for k in phase['keys']]
                expected = self.input_text(rows)
                frozen.require(phase.get('input_sha256') == common.text_sha256(expected) and phase.get('input_path') == str(self.path(name, 'input.jsonl'))
                               and self.path(name, 'input.jsonl').read_text() == expected, 'Saved Batch input body/provenance changed')
                frozen.require(phase.get('metadata') == self.metadata(name, phase['input_sha256']), 'Saved Batch metadata changed')

    def metadata(self, name, digest):
        return {'freeze_sha256': self.preflight['binding']['freeze_sha256'], 'phase': name,
                'input_sha256': digest, 'operation_id': cert.fingerprint([self.signature, name, digest])}

    def phase_rows(self, name):
        if name == 'updates':
            frozen.require(self.phases.get('baselines', {}).get('status') == 'collected', 'All baselines must be terminal and collected before updates')
        rows = []
        for family in self.freeze['family_order']:
            for unit in (u for u in self.units if u['family_id'] == family):
                for condition in (('baseline',) if name == 'baselines' else common.CONDITIONS):
                    key = (unit['trial_id'], 'baseline' if name == 'baselines' else 'update', condition)
                    entry = self.ledger.state['entries'][budget.record_id(key)]
                    row = entry['row'] or self.row(unit, condition)
                    if row['status'] == 'blocked_baseline' or (self.ledger.state['halted'] and entry['state'] == 'reserved_unattempted'):
                        if entry['state'] == 'reserved_unattempted':
                            if row['status'] != 'blocked_baseline':
                                row['error'] = {'kind': 'not_attempted_after_fatal_error'}
                            self.ledger.unattempted(row)
                        continue
                    if entry['state'] != 'unattempted_terminal':
                        rows.append(row)
        return rows

    def input_text(self, rows):
        return ''.join(common.canonical_json({'custom_id': custom_id(common.record_key(r)), 'method': 'POST',
                     'url': ENDPOINT, 'body': frozen.request_payload(r['prompt'], self.freeze)}) + '\n' for r in rows)

    def update_counts(self, rows, *, collect=False, reconciled_receipts=()):
        requests = []
        for row in rows:
            body = frozen.request_payload(row['prompt'], self.freeze)
            count = {field: body[field] for field in cert.COUNT_FIELDS}
            requests.append({'request_id': custom_id(common.record_key(row)), 'method': 'POST', 'endpoint': cert.ENDPOINT,
                             'generation_body': body, 'generation_body_sha256': cert.fingerprint(body),
                             'count_body': count, 'count_body_sha256': cert.fingerprint(count)})
        manifest = {'schema_version': 'frontier_input_count_requests_v1', 'requests': requests}
        path = self.path('updates', 'counts.json')
        binding = {**self.preflight['binding'], 'phase': 'updates', 'run_signature': self.signature}
        if collect:
            state = counter.collect_requests(manifest, path, binding, self.get_client, reconciled_receipts)
        else:
            frozen.require(path.exists(), 'Exact materialized update counts must be collected before submission')
            state = json.loads(path.read_text())
        frozen.require(state.get('status') == 'complete' and state.get('binding') == binding and state.get('requests') == manifest
                       and state.get('request_manifest_sha256') == cert.fingerprint(manifest), 'Update count collection changed')
        frozen.require(set(state['counts']) == {r['request_id'] for r in requests}
                       and all(c.get('status') == 'received' for c in state['counts'].values()), 'Missing materialized update receipts')
        cert.import_receipts(manifest, [state['counts'][r['request_id']]['receipt'] for r in requests])
        for row, request in zip(rows, requests):
            key = budget.record_id(common.record_key(row))
            try:
                cert.validate_materialized_count(request['generation_body'], state['counts'][request['request_id']]['receipt'], self.specs[key]['input_token_upper_bound'])
            except ValueError:
                self.ledger.state.update(halted=True, halt_reason='materialized_count_exceeded_or_changed_frozen_ceiling')
                self.ledger.persist()
                raise
        return {'count_requests': len(requests), 'count_collection_sha256': common.file_sha256(path)}

    def submit(self, name):
        frozen.require(not self.ledger.state['halted'], 'Ledger halted; no additional generation permitted')
        phase = self.phases.get(name)
        if phase and phase.get('batch_id'):
            return {'batch_id': phase['batch_id'], 'phase_status': phase['status'], 'new_submissions': 0}
        rows = self.phase_rows(name)
        if not rows:
            self.phases[name] = {'status': 'collected', 'keys': [], 'no_eligible_requests': True}
            self.ledger.persist()
            return {'phase_status': 'collected', 'new_submissions': 0}
        if name == 'updates':
            self.update_counts(rows)
        text = self.input_text(rows)
        digest = common.text_sha256(text)
        if phase is None:
            path = self.path(name, 'input.jsonl')
            frozen.require(not path.exists() or path.read_text() == text, 'Existing input file differs from the frozen requests')
            common.atomic_write(path, text)
            phase = {'status': 'prepared', 'keys': [list(common.record_key(r)) for r in rows],
                     'input_path': str(path), 'input_sha256': digest, 'metadata': self.metadata(name, digest)}
            self.phases[name] = phase
            self.ledger.persist()
        frozen.require(phase['status'] in ('prepared', 'uploaded'), 'Ambiguous prior submission/upload: reconcile its existing ID; never resubmit automatically')
        if phase['status'] == 'prepared':
            phase['status'] = 'upload_intent'; self.ledger.persist()
            try:
                with self.path(name, 'input.jsonl').open('rb') as handle:
                    uploaded = as_dict(self.get_client().files.create(file=handle, purpose='batch'))
                frozen.require(isinstance(uploaded.get('id'), str) and uploaded['id'], 'Upload returned no file ID')
                phase.update(status='uploaded', input_file_id=uploaded['id']); self.ledger.persist()
            except BaseException:
                phase['status'] = 'upload_unknown'; self.ledger.persist()
                raise
        # Save the creation intent and full in-flight reservations atomically
        # before the only call that can queue billable generation.
        phase['status'] = 'creation_intent'
        self.ledger.begin(rows, pilot.PROBABILITY_SCHEMA)
        try:
            result = as_dict(self.get_client().batches.create(input_file_id=phase['input_file_id'], endpoint=ENDPOINT,
                                completion_window='24h', metadata=phase['metadata']))
            batch_id = result.get('id')
            frozen.require(isinstance(batch_id, str) and batch_id, 'Batch creation returned no batch ID')
            phase.update(status='submitted', batch_id=batch_id); self.ledger.persist()
            self.validate_batch(result, phase)
        except BaseException:
            # If an ID was received, keep it for retrieve-by-ID even when a
            # later provenance check fails. Never replace an ambiguous batch.
            if not phase.get('batch_id'):
                phase['status'] = 'creation_unknown'
            self.ledger.persist()
            raise
        return {'batch_id': phase['batch_id'], 'phase_status': phase['status'], 'new_submissions': 1}

    def validate_batch(self, result, phase):
        frozen.require(result.get('id') == phase.get('batch_id') and result.get('input_file_id') == phase['input_file_id']
                       and result.get('endpoint') == ENDPOINT and result.get('metadata') == phase['metadata'], 'Remote Batch identity/provenance differs from the frozen submission')

    def adopt(self, name, kind, remote_id):
        frozen.require(isinstance(remote_id, str) and bool(remote_id), 'Reconciliation requires an explicit existing remote ID')
        phase = self.phases.get(name)
        frozen.require(phase is not None, 'There is no prior operation to reconcile')
        if kind == 'file':
            frozen.require(phase['status'] in ('upload_intent', 'upload_unknown') and not phase.get('batch_id'), 'Only an ambiguous file upload may be reconciled')
            content = file_text(self.get_client(), remote_id)
            frozen.require(common.text_sha256(content) == phase['input_sha256'], 'Existing remote file differs from the exact frozen input')
            phase.update(input_file_id=remote_id, status='uploaded'); self.ledger.persist()
        else:
            frozen.require(phase['status'] in ('creation_intent', 'creation_unknown', 'submitted'), 'No ambiguous Batch creation to reconcile')
            frozen.require(not phase.get('batch_id') or phase['batch_id'] == remote_id, 'Cannot replace a known Batch ID')
            result = as_dict(self.get_client().batches.retrieve(remote_id))
            proposed = {**phase, 'batch_id': remote_id}
            self.validate_batch(result, proposed)
            phase.update(batch_id=remote_id, status='submitted'); self.ledger.persist()
        return {'phase_status': phase['status'], 'reconciled_id': remote_id, 'new_submissions': 0}

    def collect(self, name):
        phase = self.phases.get(name)
        frozen.require(phase is not None, 'Submit a phase before collecting it')
        if phase['status'] == 'collected':
            if name == 'baselines':
                self.phase_rows('updates')  # Finish blocked branches after an interrupted local checkpoint.
            return {'phase_status': 'collected', 'new_submissions': 0}
        frozen.require(bool(phase.get('batch_id')), 'Unknown Batch result: reconcile an existing batch ID before collecting')
        cache_path = self.path(name, 'results.json')
        if phase.get('results_sha256'):
            frozen.require(cache_path.exists() and common.file_sha256(cache_path) == phase['results_sha256'], 'Archived Batch results changed')
            cache = json.loads(cache_path.read_text())
        else:
            result = as_dict(self.get_client().batches.retrieve(phase['batch_id']))
            self.validate_batch(result, phase)
            phase['remote_status'] = result.get('status'); self.ledger.persist()
            if result.get('status') not in TERMINAL:
                return {'batch_id': phase['batch_id'], 'phase_status': 'waiting', 'remote_status': result.get('status'), 'new_submissions': 0}
            files = {}
            for field in ('output_file_id', 'error_file_id'):
                if result.get(field):
                    files[field] = {'file_id': result[field], 'text': file_text(self.get_client(), result[field])}
            cache = {'batch': result, 'files': files}
            common.atomic_write(cache_path, json.dumps(cache, sort_keys=True, indent=2) + '\n')
            phase['results_sha256'] = common.file_sha256(cache_path); self.ledger.persist()
        self.validate_batch(cache['batch'], phase)
        frozen.require(cache['batch'].get('status') in TERMINAL, 'Cached batch is not terminal')
        expected = {custom_id(k): tuple(k) for k in phase['keys']}
        indexed = {}
        for file in cache['files'].values():
            for line in file['text'].splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get('custom_id')
                frozen.require(key in expected and key not in indexed, 'Unknown or duplicate Batch result ID')
                indexed[key] = record
        # Validate every downloaded identifier before accepting any completion.
        for identifier, key in expected.items():
            entry = self.ledger.state['entries'][budget.record_id(key)]
            if entry['state'] != 'inflight':
                frozen.require(entry['state'] in ('settled', 'unknown_billed'), 'Terminal Batch has an unattempted entry')
                continue  # Resume after an interrupted local collection, no call.
            row, fatal = parse_response(entry['row'], indexed.get(identifier))
            row.update(batch_id=phase['batch_id'], batch_custom_id=identifier)
            if fatal:
                self.ledger.state.update(halted=True, halt_reason='fatal_batch_model_contract_or_request_error')
            self.ledger.settle(row)
        phase['status'] = 'collected'; self.ledger.persist()
        if name == 'baselines':
            self.phase_rows('updates')  # Retain all planned blocked branches now.
        return {'batch_id': phase['batch_id'], 'phase_status': 'collected', 'remote_status': cache['batch']['status'], 'new_submissions': 0}

    def checkpoint(self, detail=None):
        if self.phases.get('baselines', {}).get('status') == 'collected':
            self.phase_rows('updates')  # Materialize every blocked or halted planned branch, including on exceptions.
        rows = self.ledger.rows()
        # In-flight placeholders belong to the durable ledger, not a completed
        # response dataset. Every terminal planned failure remains in output.
        completed = [e['row'] for e in self.ledger.state['entries'].values() if e['state'] in ('settled', 'unknown_billed', 'unattempted_terminal')]
        complete = len(completed) == len(self.specs)
        status = 'failed' if self.ledger.state['halted'] else ('complete' if complete and all(r['status'] == 'ok' for r in completed) else 'complete_with_errors' if complete else 'pending')
        summary = {'status': status, 'binding': self.preflight['binding'], 'run_signature': self.signature,
                   'planned_records': len(self.specs), 'records': len(completed), 'status_counts': dict(Counter(r['status'] for r in completed)),
                   'attempted_calls': sum(r['request_attempted'] for r in rows.values()), 'authorization_ledger': str(self.ledger.path),
                   'budget_cap_usd': budget.dollars(self.ledger.cap), 'exposure_usd': budget.dollars(self.ledger.exposure()),
                   'cohort_worst_case_usd': self.preflight['budget']['cohort_worst_case_usd'],
                   'unknown_billed_calls': sum(e['state'] == 'unknown_billed' for e in self.ledger.state['entries'].values()),
                   'inflight_calls': sum(e['state'] == 'inflight' for e in self.ledger.state['entries'].values()),
                   'halt_reason': self.ledger.state.get('halt_reason'), 'phases': self.phases, 'updated_at': common.utc_now(), **(detail or {})}
        common.atomic_write(self.output, ''.join(common.canonical_json(r) + '\n' for r in completed))
        common.atomic_write(Path(str(self.output) + '.manifest.json'), json.dumps(summary, sort_keys=True, indent=2) + '\n')
        return summary


def execute(freeze_path: Path, action='status', *, live=False, remote_id=None, client_factory=None, ledger_path=None, receipt_file=None):
    freeze_path = freeze_path.resolve()
    freeze, units, specs, preflight = frozen.validate_freeze(freeze_path)
    counter.verify_code(freeze, ('run_frozen_frontier_batch.py',))
    frozen.require(action in ACTIONS and freeze['pricing'] == 'batch', 'Use a declared Batch action and Batch prices')
    frozen.require(preflight['budget']['budget_cap_nusd'] <= 24 * budget.NANO, 'Operational cohort cap must be at most $24')
    validate_certificate_source(freeze, units)
    frozen.require(preflight['budget']['fits_entire_cohort'], 'Entire frozen cohort worst case exceeds cap; no generation permitted')
    output = Path(freeze['output_path'])
    ledger_path = ledger_path or budget.LEDGER_PATH
    generated = [output, Path(str(output) + '.manifest.json'), *[Path(str(output) + f'.batch_{phase}.{suffix}')
        for phase in ('baselines', 'updates') for suffix in ('input.jsonl', 'results.json')], Path(str(output) + '.batch_updates.counts.json')]
    frozen.validate_output_paths(freeze_path, freeze, generated, ledger_path=ledger_path)
    ledger = budget.Ledger(ledger_path, preflight['binding'], specs, preflight['budget']['budget_cap_nusd'], 'batch')
    if not live or action == 'status':
        ledger.load()
        run = BatchRun(freeze_path, freeze, units, specs, preflight, ledger, client_factory)
        run.validate_rows()
        return {**preflight, 'status': 'offline_valid', 'requested_action': action,
                'existing_exposure_usd': budget.dollars(ledger.exposure()), 'phases': run.phases,
                'api_calls': 0, 'credential_reads': 0, 'output_writes': 0}
    with ledger.session():
        frozen.require(ledger.path.exists() or not output.exists(), 'Output exists without its shared authorization ledger')
        run = BatchRun(freeze_path, freeze, units, specs, preflight, ledger, client_factory)
        run.validate_rows()
        ledger.persist()  # Full immutable cohort reserved before client/upload.
        try:
            if action == 'count-updates':
                frozen.require(not ledger.state['halted'], 'Ledger halted; no further requests permitted')
                detail = run.update_counts(run.phase_rows('updates'), collect=True, reconciled_receipts=json.loads(receipt_file.read_text()) if receipt_file else ())
            elif action.startswith('submit-'):
                detail = run.submit(action.split('-', 1)[1])
            elif action.startswith('collect-'):
                detail = run.collect(action.split('-', 1)[1])
            else:
                _, singular, kind = action.split('-')
                detail = run.adopt(singular + 's', kind, remote_id)
            return run.checkpoint(detail)
        except BaseException:
            run.checkpoint({'operation_interrupted': True})
            raise
        finally:
            if run.client is not None:
                try:
                    run.client.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-manifest', type=Path, required=True)
    parser.add_argument('--action', choices=ACTIONS, default='status')
    parser.add_argument('--live', action='store_true', help='Allow only this explicit remote step after immutable freeze and whole-cohort budget gates')
    parser.add_argument('--remote-id', help='Existing file/batch ID for an explicit adopt action; never creates a replacement')
    parser.add_argument('--receipt-file', type=Path, help='For count-updates only: archived receipts for explicit reconciliation of unknown count results')
    args = parser.parse_args()
    report = execute(args.freeze_manifest, args.action, live=args.live, remote_id=args.remote_id, receipt_file=args.receipt_file)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        print('Frozen Batch validation/execution failed; no secret-bearing exception text is logged.', file=sys.stderr)
        sys.exit(2)
