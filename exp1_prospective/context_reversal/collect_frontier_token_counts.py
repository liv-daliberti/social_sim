#!/usr/bin/env python3
"""Explicit, resumable input-count collection; offline unless --live is supplied.

The complete fresh-material freeze/review gate runs before credential access.
This calls only the official non-generation /responses/input_tokens endpoint.
Each count receipt is durable; an ambiguous count request is never repeated
silently. The exported/imported certificate keeps exact body hashes and trace IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_token_certificate as cert
from exp1_prospective.context_reversal import run_frozen_frontier as frozen
from exp1_prospective.context_reversal import run_openai as pilot
from exp1_prospective.context_reversal import run_local as common


def verify_code(freeze: dict, names=()):
    for name in ('collect_frontier_token_counts.py', *names):
        frozen.require(freeze['code_sha256'].get(name) == common.file_sha256(frozen.SOURCE / name), f'Frozen transport code mismatch: {name}')


def count_binding(freeze: dict) -> dict:
    return {'evaluation_id': freeze['evaluation_id'], 'model': freeze['model'],
            'max_output_tokens': freeze['max_output_tokens'], 'pricing': freeze['pricing'],
            'output_path': str(Path(freeze['output_path']).resolve()),
            'token_certificate_output': str(Path(freeze['token_certificate_output']).resolve()),
            'token_count_collection_output': str(Path(str(Path(freeze['token_certificate_output']).resolve()) + '.collection.json').resolve()),
            'authorization_scope': freeze['authorization_scope'],
            'budget_usd': freeze['budget_usd'], 'family_order': freeze['family_order'], 'contexts': freeze['contexts'],
            'artifacts': {key: freeze['artifacts'][key] for key in ('plan', 'protocol', 'scoring', 'review', 'freshness')}}


def collect_requests(requests: dict, state_path: Path, binding: dict, get_client, reconciled_receipts=()) -> dict:
    """Caller holds the appropriate process lock; get_client is lazy/injectable."""
    digest = cert.fingerprint(requests)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'schema_version': 'frontier_token_collection_v1', 'binding': binding,
        'request_manifest_sha256': digest, 'requests': requests, 'counts': {},
        'created_at': common.utc_now(), 'status': 'collecting'}
    frozen.require(state.get('binding') == binding and state.get('request_manifest_sha256') == digest
                   and state.get('requests') == requests, 'Count collection differs from frozen request bodies')
    by_id = {r['request_id']: r for r in requests['requests']}
    frozen.require(len(by_id) == len(requests['requests']) and set(state['counts']) <= set(by_id), 'Duplicate or unknown count request')
    for receipt in reconciled_receipts:
        key = receipt['request_id']
        entry = state['counts'].get(key)
        frozen.require(key in by_id and entry is not None and entry['status'] in ('inflight', 'result_unknown', 'received'), 'Receipt has no corresponding prior count attempt')
        cert.import_receipts({'schema_version': 'frontier_input_count_requests_v1', 'requests': [by_id[key]]}, [receipt])
        if entry['status'] == 'received':
            frozen.require(entry['receipt'] == receipt, 'Cannot replace an already received count receipt')
        else:
            state['counts'][key] = {'status': 'received', 'receipt': receipt, 'explicit_receipt_reconciliation': True}
            common.atomic_write(state_path, json.dumps(state, sort_keys=True, indent=2) + '\n')
    for request in requests['requests']:
        key = request['request_id']
        entry = state['counts'].get(key)
        if entry:
            frozen.require(entry['status'] == 'received', 'An input-count request has an unknown result; reconcile its receipt instead of resubmitting')
            # Validate body and provider response even on an already-completed resume.
            cert.import_receipts({'schema_version': 'frontier_input_count_requests_v1', 'requests': [request]}, [entry['receipt']])
            continue
        state['counts'][key] = {'status': 'inflight', 'count_body_sha256': request['count_body_sha256']}
        common.atomic_write(state_path, json.dumps(state, sort_keys=True, indent=2) + '\n')
        try:
            raw = get_client().responses.input_tokens.with_raw_response.count(**request['count_body'])
            result = raw.parse()
            response = result.model_dump(mode='json') if hasattr(result, 'model_dump') else {'object': result.object, 'input_tokens': result.input_tokens}
            receipt = {'request_id': key, 'endpoint': cert.ENDPOINT, 'http_status': raw.status_code,
                       'count_body_sha256': request['count_body_sha256'],
                       'provider_request_id': raw.headers.get('x-request-id'), 'received_at': common.utc_now(), 'response': response}
            cert.import_receipts({'schema_version': 'frontier_input_count_requests_v1', 'requests': [request]}, [receipt])
        except BaseException:
            state['counts'][key]['status'] = 'result_unknown'
            state['status'] = 'blocked_unknown_count_receipt'
            common.atomic_write(state_path, json.dumps(state, sort_keys=True, indent=2) + '\n')
            raise
        state['counts'][key] = {'status': 'received', 'receipt': receipt}
        common.atomic_write(state_path, json.dumps(state, sort_keys=True, indent=2) + '\n')
    state['status'] = 'complete'
    state['completed_at'] = state.get('completed_at', common.utc_now())
    common.atomic_write(state_path, json.dumps(state, sort_keys=True, indent=2) + '\n')
    return state


def collect(freeze_path: Path, *, live=False, client_factory=None, authorization_path: Path | None = None, receipt_file: Path | None = None) -> dict:
    freeze, units, _, _ = frozen.validate_freeze(freeze_path)
    verify_code(freeze)
    frozen.require(budget.usd_to_nusd(freeze['budget_usd']) <= 24 * budget.NANO, 'Operational cohort cap must be at most$24')
    requests = cert.request_manifest(units, freeze['artifacts']['plan']['sha256'], freeze['max_output_tokens'])
    output = Path(freeze['token_certificate_output'])
    frozen.require(output.is_absolute(), 'Token-certificate output must be frozen as an absolute path')
    output = output.resolve()
    state_path = Path(str(output) + '.collection.json').resolve()
    auth = authorization_path or budget.LEDGER_PATH.with_name('token_count_authorization.json')
    if {'token_certificate', 'token_count_collection'} & set(freeze['artifacts']):
        frozen.require({'token_certificate', 'token_count_collection'} <= set(freeze['artifacts']), 'Freeze both certificate and collection together')
        frozen.require(frozen.checked_artifact(freeze['artifacts']['token_certificate']) == output.resolve()
                       and frozen.checked_artifact(freeze['artifacts']['token_count_collection']) == state_path.resolve(), 'Final frozen count artifact paths changed')
        state = json.loads(state_path.read_text())
        frozen.require(state.get('binding') == count_binding(freeze) and state.get('requests') == requests
                       and state.get('request_manifest_sha256') == cert.fingerprint(requests) and state.get('status') == 'complete', 'Final count collection binding changed')
        frozen.require(set(state['counts']) == {r['request_id'] for r in requests['requests']}
                       and all(c.get('status') == 'received' for c in state['counts'].values()), 'Final count collection incomplete')
        certificate = cert.import_receipts(requests, [state['counts'][r['request_id']]['receipt'] for r in requests['requests']])
        frozen.require(json.loads(output.read_text()) == certificate, 'Final certificate differs from its receipts')
        return {'status': 'complete', 'count_requests': len(requests['requests']), 'certificate_path': str(output),
                'certificate_sha256': common.file_sha256(output), 'collection_path': str(state_path),
                'collection_sha256': common.file_sha256(state_path), 'generation_calls': 0, 'api_calls': 0, 'credential_reads': 0, 'output_writes': 0}
    frozen.validate_output_paths(freeze_path, freeze, [output, state_path, Path(freeze['output_path']), Path(freeze['output_path'] + '.manifest.json')], ledger_path=auth)
    if not live:
        return {'status': 'offline_count_plan', 'count_requests': len(requests['requests']),
                'output': str(output), 'request_manifest_sha256': cert.fingerprint(requests), 'api_calls': 0, 'credential_reads': 0}
    binding = count_binding(freeze)
    client = None
    def get_client():
        nonlocal client
        if client is None:
            client = (client_factory or pilot.create_client)()
        return client
    with common.output_lock(Path(str(auth) + '.lock')):
        frozen.require(not output.exists() or state_path.exists(), 'Certificate exists without its receipt collection; refusing unsafe recount')
        if auth.exists():
            frozen.require(json.loads(auth.read_text()) == binding, 'Count authorization is already bound to another evaluation')
        else:
            common.atomic_write(auth, json.dumps(binding, sort_keys=True, indent=2) + '\n')
        try:
            if output.exists():
                state = json.loads(state_path.read_text())
                frozen.require(state.get('status') == 'complete' and state.get('binding') == binding
                               and state.get('requests') == requests and state.get('request_manifest_sha256') == cert.fingerprint(requests), 'Existing certificate has no complete matching collection')
                frozen.require(set(state['counts']) == {r['request_id'] for r in requests['requests']}
                               and all(c.get('status') == 'received' for c in state['counts'].values()), 'Existing certificate collection is missing received counts')
                existing = cert.import_receipts(requests, [state['counts'][r['request_id']]['receipt'] for r in requests['requests']])
                frozen.require(json.loads(output.read_text()) == existing, 'Existing certificate differs from its collected receipts')
                return {'status': 'complete', 'count_requests': len(requests['requests']), 'certificate_path': str(output),
                        'certificate_sha256': common.file_sha256(output), 'collection_path': str(state_path),
                        'collection_sha256': common.file_sha256(state_path), 'generation_calls': 0}
            state = collect_requests(requests, state_path, binding, get_client, json.loads(receipt_file.read_text()) if receipt_file else ())
            certificate = cert.import_receipts(requests, [state['counts'][r['request_id']]['receipt'] for r in requests['requests']])
            text = json.dumps(certificate, sort_keys=True, indent=2) + '\n'
            frozen.require(not output.exists() or output.read_text() == text, 'Existing token certificate differs; refusing replacement')
            if not output.exists():
                common.atomic_write(output, text)
            return {'status': 'complete', 'count_requests': len(requests['requests']), 'certificate_path': str(output),
                    'certificate_sha256': common.file_sha256(output), 'collection_path': str(state_path),
                    'collection_sha256': common.file_sha256(state_path), 'generation_calls': 0}
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-manifest', required=True, type=Path)
    parser.add_argument('--live', action='store_true', help='Explicitly authorize count requests after all frozen review gates pass')
    parser.add_argument('--receipt-file', type=Path, help='Explicit archived receipt list to reconcile previously unknown count requests; never reissue them')
    args = parser.parse_args()
    print(json.dumps(collect(args.freeze_manifest.resolve(), live=args.live, receipt_file=args.receipt_file), sort_keys=True, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Frozen count collection failed; no secret-bearing exception text is logged.', file=sys.stderr)
        sys.exit(2)
