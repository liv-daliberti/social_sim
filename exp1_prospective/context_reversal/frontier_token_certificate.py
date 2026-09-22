#!/usr/bin/env python3
"""Offline export/import of Responses input-token count evidence.

No API calls, SDK imports, or credential reads. The export describes exact count
requests; a future authorized collector must obtain receipts from the official
endpoint. Imported receipts are evidence, not cryptographically authenticated
provider statements; archive/review their provenance before freezing them.

Baselines have exact-body counts. An update has an unknown prior until baseline
completion: its bound uses a certified identical message/schema envelope plus
one token per UTF-8 byte of its ENTIRE possible user text, with32 prior bytes.
This is a conservative content ceiling, not an exact count of a future update.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp1_prospective.context_reversal import run_openai as pilot
from exp1_prospective.context_reversal import run_local as common

ENDPOINT = 'https://api.openai.com/v1/responses/input_tokens'
DOCS = 'https://developers.openai.com/api/docs/guides/token-counting'
REFERENCE = 'https://developers.openai.com/api/reference/python/resources/responses/subresources/input_tokens/methods/count'
COUNT_FIELDS = ('model', 'input', 'reasoning', 'text', 'tools', 'truncation')
PRIOR_BYTES = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


def body(prompt: str, max_output_tokens: int) -> dict:
    return {'model': 'gpt-5.6-sol', 'input': [{'role': 'user', 'content': prompt}],
            'reasoning': {'effort': 'low'}, 'max_output_tokens': max_output_tokens,
            'text': {'format': {'type': 'json_schema', 'name': 'forecast_probability', 'strict': True,
                                'schema': pilot.PROBABILITY_SCHEMA}},
            'service_tier': 'default', 'store': False, 'tools': [], 'truncation': 'disabled'}


def fingerprint(value) -> str:
    return common.text_sha256(common.canonical_json(value))


def request_manifest(units: list[dict], plan_sha256: str, max_output_tokens: int) -> dict:
    require(type(max_output_tokens) is int and max_output_tokens in (1024, 2048), 'Unsupported output budget')
    requests = []
    # Same model, user-message shape, reasoning and full structured-output schema.
    # The anchor's own token count is retained, never subtracted.
    for request_id, prompt in [('framing_envelope', 'x'), *[(f"baseline:{u['trial_id']}", u['baseline_prompt']) for u in units]]:
        generation = body(prompt, max_output_tokens)
        count = {field: generation[field] for field in COUNT_FIELDS}
        requests.append({'request_id': request_id, 'method': 'POST', 'endpoint': ENDPOINT,
                         'generation_body': generation, 'generation_body_sha256': fingerprint(generation),
                         'count_body': count, 'count_body_sha256': fingerprint(count)})
    return {'schema_version': 'frontier_input_count_requests_v1', 'plan_sha256': plan_sha256,
            'model': 'gpt-5.6-sol', 'max_output_tokens': max_output_tokens, 'requests': requests,
            'units_sha256': fingerprint(units),
            'prior_serialized_utf8_byte_limit': PRIOR_BYTES,
            'update_bound': 'certified_identical_envelope_count_plus_full_user_prompt_utf8_bytes_including32_prior_bytes',
            'documentation': [DOCS, REFERENCE]}


def import_receipts(requests: dict, receipts: list[dict]) -> dict:
    require(requests.get('schema_version') == 'frontier_input_count_requests_v1', 'Unknown request-manifest schema')
    expected = {row['request_id']: row for row in requests['requests']}
    require(len(expected) == len(requests['requests']), 'Duplicate count request')
    indexed = {}
    for receipt in receipts:
        key = receipt['request_id']
        require(key in expected and key not in indexed, 'Unknown or duplicate count receipt')
        original = expected[key]
        require(receipt.get('endpoint') == ENDPOINT and receipt.get('http_status') == 200, 'Receipt is not a successful official input-count request')
        require(receipt.get('count_body_sha256') == original['count_body_sha256'], 'Receipt count body changed')
        require(original['count_body_sha256'] == fingerprint(original['count_body']) and original['generation_body_sha256'] == fingerprint(original['generation_body']), 'Exported request hash mismatch')
        require(original['count_body'] == {field: original['generation_body'][field] for field in COUNT_FIELDS}, 'Count body omitted or altered an input-affecting generation field')
        response = receipt.get('response')
        require(isinstance(response, dict) and response.get('object') == 'response.input_tokens'
                and type(response.get('input_tokens')) is int and 0 < response['input_tokens'] <= 272000,
                'Invalid provider input-token count response')
        require(isinstance(receipt.get('provider_request_id'), str) and bool(receipt['provider_request_id'].strip()), 'Provider request trace identifier missing')
        require(isinstance(receipt.get('received_at'), str) and bool(receipt['received_at']), 'Receipt timestamp missing')
        indexed[key] = {**receipt, 'response_sha256': fingerprint(response)}
    require(set(indexed) == set(expected), 'Input-count receipts do not cover every exact baseline and the framing envelope')
    return {'schema_version': 'frontier_input_token_certificate_v1',
            'request_manifest': requests, 'request_manifest_sha256': fingerprint(requests),
            'receipts': [indexed[row['request_id']] for row in requests['requests']],
            'source_provenance': 'Externally collected official endpoint receipts; review/retain transport request IDs',
            'updates_are_exact_counts': False, 'baseline_counts_are_exact_for_hashed_bodies': True}


def certified_bounds(units: list[dict], plan_sha256: str, max_output_tokens: int, certificate: dict) -> dict:
    expected = request_manifest(units, plan_sha256, max_output_tokens)
    require(certificate.get('schema_version') == 'frontier_input_token_certificate_v1', 'Unknown token-certificate schema')
    require(certificate.get('request_manifest') == expected and certificate.get('request_manifest_sha256') == fingerprint(expected), 'Certificate differs from the exact plan/model/schema/reasoning/output contract')
    # Revalidate every receipt rather than trusting imported derived numbers.
    imported = import_receipts(expected, certificate['receipts'])
    require(imported == certificate, 'Token certificate metadata or receipts were altered')
    counts = {r['request_id']: r['response']['input_tokens'] for r in certificate['receipts']}
    frame = counts['framing_envelope']
    result = {}
    for unit in units:
        key = common.canonical_json([unit['trial_id'], 'baseline', 'baseline'])
        result[key] = {'input_token_upper_bound': counts[f"baseline:{unit['trial_id']}"],
                       'input_bound_method': 'api_exact_baseline_body',
                       'certified_prompt_sha256': common.text_sha256(unit['baseline_prompt'])}
        for arm in common.CONDITIONS:
            template = unit['update_templates'][arm]
            prompt = template.replace(common.PRIOR_TOKEN, '0.' + '0' * (PRIOR_BYTES - 2))
            ceiling = len(prompt.encode('utf-8'))
            require(frame + ceiling <= 272000, 'Certified update bound exceeds short-context pricing')
            key = common.canonical_json([unit['trial_id'], 'update', arm])
            result[key] = {'input_token_upper_bound': frame + ceiling,
                           'input_bound_method': 'api_envelope_plus_full_utf8_content_ceiling',
                           'certified_envelope_count': frame, 'max_prompt_utf8_bytes': ceiling}
    return result


def validate_materialized_count(generation_body: dict, receipt: dict, upper_bound: int) -> int:
    """Future Batch adapter must check the actual rendered body before submission.

    This permits a second exact-body check once a baseline prior is known. It
    never silently substitutes a typical-token estimate or changes the frozen
    cohort when an unexpected provider count exceeds the reserved ceiling.
    """
    count = {field: generation_body[field] for field in COUNT_FIELDS}
    require(receipt.get('endpoint') == ENDPOINT and receipt.get('http_status') == 200,
            'Materialized count was not obtained successfully from the official endpoint')
    require(receipt.get('count_body_sha256') == fingerprint(count), 'Materialized request body does not match its count receipt')
    response = receipt.get('response', {})
    total = response.get('input_tokens')
    require(response.get('object') == 'response.input_tokens' and type(total) is int and 0 < total <= upper_bound,
            'Materialized input count exceeds the frozen certificate ceiling; no generation may be submitted')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    export = actions.add_parser('export')
    export.add_argument('--plan', type=Path, required=True)
    export.add_argument('--max-output-tokens', type=int, choices=(1024, 2048), default=2048)
    export.add_argument('--output', type=Path, required=True)
    importer = actions.add_parser('import')
    importer.add_argument('--requests', type=Path, required=True)
    importer.add_argument('--receipts', type=Path, required=True, help='JSON list of external endpoint receipts')
    importer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'export':
        result = request_manifest(common.read_units(args.plan), common.file_sha256(args.plan), args.max_output_tokens)
    else:
        result = import_receipts(json.loads(args.requests.read_text()), json.loads(args.receipts.read_text()))
    require(not args.output.exists(), 'Refusing to overwrite an existing token evidence artifact')
    common.atomic_write(args.output, json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': common.file_sha256(args.output), 'api_calls': 0, 'credential_reads': 0}))


if __name__ == '__main__':
    main()
