#!/usr/bin/env python3
"""Fail-closed runner for the Claude Opus 5 replication, total cap $25.

Same gating as the GPT-5.6 runner: the whole cohort's worst case must fit before
any request is admitted, every call is reserved before dispatch, in-flight and
unknown-billed calls keep their full reservation, and provider usage exceeding a
reservation halts the ledger. --dry-run never constructs a client or reads a key.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_budget_anthropic as authorization
from exp1_prospective.context_reversal import run_frozen_frontier as gpt
from exp1_prospective.context_reversal import run_openai as pilot
from exp1_prospective.context_reversal import run_local as common

SOURCE = Path(__file__).resolve().parent
MODEL = authorization.MODEL
require = gpt.require
parse_probability = gpt.parse_probability


def create_client(max_retries: int = 0):
    import anthropic
    return anthropic.AnthropicFoundry(api_key=os.environ['ANTHROPIC_FOUNDRY_API_KEY'],
                                      base_url=os.environ['ANTHROPIC_FOUNDRY_BASE_URL'],
                                      max_retries=max_retries)


def validate_freeze(path: Path) -> tuple[dict, list, dict, dict]:
    freeze = json.loads(path.read_text())
    require(freeze.get('schema_version') == 'frozen_frontier_evaluation_claude_v1' and freeze.get('status') == 'frozen', 'Evaluation is not frozen')
    require(freeze.get('cohort_inference_started_before_freeze') is False, 'Cohort inference must start after the freeze')
    require(freeze.get('authorization_scope') == authorization.AUTHORIZATION_SCOPE, 'Wrong single-evaluation authorization scope')
    require(freeze.get('model') == MODEL, 'Only the frozen claude-opus-5 model is permitted')
    require(freeze.get('max_output_tokens') in (1024, 2048), 'Freeze 1024 or 2048 output tokens before evaluation')
    require(freeze.get('effort') == 'low' and freeze.get('max_retries') == 0, 'Effort/retry contract changed')
    require(type(freeze.get('concurrency')) is int and 1 <= freeze['concurrency'] <= 4, 'Concurrency must be 1..4')
    require(freeze.get('pricing') == authorization.PRICING, 'Unknown pricing mode')
    require(freeze.get('prices_nusd_per_token') == authorization.PRICES_NUSD_PER_TOKEN, 'Frozen prices differ from the audited schedule')
    require(freeze.get('pricing_valid_through') == authorization.PRICE_VALID_THROUGH
            and datetime.now(timezone.utc).date() <= date.fromisoformat(authorization.PRICE_VALID_THROUGH), 'Pricing audit has expired')
    require(freeze.get('structured_output_schema') == authorization.PROBABILITY_SCHEMA, 'Frozen output schema changed')
    cap = budget.usd_to_nusd(freeze['budget_usd'])
    require(cap <= budget.ABSOLUTE_CAP_NUSD, 'Total authorization is at most $25')
    for filename, digest in freeze['code_sha256'].items():
        require(digest == common.file_sha256(SOURCE / filename), f'Frozen code mismatch: {filename}')
    artifacts = freeze['artifacts']
    require({'plan', 'protocol', 'scoring', 'review', 'freshness', 'token_certificate'} <= set(artifacts), 'Freeze requires plan, protocol, scoring, review, freshness and token certificate')
    paths = {name: gpt.checked_artifact(spec) for name, spec in artifacts.items()}
    units = common.read_units(paths['plan'])
    require(all(u['repeat'] == 0 for u in units), 'One repeat per evaluation unit must be frozen')
    contexts = set(freeze['contexts'])
    families = sorted({u['family_id'] for u in units})
    require(set(freeze['family_order']) == set(families) and len(freeze['family_order']) == len(families), 'Family order must cover each planned family exactly once')
    require(all({u['context_id'] for u in units if u['family_id'] == f} == contexts for f in families), 'Every family must have every frozen context')
    review, freshness = (json.loads(paths[name].read_text()) for name in ('review', 'freshness'))
    for name, record in (('review', review), ('freshness', freshness)):
        require(record.get('plan_sha256') == artifacts['plan']['sha256'] and sorted(record.get('family_ids', [])) == families, f'{name} does not cover the exact frozen plan')
    require(review.get('status') == 'complete' and review.get('unresolved_findings') == 0
            and review.get('target_model_outputs_used') is False and review.get('outcome_based_family_filtering') is False,
            'Review must be complete without target-model leakage or outcome-based filtering')
    require(freshness.get('status') == 'passed', 'Freshness audit missing')
    output = Path(freeze['output_path'])
    gpt.validate_output_paths(path, freeze, (output, Path(str(output) + '.manifest.json')), ledger_path=authorization.LEDGER_PATH)
    certificate = json.loads(paths['token_certificate'].read_text())
    entries = authorization.reservations(units, freeze['max_output_tokens'], certificate, artifacts['plan']['sha256'])
    summary = budget.plan_summary(entries, cap)
    binding = {'freeze_path': str(path.resolve()), 'freeze_sha256': common.file_sha256(path),
               'evaluation_id': freeze['evaluation_id'], 'output_path': str(output.resolve()),
               'plan_sha256': artifacts['plan']['sha256'],
               'schema_sha256': common.text_sha256(common.canonical_json(authorization.PROBABILITY_SCHEMA))}
    return freeze, units, entries, {'binding': binding, 'budget': summary}


def submit(client, row: dict, freeze: dict) -> tuple[dict, bool]:
    body = authorization.request_body(row['prompt'], freeze['max_output_tokens'])
    try:
        result = client.messages.create(**body)
    except Exception as exc:
        row['error'] = pilot.safe_api_error(exc)
        return row, pilot.fatal_api_error(row['error'])
    usage = getattr(result, 'usage', None)
    tokens = {'input_tokens': int(getattr(usage, 'input_tokens', 0) or 0),
              'output_tokens': int(getattr(usage, 'output_tokens', 0) or 0)}
    tokens['total_tokens'] = tokens['input_tokens'] + tokens['output_tokens']
    stop_details = getattr(result, 'stop_details', None)
    row.update(response_received=True, returned_model=getattr(result, 'model', None),
               response_id=getattr(result, 'id', None), response_status=getattr(result, 'stop_reason', None),
               stop_category=getattr(stop_details, 'category', None) if stop_details else None,
               usage=tokens, created_at=common.utc_now())
    row['raw'] = ''.join(b.text for b in result.content if getattr(b, 'type', None) == 'text')
    if row['returned_model'] != MODEL:
        row['error'] = {'kind': 'unexpected_returned_model'}
        return row, True
    if row['response_status'] == 'refusal':
        row.update(status='generation_error', error={'kind': 'model_refusal'})
        return row, False
    if row['response_status'] == 'max_tokens':
        row.update(status='generation_error', error={'kind': 'truncated_output_budget'})
        return row, False
    if row['response_status'] != 'end_turn':
        row.update(status='generation_error', error={'kind': 'unexpected_stop_reason'})
        return row, False
    try:
        row.update(probability=parse_probability(row['raw']), status='ok')
    except (ValueError, TypeError, OverflowError):
        row.update(status='parse_error', error={'kind': 'invalid_probability_json'})
    return row, False


def execute(path: Path, *, client_factory=None, ledger_path: Path | None = None) -> dict:
    freeze, units, entries, preflight = validate_freeze(path)
    require(preflight['budget']['fits_entire_cohort'], 'Entire frozen cohort worst case exceeds the cap; no calls permitted')
    output = Path(freeze['output_path'])
    ledger = authorization.Ledger(ledger_path or authorization.LEDGER_PATH, preflight['binding'], entries,
                                  preflight['budget']['budget_cap_nusd'], freeze['pricing'])
    config = {'input_sha256': freeze['artifacts']['plan']['sha256'], 'model_key': freeze['model_key'], 'model': MODEL,
              'freeze_sha256': preflight['binding']['freeze_sha256'],
              'decode': {'max_output_tokens': freeze['max_output_tokens'], 'effort': freeze['effort']}}
    by_id = {u['trial_id']: u for u in units}
    with ledger.session():
        require(ledger.path.exists() or not output.exists(), 'Output exists without its authorization ledger; refusing unsafe budget reset')
        ledger.recover_inflight()
        for key, row in ledger.rows().items():
            unit = by_id[key[0]]
            for field in ('trial_id', 'family_id', 'domain', 'context_id', 'repeat', 'material_status'):
                require(row.get(field) == unit[field], 'Persisted response metadata mismatch')
            require(row.get('run_signature') == common.text_sha256(common.canonical_json(config)), 'Persisted response signature mismatch')

        def row_for(unit, condition):
            stage = 'baseline' if condition == 'baseline' else 'update'
            if stage == 'baseline':
                return pilot.base_record(unit, stage, condition, unit['baseline_prompt'], config)
            baseline = ledger.rows().get((unit['trial_id'], 'baseline', 'baseline'))
            require(baseline is not None, 'An update cannot precede its own context baseline')
            prior = baseline['probability']
            prompt = common.render_update(unit, condition, prior) if baseline['status'] == 'ok' else None
            if prior is not None:
                require(len(json.dumps(prior).encode()) <= budget.PRIOR_SERIALIZED_BYTE_LIMIT, 'Prior exceeds the frozen serialization bound')
            row = pilot.base_record(unit, stage, condition, prompt, config, prior)
            if baseline['status'] != 'ok':
                row.update(status='blocked_baseline', error={'kind': 'baseline_has_no_valid_probability'})
            return row

        def checkpoint(status):
            if ledger.state['halted']:
                status = 'failed'
            rows = ledger.rows()
            common.atomic_write(output, ''.join(common.canonical_json(r) + '\n' for r in rows.values()))
            summary = {'status': status, 'binding': preflight['binding'], 'freeze': str(path.resolve()),
                       'authorization_ledger': str(ledger.path), 'planned_records': len(entries), 'records': len(rows),
                       'status_counts': dict(Counter(r['status'] for r in rows.values())),
                       'stop_reasons': dict(Counter(r.get('response_status') or 'not_attempted' for r in rows.values())),
                       'attempted_calls': sum(r['request_attempted'] for r in rows.values()),
                       'budget_cap_usd': budget.dollars(ledger.cap), 'exposure_usd': budget.dollars(ledger.exposure()),
                       'halted': ledger.state['halted'], 'halt_reason': ledger.state.get('halt_reason'),
                       'unknown_billed_calls': sum(e['state'] == 'unknown_billed' for e in ledger.state['entries'].values()),
                       'cohort_worst_case_usd': preflight['budget']['cohort_worst_case_usd'], 'updated_at': common.utc_now()}
            common.atomic_write(Path(str(output) + '.manifest.json'), json.dumps(summary, sort_keys=True, indent=2) + '\n')
            return summary

        def terminalize_unattempted():
            for unit in units:
                for condition in ('baseline', *common.CONDITIONS):
                    key = (unit['trial_id'], 'baseline' if condition == 'baseline' else 'update', condition)
                    if key not in ledger.rows():
                        row = row_for(unit, condition)
                        if row['status'] != 'blocked_baseline':
                            row['error'] = {'kind': 'not_attempted_after_fatal_error'}
                        ledger.unattempted(row)

        if ledger.state['halted']:
            terminalize_unattempted()
            return checkpoint('failed')
        if all(e['state'] != 'reserved_unattempted' for e in ledger.state['entries'].values()):
            return checkpoint('complete' if all(r['status'] == 'ok' for r in ledger.rows().values()) else 'complete_with_errors')
        client = None
        try:
            client = (client_factory or create_client)()
            fatal = False
            with ThreadPoolExecutor(max_workers=freeze['concurrency']) as executor:
                for family in freeze['family_order']:
                    family_units = [u for u in units if u['family_id'] == family]
                    for conditions in [('baseline',), common.CONDITIONS]:
                        pending = []
                        for unit in family_units:
                            for condition in conditions:
                                key = (unit['trial_id'], 'baseline' if condition == 'baseline' else 'update', condition)
                                if budget.record_id(key) not in entries or key in ledger.rows():
                                    continue
                                row = row_for(unit, condition)
                                if row['status'] == 'blocked_baseline':
                                    ledger.unattempted(row)
                                else:
                                    pending.append(row)
                        for offset in range(0, len(pending), freeze['concurrency']):
                            rows = pending[offset:offset + freeze['concurrency']]
                            ledger.begin(rows, authorization.PROBABILITY_SCHEMA)
                            for row, failed in list(executor.map(lambda r: submit(client, r, freeze), rows)):
                                if failed:
                                    ledger.state['halted'] = True
                                    ledger.state.setdefault('halt_reason', (row.get('error') or {}).get('kind') or 'fatal_api_or_billing_error')
                                ledger.settle(row)
                                fatal = fatal or failed or ledger.state['halted']
                            checkpoint('running')
                            if fatal:
                                break
                        if fatal:
                            break
                    if fatal:
                        break
            if fatal:
                ledger.state['halted'] = True
                ledger.state['halt_reason'] = ledger.state.get('halt_reason', 'fatal_api_or_billing_error')
                ledger.persist()
                terminalize_unattempted()
                return checkpoint('failed')
            return checkpoint('complete' if all(r['status'] == 'ok' for r in ledger.rows().values()) else 'complete_with_errors')
        except BaseException:
            checkpoint('interrupted')
            raise
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-manifest', type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        report = execute(args.freeze_manifest.resolve())
    else:
        freeze, units, entries, report = validate_freeze(args.freeze_manifest.resolve())
        report.update(status='dry_run_valid' if report['budget']['fits_entire_cohort'] else 'budget_ineligible',
                      pricing=freeze['pricing'], api_calls=0, credential_reads=0, output_writes=0)
        report['budget'].pop('family_reservations_usd', None)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report['status'] in ('dry_run_valid', 'complete', 'complete_with_errors') else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        print('Frozen evaluation validation/execution failed; no secret-bearing exception text is logged.', file=sys.stderr)
        sys.exit(2)
