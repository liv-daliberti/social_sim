#!/usr/bin/env python3
"""Fail-closed runner for one new frozen frontier evaluation, total cap $25.

The default/--dry-run path is offline and never constructs a client or reads a
key. --execute supports standard Responses only. Batch pricing can be planned,
but Batch execution is intentionally rejected until a separate reviewed adapter
exists. Whole-cohort worst-case cost must fit before ANY request is admitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_token_certificate as token_certificate
from exp1_prospective.context_reversal import run_openai as pilot
from exp1_prospective.context_reversal import run_local as common

SOURCE = Path(__file__).resolve().parent
MODEL = 'gpt-5.6-sol'
CODE_FILES = ('run_frozen_frontier.py', 'frontier_budget.py', 'run_openai.py', 'run_local.py', 'frontier_token_certificate.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_artifact(spec: dict) -> Path:
    path = Path(spec['path']).resolve()
    require(path.is_file() and common.file_sha256(path) == spec['sha256'], f'Frozen artifact hash mismatch: {path.name}')
    return path


def validate_output_paths(freeze_path: Path, freeze: dict, generated_paths: Iterable[Path],
                          ledger_path: Path = budget.LEDGER_PATH) -> None:
    """Reject generated files that could replace frozen inputs or accounting.

    Callers include every generated sidecar, including paths used only on
    resume. This helper reads artifact metadata but never writes any file.
    """
    count_authorization = budget.LEDGER_PATH.with_name('token_count_authorization.json')
    protected = [Path(freeze_path)]
    for authorization in (budget.LEDGER_PATH, Path(ledger_path), count_authorization):
        protected.extend((authorization, Path(str(authorization) + '.lock')))
    protected.extend(SOURCE / name for name in set(CODE_FILES) | set(freeze.get('code_sha256', {})))
    artifacts = freeze.get('artifacts', {})
    protected.extend(Path(spec['path']) for spec in artifacts.values())
    for name, field in (('review', 'development_results'), ('freshness', 'excluded_plans')):
        if name in artifacts:
            record = json.loads(Path(artifacts[name]['path']).read_text())
            protected.extend(Path(spec['path']) for spec in record.get(field, []))

    def identities(path):
        # Keep lexical identity too: atomic replacement replaces the final
        # symlink itself, whereas parent-directory symlinks are followed.
        return {Path(os.path.abspath(path)), path.resolve()}

    protected_ids = set().union(*(identities(path) for path in protected))
    generated_ids = set()
    for value in generated_paths:
        path = Path(value)
        require(path.is_absolute(), 'Generated output paths must be absolute')
        ids = identities(path)
        require(not ids & protected_ids, 'Generated output collides with a protected artifact or authorization path')
        require(not ids & generated_ids, 'Generated output paths collide with each other')
        generated_ids.update(ids)


def parse_probability(raw: str) -> float:
    """Check the exact JSON number's range before conversion to binary float."""
    def object_pairs(pairs):
        require(len({key for key, _ in pairs}) == len(pairs), 'Duplicate JSON fields')
        return dict(pairs)

    def invalid_constant(_):
        raise ValueError('Probability must be finite')

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs, parse_int=Decimal,
                           parse_float=Decimal, parse_constant=invalid_constant)
    except InvalidOperation as exc:
        raise ValueError('Invalid probability number') from exc
    require(isinstance(value, dict) and set(value) == {'probability'}, 'Expected exactly one probability field')
    probability = value['probability']
    require(type(probability) is Decimal and probability.is_finite() and 0 <= probability <= 1,
            'Probability must be numeric, finite, and in [0, 1]')
    return float(probability)


def validate_freeze(path: Path) -> tuple[dict, list[dict], dict, dict]:
    freeze = json.loads(path.read_text())
    require(freeze.get('schema_version') == 'frozen_frontier_evaluation_v1' and freeze.get('status') == 'frozen', 'Evaluation is not frozen')
    require(freeze.get('freeze_stage') == 'after_local_development_before_frontier', 'Local development must precede the frontier freeze')
    require(freeze.get('target_inference_started_before_freeze') is False, 'All target-model inference must start after the evaluation freeze')
    require(isinstance(freeze.get('evaluation_id'), str) and bool(freeze['evaluation_id'].strip()), 'Missing evaluation identity')
    require(freeze.get('authorization_scope') == budget.AUTHORIZATION_SCOPE, 'Wrong single-evaluation authorization scope')
    require(freeze.get('model') == MODEL, 'Only the frozen gpt-5.6-sol model is permitted')
    require(freeze.get('max_output_tokens') in (1024, 2048), 'Freeze 1024 or2048 output tokens before evaluation')
    require(freeze.get('reasoning_effort') == 'low' and freeze.get('max_retries') == 0, 'Reasoning/retry contract changed')
    require(type(freeze.get('concurrency')) is int and 1 <= freeze['concurrency'] <= 4, 'Concurrency must be1..4')
    require(freeze.get('pricing') in budget.PRICES_NUSD_PER_TOKEN, 'Unknown pricing mode')
    require(freeze.get('prices_nusd_per_token') == budget.PRICES_NUSD_PER_TOKEN[freeze['pricing']], 'Frozen prices differ from the audited conservative schedule')
    require(freeze.get('pricing_valid_through') == budget.PRICE_VALID_THROUGH and datetime.now(timezone.utc).date() <= date.fromisoformat(budget.PRICE_VALID_THROUGH), 'Pricing audit has expired or is missing')
    cap = budget.usd_to_nusd(freeze['budget_usd'])
    require(cap <= budget.ABSOLUTE_CAP_NUSD, 'Total authorization is at most$25')
    for filename in CODE_FILES:
        require(freeze['code_sha256'].get(filename) == common.file_sha256(SOURCE / filename), f'Frozen code mismatch: {filename}')
    artifacts = freeze['artifacts']
    require({'plan', 'protocol', 'scoring', 'review', 'freshness'} <= set(artifacts), 'Freeze requires plan, protocol, scoring, review, and freshness artifacts')
    paths = {name: checked_artifact(spec) for name, spec in artifacts.items()}
    require(all(paths[name].stat().st_size > 0 for name in ('protocol', 'scoring')), 'Protocol and scoring cannot be empty')
    units = common.read_units(paths['plan'])
    require(all(u['repeat'] == 0 for u in units), 'One repeat per evaluation unit must be frozen')
    contexts = set(freeze['contexts'])
    require(contexts in ({'positive', 'negative', 'broken'}, {'positive', 'negative', 'broken', 'masked'}), 'Freeze three scored contexts with optional masked context')
    families = sorted({u['family_id'] for u in units})
    require(len(freeze['family_order']) == len(families) and set(freeze['family_order']) == set(families), 'Family order must cover each planned family exactly once')
    require(all({u['context_id'] for u in units if u['family_id'] == family} == contexts for family in families), 'Every family must have every frozen context')
    review, freshness = (json.loads(paths[name].read_text()) for name in ('review', 'freshness'))
    for name, record in (('review', review), ('freshness', freshness)):
        require(record.get('plan_sha256') == artifacts['plan']['sha256'] and sorted(record.get('family_ids', [])) == families, f'{name} does not cover the exact frozen plan')
    require(review.get('status') == 'complete' and review.get('unresolved_findings') == 0 and review.get('frontier_model_outputs_used') is False and review.get('target_model_outputs_used') is False
            and review.get('outcome_based_family_filtering') is False,
            'Review/selection must be complete without any target-model output leakage or outcome-based family filtering')
    require(review.get('local_development_complete') is True and isinstance(review.get('reviewer'), str) and bool(review['reviewer'].strip()), 'Local development/reviewer attestation missing')
    require(bool(review.get('development_results')), 'Local development result artifacts must be frozen before evaluation')
    for artifact in review['development_results']:
        checked_artifact(artifact)
    require(freshness.get('status') == 'passed' and bool(freshness.get('excluded_plans')), 'Freshness audit/excluded development plans missing')
    excluded_paths, old_families, old_prompts = set(), set(), set()
    for spec in freshness['excluded_plans']:
        excluded = checked_artifact(spec)
        excluded_paths.add(excluded)
        for unit in common.read_units(excluded):
            old_families.add(unit['family_id'])
            old_prompts.add(unit['baseline_prompt'])
            old_prompts.update(unit['update_templates'].values())
    original_plan = (SOURCE / 'runs/frontier_gpt56_pilot_v1/plan.jsonl').resolve()
    require(original_plan in excluded_paths, 'Freshness audit must exclude the original evaluated development plan')
    require(not set(families) & old_families, 'Evaluation families overlap an excluded development/evaluated plan')
    require(not any(u['baseline_prompt'] in old_prompts or any(t in old_prompts for t in u['update_templates'].values()) for u in units), 'Evaluation prompts duplicate an excluded development/evaluated prompt')
    output = Path(freeze['output_path'])
    validate_output_paths(path, freeze, (output, Path(str(output) + '.manifest.json')))
    entries = budget.reservations(units, pilot.PROBABILITY_SCHEMA, freeze['max_output_tokens'], freeze['pricing'])
    if 'token_certificate' in paths:
        certificate = json.loads(paths['token_certificate'].read_text())
        bounds = token_certificate.certified_bounds(units, artifacts['plan']['sha256'], freeze['max_output_tokens'], certificate)
        for key, values in bounds.items():
            entries[key].update(values)
            entries[key]['reservation_nusd'] = budget.upper_cost(values['input_token_upper_bound'], freeze['max_output_tokens'], freeze['pricing'])
    summary = budget.plan_summary(entries, cap)
    binding = {'freeze_path': str(path.resolve()), 'freeze_sha256': common.file_sha256(path),
               'evaluation_id': freeze['evaluation_id'], 'output_path': str(output.resolve()),
               'plan_sha256': artifacts['plan']['sha256'], 'schema_sha256': common.text_sha256(common.canonical_json(pilot.PROBABILITY_SCHEMA))}
    return freeze, units, entries, {'binding': binding, 'budget': summary}


def request_payload(prompt: str, freeze: dict) -> dict:
    return token_certificate.body(prompt, freeze['max_output_tokens'])


def submit(client, row: dict, freeze: dict) -> tuple[dict, bool]:
    try:
        result = client.responses.create(**request_payload(row['prompt'], freeze))
    except Exception as exc:
        row['error'] = pilot.safe_api_error(exc)
        return row, pilot.fatal_api_error(row['error'])
    row.update(response_received=True, returned_model=getattr(result, 'model', None), response_id=getattr(result, 'id', None),
               response_status=getattr(result, 'status', None), usage=pilot.response_usage(result), created_at=common.utc_now(),
               returned_service_tier=getattr(result, 'service_tier', None))
    text = getattr(result, 'output_text', '')
    row['raw'] = text if isinstance(text, str) else ''
    if row['returned_model'] != MODEL:
        row['error'] = {'kind': 'unexpected_returned_model'}
        return row, True
    if row['returned_service_tier'] not in (None, 'default'):
        row['error'] = {'kind': 'unexpected_billing_service_tier'}
        return row, True
    if row['response_status'] != 'completed':
        row['error'] = {'kind': 'response_not_completed'}
        return row, False
    try:
        row.update(probability=parse_probability(row['raw']), status='ok')
    except (ValueError, TypeError, OverflowError):
        row.update(status='parse_error', error={'kind': 'invalid_probability_json'})
    return row, False


def execute(path: Path, *, client_factory=None, ledger_path: Path | None = None) -> dict:
    # Injection points are solely for offline tests; the CLI always uses the
    # fixed authorization ledger and the SDK client with max_retries=0.
    freeze, units, entries, preflight = validate_freeze(path)
    require(freeze['pricing'] == 'standard', 'Batch execution is not implemented; no credentials or API client were accessed')
    require(preflight['budget']['fits_entire_cohort'], 'Entire frozen cohort worst case exceeds the cap; no calls permitted')
    output = Path(freeze['output_path'])
    validate_output_paths(path, freeze, (output, Path(str(output) + '.manifest.json')),
                          ledger_path=ledger_path or budget.LEDGER_PATH)
    ledger = budget.Ledger(ledger_path or budget.LEDGER_PATH, preflight['binding'], entries,
                           preflight['budget']['budget_cap_nusd'], freeze['pricing'])
    config = {'input_sha256': freeze['artifacts']['plan']['sha256'], 'model_key': freeze['model_key'], 'model': MODEL,
              'freeze_sha256': preflight['binding']['freeze_sha256'], 'decode': {'max_output_tokens': freeze['max_output_tokens']}}
    by_id = {u['trial_id']: u for u in units}
    with ledger.session():
        # An existing output without the shared ledger cannot reset spending.
        require(ledger.path.exists() or not output.exists(), 'Output exists without its authorization ledger; refusing unsafe budget reset')
        ledger.recover_inflight()
        # Revalidate persisted sampled content and context-specific priors before
        # constructing a client. The ledger is the authoritative response store.
        existing = ledger.rows()
        for key, row in existing.items():
            unit = by_id[key[0]]
            for field in ('trial_id', 'family_id', 'domain', 'context_id', 'repeat', 'material_status'):
                require(row.get(field) == unit[field], 'Persisted response metadata mismatch')
            require(row.get('run_signature') == common.text_sha256(common.canonical_json(config)), 'Persisted response signature mismatch')
            require(row.get('status') in {'ok', 'parse_error', 'generation_error', 'blocked_baseline'}, 'Invalid persisted response status')
            if row['status'] == 'ok':
                require(row.get('returned_model') == MODEL, 'Persisted response model mismatch')
                require(type(row.get('probability')) in (int, float) and parse_probability(row['raw']) == row['probability'], 'Persisted probability mismatch')
            else:
                require(row.get('probability') is None, 'Failed persisted response has a probability')
            if key[1] == 'baseline':
                expected_prompt = unit['baseline_prompt']
                require(row.get('prior_probability') is None and row['status'] != 'blocked_baseline', 'Invalid baseline metadata')
            else:
                baseline = existing.get((key[0], 'baseline', 'baseline'))
                require(baseline is not None, 'Persisted update lacks its baseline')
                require(row.get('prior_probability') == baseline['probability'], 'Persisted update changed its own baseline prior')
                valid = baseline['status'] == 'ok'
                require((row['status'] == 'blocked_baseline') != valid, 'Persisted update/baseline status mismatch')
                expected_prompt = common.render_update(unit, key[2], baseline['probability']) if valid else None
            require(row.get('prompt') == expected_prompt and row.get('prompt_sha256') == (common.text_sha256(expected_prompt) if expected_prompt is not None else None), 'Persisted prompt mismatch')
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
            common.atomic_write(output, ''.join(common.canonical_json(row) + '\n' for row in rows.values()))
            summary = {'status': status, 'binding': preflight['binding'], 'freeze': str(path.resolve()),
                       'authorization_ledger': str(ledger.path), 'planned_records': len(entries), 'records': len(rows),
                       'status_counts': dict(Counter(r['status'] for r in rows.values())),
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
        require(not ledger.state['halted'], 'Ledger halted; cannot resume API calls')
        client = None
        try:
            # Whole cohort is now durably reserved, including all future updates.
            client = (client_factory or pilot.create_client)()
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
                            ledger.begin(rows, pilot.PROBABILITY_SCHEMA)
                            results = list(executor.map(lambda row: submit(client, row, freeze), rows))
                            for row, failed in results:
                                if failed:
                                    # Persist the fatal flag together with this
                                    # result, before any interruptible checkpoint.
                                    ledger.state['halted'] = True
                                    reason = (row.get('error') or {}).get('kind')
                                    ledger.state.setdefault('halt_reason', reason or 'fatal_api_or_billing_error')
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
            # In-flight reservations remain fully charged and cannot be retried.
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
    action.add_argument('--dry-run', action='store_true', help='Default: validate and estimate only; no API/key access')
    action.add_argument('--execute', action='store_true', help='Run the single fully frozen evaluation after all gates pass')
    args = parser.parse_args()
    if args.execute:
        report = execute(args.freeze_manifest.resolve())
    else:
        freeze, units, entries, report = validate_freeze(args.freeze_manifest.resolve())
        report.update(status='dry_run_valid' if report['budget']['fits_entire_cohort'] else 'budget_ineligible',
                      pricing=freeze['pricing'], transport_implemented=freeze['pricing'] == 'standard',
                      api_calls=0, credential_reads=0, output_writes=0)
        if budget.LEDGER_PATH.exists():
            ledger = budget.Ledger(budget.LEDGER_PATH, report['binding'], entries, report['budget']['budget_cap_nusd'], freeze['pricing'])
            ledger.load()
            report['existing_exposure_usd'] = budget.dollars(ledger.exposure())
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report['status'] in ('dry_run_valid', 'complete', 'complete_with_errors') else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        # Never stringify an exception from SDK/client setup or a request.
        print('Frozen evaluation validation/execution failed; no secret-bearing exception text is logged.', file=sys.stderr)
        sys.exit(2)
