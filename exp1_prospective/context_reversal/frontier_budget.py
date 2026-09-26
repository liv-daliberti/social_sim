"""Persistent whole-cohort budget accounting for ONE future frozen evaluation.

All currency is integer nanodollars. Reserve the entire fixed cohort before its
first call; unknown-billed and in-flight calls retain the full reservation.
This module never imports an API client, reads credentials, or calls a service.
"""
from __future__ import annotations

from collections import Counter
import copy
from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path

from exp1_prospective.context_reversal import run_local as common

NANO = 1_000_000_000
ABSOLUTE_CAP_NUSD = 25 * NANO
AUTHORIZATION_SCOPE = 'one_future_frozen_frontier_evaluation_25usd_v1'
# A single fixed ledger binds this authorization to one freeze and output path.
LEDGER_PATH = Path(__file__).resolve().parent / 'runs/frontier_budget_authorization_v1/ledger.json'
PRICES_NUSD_PER_TOKEN = {'standard': {'input_upper': 5000, 'output': 20000},
                         'batch': {'input_upper': 2500, 'output': 10000}}
PRICE_SOURCE = 'https://developers.openai.com/api/docs/pricing'
PRICE_VALID_THROUGH = '2026-11-21'
INPUT_FRAMING_TOKEN_ALLOWANCE = 4096
PRIOR_SERIALIZED_BYTE_LIMIT = 32


def dollars(value: int) -> str:
    return str(Decimal(value) / NANO)


def usd_to_nusd(value) -> int:
    amount = Decimal(str(value)) * NANO
    if not amount.is_finite() or amount != amount.to_integral_value() or amount <= 0:
        raise ValueError('Budget must be a positive exact nanodollar amount')
    return int(amount)


def record_id(key) -> str:
    return common.canonical_json(list(key))


def upper_cost(input_tokens: int, output_tokens: int, pricing: str) -> int:
    if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens)):
        raise ValueError('Token counts must be nonnegative integers')
    rates = PRICES_NUSD_PER_TOKEN[pricing]
    return input_tokens * rates['input_upper'] + output_tokens * rates['output']


def prompt_input_bound(prompt: str, schema: dict) -> int:
    # Conservative tokenizer-independent local bound. Deliberately retain the
    # previous runner's 4096-token allowance for server-owned framing. A smaller
    # empirically typical token count is not accepted as a hard ceiling.
    return len(prompt.encode('utf-8')) + len(common.canonical_json(schema).encode('utf-8')) + INPUT_FRAMING_TOKEN_ALLOWANCE


def reservations(units: list[dict], schema: dict, max_output_tokens: int, pricing: str) -> dict:
    entries = {}
    for unit in units:
        for stage, condition in [('baseline', 'baseline'), *[('update', c) for c in common.CONDITIONS]]:
            template = unit['baseline_prompt'] if stage == 'baseline' else unit['update_templates'][condition]
            prompt = template if stage == 'baseline' else template.replace(common.PRIOR_TOKEN, '0.' + '0' * (PRIOR_SERIALIZED_BYTE_LIMIT - 2))
            bound = prompt_input_bound(prompt, schema)
            if bound > 272000:
                raise ValueError('Only short-context input pricing is permitted')
            key = (unit['trial_id'], stage, condition)
            entries[record_id(key)] = {'key': list(key), 'family_id': unit['family_id'],
                                      'template_sha256': common.text_sha256(template),
                                      'input_token_upper_bound': bound, 'output_token_upper_bound': max_output_tokens,
                                      'reservation_nusd': upper_cost(bound, max_output_tokens, pricing)}
    return entries


def plan_summary(entries: dict, cap_nusd: int) -> dict:
    by_family = Counter()
    for entry in entries.values():
        by_family[entry['family_id']] += entry['reservation_nusd']
    total = sum(by_family.values())
    return {'planned_calls': len(entries), 'planned_families': len(by_family),
            'cohort_worst_case_nusd': total, 'cohort_worst_case_usd': dollars(total),
            'budget_cap_nusd': cap_nusd, 'budget_cap_usd': dollars(cap_nusd),
            'fits_entire_cohort': 0 < total <= cap_nusd <= ABSOLUTE_CAP_NUSD,
            'family_reservations_usd': {key: dollars(value) for key, value in sorted(by_family.items())},
            'input_bound_methods': dict(Counter(e.get('input_bound_method', 'utf8_plus_schema_plus4096') for e in entries.values())),
            'input_charge': 'All input charged at the cache-write ceiling; no cache discount',
            'prior_pilot_charged_to_this_authorization': False}


class Ledger:
    """Caller must hold session() for mutations; one writer across processes."""
    def __init__(self, path: Path, binding: dict, entries: dict, cap_nusd: int, pricing: str):
        self.path, self.binding, self.specs = path, binding, entries
        self.cap, self.pricing = cap_nusd, pricing
        summary = plan_summary(entries, cap_nusd)
        if not summary['fits_entire_cohort']:
            raise ValueError('The entire frozen cohort worst case exceeds the budget; no calls may begin')
        for key, spec in entries.items():
            if spec['reservation_nusd'] != upper_cost(spec['input_token_upper_bound'], spec['output_token_upper_bound'], pricing):
                raise ValueError('Reservation does not equal its conservative token-cost bound')
        self.state = None
        self.locked = False

    @contextmanager
    def session(self):
        with common.output_lock(Path(str(self.path) + '.lock')):
            self.locked = True
            try:
                self.load()
                yield self
            finally:
                self.locked = False

    def load(self):
        if not self.path.exists():
            self.state = {'schema_version': 1, 'authorization_scope': AUTHORIZATION_SCOPE,
                          'binding': self.binding, 'cap_nusd': self.cap, 'pricing': self.pricing,
                          'created_at': common.utc_now(), 'halted': False,
                          'entries': {key: {**spec, 'state': 'reserved_unattempted', 'charge_nusd': spec['reservation_nusd'], 'row': None}
                                      for key, spec in self.specs.items()}}
            return
        self.state = json.loads(self.path.read_text())
        if (self.state.get('authorization_scope') != AUTHORIZATION_SCOPE or self.state.get('binding') != self.binding
                or self.state.get('cap_nusd') != self.cap or self.state.get('pricing') != self.pricing
                or set(self.state.get('entries', {})) != set(self.specs)):
            raise ValueError('Authorization is already bound to another freeze, output, pricing, or cohort')
        for key, entry in self.state['entries'].items():
            if any(entry.get(field) != expected for field, expected in self.specs[key].items()):
                raise ValueError('Immutable ledger reservation changed')
            state, charge = entry['state'], entry['charge_nusd']
            if state not in {'reserved_unattempted', 'inflight', 'unknown_billed', 'settled', 'unattempted_terminal'}:
                raise ValueError('Unknown ledger state')
            if type(charge) is not int or charge < 0:
                raise ValueError('Invalid ledger charge')
            if state in {'reserved_unattempted', 'inflight', 'unknown_billed'} and charge != entry['reservation_nusd']:
                raise ValueError('An uncertain or pending call lost its full reservation')
            if state == 'unattempted_terminal' and charge != 0:
                raise ValueError('An unattempted terminal row has a charge')
            if state == 'settled':
                usage = entry.get('usage')
                if not self.valid_usage(usage) or charge != upper_cost(usage['input_tokens'], usage['output_tokens'], self.pricing):
                    raise ValueError('Settled charge does not match conservative usage accounting')
            row = entry.get('row')
            if state == 'reserved_unattempted' and row is not None:
                raise ValueError('Unattempted reservation already contains a response')
            if state != 'reserved_unattempted' and (not isinstance(row, dict) or record_id(common.record_key(row)) != key):
                raise ValueError('Ledger row identity mismatch')
        if self.exposure() > self.cap and not self.state['halted']:
            raise ValueError('Ledger exposure exceeds its cap')

    def persist(self):
        if not self.locked:
            raise RuntimeError('Budget mutation requires the exclusive authorization lock')
        self.state['updated_at'] = common.utc_now()
        self.state['exposure_nusd'] = self.exposure()
        common.atomic_write(self.path, json.dumps(self.state, sort_keys=True, indent=2) + '\n')

    def exposure(self) -> int:
        return sum(entry['charge_nusd'] for entry in self.state['entries'].values())

    @staticmethod
    def valid_usage(usage):
        return (isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens', 'total_tokens'))
                and usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens'])

    def recover_inflight(self):
        for entry in self.state['entries'].values():
            if entry['state'] == 'inflight':
                entry['state'] = 'unknown_billed'
                entry['row'].update(status='generation_error', probability=None,
                                    error={'kind': 'interrupted_result_unknown_no_retry'})
        self.persist()

    def begin(self, rows: list[dict], schema: dict):
        if self.state['halted']:
            raise ValueError('Budget ledger halted; no additional calls are permitted')
        ids = [record_id(common.record_key(row)) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate request within concurrent batch')
        for key, row in zip(ids, rows):
            entry = self.state['entries'][key]
            if entry['state'] != 'reserved_unattempted':
                raise ValueError('Previously attempted or completed calls cannot be retried')
            method = entry.get('input_bound_method', 'utf8_plus_schema_plus4096')
            if method == 'api_exact_baseline_body':
                if common.text_sha256(row['prompt']) != entry['certified_prompt_sha256']:
                    raise ValueError('Baseline prompt differs from its certified exact body')
            elif method == 'api_envelope_plus_full_utf8_content_ceiling':
                if len(row['prompt'].encode('utf-8')) > entry['max_prompt_utf8_bytes']:
                    raise ValueError('Update exceeds the full-content UTF-8 certificate ceiling')
            elif method != 'utf8_plus_schema_plus4096' or prompt_input_bound(row['prompt'], schema) > entry['input_token_upper_bound']:
                raise ValueError('Rendered prompt exceeds its frozen input reservation')
        for key, row in zip(ids, rows):
            row['request_attempted'] = True
            self.state['entries'][key].update(state='inflight', row=copy.deepcopy(row), request_prompt_sha256=common.text_sha256(row['prompt']))
        # All concurrently submitted calls remain fully charged before dispatch.
        self.persist()

    def settle(self, row: dict):
        entry = self.state['entries'][record_id(common.record_key(row))]
        if entry['state'] != 'inflight':
            raise ValueError('Only a once-dispatched call may settle')
        usage = row.get('usage')
        entry['row'] = row
        if self.valid_usage(usage):
            charge = upper_cost(usage['input_tokens'], usage['output_tokens'], self.pricing)
            if (usage['input_tokens'] > entry['input_token_upper_bound'] or usage['output_tokens'] > entry['output_token_upper_bound']):
                self.state['halted'] = True
                self.state['halt_reason'] = 'provider_reported_usage_exceeded_reserved_token_bound'
            entry.update(state='settled', charge_nusd=charge, usage=usage)
        else:
            entry.update(state='unknown_billed', charge_nusd=entry['reservation_nusd'])
        self.persist()

    def unattempted(self, row: dict):
        entry = self.state['entries'][record_id(common.record_key(row))]
        if entry['state'] != 'reserved_unattempted':
            raise ValueError('Cannot release an attempted or already terminal request')
        entry.update(state='unattempted_terminal', charge_nusd=0, row=row)
        self.persist()

    def rows(self) -> dict:
        return {tuple(entry['key']): entry['row'] for entry in self.state['entries'].values() if entry['row'] is not None}
