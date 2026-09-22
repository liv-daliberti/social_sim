"""Second-model authorization: Claude Opus 5 on Microsoft Foundry, cap $25.

A separate scope, ledger and price schedule from the GPT-5.6 authorization.
The existing `frontier_budget` module is reused unchanged -- its reserve-before-
dispatch ledger, full charging of in-flight and unknown-billed calls, and halt
on usage exceeding a reservation are the audited parts and are not reimplemented.
This module only registers Anthropic prices, binds a distinct authorization
scope, and collects exact input-token receipts.

Message Batches are unavailable on Foundry, so no batch discount exists here and
standard pricing is the only mode. Input counts come from the provider's own
free `count_tokens` endpoint rather than the byte-based ceiling, which is what
makes the whole cohort fit; the same receipt pattern as the GPT arm.
"""
from __future__ import annotations

import json
from pathlib import Path

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import run_local as common

MODEL = 'claude-opus-5'
PRICING = 'anthropic_opus5_standard'
AUTHORIZATION_SCOPE = 'one_future_frozen_frontier_evaluation_claude_opus5_25usd_v1'
LEDGER_PATH = Path(__file__).resolve().parent / 'runs/frontier_claude_authorization_v1/ledger.json'
# Opus 5 list price, $5.00 / $25.00 per million tokens. Foundry bills Claude at
# standard first-party API rates through the Microsoft Marketplace.
PRICES_NUSD_PER_TOKEN = {'input_upper': 5000, 'output': 25000}
PRICE_SOURCE = 'https://www.anthropic.com/pricing (Claude Opus 5; Foundry billed at standard API rates)'
PRICE_VALID_THROUGH = '2026-11-21'
# Provider framing that a counted user-message body does not include.
COUNT_FRAMING_TOKEN_ALLOWANCE = 64
# Structured outputs on this platform reject `minimum`/`maximum` on a number, so
# the schema drops them and `parse_probability` enforces the [0, 1] range.
PROBABILITY_SCHEMA = {'type': 'object', 'properties': {'probability': {'type': 'number'}},
                      'required': ['probability'], 'additionalProperties': False}

# Additive registration; existing pricing modes and the GPT ledger are untouched.
budget.PRICES_NUSD_PER_TOKEN.setdefault(PRICING, dict(PRICES_NUSD_PER_TOKEN))
if budget.PRICES_NUSD_PER_TOKEN[PRICING] != PRICES_NUSD_PER_TOKEN:
    raise ValueError('Anthropic price schedule collides with an existing pricing mode')


def request_body(prompt: str, max_output_tokens: int) -> dict:
    """The exact generation body. Counting uses the same messages block."""
    return {'model': MODEL, 'max_tokens': max_output_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
            'output_config': {'effort': 'low',
                              'format': {'type': 'json_schema', 'schema': PROBABILITY_SCHEMA}}}


def worst_case_prompt(unit: dict, stage: str, condition: str) -> str:
    """Longest prompt this call can render: the maximal serialized prior."""
    if stage == 'baseline':
        return unit['baseline_prompt']
    template = unit['update_templates'][condition]
    return template.replace(common.PRIOR_TOKEN, '0.' + '0' * (budget.PRIOR_SERIALIZED_BYTE_LIMIT - 2))


def certified_bounds(units: list[dict], plan_sha256: str, certificate: dict) -> dict:
    """Map ledger keys to exact-count input bounds from a verified certificate."""
    if certificate.get('schema_version') != 'anthropic_frontier_input_token_certificate_v1':
        raise ValueError('Unknown certificate schema')
    if certificate.get('plan_sha256') != plan_sha256 or certificate.get('model') != MODEL:
        raise ValueError('Certificate does not bind this plan and model')
    if certificate.get('allowance') != COUNT_FRAMING_TOKEN_ALLOWANCE:
        raise ValueError('Certificate framing allowance differs')
    counts = certificate['counts']
    bounds = {}
    for unit in units:
        for stage, condition in [('baseline', 'baseline'), *[('update', c) for c in common.CONDITIONS]]:
            key = budget.record_id((unit['trial_id'], stage, condition))
            entry = counts.get(key)
            if entry is None:
                raise ValueError('Certificate does not cover every planned call')
            prompt = worst_case_prompt(unit, stage, condition)
            if entry['prompt_sha256'] != common.text_sha256(prompt):
                raise ValueError('Certified body differs from the planned worst-case prompt')
            if type(entry['input_tokens']) is not int or entry['input_tokens'] < 1:
                raise ValueError('Invalid certified token count')
            # Reuse the audited ledger's own admission checks: a baseline must
            # match its certified body exactly; an update's rendered prompt must
            # stay inside the worst-case-prior byte ceiling that was counted.
            method = 'api_exact_baseline_body' if stage == 'baseline' else 'api_envelope_plus_full_utf8_content_ceiling'
            bounds[key] = {'input_bound_method': method,
                           'input_token_upper_bound': entry['input_tokens'] + COUNT_FRAMING_TOKEN_ALLOWANCE,
                           'certified_prompt_sha256': entry['prompt_sha256'],
                           'max_prompt_utf8_bytes': len(prompt.encode('utf-8'))}
    return bounds


class Ledger(budget.Ledger):
    """Identical accounting, bound to the Claude authorization scope.

    `load` is restated rather than delegated: the base method's scope check is
    fused into the same condition as the binding, cap, pricing and cohort checks,
    so a Claude-scoped file cannot pass through it. Every other rule below is the
    base method's, unchanged, and `begin`/`settle`/`unattempted`/`persist` are
    inherited untouched.
    """

    def load(self):
        if not self.path.exists():
            self.state = {'schema_version': 1, 'authorization_scope': AUTHORIZATION_SCOPE,
                          'binding': self.binding, 'cap_nusd': self.cap, 'pricing': self.pricing,
                          'created_at': common.utc_now(), 'halted': False,
                          'entries': {key: {**spec, 'state': 'reserved_unattempted',
                                            'charge_nusd': spec['reservation_nusd'], 'row': None}
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
                if not self.valid_usage(usage) or charge != budget.upper_cost(usage['input_tokens'], usage['output_tokens'], self.pricing):
                    raise ValueError('Settled charge does not match conservative usage accounting')
            row = entry.get('row')
            if state == 'reserved_unattempted' and row is not None:
                raise ValueError('Unattempted reservation already contains a response')
            if state != 'reserved_unattempted' and (not isinstance(row, dict) or budget.record_id(common.record_key(row)) != key):
                raise ValueError('Ledger row identity mismatch')
        if self.exposure() > self.cap and not self.state['halted']:
            raise ValueError('Ledger exposure exceeds its cap')


def reservations(units: list[dict], max_output_tokens: int, certificate: dict, plan_sha256: str) -> dict:
    entries = budget.reservations(units, PROBABILITY_SCHEMA, max_output_tokens, PRICING)
    for key, values in certified_bounds(units, plan_sha256, certificate).items():
        entries[key].update(values)
        entries[key]['reservation_nusd'] = budget.upper_cost(values['input_token_upper_bound'],
                                                             max_output_tokens, PRICING)
    return entries
