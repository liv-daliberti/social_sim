"""Freeze the Claude Opus 5 replication of the frozen frontier evaluation.

Reuses the existing frozen materials, protocol, scorer, review and freshness
audit byte for byte; only the model, transport, price schedule and output budget
are new. Nothing about the cohort, its families or its scoring changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_budget_anthropic as authorization
from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal.fresh.freeze import artifact, check_artifact

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/fresh_evaluation_v1/frozen_v1/frontier_design_freeze.json'
DEST = ROOT / 'data/fresh_evaluation_v1/frozen_v1_claude/frontier_design_freeze_claude_opus5.json'
CERTIFICATE = ROOT / 'runs/frontier_claude_authorization_v1/input_token_certificate.json'
MAX_OUTPUT_TOKENS = 1024
CODE_FILES = ('run_frozen_frontier_claude.py', 'frontier_budget_anthropic.py', 'frontier_budget.py',
              'collect_anthropic_token_certificate.py', 'run_local.py')

# Four paid calls were made on frozen paraphrase-arm baseline prompts before this
# freeze, to establish that the platform's structured-output path avoids the
# refusal classifier and to size the output budget. Their outputs selected no
# material, changed no prompt and entered no analysis; the cohort is re-run in
# full under the ledger. Disclosed rather than asserted away.
PROBE = {'calls': 4, 'stage': 'baseline', 'arm': 'paraphrase',
         'families': ['fresh001_paraphrase', 'fresh004_paraphrase'],
         'purpose': ['refusal_behaviour_under_structured_outputs', 'output_token_budget_sizing'],
         'outputs_used_for_material_selection': False, 'outputs_used_in_analysis': False,
         'responses_retained_in_cohort': False,
         'observed_output_tokens': [219, 226, 229, 285],
         'free_text_schema_refusal_category_observed': 'cyber'}

# A first execution attempt aborted after 19 settled calls on a ledger-resume
# defect in this module, not in the materials or the accounting. Its records and
# ledger are retained under runs/frontier_claude_authorization_v1/
# superseded_pre_resume_fix/ and take no part in the evaluation; the cohort is
# re-run in full. Actual spend on that attempt was $0.1733.
ABORTED_ATTEMPT = {'settled_calls': 19, 'input_tokens': 19651, 'output_tokens': 3003,
                   'actual_usd': '0.1733', 'cause': 'ledger_resume_scope_check_defect',
                   'records_retained_outside_cohort': True, 'outputs_used_in_analysis': False,
                   'archive': 'runs/frontier_claude_authorization_v1/superseded_pre_resume_fix'}


def main():
    source = json.loads(SOURCE.read_text())
    for spec in source['artifacts'].values():
        check_artifact(spec)
    plan = check_artifact(source['artifacts']['plan'])
    units = common.read_units(plan)
    certificate = json.loads(CERTIFICATE.read_text())
    entries = authorization.reservations(units, MAX_OUTPUT_TOKENS, certificate, common.file_sha256(plan))
    cap = budget.usd_to_nusd('24')
    summary = budget.plan_summary(entries, cap)
    if not summary['fits_entire_cohort']:
        raise ValueError('Cohort worst case exceeds the cap; refusing to freeze')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    freeze = {
        'schema_version': 'frozen_frontier_evaluation_claude_v1', 'status': 'frozen',
        'created_at': common.utc_now(), 'freeze_stage': 'after_local_development_after_gpt56_evaluation',
        'cohort_inference_started_before_freeze': False, 'pre_freeze_feasibility_probe': PROBE,
        'superseded_aborted_attempt': ABORTED_ATTEMPT,
        'evaluation_id': 'fresh_finite_context_reversal_claude_opus5_v1',
        'authorization_scope': authorization.AUTHORIZATION_SCOPE,
        'model': authorization.MODEL, 'model_key': 'claude_opus5_fresh_foundry',
        'transport': 'anthropic.AnthropicFoundry', 'platform': 'microsoft_foundry',
        'batch_pricing_available': False,
        'max_output_tokens': MAX_OUTPUT_TOKENS, 'effort': 'low', 'thinking': 'adaptive_default',
        'structured_output_schema': authorization.PROBABILITY_SCHEMA,
        'schema_note': 'Platform rejects minimum/maximum on a number; the [0, 1] range is enforced by the scorer.',
        'max_retries': 0, 'concurrency': 4,
        'pricing': authorization.PRICING, 'prices_nusd_per_token': authorization.PRICES_NUSD_PER_TOKEN,
        'price_source': authorization.PRICE_SOURCE, 'pricing_valid_through': authorization.PRICE_VALID_THROUGH,
        'budget_usd': '24', 'absolute_cap_usd': '25',
        'contexts': source['contexts'], 'family_order': source['family_order'],
        'output_path': str((ROOT / 'responses/fresh_evaluation_v1/frontier_claude_opus5.jsonl').resolve()),
        'artifacts': {**source['artifacts'], 'token_certificate': artifact(CERTIFICATE)},
        'code_sha256': {name: common.file_sha256(ROOT / name) for name in CODE_FILES},
        'gpt56_evaluation': artifact(SOURCE),
        'cohort_worst_case_usd': summary['cohort_worst_case_usd'],
    }
    common.atomic_write(DEST, json.dumps(freeze, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'freeze': str(DEST), 'planned_calls': summary['planned_calls'],
                      'cohort_worst_case_usd': summary['cohort_worst_case_usd'],
                      'budget_cap_usd': summary['budget_cap_usd']}, indent=2))


if __name__ == '__main__':
    main()
