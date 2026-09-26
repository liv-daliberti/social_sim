"""Collect exact input-token receipts for the frozen Claude cohort.

`count_tokens` is a free, non-generating endpoint. Every planned call is counted
on its worst-case rendered prompt, so the resulting bound holds for the actual
request whatever prior an update ends up carrying.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import frontier_budget_anthropic as authorization
from exp1_prospective.context_reversal import run_local as common


def collect(plan: Path, output: Path, concurrency: int = 4) -> dict:
    import anthropic
    units = common.read_units(plan)
    client = anthropic.AnthropicFoundry(api_key=os.environ['ANTHROPIC_FOUNDRY_API_KEY'],
                                        base_url=os.environ['ANTHROPIC_FOUNDRY_BASE_URL'],
                                        max_retries=6)
    requests = []
    for unit in units:
        for stage, condition in [('baseline', 'baseline'), *[('update', c) for c in common.CONDITIONS]]:
            prompt = authorization.worst_case_prompt(unit, stage, condition)
            requests.append((budget.record_id((unit['trial_id'], stage, condition)), prompt))

    def count(item):
        key, prompt = item
        body = authorization.request_body(prompt, 1024)
        result = client.messages.count_tokens(model=body['model'], messages=body['messages'],
                                              output_config=body['output_config'])
        return key, {'prompt_sha256': common.text_sha256(prompt), 'input_tokens': int(result.input_tokens)}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        counts = dict(pool.map(count, requests))
    client.close()
    certificate = {'schema_version': 'anthropic_frontier_input_token_certificate_v1',
                   'model': authorization.MODEL, 'plan_sha256': common.file_sha256(plan),
                   'plan_path': str(plan.resolve()), 'allowance': authorization.COUNT_FRAMING_TOKEN_ALLOWANCE,
                   'endpoint': 'messages.count_tokens', 'billed': False, 'calls': len(counts),
                   'counted_body': 'worst_case_rendered_prompt_with_maximal_serialized_prior',
                   'collector_sha256': common.file_sha256(Path(__file__)), 'created_at': common.utc_now(),
                   'counts': counts}
    common.atomic_write(output, json.dumps(certificate, sort_keys=True, indent=2) + '\n')
    values = [c['input_tokens'] for c in counts.values()]
    return {'calls': len(values), 'min': min(values), 'max': max(values),
            'mean': sum(values) / len(values), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(collect(args.plan, args.output, args.concurrency), indent=2))


if __name__ == '__main__':
    main()
