"""Record the output-token-budget amendment and derive the amended local freeze.

The frozen local settings gave both reasoning arms the same 8,192-token output
allowance. That allowance is non-binding for the disabled arm and strongly
binding for the enabled arm, so it censored enabled responses differentially by
mechanism class. This raises the allowance for the enabled arm only, using
context that the original freeze already reserved but never spent.

Only instrument fields were inspected to justify this: finish_reason,
output_token_count, prompt_token_count, condition and domain. No probability,
update, reversal or direction outcome was computed from target responses before
the amendment. Materials, prompts, scoring code, per-request seeds, temperature,
top_p, model snapshot and every engine setting are unchanged.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .. import run_local as c
from .freeze import artifact, check_artifact

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'data/fresh_evaluation_v1/frozen_v1/local_freeze.json'
SUPERSEDED = ROOT / 'responses/fresh_evaluation_v1/_superseded_max_tokens_8192_20260922'
DEST = ROOT / 'data/fresh_evaluation_v1/frozen_v1_amended'
RUN = ROOT / 'runs/fresh_evaluation_v1'
NEW_MAX_TOKENS = 15360


def read(paths):
    rows = []
    for path in sorted(paths):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def quantiles(values):
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {'n': len(v), 'min': v[0], 'p50': q(.5), 'p90': q(.9), 'p99': q(.99), 'max': v[-1]}


def evidence():
    enabled = read(SUPERSEDED.glob('enabled_shard*.jsonl'))
    disabled = read((ROOT / 'responses/fresh_evaluation_v1').glob('disabled_shard*.jsonl'))
    truncated = [r for r in enabled if r.get('finish_reason') == 'length']
    by_domain = defaultdict(lambda: [0, 0])
    for r in enabled:
        cell = by_domain[r['domain']]
        cell[1] += 1
        cell[0] += r.get('finish_reason') == 'length'
    prompts = [r['prompt_token_count'] for r in disabled + enabled if isinstance(r.get('prompt_token_count'), int)]
    return {
        'enabled_arm': {
            'records': len(enabled), 'conditions': dict(Counter(r['condition'] for r in enabled)),
            'finish_reason': dict(Counter(r.get('finish_reason') for r in enabled)),
            'truncated': len(truncated), 'truncated_fraction': round(len(truncated) / len(enabled), 4),
            'output_tokens_completed': quantiles([r['output_token_count'] for r in enabled
                                                  if r.get('finish_reason') == 'stop' and r.get('output_token_count')]),
            'truncation_by_domain': {k: {'truncated': t, 'records': n, 'fraction': round(t / n, 4)}
                                     for k, (t, n) in sorted(by_domain.items())},
        },
        'disabled_arm': {
            'records': len(disabled), 'finish_reason': dict(Counter(r.get('finish_reason') for r in disabled)),
            'truncated': sum(r.get('finish_reason') == 'length' for r in disabled),
            'output_tokens': quantiles([r['output_token_count'] for r in disabled if r.get('output_token_count')]),
        },
        'max_prompt_tokens_observed': max(prompts),
        'inspected_fields_only': ['finish_reason', 'output_token_count', 'prompt_token_count', 'condition', 'domain'],
        'outcome_fields_inspected_before_amendment': [],
    }


def main():
    source = json.loads(FREEZE.read_text())
    for spec in list(source['artifacts'].values()) + list(source['code'].values()) + source['plans']:
        check_artifact(spec)
    ev = evidence()
    old = source['local_settings']['max_tokens']
    headroom = source['local_settings']['max_model_len'] - ev['max_prompt_tokens_observed']
    if not ev['max_prompt_tokens_observed'] + NEW_MAX_TOKENS <= source['local_settings']['max_model_len']:
        raise ValueError('amended output budget does not fit the frozen context window')
    if ev['disabled_arm']['truncated']:
        raise ValueError('disabled arm is not provably uncensored; it may not be retained unchanged')
    if ev['disabled_arm']['output_tokens']['max'] >= old:
        raise ValueError('disabled arm reached the original allowance; retaining it unchanged is unjustified')

    DEST.mkdir(parents=True, exist_ok=True)
    amended = dict(source)
    amended['local_settings'] = dict(source['local_settings'], max_tokens=NEW_MAX_TOKENS)
    amended['created_at'] = c.utc_now()
    amended['amendment'] = {
        'kind': 'output_token_budget', 'applies_to_modes': ['enabled'],
        'supersedes': artifact(FREEZE), 'previous_max_tokens': old, 'max_tokens': NEW_MAX_TOKENS,
        'disabled_arm_retained_under_original_freeze': True,
        'disabled_arm_responses': sorted(str(p.relative_to(ROOT)) for p in
                                         (ROOT / 'responses/fresh_evaluation_v1').glob('disabled_shard*.jsonl')),
        'superseded_enabled_responses': str(SUPERSEDED.relative_to(ROOT)),
        'unchanged': ['materials', 'plans', 'prompts', 'scoring', 'per_request_seeds', 'temperature', 'top_p',
                      'max_model_len', 'batch_size', 'tensor_parallel_size', 'dtype', 'model_snapshot'],
        'evidence': ev, 'context_headroom_tokens': headroom,
    }
    path = DEST / 'local_freeze_max_tokens_15360.json'
    c.atomic_write(path, json.dumps(amended, indent=2) + '\n')
    record = {'status': 'recorded', 'created_at': amended['created_at'], 'amended_freeze': artifact(path),
              'source_freeze': artifact(FREEZE), 'amendment': amended['amendment'],
              'amendment_code': artifact(Path(__file__))}
    c.atomic_write(RUN / 'TOKEN_BUDGET_AMENDMENT.json', json.dumps(record, indent=2) + '\n')
    print(json.dumps({'amended_freeze': str(path), 'max_tokens': NEW_MAX_TOKENS,
                      'enabled_truncated': ev['enabled_arm']['truncated'],
                      'enabled_records': ev['enabled_arm']['records'],
                      'disabled_max_output_tokens': ev['disabled_arm']['output_tokens']['max']}, indent=2))


if __name__ == '__main__':
    main()
