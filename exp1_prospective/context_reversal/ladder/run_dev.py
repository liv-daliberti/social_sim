#!/usr/bin/env python3
"""Development calibration for the difficulty ladder. Unfrozen by design.

This is a development set, not an evaluation: there is no screening, no freeze
and no authorization ledger. Its only question is whether accuracy falls as the
actor-to-channel binding moves away from the surface, which is what decides
whether a frozen ladder cohort is worth building. It did not, so the outcome is
reported in App. app:exp1-ladder as an exploratory probe, labelled there as
development material that supports no claim about model capability.

Chat rendering and completion parsing are imported from the frozen runner rather
than restated, so a calibration result cannot diverge from how the cohort is
scored. Only the baseline and the new-news branch are generated; the controls add
nothing to the triple measure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal.fresh import reasoning


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--thinking', choices=('enabled', 'disabled'), default='enabled')
    p.add_argument('--max-tokens', type=int, default=15360)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--tensor-parallel-size', type=int, default=2)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--stop-after-seconds', type=int, default=2700)
    a = p.parse_args()

    units = [u for i, u in enumerate(common.read_units(a.plan)) if i % a.shards == a.shard]
    started = time.monotonic()
    manifest_path = Path(str(a.output) + '.manifest.json')
    records = {}
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                records[(row['trial_id'], row['condition'])] = row

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', VLLM_USE_V1='0')
    from vllm import LLM, SamplingParams
    engine = LLM(model=str(a.model_path), tokenizer=str(a.model_path), trust_remote_code=False,
                 tensor_parallel_size=a.tensor_parallel_size, max_model_len=a.max_model_len,
                 gpu_memory_utilization=.9, dtype='bfloat16', seed=a.seed, enable_prefix_caching=True,
                 enforce_eager=True, max_num_seqs=a.batch_size, disable_custom_all_reduce=True)
    tok = engine.get_tokenizer()

    def checkpoint(status='running'):
        common.atomic_write(a.output, ''.join(common.canonical_json(r) + '\n' for r in records.values()))
        common.atomic_write(manifest_path, json.dumps({
            'status': status, 'records': len(records), 'planned': 2 * len(units),
            'thinking': a.thinking, 'max_tokens': a.max_tokens, 'plan_sha256': common.file_sha256(a.plan),
            'runner_sha256': common.file_sha256(Path(__file__)),
            'counts': dict(Counter(r['status'] for r in records.values())),
            'development_only': True, 'updated_at': common.utc_now()}, indent=2) + '\n')

    def run(batch):
        if time.monotonic() - started >= a.stop_after_seconds:
            raise TimeoutError('walltime budget reached')
        prompts, params = [], []
        for unit, condition, prompt in batch:
            chat, opened = reasoning.render_chat(tok, prompt, a.thinking)
            tokens = tok.encode(chat, add_special_tokens=False)
            if len(tokens) + a.max_tokens > a.max_model_len:
                raise ValueError(f'{unit["trial_id"]}: prompt plus output exceeds the context window')
            prompts.append({'prompt_token_ids': tokens})
            stage = 'baseline' if condition == 'baseline' else 'update'
            params.append(SamplingParams(n=1, temperature=a.temperature, top_p=1., max_tokens=a.max_tokens,
                                         seed=common.request_seed(a.seed, unit['trial_id'], stage, condition),
                                         skip_special_tokens=False))
        for unit, condition, prompt in batch:
            records[(unit['trial_id'], condition)] = {
                'trial_id': unit['trial_id'], 'condition': condition, 'status': 'generation_error',
                'failure_kind': 'interrupted_unknown', 'probability': None, 'rung': unit['rung'],
                'variant': unit['variant'], 'domain': unit['domain'], 'context_id': unit['context_id'],
                'expected_sign': unit['expected_sign'], 'oracle_baseline': unit['oracle_baseline'],
                'oracle_update': unit['oracle_update'], 'parent_family_id': unit['parent_family_id']}
        checkpoint()
        for (unit, condition, prompt), result in zip(batch, engine.generate(prompts, sampling_params=params, use_tqdm=False)):
            out = result.outputs[0]
            row = records[(unit['trial_id'], condition)]
            row.update(reasoning.inspect_completion(out.text, a.thinking, out.finish_reason,
                                                    reasoning_open_in_prompt=chat_opened(tok, prompt, a.thinking)),
                       finish_reason=out.finish_reason, output_token_count=len(out.token_ids))
            row['status'] = 'ok' if row['probability'] is not None else row.get('status', 'parse_error')
        checkpoint()
        print(f"Checkpoint: {len(records)}/{2 * len(units)}; {dict(Counter(r['status'] for r in records.values()))}", flush=True)

    def chat_opened(tokenizer, prompt, thinking):
        return reasoning.render_chat(tokenizer, prompt, thinking)[1]

    try:
        pending = [(u, 'baseline', u['baseline_prompt']) for u in units if (u['trial_id'], 'baseline') not in records]
        random.Random(a.seed).shuffle(pending)
        for i in range(0, len(pending), a.batch_size):
            run(pending[i:i + a.batch_size])
        pending = []
        for unit in units:
            if (unit['trial_id'], 'new_news') in records:
                continue
            baseline = records.get((unit['trial_id'], 'baseline'))
            if not baseline or baseline['status'] != 'ok':
                records[(unit['trial_id'], 'new_news')] = {
                    'trial_id': unit['trial_id'], 'condition': 'new_news', 'status': 'blocked_baseline',
                    'probability': None, 'rung': unit['rung'], 'variant': unit['variant'],
                    'domain': unit['domain'], 'context_id': unit['context_id'],
                    'expected_sign': unit['expected_sign'], 'oracle_baseline': unit['oracle_baseline'],
                    'oracle_update': unit['oracle_update'], 'parent_family_id': unit['parent_family_id']}
                continue
            pending.append((unit, 'new_news', common.render_update(unit, 'new_news', baseline['probability'])))
        random.Random(a.seed + 1).shuffle(pending)
        for i in range(0, len(pending), a.batch_size):
            run(pending[i:i + a.batch_size])
        checkpoint('complete')
        print('complete', flush=True)
    except TimeoutError:
        checkpoint('paused_for_walltime')
        print('paused for walltime; resume continues from saved records', flush=True)
    except BaseException:
        checkpoint('interrupted')
        raise


if __name__ == '__main__':
    main()
