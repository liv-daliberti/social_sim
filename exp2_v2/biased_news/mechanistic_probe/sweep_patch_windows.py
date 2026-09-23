#!/usr/bin/env python3
"""Causal patch sweep across depth, on development episodes only.

The registered protocol tests one window, chosen at the argmax of development
decoding CV. `layer_decoding_sweep.py` shows why that misfires on deep models:
decoding rises while the binding is computed and then sits on a long flat
plateau, so the argmax is noise-dominated and can land far past the layers where
patching still transfers anything.

This sweeps several windows and reports a causal profile across depth instead of
a single point. It reads DEVELOPMENT episodes only and writes outside the
registered artifacts, so the sealed test is untouched and a confirmatory run at a
chosen window remains available.

Generation and state capture are imported from the registered patching module
rather than restated, so a sweep number means the same thing as a protocol
number. The unpatched baseline does not depend on the window, so it is generated
once per episode and shared across windows.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_symbol_label_activations as base

ARMS = ('abc_context', 'abc_wrong_context')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', type=int, nargs='+', required=True,
                    help='window START layers; each window is three contiguous layers')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--depths', type=int, nargs='+', default=[0])
    ap.add_argument('--device-map', default='balanced')
    ap.add_argument('--gpu-memory-gib', type=int, default=44)
    ap.add_argument('--max-new-tokens', type=int, default=160)
    ap.add_argument('--stop-after-seconds', type=int, default=2700)
    a = ap.parse_args()

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks, _ = base.load_tasks(a.run_dir)
    checkpoint = base.checkpoint_spec(a.run_dir)
    model_id = str(checkpoint['model_id'])
    n_layers = int(checkpoint['transformer_layers'])
    windows = [list(range(w, w + 3)) for w in a.windows]
    for w in windows:
        if w[0] < 1 or w[-1] > n_layers - 1:
            raise ValueError(f'window {w} includes a causally ineligible layer')

    by_key = {(int(r['episode']), int(r['c_cases']), str(r['arm'])): r
              for r in tasks if r['split'] == 'dev' and r['arm'] in ARMS}
    episodes = sorted({k[0] for k in by_key})
    episodes = [e for i, e in enumerate(episodes) if i % a.shards == a.shard]
    print(f'development episodes in this shard: {len(episodes)}; windows: {a.windows}', flush=True)

    done = set()
    records = []
    if a.out.exists():
        records = [json.loads(l) for l in a.out.read_text().splitlines() if l.strip()]
        done = {(r['episode'], r['c_cases'], r['recipient_arm'], r['condition']) for r in records}

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(base.PATCH_SEED); torch.cuda.manual_seed_all(base.PATCH_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map=a.device_map,
        max_memory={i: f'{a.gpu_memory_gib}GiB' for i in range(torch.cuda.device_count())})
    model.eval()
    if int(model.config.num_hidden_layers) != n_layers:
        raise RuntimeError('model depth differs from the frozen protocol')

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
        tmp.replace(a.out)

    started = time.time()
    for episode in episodes:
        for depth in a.depths:
            pair = {arm: by_key.get((episode, depth, arm)) for arm in ARMS}
            if any(v is None for v in pair.values()):
                continue
            states, positions, ids = {}, {}, {}
            for arm, task in pair.items():
                st, pos, iids = base.capture_city_c_states(model=model, tokenizer=tokenizer, task=task)
                states[arm], positions[arm], ids[arm] = st, pos, iids
            base.validate_matched_pair(correct_positions=positions[ARMS[0]],
                                       wrong_positions=positions[ARMS[1]],
                                       correct_ids=ids[ARMS[0]], wrong_ids=ids[ARMS[1]])
            for recipient, task in pair.items():
                donor = ARMS[1] if recipient == ARMS[0] else ARMS[0]
                jobs = [('unpatched', None, [])]
                jobs += [(f'cross_window_{w[0]}', states[donor], w) for w in windows]
                for name, source, layers in jobs:
                    key = (episode, depth, recipient, name)
                    if key in done:
                        continue
                    out = base.generate_one(model=model, tokenizer=tokenizer, task=task,
                                            source_states=source, hidden_state_layers=layers,
                                            max_new_tokens=a.max_new_tokens)
                    out.update(episode=episode, c_cases=depth, recipient_arm=recipient,
                               source_arm=None if source is None else donor, condition=name,
                               patched_hidden_state_layers=layers, split='dev',
                               sample_id=task['sample_id'],
                               query_starting_poll=task['query_starting_poll'],
                               query_net_news=task['query_net_news'],
                               strong_reference_city=task['strong_reference_city'],
                               target_strong=task['target_strong'])
                    records.append(out); done.add(key)
            del states
            flush()
            elapsed = time.time() - started
            print(f'episode {episode} depth {depth}: {len(records)} records, {elapsed/60:.1f} min', flush=True)
            if elapsed >= a.stop_after_seconds:
                print('walltime budget reached; resume continues from saved records', flush=True)
                return
    print('sweep complete', flush=True)


if __name__ == '__main__':
    main()
