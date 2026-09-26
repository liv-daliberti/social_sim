#!/usr/bin/env python3
"""Seed format-only Qwen3 v2 recovery from a terminal, immutable v1 arm.

No model calls. Every received completion is reused, including invalid/truncated
ones. Only blocked, unattempted updates whose baseline becomes valid are omitted
so that v2 can generate them once. No baseline is regenerated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

try:
    from . import run_reasoning_local as v1, run_reasoning_local_v2 as v2
except ImportError:
    import run_reasoning_local as v1
    import run_reasoning_local_v2 as v2

common = v1.common


def identity(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': common.file_sha256(path)}


def arguments(config: dict, output: Path) -> argparse.Namespace:
    decode, engine = config['decode'], config['engine']
    return argparse.Namespace(input=Path(config['input_path']), output=output,
                              model_path=Path(config['local_model']['path']), model_key=config['model_key'],
                              thinking=decode['thinking'], seed=decode['seed'], temperature=decode['temperature'],
                              max_tokens=decode['max_tokens'], tensor_parallel_size=engine['tensor_parallel_size'],
                              batch_size=engine['batch_size'], max_model_len=engine['max_model_len'],
                              gpu_memory_utilization=engine['gpu_memory_utilization'], dtype=engine['dtype'],
                              enforce_eager=engine['enforce_eager'], dry_run=False)


def reparse_records(source_records: dict, config: dict, source: dict) -> tuple[dict, list[dict]]:
    """Pure transformation; source records remain unchanged."""
    records, omitted = {}, []
    signature = common.text_sha256(common.canonical_json(config))
    for key, original in source_records.items():
        row = copy.deepcopy(original)
        row.update(run_signature=signature, parser_version=v2.PARSER_VERSION,
                   final_output_raw=None, final_format=None,
                   source_record_sha256=common.text_sha256(common.canonical_json(original)),
                   source_response_path=source['source_response']['path'],
                   source_response_sha256=source['source_response']['sha256'],
                   source_manifest_sha256=source['source_manifest']['sha256'],
                   source_run_signature=source['source_run_signature'],
                   source_status=original['status'], source_probability=original['probability'],
                   source_error=original.get('error'), source_parser_version=v1.PARSER_VERSION,
                   recovered_at=source['created_at'],
                   recovery_origin='reparsed_received_completion' if original['response_received'] else 'retained_unattempted_failure')
        if original['response_received']:
            row.update(v2.inspect_completion(original['raw'], config['decode']['thinking'], original['finish_reason'],
                                              reasoning_open_in_prompt=original['reasoning_open_in_prompt']))
        if original['status'] == 'ok' and (row['status'] != 'ok' or row['probability'] != original['probability']):
            raise ValueError(f'Parser revision changed an already valid probability: {key}')
        for field in ('raw', 'prompt', 'prompt_sha256', 'seed', 'prior_probability', 'chat_prompt_sha256',
                      'prompt_token_count', 'output_token_count', 'finish_reason', 'stop_reason', 'created_at'):
            if row.get(field) != original.get(field):
                raise ValueError(f'Recovery changed immutable sampled content/settings: {key}/{field}')
        records[key] = row
    for key, row in list(records.items()):
        original = source_records[key]
        if original['status'] != 'blocked_baseline':
            continue
        baseline = records[(key[0], 'baseline', 'baseline')]
        if baseline['status'] == 'ok':
            if original['response_received'] or original['raw'] or original['prompt'] is not None or key[1] != 'update':
                raise ValueError(f'Cannot regenerate a previously attempted response: {key}')
            omitted.append({'key': list(key), 'source_record_sha256': row['source_record_sha256'],
                            'source_status': original['status'], 'prior_probability': baseline['probability']})
            del records[key]
    expected_received = {key for key, row in source_records.items() if row['response_received']}
    actual_received = {key for key, row in records.items() if row['response_received']}
    if expected_received != actual_received or any(key[1] == 'baseline' for key in (tuple(r['key']) for r in omitted)):
        raise ValueError('Recovery would drop a sampled output or regenerate a baseline')
    return records, omitted


def seed(source_path: Path, output_path: Path, *, dry_run: bool = False) -> dict:
    source_path, output_path = source_path.resolve(), output_path.resolve()
    if source_path == output_path:
        raise ValueError('Recovery output must differ from the frozen source')
    source_manifest_path = Path(str(source_path) + '.manifest.json')
    # Check terminal status before touching any potentially in-progress raw file.
    preliminary = json.loads(source_manifest_path.read_text())
    if preliminary.get('status') not in ('complete', 'complete_with_errors'):
        raise ValueError('Source arm must be terminal; do not seed from an unfinished output')
    with common.output_lock(Path(str(source_path) + '.lock')):
        manifest = json.loads(source_manifest_path.read_text())
        if manifest.get('status') not in ('complete', 'complete_with_errors'):
            raise ValueError('Source became nonterminal before the snapshot lock')
        source_config = manifest['config']
        args = arguments(source_config, source_path)
        units = common.read_units(args.input)
        if v1.make_config(args, units) != source_config:
            raise ValueError('Source config/code/model/software differs from the frozen v1 runner')
        source_records = v1.load_existing(source_path, source_manifest_path, source_config, units)
        expected = {(u['trial_id'], stage, arm) for u in units
                    for stage, arm in [('baseline', 'baseline'), *[('update', c) for c in v1.CONDITIONS]]}
        if set(source_records) != expected or manifest.get('records') != len(expected):
            raise ValueError('Source terminal arm lacks its exact complete planned record set')
        if manifest.get('counts') != dict(Counter(r['status'] for r in source_records.values())):
            raise ValueError('Source manifest status counts do not match raw records')
        args.output = output_path
        config = v2.make_config(args, units)
        changes = {key for key in config if config[key] != source_config.get(key)}
        if changes != {'schema_version', 'parser_version', 'runner_sha256'}:
            raise ValueError(f'Recovery changed inference settings beyond the parser runner: {changes}')
        recovery = {'schema_version': 'qwen3_format_recovery_v2', 'created_at': common.utc_now(),
                    'source_response': identity(source_path), 'source_manifest': identity(source_manifest_path),
                    'source_run_signature': manifest['run_signature'], 'source_status_counts': manifest['counts'],
                    'source_parser_version': source_config['parser_version'], 'target_parser_version': config['parser_version'],
                    'seeder': identity(Path(__file__)), 'policy': 'Reuse all received raw outputs; only generate previously blocked unattempted updates; no baseline regeneration'}
        records, omitted = reparse_records(source_records, config, recovery)
        recovery.update(previously_blocked_unattempted_updates=omitted, expected_records=len(expected),
                        reused_records=len(records), reused_received_completions=sum(r['response_received'] for r in records.values()),
                        recovered_format_records=sum(r['source_status'] == 'parse_error' and r['status'] == 'ok' for r in records.values()),
                        additional_generation_requests=len(omitted), baseline_generation_requests=0,
                        seeded_status_counts=dict(Counter(r['status'] for r in records.values())),
                        target_run_signature=common.text_sha256(common.canonical_json(config)))
        target_manifest = {'run_signature': recovery['target_run_signature'], 'config': config,
                           'started_at': recovery['created_at'], 'updated_at': recovery['created_at'],
                           'slurm_job_ids': [], 'status': 'recovery_seeded', 'expected_records': len(expected),
                           'records': len(records), 'counts': recovery['seeded_status_counts'], 'recovery': recovery}
        for field in ('hardware', 'tokenizer_chat_template_sha256'):
            if field in manifest:
                target_manifest[field] = manifest[field]
        # Validate own-context priors/prompts/seeds and v2 diagnostics before writes.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='qwen3-recovery-verify-') as directory:
            trial_output = Path(directory) / 'responses.jsonl'
            trial_manifest = Path(str(trial_output) + '.manifest.json')
            trial_output.write_text(''.join(common.canonical_json(r) + '\n' for r in records.values()))
            trial_manifest.write_text(common.canonical_json(target_manifest))
            v2.load_existing(trial_output, trial_manifest, config, units)
        if not dry_run:
            output_manifest = Path(str(output_path) + '.manifest.json')
            recovery_path = Path(str(output_path) + '.recovery.json')
            with common.output_lock(Path(str(output_path) + '.lock')):
                if any(path.exists() for path in (output_path, output_manifest, recovery_path)):
                    raise ValueError('Target recovery output already exists; refusing to overwrite or reseed')
                common.atomic_write(output_manifest, json.dumps(target_manifest, sort_keys=True, indent=2) + '\n')
                common.atomic_write(output_path, ''.join(common.canonical_json(r) + '\n' for r in records.values()))
                common.atomic_write(recovery_path, json.dumps(recovery, sort_keys=True, indent=2) + '\n')
        return {key: recovery[key] for key in ('source_response', 'target_run_signature', 'reused_records',
                'reused_received_completions', 'recovered_format_records', 'additional_generation_requests',
                'baseline_generation_requests', 'seeded_status_counts')} | {'dry_run': dry_run, 'output': str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate and report the recovery; no output writes or inference')
    args = parser.parse_args()
    print(json.dumps(seed(args.source, args.output, dry_run=args.dry_run), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
