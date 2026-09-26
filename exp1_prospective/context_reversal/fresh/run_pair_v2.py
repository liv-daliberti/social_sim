"""Run one shard's enabled-thinking arm under the amended output-token budget.

The disabled arm is complete under the original freeze and is provably
uncensored there (31 output tokens at most against an 8,192 allowance), so this
launcher never regenerates it. Longer allocations are used because the enabled
arm now carries the whole job: the original wrapper spent two model loads and a
1,200-second budget per hour to advance one arm.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

from .. import run_local as c
from .freeze import check_artifact

STOP_AFTER_SECONDS = 2700
WALLTIME = '01:00:00'
MAX_CONTINUATIONS = 23


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--continuation', type=int, default=0)
    a = p.parse_args()
    f = json.loads(a.freeze.read_text())
    root = Path(__file__).resolve().parents[1]
    amendment = f.get('amendment')
    if not amendment or amendment.get('applies_to_modes') != ['enabled']:
        raise ValueError('this launcher requires the enabled-arm token-budget amendment')
    for spec in f['code'].values():
        check_artifact(spec)
    check_artifact(amendment['supersedes'])
    plan = check_artifact(f['plans'][a.shard])
    settings = f['local_settings']
    mode = 'enabled'
    output = Path(f['responses_dir']) / (mode + '_' + plan.stem + '.jsonl')
    state = {'shard': a.shard, 'continuation': a.continuation, 'job_id': os.getenv('SLURM_JOB_ID'),
             'freeze_sha256': c.file_sha256(a.freeze), 'launcher_sha256': c.file_sha256(Path(__file__)),
             'arm': mode, 'max_tokens': settings['max_tokens'], 'started_at': c.utc_now(), 'modes': {}}
    state_path = root / 'runs/fresh_evaluation_v1/local_jobs_amended' / f'shard{a.shard:02d}_attempt{a.continuation:02d}.json'
    c.atomic_write(state_path, json.dumps(state, indent=2) + '\n')

    cmd = [sys.executable, '-u', '-m', 'exp1_prospective.context_reversal.fresh.reasoning',
           '--model-path', f['local_model']['path'], '--model-key', f['model_keys'][mode],
           '--input', str(plan), '--output', str(output), '--thinking', mode,
           '--design-freeze', str(a.freeze.resolve()), '--stop-after-seconds', str(STOP_AFTER_SECONDS),
           '--enforce-eager']
    for key, flag in [('temperature', '--temperature'), ('max_tokens', '--max-tokens'), ('seed', '--seed'),
                      ('max_model_len', '--max-model-len'), ('batch_size', '--batch-size'),
                      ('tensor_parallel_size', '--tensor-parallel-size'), ('dtype', '--dtype')]:
        cmd.extend([flag, str(settings[key])])
    result = subprocess.run(cmd, check=False)
    mp = Path(str(output) + '.manifest.json')
    manifest = json.loads(mp.read_text()) if mp.exists() else {}
    state['modes'][mode] = {'command': cmd, 'returncode': result.returncode, 'status': manifest.get('status'),
                            'records': manifest.get('records', 0), 'expected_records': manifest.get('expected_records')}
    c.atomic_write(state_path, json.dumps(state, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    x = state['modes'][mode]
    complete = x['records'] == x['expected_records'] and x['status'] in ('complete', 'complete_with_errors')
    if not complete:
        if a.continuation >= MAX_CONTINUATIONS:
            raise RuntimeError('allocation limit reached; inspect remaining requests')
        log = root / 'logs/fresh_evaluation_v1'
        sub = ['sbatch', '--parsable', '--partition=all', '--gres=gpu:a6000:2', '--mem=120G',
               '--time=' + WALLTIME, '--job-name=fresh-pair-' + str(a.shard),
               '--output=' + str(log / f'pair{a.shard:02d}_%j.out'), '--error=' + str(log / f'pair{a.shard:02d}_%j.err')]
        if os.getenv('SLURM_JOB_ID'):
            sub.append('--dependency=afterany:' + os.environ['SLURM_JOB_ID'])
        sub.extend([str(root / 'fresh/run.sbatch'), 'exp1_prospective.context_reversal.fresh.run_pair_v2',
                    '--freeze', str(a.freeze.resolve()), '--shard', str(a.shard), '--continuation', str(a.continuation + 1)])
        # Durable submission intent: do not rerun this launcher if the reply is lost.
        state['continuation_submission'] = {'status': 'intent', 'command': sub}
        c.atomic_write(state_path, json.dumps(state, indent=2) + '\n')
        result = subprocess.run(sub, capture_output=True, text=True)
        state['continuation_submission'].update(status='submitted' if result.returncode == 0 else 'failed',
                                                job_id=result.stdout.strip(), stderr=result.stderr,
                                                returncode=result.returncode)
        c.atomic_write(state_path, json.dumps(state, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError('continuation submission failed')
    state.update(status='complete' if complete else 'continuation_submitted', updated_at=c.utc_now())
    c.atomic_write(state_path, json.dumps(state, indent=2) + '\n')
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
