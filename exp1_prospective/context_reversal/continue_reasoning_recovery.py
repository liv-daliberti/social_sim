"""Resume missing strict-v1 records, then apply the symmetric v2 format recovery.

Infrastructure continuation only: both runners retain all saved sampled outputs,
including failures. The seeder allows new calls only for never-attempted updates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .seed_reasoning_recovery_v2 import seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--model-key', required=True)
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text())
    root = Path(__file__).resolve().parent
    for filename, expected in submission['code_sha256'].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen code changed: {filename}')
    job = next(j for j in submission['jobs'] if j['model_key'] == args.model_key)
    original = job['strict_v1_source']
    if original.get('thinking') != job.get('thinking'):
        raise ValueError('Thinking mode changed during continuation')
    print('Continuing only missing v1 records; saved failures are terminal.', flush=True)
    subprocess.run(original['inference_command'], check=True)
    subprocess.run(original['analysis_command'], check=True)
    output = Path(job['response_path'])
    if not output.exists():
        print(json.dumps(seed(Path(original['response_path']), output), indent=2), flush=True)
    else:
        recovery_path = Path(str(output) + '.recovery.json')
        recovery = json.loads(recovery_path.read_text())
        source = recovery['source_response']
        if (source['path'] != original['response_path'] or
                source['sha256'] != hashlib.sha256(Path(original['response_path']).read_bytes()).hexdigest()):
            raise ValueError('Recovery source differs from immutable v1 output')
    subprocess.run(job['recovery_inference_command'], check=True)


if __name__ == '__main__':
    main()
