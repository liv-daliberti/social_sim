#!/usr/bin/env python3
"""CPU-only audited paper completion. Default is a read-only paper preflight.

Configure freezes code and canonical paper hashes in a separate config. Execute
requires every model and finalizer complete, builds in /tmp, and publishes only
against that unchanged snapshot. No inference, submissions, or frozen-worker edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = HERE / 'runs/robustness_development_v2'
MODELS = ('qwen2_5_72b_instruct', 'llama3_1_70b_instruct', 'qwen3_32b')
APPENDIX = 'experiment1_context_reversal_appendix.tex'
TABLES = tuple(f'tables/exp1_context_robustness_v2_{name}.tex' for name in ('performance', 'edits', 'controls'))
TARGETS = (APPENDIX, *TABLES, 'main.pdf')
HEADING = 'Controlled robustness development on the original families.'
CODE = ('finish_robustness_pipeline.py', 'finish_robustness_pipeline.sbatch', 'audit_robustness_results.py', 'write_robustness_paper.py', 'run_local.py')
EXCLUDED_DIRS = {'iclr', '_archive', 'socialagent2026', 'neurips', 'workshop', '__pycache__', '.pytest_cache', '.git'}
AUX_SUFFIXES = ('.aux', '.log', '.bbl', '.blg', '.out', '.toc', '.fls', '.fdb_latexmk', '.synctex.gz', '.bcf', '.run.xml', '.nav', '.snm', '.vrb', '.orig', '.bak', '.zip', '.zip.sha256')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save(path, value):
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def paper_snapshot(paper):
    """Protect all canonical regular sources/assets, including untracked inputs."""
    paper = Path(paper).resolve()
    sources = {}
    for root, directories, files in os.walk(paper, followlinks=False):
        directories[:] = sorted(name for name in directories if name.lower() not in EXCLUDED_DIRS)
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError('Unrecognized canonical input symlink: ' + str(Path(root) / name))
        for name in sorted(files):
            path = Path(root) / name
            relative = path.relative_to(paper).as_posix()
            if relative == 'main.pdf' or name.endswith(AUX_SUFFIXES) or 'neurips' in name.lower() or 'workshop' in name.lower():
                continue
            if path.is_symlink():
                raise ValueError('Unrecognized canonical input symlink: ' + str(path))
            if path.is_file():
                sources[relative] = digest(path)
    pdf = paper / 'main.pdf'
    if pdf.is_symlink():
        raise ValueError('Canonical main.pdf must not be a symlink')
    return {'sources_sha256': sources, 'main_pdf_sha256': digest(pdf) if pdf.is_file() else None}


def assert_snapshot(paper, expected):
    actual = paper_snapshot(paper)
    if actual != expected:
        old, new = expected['sources_sha256'], actual['sources_sha256']
        changed = sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))
        if actual['main_pdf_sha256'] != expected['main_pdf_sha256']:
            changed.append('main.pdf')
        raise ValueError('Canonical paper changed since configuration: ' + ', '.join(changed[:20]))


def ai_statement_page(text):
    pages = [index + 1 for index, page in enumerate(text.split('\f')) if 'ai use statement' in ' '.join(page.lower().split())]
    if len(pages) != 1:
        raise ValueError('Expected exactly one AI use statement in PDF text')
    return pages[0]


def submission_projection(path):
    submission = load(path)
    keys = ('run_id', 'sampling_families', 'variants', 'expected_records', 'code_sha256',
            'protocol_path', 'protocol_sha256', 'probability_design', 'probability_design_sha256',
            'direction_design', 'direction_design_sha256')
    group_keys = ('model_key', 'model_path', 'tensor_parallel_size', 'expected_probability_records',
                  'expected_direction_records', 'results_dir', 'probability_responses', 'direction_responses',
                  'probability_command', 'direction_command', 'analysis_command')
    projection = {key: submission[key] for key in keys}
    projection['groups'] = {key: {field: group[field] for field in group_keys}
                            for key, group in submission['groups'].items()}
    return projection


def make_config(run_dir, paper):
    run_dir, paper = Path(run_dir).resolve(), Path(paper).resolve()
    snapshot = paper_snapshot(paper)
    if not (paper / 'main.tex').is_file() or snapshot['main_pdf_sha256'] is None:
        raise ValueError('Canonical main.tex and existing main.pdf are required')
    if any((paper / name).exists() or (paper / name).is_symlink() for name in TABLES):
        raise ValueError('All three new canonical robustness tables must be absent at configuration')
    tools = {name: shutil.which(name) for name in ('latexmk', 'pdftotext')}
    if not all(tools.values()):
        raise ValueError('latexmk and pdftotext must be available')
    text = subprocess.check_output([tools['pdftotext'], str(paper / 'main.pdf'), '-'], text=True, timeout=60)
    page = ai_statement_page(text)
    if page != 10:
        raise ValueError('Existing main paper must have nine pages before the AI statement')
    config = {'schema_version': 'robustness_finish_pipeline_config_v1', 'created_at': now(),
              'run_dir': str(run_dir), 'paper_dir': str(paper), 'repository': str(REPO),
              'submission_path': str(run_dir / 'submission.json'), 'python_executable': sys.executable,
              'submission_projection': submission_projection(run_dir / 'submission.json'),
              'paper_snapshot': snapshot, 'publication_targets': list(TARGETS),
              'absent_table_paths': list(TABLES), 'ai_statement_page': page,
              'code_sha256': {name: digest(HERE / name) for name in CODE},
              'tools': {name: {'path': path, 'sha256': digest(path)} for name, path in tools.items()},
              'frontier_api_calls': 0, 'new_human_review_requested': False}
    assert_snapshot(paper, snapshot)
    return config


def verify_code(config):
    if config.get('schema_version') != 'robustness_finish_pipeline_config_v1' or config['publication_targets'] != list(TARGETS):
        raise ValueError('Unknown pipeline schema or publication targets')
    if config.get('absent_table_paths') != list(TABLES) or any(name in config['paper_snapshot']['sources_sha256'] for name in TABLES):
        raise ValueError('Configured absence of the new tables changed')
    if submission_projection(config['submission_path']) != config['submission_projection']:
        raise ValueError('Frozen submission design, model, settings, or source roster changed')
    for field in ('probability_design', 'direction_design'):
        projection = config['submission_projection']
        if digest(projection[field]) != projection[field + '_sha256']:
            raise ValueError('Frozen design changed: ' + field)
    projection = config['submission_projection']
    if digest(projection['protocol_path']) != projection['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    for name, expected in projection['code_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError('Frozen analysis/runner code changed: ' + name)
    if set(config['code_sha256']) != set(CODE):
        raise ValueError('Pipeline code roster changed')
    for name, expected in config['code_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError('Pipeline code hash changed: ' + name)
    for name, spec in config['tools'].items():
        if digest(spec['path']) != spec['sha256']:
            raise ValueError('Build tool hash changed: ' + name)


def readiness(config):
    submission = load(config['submission_path'])
    if set(submission['groups']) != set(MODELS):
        raise ValueError('Submission does not contain exactly the prescribed three models')
    missing, models = [], {}
    for model in MODELS:
        group = submission['groups'][model]
        path = Path(group['results_dir']) / 'summary.json'
        reasons = []
        if not path.is_file():
            reasons.append('summary_missing')
        else:
            summary = load(path)
            if summary.get('model_key') != model or not summary.get('record_complete'):
                reasons.append('summary_incomplete_or_wrong_model')
            if (summary.get('n_sampling_families'), summary.get('n_variants'), summary.get('n_units')) != (20, 4, 320):
                reasons.append('wrong_parent_variant_coverage')
        for task, count in (('probability', 1280), ('direction', 960)):
            manifest_path = Path(group[task + '_responses'] + '.manifest.json')
            if not manifest_path.is_file():
                reasons.append(task + '_manifest_missing')
            else:
                manifest = load(manifest_path)
                if manifest.get('status') != 'complete' or manifest.get('records') != count:
                    reasons.append(task + '_incomplete')
        models[model] = {'ready': not reasons, 'reasons': reasons, 'summary_path': str(path)}
        if reasons:
            missing.append(model)
    completion_path = Path(config['run_dir']) / 'completion.json'
    completion_ready = False
    if completion_path.is_file():
        completion = load(completion_path)
        completion_ready = (completion.get('analysis_complete') is True and completion.get('record_complete') is True
                            and completion.get('expected_records') == 6720 and set(completion.get('models', {})) == set(MODELS))
        if completion_ready:
            for model in MODELS:
                item = completion['models'][model]
                completion_ready &= item.get('summary_sha256') == digest(Path(models[model]['summary_path']))
    return {'ready': not missing and completion_ready, 'missing_models': missing,
            'finalizer_complete': bool(completion_ready), 'models': models}


def verify_draft(draft, expected_manifest_sha256=None):
    if expected_manifest_sha256 is not None and digest(draft / 'draft_manifest.json') != expected_manifest_sha256:
        raise ValueError('Generated draft manifest changed since writer execution')
    manifest = load(draft / 'draft_manifest.json')
    if manifest.get('status') != 'complete_audited_draft_for_root_review':
        raise ValueError('Writer did not produce a complete audited draft')
    for path, expected in manifest['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Draft provenance source changed: ' + path)
    if digest(manifest['writer']['path']) != manifest['writer']['sha256']:
        raise ValueError('Draft writer source changed')
    if set(manifest.get('generated_sha256', {})) != {'exp1_context_robustness_v2.tex', *TABLES}:
        raise ValueError('Writer output hash roster is incomplete')
    for relative, expected in manifest['generated_sha256'].items():
        if (draft / relative).is_symlink() or not (draft / relative).is_file():
            raise ValueError('Draft output is missing or symlinked: ' + relative)
        if digest(draft / relative) != expected:
            raise ValueError('Generated draft output changed: ' + relative)
    return manifest


def stage_paper(config, draft, generated_sha256):
    paper = Path(config['paper_dir'])
    assert_snapshot(paper, config['paper_snapshot'])
    stage = Path(tempfile.mkdtemp(prefix='robustness-paper-', dir='/tmp'))
    for relative, expected in config['paper_snapshot']['sources_sha256'].items():
        source, target = paper / relative, stage / relative
        if source.is_symlink() or digest(source) != expected:
            raise ValueError('Canonical input changed while staging: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected:
            raise ValueError('Canonical input changed during staging copy: ' + relative)
    appendix = stage / APPENDIX
    text = appendix.read_text()
    marker = r'\paragraph{Interpretive limits.}'
    if text.count(marker) != 1 or HEADING in text:
        raise ValueError('Appendix insertion marker is absent, duplicated, or already expanded')
    before, original_limits = text.split(marker)
    if original_limits.count('Baseline probabilities at the relevant endpoint') != 1 or original_limits.count('Material ambiguities also remain:') != 1:
        raise ValueError('Original-pilot limits no longer match the reviewed scope edit')
    original_limits = original_limits.replace('Baseline probabilities at the relevant endpoint', 'In the original three-repeat pilot, baseline probabilities at the relevant endpoint', 1)
    original_limits = original_limits.replace('Material ambiguities also remain:', 'The original pilot materials retain ambiguities:', 1)
    fragment_bytes = (draft / 'exp1_context_robustness_v2.tex').read_bytes()
    if hashlib.sha256(fragment_bytes).hexdigest() != generated_sha256['exp1_context_robustness_v2.tex']:
        raise ValueError('Draft fragment changed during staging')
    fragment = fragment_bytes.decode('utf-8')
    if fragment.count(HEADING) != 1:
        raise ValueError('Unexpected robustness fragment heading')
    appendix.write_text(before + fragment.rstrip() + '\n\n' + r'\paragraph{Interpretive limits of the original pilot.}' + original_limits)
    for relative in TABLES:
        (stage / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(draft / relative, stage / relative)
        if digest(stage / relative) != generated_sha256[relative]:
            raise ValueError('Draft table changed during staging copy: ' + relative)
    assert_snapshot(paper, config['paper_snapshot'])
    return stage


def validate_build(config, stage):
    log = (stage / 'main.log').read_text(errors='replace')
    patterns = (r'^!', r'LaTeX Error', r'Package .* Error', r'Undefined control sequence', r'Emergency stop',
                r'Fatal error', r'(?:Reference|Citation).*undefined', r'There were undefined (?:references|citations)',
                r'Rerun to get cross-references right', r'Label\(s\) may have changed')
    failures = [pattern for pattern in patterns if re.search(pattern, log, re.MULTILINE | re.IGNORECASE)]
    if failures:
        raise ValueError('Staged LaTeX log failed checks: ' + ', '.join(failures))
    pdf = stage / 'main.pdf'
    pdf_hash = digest(pdf)
    if not pdf.read_bytes().startswith(b'%PDF-'):
        raise ValueError('Staged main.pdf is not a PDF')
    text = subprocess.check_output([config['tools']['pdftotext']['path'], str(pdf), '-'], text=True, timeout=60)
    normalized = ' '.join(text.lower().split())
    if HEADING.lower() not in normalized:
        raise ValueError('Staged PDF does not contain the robustness section')
    page = ai_statement_page(text)
    if page != config['ai_statement_page'] or page - 1 != 9:
        raise ValueError('The nine-page main-paper boundary changed')
    for line in (stage / 'main.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            path = Path(line[6:])
            path = (stage / path).resolve() if not path.is_absolute() else path.resolve()
            if path.is_relative_to(Path(config['paper_dir'])) or (path.is_relative_to(REPO) and not path.is_relative_to(stage)):
                raise ValueError('Isolated build read an unstaged repository input: ' + str(path))
    if digest(pdf) != pdf_hash:
        raise ValueError('Staged PDF changed during validation')
    return {'pdf_sha256': pdf_hash, 'log_sha256': digest(stage / 'main.log'), 'robustness_heading_present': True,
            'ai_statement_page': page, 'main_paper_pages': page - 1, 'undefined_references': False}


def publish(config, stage, backup_dir, progress, publication_sha256):
    """Preserve displaced inodes; install and restore only into absent paths.

    This is a journaled multi-file update, not a transaction against uncooperative
    writers. Every displaced version survives, and conflicts stop publication.
    """
    paper = Path(config['paper_dir'])
    expected = json.loads(json.dumps(config['paper_snapshot']))
    assert_snapshot(paper, expected)
    backup_dir.mkdir(parents=True, exist_ok=False)
    if backup_dir.stat().st_dev != paper.stat().st_dev:
        raise ValueError('Persistent publication backups must share the canonical filesystem')
    if set(publication_sha256) != set(TARGETS):
        raise ValueError('Validated publication hash roster differs from targets')
    records = {}
    journal = {'created_at': now(), 'targets': records, 'state': 'preparing'}
    journal_path = backup_dir / 'transaction.json'
    def checkpoint(state):
        journal.update(state=state, updated_at=now())
        save(journal_path, journal)
    for relative in TARGETS:
        prepared = backup_dir / 'prepared' / relative
        prepared.parent.mkdir(parents=True, exist_ok=True)
        after = publication_sha256[relative]
        if digest(stage / relative) != after:
            raise ValueError('Validated staged file changed before publication: ' + relative)
        shutil.copy2(stage / relative, prepared)
        if digest(prepared) != after:
            raise ValueError('Staged publication file changed during preparation: ' + relative)
        before = expected['main_pdf_sha256'] if relative == 'main.pdf' else expected['sources_sha256'].get(relative)
        records[relative] = {'before': before, 'after': after, 'prepared': str(prepared),
                             'displaced': str(backup_dir / 'displaced' / relative), 'installed': False}
    checkpoint('prepared')
    # The potentially long copying is over; guard the complete paper again.
    assert_snapshot(paper, expected)
    progress('publishing', backups=str(backup_dir), publication_hashes=records, publication_started=True,
             canonical_paper_modified=True, canonical_changes_possible=True)
    try:
        for relative, record in records.items():
            verify_code(config)
            assert_snapshot(paper, expected)
            target, displaced = paper / relative, Path(record['displaced'])
            displaced.parent.mkdir(parents=True, exist_ok=True)
            record['step'] = 'before_displacement'
            checkpoint('publishing')
            if record['before'] is not None:
                # Atomic move captures the actual current inode, including a racing edit.
                os.rename(target, displaced)
                record['step'] = 'displaced'
                checkpoint('publishing')
                if displaced.is_symlink() or digest(displaced) != record['before']:
                    raise ValueError('Canonical target changed during displacement: ' + relative)
            record['step'] = 'before_exclusive_install'
            checkpoint('publishing')
            if digest(record['prepared']) != record['after']:
                raise ValueError('Prepared publication file changed: ' + relative)
            # EEXIST preserves any concurrent creation of the canonical path.
            os.link(record['prepared'], target)
            record.update(installed=True, step='installed')
            checkpoint('publishing')
            if relative == 'main.pdf':
                expected['main_pdf_sha256'] = record['after']
            else:
                expected['sources_sha256'][relative] = record['after']
            assert_snapshot(paper, expected)
        checkpoint('published')
        progress('published', published_at=now(), publication_hashes=records, published_snapshot=expected,
                 canonical_changes_possible=False)
    except BaseException:
        conflicts = []
        for relative, record in reversed(list(records.items())):
            target, displaced = paper / relative, Path(record['displaced'])
            restore = displaced if displaced.exists() or displaced.is_symlink() else None
            # A signal/error can arrive after link succeeds but before the next
            # journal update. Reconcile the inode before treating it as uninstalled.
            if not record['installed'] and record.get('step') == 'before_exclusive_install':
                try:
                    record['installed'] = os.path.samefile(target, record['prepared'])
                except FileNotFoundError:
                    pass
            if record['installed']:
                captured = backup_dir / 'rollback_capture' / relative
                captured.parent.mkdir(parents=True, exist_ok=True)
                record['rollback_capture'] = str(captured)
                record['step'] = 'before_rollback_capture'
                checkpoint('rolling_back')
                try:
                    os.rename(target, captured)
                except FileNotFoundError:
                    conflicts.append(relative + ': installed path disappeared')
                else:
                    if captured.is_symlink() or digest(captured) != record['after']:
                        conflicts.append(relative + ': concurrent version retained')
                        restore = captured
            if restore is not None:
                # A modified original backup is still preserved/restored as the actual
                # displaced version; never substitute a stale copied baseline.
                restore_hash = None if restore.is_symlink() else digest(restore)
                if restore == displaced and restore_hash != record['before']:
                    conflicts.append(relative + ': displaced version differs from snapshot')
                record['restoring_sha256'] = restore_hash
                record['step'] = 'before_exclusive_restore'
                checkpoint('rolling_back')
                try:
                    os.link(restore, target, follow_symlinks=False)
                except FileExistsError:
                    conflicts.append(relative + ': canonical path occupied; preserved backup')
                else:
                    if restore_hash is not None and digest(target) != restore_hash:
                        conflicts.append(relative + ': restored inode changed concurrently')
            record['step'] = 'rollback_processed'
            checkpoint('rolling_back')
        try:
            assert_snapshot(paper, config['paper_snapshot'])
        except (OSError, ValueError) as exc:
            conflicts.append('Original snapshot not restored: ' + str(exc))
        journal['external_change_conflicts'] = conflicts
        checkpoint('rollback_conflict' if conflicts else 'rolled_back')
        progress('publication_rollback_conflict' if conflicts else 'publication_rolled_back',
                 rollback_conflicts=conflicts, backups=str(backup_dir),
                 canonical_paper_modified=bool(conflicts), canonical_changes_possible=bool(conflicts))
        raise


def execute(config, config_path, state_dir, progress):
    verify_code(config)
    assert_snapshot(config['paper_dir'], config['paper_snapshot'])
    check = readiness(config)
    progress('preflight', readiness=check)
    if not check['ready']:
        raise ValueError('All prescribed models and the finalizer must be complete before execution')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    logs = state_dir / 'attempts' / attempt
    logs.mkdir(parents=True)
    def command(name, arguments, cwd=REPO, timeout=600):
        verify_code(config)
        progress(name)
        environment = os.environ.copy()
        environment.update(TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run(arguments, cwd=cwd, env=environment, text=True, capture_output=True, timeout=timeout, check=False)
        atomic(logs / (name + '.stdout'), result.stdout)
        atomic(logs / (name + '.stderr'), result.stderr)
        progress(name, last_command={'arguments': arguments, 'returncode': result.returncode, 'logs': str(logs)})
        return result.returncode
    failures = []
    for model in MODELS:
        rc = command('audit_' + model, [config['python_executable'], '-m', 'exp1_prospective.context_reversal.audit_robustness_results', '--run-dir', config['run_dir'], '--model-key', model])
        if rc:
            failures.append(model)
    if failures:
        raise ValueError('Independent audits failed: ' + ', '.join(failures))
    draft = Path(config['run_dir']) / 'paper_draft' / ('pipeline_' + attempt)
    if command('write_draft', [config['python_executable'], '-m', 'exp1_prospective.context_reversal.write_robustness_paper', '--submission', config['submission_path'], '--output', str(draft)]):
        raise ValueError('Audited draft writer failed')
    writer_receipt = json.loads((logs / 'write_draft.stdout').read_text())
    manifest_hash = writer_receipt['draft_manifest_sha256']
    manifest = verify_draft(draft, manifest_hash)
    if writer_receipt['generated_sha256'] != manifest['generated_sha256']:
        raise ValueError('Writer receipt differs from its output manifest')
    draft_files = {relative: digest(draft / relative) for relative in ('exp1_context_robustness_v2.tex', *TABLES, 'draft_manifest.json')}
    stage = stage_paper(config, draft, manifest['generated_sha256'])
    verify_draft(draft, manifest_hash)
    stage_sources = paper_snapshot(stage)['sources_sha256']
    progress('staged', stage_dir=str(stage), draft_dir=str(draft), draft_files_sha256=draft_files,
             staged_source_sha256=stage_sources)
    latex_command = [config['tools']['latexmk']['path'], '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', 'main.tex']
    if command('latex_build', latex_command, cwd=stage, timeout=1200):
        raise ValueError('Isolated LaTeX build failed; canonical paper unchanged')
    build = validate_build(config, stage)
    progress('build_verified', build=build)
    verify_code(config)
    verify_draft(draft, manifest_hash)
    if any(digest(draft / relative) != expected for relative, expected in draft_files.items()):
        raise ValueError('Generated draft files changed after writer execution')
    if paper_snapshot(stage)['sources_sha256'] != stage_sources:
        raise ValueError('Staged paper sources changed during the build')
    assert_snapshot(config['paper_dir'], config['paper_snapshot'])
    if digest(config_path) != load(state_dir / 'status.json')['config_sha256']:
        raise ValueError('Pipeline configuration changed during execution')
    publication_sha256 = {relative: stage_sources[relative] for relative in (APPENDIX, *TABLES)}
    publication_sha256['main.pdf'] = build['pdf_sha256']
    publish(config, stage, state_dir / 'backups' / attempt, progress, publication_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=RUN / 'finish_pipeline/config.json')
    parser.add_argument('--expected-config-sha256', help='Required for execution; bind this digest in the scheduled command')
    parser.add_argument('--run-dir', type=Path, default=RUN, help='Used only while configuring or for an unconfigured dry run')
    parser.add_argument('--paper-dir', type=Path, default=REPO / 'paper')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--configure', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    state_dir = config_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = state_dir / 'status.json'
        previous = load(status_path) if status_path.exists() else {}
        status = {'schema_version': 'robustness_finish_pipeline_status_v1', 'started_at': now(), 'config_path': str(config_path), 'events': [], 'frontier_api_calls': 0,
                  'canonical_paper_modified': previous.get('canonical_paper_modified', False),
                  'canonical_changes_possible': previous.get('canonical_changes_possible', False)}
        def progress(state, **fields):
            status.update(fields, state=state, updated_at=now())
            status['events'].append({'at': now(), 'state': state})
            if state == 'published':
                status['canonical_paper_modified'] = True
            save(status_path, status)
        try:
            if args.configure:
                if config_path.exists():
                    raise ValueError('Configuration already exists; use a new config path for an explicit new snapshot')
                config = make_config(args.run_dir, args.paper_dir)
                save(config_path, config)
                status['config_sha256'] = digest(config_path)
                save(state_dir / 'config_binding.json', {'config_path': str(config_path), 'sha256': status['config_sha256'], 'configured_at': now()})
                progress('configured', readiness=readiness(config))
            else:
                if args.execute and not config_path.exists():
                    raise ValueError('Execute requires an explicit configured snapshot')
                if args.execute and not args.expected_config_sha256:
                    raise ValueError('Execute requires --expected-config-sha256 from the configured snapshot')
                binding_path = state_dir / 'config_binding.json'
                binding = load(binding_path) if binding_path.exists() else None
                expected_hash = args.expected_config_sha256 or (binding or {}).get('sha256')
                status['config_sha256'] = expected_hash
                if config_path.exists():
                    if expected_hash is None or digest(config_path) != expected_hash:
                        raise ValueError('Configuration differs from its immutable configured digest')
                    if binding is None or binding.get('sha256') != expected_hash or binding.get('config_path') != str(config_path):
                        raise ValueError('Configuration binding is absent or inconsistent')
                config = load(config_path) if config_path.exists() else make_config(args.run_dir, args.paper_dir)
                verify_code(config)
                interrupted = []
                for journal_path in (state_dir / 'backups').glob('*/transaction.json'):
                    journal = load(journal_path)
                    if journal.get('state') not in ('published', 'rolled_back'):
                        interrupted.append(str(journal_path))
                if interrupted:
                    progress('publication_interrupted', canonical_paper_modified=True, canonical_changes_possible=True,
                             interrupted_publication_journals=interrupted)
                    raise ValueError('Interrupted or conflicted publication requires inspection of preserved versions: ' + ', '.join(interrupted))
                if previous.get('state') == 'published':
                    if previous.get('config_sha256') != status['config_sha256']:
                        raise ValueError('Previously published state belongs to a different configuration')
                    assert_snapshot(config['paper_dir'], previous['published_snapshot'])
                    print(json.dumps({'state': 'already_published', 'status': str(status_path)}))
                    return 0
                if args.execute:
                    execute(config, config_path, state_dir, progress)
                else:
                    assert_snapshot(config['paper_dir'], config['paper_snapshot'])
                    progress('dry_run', readiness=readiness(config), canonical_source_count=len(config['paper_snapshot']['sources_sha256']), configured=config_path.exists())
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            progress('failed', error=str(exc))
            print(json.dumps({'state': 'failed', 'error': str(exc), 'status': str(status_path)}, indent=2))
            return 1
        print(json.dumps({'state': status['state'], 'config_sha256': status.get('config_sha256'), 'readiness': status.get('readiness'), 'status': str(status_path)}, indent=2))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
