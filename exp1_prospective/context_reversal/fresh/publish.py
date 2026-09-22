"""Isolated ICLR build and guarded publication of audited fresh results.

Publication transaction reused from finish_robustness_pipeline.py; no inference.
"""
import argparse,fcntl,json,os,re,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from ..finish_robustness_pipeline import (now,digest,atomic,save,load,paper_snapshot,assert_snapshot,ai_statement_page)
from . import finish,paper as writer
from .freeze import check_artifact,artifact

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
STATE=ROOT/'runs/fresh_evaluation_v1/publication'
APPENDIX='experiment1_fresh_context_appendix.tex'
TABLES=tuple('tables/exp1_fresh_'+x+'.tex' for x in ('performance','controls','reasoning','edits','frontier','mechanisms'))
TARGETS=('main.tex','evidence_ladder.tex','appendix_guide.tex',APPENDIX,*TABLES,'main.pdf')


def verify_code(config):
    if config['schema_version']!='fresh_publication_v1' or config['publication_targets']!=list(TARGETS):raise ValueError('Unexpected publication config')
    for spec in config['code'].values():check_artifact(spec)
    check_artifact(config['local_freeze']);check_artifact(config['frontier_freeze'])


def configure():
    STATE.mkdir(parents=True,exist_ok=True)
    path=STATE/'config.json'
    if path.exists():raise ValueError('Do not overwrite a configured publication')
    paper=(REPO/'paper/ICLR').resolve()
    tools={name:shutil.which(name) for name in ('latexmk','pdftotext')}
    if not all(tools.values()):raise ValueError('Missing LaTeX tools')
    names=['publish.py','finish.py','paper.py','publish.sbatch']
    config={'schema_version':'fresh_publication_v1','created_at':now(),'paper_dir':str(paper),'paper_snapshot':paper_snapshot(paper),'publication_targets':list(TARGETS),'code':{name:artifact(Path(__file__).parent/name) for name in names},'helper':artifact(ROOT/'finish_robustness_pipeline.py'),'local_freeze':artifact(finish.FROZEN/'local_freeze.json'),'frontier_freeze':artifact(finish.FROZEN/'frontier_generation_freeze.json'),'tools':tools}
    config['code']['publication_helper']=config['helper']
    save(path,config);return {'config':str(path),'sha256':digest(path)}


def takeaway(data):
    off,on,gpt=[data[x]['by_variant']['base']['paired_reversal'] for x in ('disabled.json','enabled.json','frontier.json')]
    return (r'\noindent\textbf{Takeaway.}'+'\n'+
        r'Lexical cues identify original-packet relevance ($.989$ accuracy). A frozen'+'\n'+
        r'follow-up uses identical news across relational contexts in 80 finite-rule'+'\n'+
        r'instances spanning eight mechanism classes. Qwen3-32B reverses both directions'+'\n'+
        f"in {off['n_success']}/80 pairs without thinking and {on['n_success']}/80 with thinking; GPT-5.6 reaches "+
        f"{gpt['n_success']}/24 on a fixed subset, with "+
        f"{data['frontier.json']['by_variant']['base']['broken_absolute_pp']['estimate']:.1f}-point broken-link movement "+
        "(base wording; all planned failures count). "+'\n'+
        r'Wording, control drift, and failure analyses appear in App.~\ref{app:exp1-fresh-factorial}.'+'\n'+
        r'This formally checked diagnostic has no new human validation and does not'+'\n'+
        r'establish natural-news coverage. Experiment~2 tests contextual relationship'+'\n'+
        r'selection under a controlled generative process.'+'\n\n')


def stage_paper(config,draft,directory=None):
    verify_code(config);assert_snapshot(Path(config['paper_dir']),config['paper_snapshot'])
    manifest=load(draft/'draft_manifest.json');check_artifact(manifest['technical_audit']);check_artifact(manifest['writer'])
    for spec in manifest['inputs'].values():check_artifact(spec)
    for name,sha in manifest['files'].items():
        if digest(draft/name)!=sha:raise ValueError('Draft changed: '+name)
    stage=Path(tempfile.mkdtemp(prefix='fresh_iclr_')) if directory is None else Path(directory)
    stage.mkdir(parents=True,exist_ok=True)
    for name,sha in config['paper_snapshot']['sources_sha256'].items():
        destination=stage/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(config['paper_dir'])/name,destination)
        if digest(destination)!=sha:raise ValueError('Source changed while staging: '+name)
    for relative in (APPENDIX,*TABLES):shutil.copy2(draft/Path(relative).name,stage/relative)
    main=stage/'main.tex';text=main.read_text();anchor=r'\input{experiment1_context_reversal_appendix}'
    if text.count(anchor)!=1 or r'\input{experiment1_fresh_context_appendix}' in text:raise ValueError('Unexpected appendix inclusion state')
    main.write_text(text.replace(anchor,anchor+'\n'+r'\input{experiment1_fresh_context_appendix}'))
    guide=stage/'appendix_guide.tex';text=guide.read_text();anchor=r'\appendixguidesection{app:exp1-context-reversal}'
    if text.count(anchor)!=1 or 'app:exp1-fresh-factorial' in text:raise ValueError('Unexpected appendix guide state')
    guide.write_text(text.replace(anchor,anchor+'\n'+r'\appendixguidesection{app:exp1-fresh-factorial}'))
    ladder=stage/'evidence_ladder.tex';text=ladder.read_text();start=text.index(r'\noindent\textbf{Takeaway.}');end=text.index(r'\subsection{Exp.~2:',start)
    data={name:load(check_artifact(spec)) for name,spec in manifest['inputs'].items()};ladder.write_text(text[:start]+takeaway(data)+text[end:])
    assert_snapshot(Path(config['paper_dir']),config['paper_snapshot']);return stage


def validate_build(config,stage):
    log=(stage/'main.log').read_text(errors='replace')
    patterns=(r'^!',r'LaTeX Error',r'Package .* Error',r'Undefined control sequence',r'Emergency stop',r'Fatal error',r'(?:Reference|Citation).*undefined',r'There were undefined (?:references|citations)',r'Rerun to get cross-references right',r'Label\(s\) may have changed',r'Overfull \\[hv]box')
    failures=[p for p in patterns if re.search(p,log,re.MULTILINE|re.IGNORECASE)]
    if failures:raise ValueError('Staged LaTeX validation failed: '+', '.join(failures))
    pdf=stage/'main.pdf';sha=digest(pdf)
    if not pdf.read_bytes().startswith(b'%PDF-'):raise ValueError('Not a PDF')
    text=subprocess.check_output([config['tools']['pdftotext'],str(pdf),'-'],text=True,timeout=60)
    if 'freshfinite-rulecontextreversal' not in re.sub(r'\s+', '', text.lower()):raise ValueError('Missing fresh appendix')
    if ai_statement_page(text)!=10:raise ValueError('Nine-page main-paper boundary changed')
    for line in (stage/'main.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            p=Path(line[6:]);p=(stage/p).resolve() if not p.is_absolute() else p.resolve()
            if p.is_relative_to(REPO) and not p.is_relative_to(stage):raise ValueError('Unstaged repository input: '+str(p))
    if digest(pdf)!=sha:raise ValueError('PDF changed during validation')
    return {'pdf_sha256':sha,'log_sha256':digest(stage/'main.log'),'main_paper_pages':9,'ai_statement_page':10,'undefined_references':False,'overfull_boxes':0,'fresh_heading_present':True,'total_pages':len(text.rstrip('\f').split('\f'))}


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
    if set(publication_sha256) != set(config['publication_targets']):
        raise ValueError('Validated publication hash roster differs from targets')
    records = {}
    journal = {'created_at': now(), 'targets': records, 'state': 'preparing'}
    journal_path = backup_dir / 'transaction.json'
    def checkpoint(state):
        journal.update(state=state, updated_at=now())
        save(journal_path, journal)
    for relative in config['publication_targets']:
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

def execute(config,progress):
    verify_code(config)
    result=finish.finish();progress('auditing',completion=result)
    if result['status']!='complete':return False
    draft_manifest=writer.write();progress('draft_ready',draft_manifest=artifact(writer.DRAFT/'draft_manifest.json'))
    stage=stage_paper(config,writer.DRAFT);progress('building',stage_dir=str(stage))
    with (STATE/'build.stdout.log').open('w') as stream:
        subprocess.run([config['tools']['latexmk'],'-pdf','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],cwd=stage,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1200)
    checks=validate_build(config,stage);save(STATE/'build_validation.json',checks)
    verify_code(config);publication={name:digest(stage/name) for name in TARGETS}
    save(STATE/'validated_files.json',publication)
    publish(config,stage,STATE/'backups',progress,publication)
    atomic(ROOT/'runs/fresh_evaluation_v1/STATUS.md','# Fresh evaluation complete\n\nLocal and frontier collection, technical audit, isolated ICLR build, and guarded publication are complete.\n\n- Results: `../../results/fresh_evaluation_v1/summary.md` and `technical_audit.json`.\n- Publication evidence: `publication/status.json`, `build_validation.json`, and `backups/transaction.json`.\n- Canonical paper: `paper/ICLR/main.tex` and `paper/ICLR/main.pdf`; the main paper remains nine pages.\n- Materials: 80 finite-rule instances of eight mechanisms; model screening and formal author adjudication, without new human validation.\n')
    return True


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--configure',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--expected-config-sha256');p.add_argument('--execute',action='store_true');p.add_argument('--reschedule',action='store_true');p.add_argument('--attempt',type=int,default=0);a=p.parse_args(argv)
    if a.configure:print(json.dumps(configure(),indent=2));return
    if not a.config or digest(a.config)!=a.expected_config_sha256:raise ValueError('Configuration hash required')
    config=load(a.config);verify_code(config);STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=STATE/'status.json';state=load(status_path) if status_path.exists() else {}
        if state.get('status')=='published':
            assert_snapshot(Path(config['paper_dir']),state['published_snapshot']);print('Already published and verified');return
        if state.get('publication_started'):raise ValueError('Prior publication attempt requires journal reconciliation')
        if state.get('status')=='submission_intent':raise ValueError('Prior scheduler submission requires reconciliation')
        if state.get('status')=='waiting' and state.get('continuation_job_id') and a.attempt<=state.get('attempt',-1):
            print('A later publication check is already scheduled');return
        def progress(status,**kwargs):
            state.update(status=status,updated_at=now(),attempt=a.attempt,**kwargs);save(status_path,state)
        try:
            if not a.execute:assert_snapshot(Path(config['paper_dir']),config['paper_snapshot']);progress('preflight_passed');return
            if execute(config,progress):return
            if not a.reschedule:progress('waiting');return
            if a.attempt>=95:raise ValueError('Automatic publication polling limit reached')
            cmd=['sbatch','--parsable','--begin=now+10minutes','--job-name=fresh-exp1-publish','--output='+str(STATE/'job_%j.out'),'--error='+str(STATE/'job_%j.err'),str(Path(__file__).with_name('publish.sbatch')),str(a.config.resolve()),a.expected_config_sha256,str(a.attempt+1)]
            progress('submission_intent',continuation_command=cmd)
            submitted=subprocess.run(cmd,capture_output=True,text=True,check=False)
            progress('waiting' if submitted.returncode==0 else 'submission_failed',continuation_job_id=submitted.stdout.strip(),submission_stderr=submitted.stderr)
            if submitted.returncode:raise RuntimeError('Publication continuation submission failed')
        except BaseException as exc:
            progress('failed',error=repr(exc));raise

if __name__=='__main__':main()
