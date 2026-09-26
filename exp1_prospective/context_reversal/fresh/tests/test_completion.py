"""Postcollection guards: synthetic fixtures, never real publication/inference."""
import copy,json
from pathlib import Path
from fractions import Fraction
import pytest
from exp1_prospective.context_reversal.fresh import analyze,finish,paper,publish,freeze
from exp1_prospective.context_reversal import run_local as c


def synthetic_rows(units):
    rows=[]
    for u in units:
        prior=float(Fraction(u['oracle_baseline']))
        for stage,condition in [('baseline','baseline'),*[('update',x) for x in c.CONDITIONS]]:
            p=prior if condition!='new_news' else float(Fraction(u['oracle_update']))
            prompt=u['baseline_prompt'] if stage=='baseline' else c.render_update(u,condition,prior)
            rows.append({**{k:u[k] for k in ('trial_id','family_id','domain','context_id','repeat','material_status')},'stage':stage,'condition':condition,'probability':p,'status':'ok','prior_probability':None if stage=='baseline' else prior,'prompt':prompt,'prompt_sha256':c.text_sha256(prompt),'model_key':'synthetic_test_only'})
    return rows


def fixture_results(directory):
    directory.mkdir(parents=True,exist_ok=True)
    units=c.read_units(finish.FROZEN/'probability_plan.jsonl');paid=c.read_units(finish.FROZEN/'frontier_plan.jsonl')
    full=analyze.analyze(units,synthetic_rows(units),'synthetic_test_only');subset=analyze.analyze(paid,synthetic_rows(paid),'synthetic_test_only')
    data={'disabled.json':full,'enabled.json':full,'reasoning_comparison.json':analyze.compare(full,full),'frontier.json':subset,'disabled_frontier_subset.json':subset,'enabled_frontier_subset.json':subset}
    for name,value in data.items():(directory/name).write_text(json.dumps(value))
    audit={'status':'passed','frontier_status':'complete','audit_code':freeze.artifact(Path(finish.__file__)),'summary_artifacts':{name:freeze.artifact(directory/name) for name in data}}
    (directory/'technical_audit.json').write_text(json.dumps(audit));(directory/'status.json').write_text(json.dumps({'status':'complete'}))
    (directory/'SYNTHETIC_TEST_ONLY.txt').write_text('Synthetic oracle fixtures for layout and guard tests. NOT experimental results. Never publish.')
    return data


def test_independent_audit_rejects_wrong_counts():
    units=c.read_units(finish.FROZEN/'frontier_plan.jsonl');rows=synthetic_rows(units);summary=analyze.analyze(units,rows,'synthetic_test_only')
    assert finish.independent_primary(units,rows,summary)[0]>0
    summary['by_variant']['base']['paired_reversal']['n_success']-=1
    with pytest.raises(AssertionError):finish.independent_primary(units,rows,summary)


def test_paper_rejects_incomplete_and_tampered_results(tmp_path,monkeypatch):
    results=tmp_path/'results';fixture_results(results);monkeypatch.setattr(paper,'RESULTS',results);monkeypatch.setattr(paper,'DRAFT',tmp_path/'draft')
    manifest=paper.write();assert len(manifest['files'])==7
    (results/'status.json').write_text('{"status":"waiting"}')
    with pytest.raises(ValueError,match='complete'):paper.write()
    (results/'status.json').write_text('{"status":"complete"}');(results/'frontier.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):paper.write()


def publication_fixture(tmp_path,monkeypatch):
    canonical=tmp_path/'canonical';stage=tmp_path/'stage';canonical.mkdir();stage.mkdir()
    for name in ('main.tex','main.pdf'):
        (canonical/name).write_text('old '+name);(stage/name).write_text('new '+name)
    monkeypatch.setattr(publish,'verify_code',lambda config:None)
    cfg={'paper_dir':str(canonical),'paper_snapshot':publish.paper_snapshot(canonical),'publication_targets':['main.tex','main.pdf']}
    hashes={name:publish.digest(stage/name) for name in cfg['publication_targets']}
    return canonical,stage,cfg,hashes


def test_publication_preserves_backup_and_rejects_changed_snapshot(tmp_path,monkeypatch):
    canonical,stage,cfg,hashes=publication_fixture(tmp_path,monkeypatch)
    (canonical/'main.tex').write_text('user edit')
    with pytest.raises(ValueError,match='changed'):publish.publish(cfg,stage,tmp_path/'backup',lambda *a,**k:None,hashes)
    assert (canonical/'main.tex').read_text()=='user edit'
    cfg['paper_snapshot']=publish.paper_snapshot(canonical)
    publish.publish(cfg,stage,tmp_path/'backup',lambda *a,**k:None,hashes)
    assert (canonical/'main.tex').read_text()=='new main.tex'
    assert (tmp_path/'backup/displaced/main.tex').read_text()=='user edit'


def test_publication_rolls_back_on_second_install_failure(tmp_path,monkeypatch):
    canonical,stage,cfg,hashes=publication_fixture(tmp_path,monkeypatch);real_link=publish.os.link
    def fail_second(source,target,**kwargs):
        if 'prepared' in str(source) and Path(target).name=='main.pdf':raise OSError('injected failure')
        return real_link(source,target,**kwargs)
    monkeypatch.setattr(publish.os,'link',fail_second)
    with pytest.raises(OSError,match='injected'):publish.publish(cfg,stage,tmp_path/'backup',lambda *a,**k:None,hashes)
    publish.assert_snapshot(canonical,cfg['paper_snapshot'])
    assert json.loads((tmp_path/'backup/transaction.json').read_text())['state']=='rolled_back'
