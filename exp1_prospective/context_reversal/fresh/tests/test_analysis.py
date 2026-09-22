import copy
from pathlib import Path
import pytest
from exp1_prospective.context_reversal.fresh import materials as m,analyze as a
from exp1_prospective.context_reversal import run_local as c


def fixture():
    # Two instances in each class; all variants and all contexts.
    units=[m.unit(m.make_family(k,j),v,ctx) for k in range(8) for j in range(2) for v in m.VARIANTS for ctx in m.CONTEXTS]
    rows=[]
    for u in units:
        prior=float(a.Fraction(u['oracle_baseline']))
        for stage,condition in [('baseline','baseline'),*[('update',x) for x in c.CONDITIONS]]:
            p=prior if condition!='new_news' else float(a.Fraction(u['oracle_update']))
            prompt=u['baseline_prompt'] if stage=='baseline' else c.render_update(u,condition,prior)
            rows.append({**{k:u[k] for k in ('trial_id','family_id','domain','context_id','repeat','material_status')},'stage':stage,'condition':condition,'probability':p,'status':'ok','prior_probability':None if stage=='baseline' else prior,'prompt':prompt,'prompt_sha256':c.text_sha256(prompt),'model_key':'test'})
    return units,rows


def test_oracle_scores_perfect_without_pooling_variants():
    u,r=fixture();s=a.analyze(u,r,'test',100)
    assert s['parent_instances']==16 and s['coverage']['planned']==768
    for v,metrics in s['by_variant'].items():
        assert metrics['paired_reversal']['estimate']==1
        assert metrics['paired_reversal']['n_planned']==16
        assert metrics['sign_accuracy']['n_planned']==32
        assert metrics['broken_absolute_pp']['estimate']==0
        assert metrics['broken_stability']['status']=='criterion_met'
        assert metrics['update_oracle_error_pp']['estimate']==0
    assert a.compare(s,s,100)['paired_reversal_on_minus_off_pp']['base']['estimate']==0


def test_missing_broken_measurement_is_not_equivalence():
    u,r=fixture();r=[x for x in r if not (x['trial_id']=='fresh001_base_broken' and x['condition']=='new_news')]
    s=a.analyze(u,r,'test',100);assert s['coverage']['missing']==1
    assert s['by_variant']['base']['broken_stability']['status']=='indeterminate'
    assert s['by_variant']['base']['broken_absolute_pp']['n_missing']==1


def test_zero_and_invalid_signed_updates_count_as_failures():
    u,r=fixture()
    for x in r:
        if x['trial_id']=='fresh001_base_positive' and x['condition']=='new_news':x['probability']=x['prior_probability']
        if x['trial_id']=='fresh002_base_negative' and x['condition']=='new_news':x.update(status='parse_error',probability=None)
    s=a.analyze(u,r,'test',100)
    assert s['by_variant']['base']['paired_reversal']['n_success']==14
    assert s['by_variant']['base']['paired_reversal']['denominator']==16
    assert s['by_variant']['base']['sign_accuracy']['n_success']==30


def test_wrong_prior_and_duplicate_are_rejected():
    u,r=fixture();bad=copy.deepcopy(r);bad[1]['prior_probability']+=.01
    with pytest.raises(ValueError,match='another prior'):a.analyze(u,bad,'test',100)
    with pytest.raises(ValueError,match='duplicate'):a.analyze(u,r+[r[0]],'test',100)


def test_reasoning_contrast_and_interaction_are_paired():
    u,r=fixture();on=a.analyze(u,r,'test',100)
    for row in r:
        if row['trial_id'].startswith('fresh001_base') and row['condition']=='new_news':row['probability']=row['prior_probability']
    off=a.analyze(u,r,'test',100);s=a.compare(off,on,100)
    assert s['paired_reversal_on_minus_off_pp']['base']['estimate']==6.25
    assert s['interaction_relative_to_base_pp']['paraphrase']['estimate']==-6.25
