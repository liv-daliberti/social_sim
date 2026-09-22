"""Frozen scoring for the fresh reasoning x wording study (no inference)."""
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import argparse,json
import numpy as np
from .. import run_local as c
from ..analyze import FamilyBootstrap,Sample,stability_diagnostic


def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def index(units,records,model_key):
    lookup={u['trial_id']:u for u in units};by={}
    for r in records:
        if r['model_key']!=model_key:raise ValueError('unexpected model in response file')
        key=c.record_key(r)
        if key in by or key[0] not in lookup:raise ValueError('duplicate or unknown response key')
        if key[1:] not in [('baseline','baseline'),*[('update',x) for x in c.CONDITIONS]]:raise ValueError('unexpected condition')
        u=lookup[key[0]]
        for f in ('family_id','context_id','repeat','domain','material_status'):
            if r[f]!=u[f]:raise ValueError('response metadata differs from frozen design')
        if r['status']=='ok' and (type(r['probability']) not in (float,int) or not 0<=r['probability']<=1):raise ValueError('invalid probability')
        by[key]=r
    for key,r in by.items():
        u=lookup[key[0]]
        if key[1]=='baseline':expected=u['baseline_prompt']
        else:
            baseline=by.get((key[0],'baseline','baseline'))
            if baseline is None:raise ValueError('update without own baseline')
            prior=baseline.get('probability') if baseline['status']=='ok' else None
            if r.get('prior_probability')!=prior:raise ValueError('update uses another prior')
            expected=c.render_update(u,key[2],prior) if prior is not None else None
            if prior is None and r['status']!='blocked_baseline':raise ValueError('update not blocked by failed baseline')
        if r.get('prompt')!=expected or r.get('prompt_sha256')!=(c.text_sha256(expected) if expected is not None else None):raise ValueError('response prompt differs')
    return by


def analyze(units,records,model_key,draws=2000):
    by=index(units,records,model_key)
    parents=sorted({u['parent_family_id'] for u in units});variants=sorted({u['variant'] for u in units});boot=FamilyBootstrap(parents,draws,20260922)
    metrics={};trial={};coverage=Counter();deltas={};signed_ok={}
    def metric(items,binary=False):return boot.summarize([Sample(f,v) for f,v in items],binary=binary)
    for u in units:
        key=(u['parent_family_id'],u['variant'],u['context_id']);t=u['trial_id'];b=by.get((t,'baseline','baseline'));prior=b['probability'] if b and b['status']=='ok' else None
        for stage,cond in [('baseline','baseline'),*[('update',x) for x in c.CONDITIONS]]:
            r=by.get((t,stage,cond));coverage['planned']+=1;coverage['missing' if r is None else r['status']]+=1
        ds={};posts={}
        for cond in c.CONDITIONS:
            r=by.get((t,'update',cond));post=r['probability'] if r and r['status']=='ok' else None
            posts[cond]=post;ds[cond]=None if prior is None or post is None else 100*(post-prior)
        deltas[key]=ds
        correct=None if ds['new_news'] is None else float(ds['new_news']*u['expected_sign']>0)
        if u['expected_sign']:signed_ok[key]=correct
        trial[key]={'parent':key[0],'variant':key[1],'context':key[2],'stratum':u['domain'],'prior':prior,'post':posts,'delta_pp':ds,'strict_sign_correct':correct,'baseline_oracle':float(Fraction(u['oracle_baseline'])),'new_oracle':float(Fraction(u['oracle_update'])),'endpoint_blocked':None if prior is None else bool((u['expected_sign']==1 and prior==1) or (u['expected_sign']==-1 and prior==0))}
    pair_values={}
    for v in variants:
        ps=sorted({u['parent_family_id'] for u in units if u['variant']==v});pairs=[];signs=[];broken=[];control={a:[] for a in ('no_news','repeated_news')};endpoint=[];baseline_error=[];post_error=[];update_error=[];adjusted=[]
        for f in ps:
            both=[signed_ok.get((f,v,ctx)) for ctx in ('positive','negative')]
            pair=None if any(x is None for x in both) else float(all(both));pairs.append((f,pair));pair_values[f,v]=0. if pair is None else pair
            for ctx in ('positive','negative','broken'):
                row=trial[f,v,ctx];ds=row['delta_pp'];prior=row['prior'];post=row['post']['new_news']
                if ctx!='broken':
                    signs.append((f,row['strict_sign_correct']));endpoint.append((f,None if prior is None else float(row['endpoint_blocked'])))
                    adjusted.append((f,None if ds['new_news'] is None or ds['no_news'] is None else float((ds['new_news']-ds['no_news'])*(1 if ctx=='positive' else -1)>0)))
                else:broken.append((f,ds['new_news']))
                for a in control:control[a].append((f,None if ds[a] is None else abs(ds[a])))
                baseline_error.append((f,None if prior is None else 100*abs(prior-row['baseline_oracle'])))
                post_error.append((f,None if post is None else 100*abs(post-row['new_oracle'])))
                update_error.append((f,None if ds['new_news'] is None else abs(ds['new_news']-100*(row['new_oracle']-row['baseline_oracle']))))
        bs=metric(broken);ba=metric([(f,None if x is None else abs(x)) for f,x in broken]);stability=stability_diagnostic(bs,ba);stability['interpretation']='Prespecified finite-corpus diagnostic; model screening is not human validation.'
        byclass={}
        for cl in sorted({u['domain'] for u in units}):
            subset={u['parent_family_id'] for u in units if u['domain']==cl}
            observed=[x for f,x in pairs if f in subset];byclass[cl]={'planned':len(observed),'success':sum(x or 0 for x in observed),'rate':sum(x or 0 for x in observed)/len(observed)}
        effect_bins={}
        for label,low,high in [('below_1pp',0,1),('1_to_2pp',1,2),('at_least_2pp',2,float('inf'))]:
            items=[]
            for f in ps:
                for ctx in ('positive','negative'):
                    row=trial[f,v,ctx];effect=100*abs(row['new_oracle']-row['baseline_oracle'])
                    if low<=effect<high:items.append((f,row['strict_sign_correct']))
            effect_bins[label]=metric(items,True)
        metrics[v]={'normative_effect_size_bins':effect_bins,'paired_reversal':metric(pairs,True),'sign_accuracy':metric(signs,True),'no_news_adjusted_sign_accuracy':metric(adjusted,True),'broken_signed_pp':bs,'broken_absolute_pp':ba,'broken_stability':stability,'controls_absolute_pp':{a:metric(x) for a,x in control.items()},'endpoint_blocked_fraction_observed':metric(endpoint),'baseline_oracle_error_pp':metric(baseline_error),'new_oracle_error_pp':metric(post_error),'update_oracle_error_pp':metric(update_error),'by_mechanism':byclass,'equal_mechanism_macro_pair_rate':float(np.mean([x['rate'] for x in byclass.values()]))}
    edits={}
    for v in ('names','paraphrase','resample'):
        if v not in variants:continue
        discrepancy=[];excess=[];joint=[]
        for f in parents:
            for ctx in ('positive','negative','broken'):
                if (f,v,ctx) not in deltas or (f,'base',ctx) not in deltas:continue
                a=deltas[f,'base',ctx]['new_news'];b=deltas[f,v,ctx]['new_news'];res=deltas.get((f,'resample',ctx),{}).get('new_news')
                diff=None if a is None or b is None else abs(a-b);discrepancy.append((f,diff));excess.append((f,None if diff is None or res is None else diff-abs(a-res)))
            if (f,v) in pair_values and (f,'base') in pair_values:joint.append((f,pair_values[f,v]*pair_values[f,'base']))
        edits[v]={'absolute_update_discrepancy_pp':metric(discrepancy),'excess_over_resampling_pp':metric(excess) if 'resample' in variants else None,'joint_pair_correct':metric(joint,True)}
    return {'schema_version':'fresh_factorial_v1','model_key':model_key,'parent_instances':len(parents),'mechanism_classes':len({u['domain'] for u in units}),'coverage':dict(coverage),'all_planned_received':coverage['missing']==0,'by_variant':metrics,'edits':edits,'trial_rows':list(trial.values()),'pair_values':[{'parent':f,'variant':v,'value':x} for (f,v),x in pair_values.items()],'bootstrap':{'draws':draws,'seed':20260922,'unit':'parent instance; contexts, variants and modes remain paired','interpretation':'descriptive only; shared template/mechanism dependence is not removed'}}


def compare(off,on,draws=2000):
    a={(r['parent'],r['variant']):r['value'] for r in off['pair_values']};b={(r['parent'],r['variant']):r['value'] for r in on['pair_values']}
    if a.keys()!=b.keys():raise ValueError('reasoning cohorts differ')
    parents=sorted({f for f,v in a});boot=FamilyBootstrap(parents,draws,20260922)
    metric=lambda vals:boot.summarize([Sample(f,x) for f,x in vals])
    variants=sorted({v for f,v in a});effects={v:metric([(f,100*(b[f,v]-a[f,v])) for f in parents]) for v in variants}
    interactions={v:metric([(f,100*((b[f,v]-a[f,v])-(b[f,'base']-a[f,'base']))) for f in parents]) for v in variants if v!='base'}
    strata={r['parent']:r['stratum'] for r in off['trial_rows']};leave={s:float(np.mean([100*(b[f,'base']-a[f,'base']) for f in parents if strata[f]!=s])) for s in sorted(set(strata.values()))}
    return {'paired_reversal_on_minus_off_pp':effects,'interaction_relative_to_base_pp':interactions,'leave_one_mechanism_out_base_difference_pp':leave,'missing_binary_outcomes_count_as_failure':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--responses',type=Path,nargs='+',required=True);p.add_argument('--model-key',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(c.read_units(a.plan),[r for path in a.responses for r in read(path)],a.model_key)
    result['provenance']={'plan_sha256':c.file_sha256(a.plan),'responses':{str(path):c.file_sha256(path) for path in a.responses},'scorer_sha256':c.file_sha256(Path(__file__))}
    c.atomic_write(a.output,json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
