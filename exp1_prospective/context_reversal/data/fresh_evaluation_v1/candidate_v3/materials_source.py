"""Author finite mechanisms with exhaustive directional and uncertainty checks.

These are 80 new instances of eight mechanism classes, not 80 distinct mechanisms.
All uncertainty is specified; numerical forecasts are elicited, never supplied.
"""
from collections import Counter
from fractions import Fraction as Q
from itertools import permutations, product
from pathlib import Path
import json, math, re
from .. import run_local as common
from ..direction_design import BASELINE_SUFFIX, UPDATE_SUFFIX

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fresh_evaluation_v1/candidate_v3'
CLASSES=('calibration','time_windows','quorums','procedural_gates','broker_allocation','directed_routes','bounded_feedback','evidence_dependence')
SETTINGS=(
 ('exhibit sealing','panel inspection','watermark authentication','transit detection','survey positioning','fermentation assay','reservoir monitoring','cable commissioning','wildlife detection','cargo scanning'),
 ('article shipment','probe launch','adhesive certification','performance opening','sample measurement','rail inspection','simulator synchronization','restoration opening','software upload','prototype acceptance'),
 ('ceramic processing','beacon operation','replicated log','remote platform','pressure fixture','display installation','ferry navigation','museum access','payload recovery','bridge trial'),
 ('instrument admission','design showcase','workshop opening','protocol activation','radio clearance','material qualification','restoration authorization','program renewal','report certification','cart activation'),
 ('workshop allocation','survey dispatch','compute allocation','fabrication matching','field equipment','test admission','digitization allocation','sensor shipment','repair dispatch','survey permits'),
 ('station delivery','control connection','remote command','network transfer','depot access','gateway delivery','courier transfer','server receipt','shelter alert','vehicle return'),
 ('kiln regulation','stage positioning','cabinet cooling','winding control','chamber regulation','platform tracking','water regulation','conveyor placement','robot clearance','incubator regulation'),
 ('outpost sampling','component sampling','registration sampling','population sampling','fleet sampling','delivery sampling','restoration sampling','mapping sampling','building sampling','seed sampling'))
NAMES=(('Aster','Beryl','Cedar'),('Doran','Elwin','Faris'),('Galen','Hedra','Ivara'),('Jalen','Korin','Lumen'))
VARIANTS=('base','names','paraphrase','resample')
CONTEXTS=('positive','negative','broken')


def mechanism(k,j):
    """Return a visible, executable finite rule and its two exact firing chances."""
    # All U states are equiprobable; X is the only quantity whose distribution changes.
    if k==0:
        # Calibration changes a likelihood ratio for a stored positive reading.
        sensitivity=Q(3+j%3,5+j%3); fp0=Q(2+j//3,5+j//3); fp1=fp0/2
        text=f'Under either value of X, before observing reports, a hidden state H was equally likely to be 0 or 1. A detector has already reported 1. Its probability of reporting 1 when H=1 is {sensitivity}, regardless of X. Its probability of reporting 1 when H=0 is {fp0} if X=0 and {fp1} if X=1. It made no other observations. The channel fires exactly when H=1. Probabilities for X below are already conditional on this stored report; do not condition X on it a second time.'
        rates=[sensitivity/(sensitivity+fp) for fp in (fp0,fp1)]
        return dict(text=text,conditional_fire=[str(x) for x in rates],state_count=None,rule_class=CLASSES[k],derivation='Bayes with equal H prior: sensitivity/(sensitivity+false-positive rate); X mixture already conditioned on the report')
    elif k==1:
        n=6+j; slack=2+j%3; deadline=n+slack
        text=f'A prerequisite completion time V and a task duration U are independently drawn uniformly from the integers 1 through {n}. Access opens at time {slack+2}*(1-X). The task starts at max(V,access opening time), lasts U units, and cannot be interrupted. The channel fires exactly when the task finishes by time {deadline}.'
        states=list(product(range(1,n+1),repeat=2)); test=lambda x,u:max(u[1],(slack+2)*(1-x))+u[0]<=deadline
    elif k==2:
        n=2+j//2; threshold=1+j%2+n//2
        text=f'There are {n} ordinary components, each independently working with probability 1/2, and one additional component that works exactly when X=1. The channel fires exactly when at least {threshold} of these {n+1} components work.'
        states=list(product((0,1),repeat=n)); test=lambda x,u: sum(u)+x>=threshold
    elif k==3:
        n=8+j; threshold=3+j%4
        text=f'A score U is drawn uniformly from the integers 1 through {n}. A mandatory inspection independently passes with probability 1/2. The channel fires exactly when the inspection passes AND either U is at least {threshold} or a waiver is valid. The waiver is valid exactly when X=1; it cannot bypass the inspection.'
        states=list(product(range(1,n+1),(0,1))); test=lambda x,u:u[1] and (u[0]>=threshold or x)
    elif k==4:
        n=6+j; stock=12+j; target=3+j%3
        text=f'A competing account weight U is drawn uniformly from the integers 1 through {n}. A broker splits {stock} divisible units between this channel with weight 1+X and that account with weight U, in exact proportion to their weights. It assigns this channel {stock}*(1+X)/(1+X+U) units. The channel fires exactly when this allocation is at least {target}. Each channel has its own separate broker and account.'
        states=list(range(1,n+1)); test=lambda x,u:Q(stock*(1+x),1+x+u)>=target
    elif k==5:
        # Direct extra edge is in parallel with a serial route; all edge draws independent.
        n=2+j%4; m=1+j//4
        text=f'A directed network has a fixed edge from start to junction, a route of {n} edges in series from junction to end, and a second route of {m+1} edges in series from junction to end. One edge of the second route is open exactly when X=1; every other nonfixed edge is independently open with probability 1/2. All routes point toward end, share no edges, and there are no other edges. The channel fires exactly when end is reachable from start.'
        states=list(product((0,1),repeat=n+m)); test=lambda x,u:all(u[:n]) or (x and all(u[n:]))
    elif k==6:
        n=6+j; correction=2+j%3; limit=n
        text=f'A disturbance U and a later disturbance V are independently drawn uniformly from the integers 1 through {n}. First the controller applies correction {correction}*X to U, leaving error max(0,U-{correction}*X). Then V is added, with no further correction. The channel fires exactly when the final error is at most {limit}. Correction cannot overshoot below zero and has no other effects.'
        states=list(product(range(1,n+1),repeat=2)); test=lambda x,u:max(0,u[0]-correction*x)+u[1]<=limit
    else:
        accuracy=Q(3+j%3,5+j%3); count=2+j//3
        text=f'Under either value of X, before observing reports, a hidden state H was equally likely to be 0 or 1. There are {count} stored reports, all reporting 1. If X=0, they are exact copies of one original report with symmetric accuracy {accuracy}. If X=1, all {count} reports were produced independently conditional on H, each with symmetric accuracy {accuracy}. Symmetric accuracy means P(report=H given either value of H). The channel fires exactly when H=1. Probabilities for X below already condition on all stored reports; do not use them to update X again.'
        rates=[accuracy,accuracy**count/(accuracy**count+(1-accuracy)**count)]
        return dict(text=text,conditional_fire=[str(x) for x in rates],state_count=None,rule_class=CLASSES[k],derivation='Copied likelihood ratio a/(1-a); independent likelihood ratio raised to number of reports; equal H prior; X mixture already conditioned on reports')
    rates=[Q(sum(bool(test(x,u)) for u in states),len(states)) for x in (0,1)]
    assert 0<=rates[0]<rates[1]<=1
    return dict(text=text,conditional_fire=[str(x) for x in rates],state_count=len(states),rule_class=CLASSES[k])


def probability(a,b,negated=False):
    # Inspection required for reserve route and ordinary route alike.
    p=Q(4,5)*(Q(1,5)+Q(4,5)*a*(1-b))
    return 1-p if negated else p


def make_family(k,j):
    i=10*k+j; fid=f'fresh{i+1:03d}'; spec=mechanism(k,j)
    falling=(j+k)%2==1; negated=j%2==1
    p0,p1=(Q(3,4),Q(1,4)) if falling else (Q(1,4),Q(3,4))
    lo,hi=map(Q,spec['conditional_fire']); a=lo+(hi-lo)*p0; a1=lo+(hi-lo)*p1
    base=probability(a,a,negated)
    role_values={'ordinary':probability(a1,a,negated),'cancellation':probability(a,a1,negated),'archive':base}
    role_sign={r: 1 if v>base else -1 if v<base else 0 for r,v in role_values.items()}
    expected={c:next(r for r,s in role_sign.items() if s==sign) for c,sign in zip(CONTEXTS,(1,-1,0))}
    return dict(family_id=fid,stratum=CLASSES[k],instance=j+1,setting=SETTINGS[k][j],mechanism=spec,falling=falling,negated=negated,p0=str(p0),p1=str(p1),role_for_context=expected,oracle_baseline=str(base),oracle_by_role={r:str(v) for r,v in role_values.items()},source_blueprint_id=f'eval{i+1:03d}',disposition='replaced prospective qualitative concept with an explicit finite rule; not claimed to instantiate every original narrative detail',review_status='candidate_unreviewed')


def unit(f,variant,c):
    i=int(f['family_id'][5:])-1; k=i//10;j=i%10
    names=list(NAMES[(i//3)%len(NAMES)])
    if variant=='names': names=[{'Aster':'Maren','Beryl':'Neris','Cedar':'Oriel','Doran':'Pella','Elwin':'Riven','Faris':'Soren','Galen':'Tarin','Hedra':'Ulwen','Ivara':'Varen','Jalen':'Wenda','Korin':'Xarin','Lumen':'Yoren'}[x] for x in names]
    reported=names[i%3]
    target_role=f['role_for_context'][c]
    remaining_names=[x for x in names if x!=reported]
    remaining_roles=[x for x in ('ordinary','cancellation','archive') if x!=target_role]
    if i%2:remaining_roles.reverse()
    bindings={reported:target_role,**dict(zip(remaining_names,remaining_roles))}
    order=list(permutations(names))[i%6]
    assignment=' '.join(f'{n} operates the {bindings[n]} channel.' for n in order)
    question=f"Will this fictional {f['setting']} project {'fail to complete' if f['negated'] else 'complete'} by its final assessment?"
    if variant=='paraphrase':
        intro='Use only these stipulated rules for a fictional project. No unstated real-world relationships apply.'
        wrapper='Completion requires the final inspection to pass AND at least one of the following: the reserve team succeeds; or the ordinary channel fires and the cancellation channel stays silent. The reserve team has a 1/5 chance of success. The final inspection has a 4/5 chance of passing. The archive channel only writes a separate log; it changes neither completion nor any other variable. Each channel independently implements the rule below with its own X and its own random draws.'
        independence='All channel variables, the reserve result, and the inspection result are mutually independent. An audit supplies only a revised chance for the named channel X; all other conditional rules remain unchanged. The audit is independent of the other channels, reserve, and inspection. There are no shared resources, hidden causes, or additional effects.'
    else:
        intro='This is a fictional forecasting scenario. Use only the stated rules; do not import real-world facts.'
        wrapper='The project completes exactly when the final inspection passes and either the reserve team succeeds or both the ordinary channel fires and the cancellation channel does not fire. The reserve succeeds with probability 1/5 and the inspection passes with probability 4/5. The archive channel merely writes a separate log and has no effect on the project or other variables. Every channel uses the following rule, with its own X and fresh random draws.'
        independence='Variables in different channels and the reserve and inspection outcomes are mutually independent. An audit reports only a revised chance for X in the named channel, leaving all other conditional rules fixed. It is independent of other channels and the reserve and inspection. There are no shared budgets, latent common causes, or other pathways.'
    prior=f"In each channel X is either 0 or 1, initially, conditional on all observations stated above, with probability {f['p0']} of X=1. The actual hidden states and all channel outcomes, the reserve result and the inspection result remain unobserved. The project identifier has already been registered."
    text=f'{intro}\nQUESTION\n{question}\nRULES\n{wrapper}\n{f["mechanism"]["text"]}\n{independence}\nASSIGNMENTS\n{assignment}\nALREADY KNOWN\n{prior}'
    messages={'new_news':f"An audit of {reported} establishes that its probability of X=1 is {f['p1']}, replacing the initial {f['p0']}. The audit supplies no realized value of X or of a channel outcome.",'no_news':'No new information about this scenario has become available.','repeated_news':'The project identifier has already been registered.'}
    updates={cond:text+'\n\nYOUR PREVIOUS FORECAST\n'+common.PRIOR_TOKEN+'\n\nMESSAGE\n'+news+UPDATE_SUFFIX for cond,news in messages.items()}
    fam=f['family_id']+'_'+variant
    return dict(trial_id=fam+'_'+c,family_id=fam,parent_family_id=f['family_id'],variant=variant,domain=f['stratum'],context_id=c,repeat=0,material_status='fresh_finite_rule_model_screening_pending',scenario_text=text,messages=messages,baseline_prompt=text+BASELINE_SUFFIX,update_templates=updates,expected_sign={'positive':1,'negative':-1,'broken':0}[c],oracle_baseline=f['oracle_baseline'],oracle_update=f['oracle_by_role'][target_role],reported_entity=reported)


def build():
    families=[make_family(k,j) for k in range(8) for j in range(10)]
    units=[unit(f,v,c) for f in families for v in VARIANTS for c in CONTEXTS]
    audit=[]
    for f in families:
        for v in VARIANTS:
            rows=[u for u in units if u['parent_family_id']==f['family_id'] and u['variant']==v]
            inventories=[Counter(re.findall(r'\w+|[^\w\s]',u['scenario_text'].lower())) for u in rows]
            assert inventories[0]==inventories[1]==inventories[2]
            assert len({u['messages']['new_news'] for u in rows})==1
            for u in rows:
                a,b=Q(u['oracle_baseline']),Q(u['oracle_update'])
                assert 0<a<1 and 0<b<1
                assert ((b>a)-(b<a))==u['expected_sign']
                audit.append({'trial_id':u['trial_id'],'baseline':str(a),'updated':str(b),'delta_pp':float(100*(b-a))})
    OUT.mkdir(parents=True,exist_ok=True)
    def save(name,rows):
        p=OUT/name
        payload=''.join(common.canonical_json(r)+'\n' for r in rows)
        if p.exists() and p.read_text()!=payload:raise RuntimeError('Candidate version already exists; create a new version')
        p.write_text(payload)
    save('families.jsonl',families); save('probability_plan.jsonl',units);save('oracle_audit.jsonl',audit)
    # Three fixed positions per stratum; chosen before any review or target output.
    paid=[u for u in units if int(u['parent_family_id'][5:])%10 in ((1,4,8) if ((int(u['parent_family_id'][5:])-1)//10)%4<2 else (2,5,9)) and u['variant'] in ('base','paraphrase')]
    save('frontier_plan.jsonl',paid)
    # Review every distinct visible packet; same-text resample is covered by exact identity.
    review=[]; key=[]
    for u in units:
        if u['variant']=='resample':continue
        rid=common.text_sha256('blind-review-v1:'+u['trial_id'])[:20]
        prompt=u['scenario_text']+'\nMESSAGE\n'+u['messages']['new_news']+'\n\nReview this fictional scenario on its own. Does this message increase, decrease, or leave unchanged the probability of YES, or is the direction unclear? Are both YES and NO still possible before and after it? Identify any missing information, contradiction, or ambiguous mechanism. Do not assume an intended answer. Return JSON with fields direction (increase/decrease/unchanged/unclear), endpoints_possible (boolean; true means both outcomes remain possible), and concern (a short explanation, or none).'
        review.append({'review_id':rid,'prompt':prompt})
        key.append({'review_id':rid,'trial_id':u['trial_id'],'expected_direction':{1:'increase',-1:'decrease',0:'unchanged'}[u['expected_sign']]})
    # Pseudorandom opaque order, independent of labels. Review runner reads packet only.
    review.sort(key=lambda r:r['review_id'])
    save('blind_review_packets.jsonl',review);save('private_review_key.jsonl',key)
    print(json.dumps({'families':len(families),'local_units':len(units),'local_records_per_reasoning_arm':4*len(units),'frontier_units':len(paid),'frontier_records':4*len(paid),'blind_review_packets':len(review),'min_nonzero_oracle_delta_pp':min(abs(x['delta_pp']) for x in audit if x['delta_pp']),'max_prompt_bytes':max(len(u['baseline_prompt'].encode()) for u in units)},indent=2))

if __name__=='__main__':build()
