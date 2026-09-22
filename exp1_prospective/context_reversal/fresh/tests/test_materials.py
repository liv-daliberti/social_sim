from collections import Counter
from fractions import Fraction as Q
from itertools import product
from math import comb
import json,re
from exp1_prospective.context_reversal.fresh import materials as m
from exp1_prospective.context_reversal import run_local as c


def test_closed_form_independent_of_authorship_enumeration():
    for k,j in product(range(8),range(10)):
        got=list(map(Q,m.mechanism(k,j)['conditional_fire']))
        expected=[]
        for x in (0,1):
            if k==0:
                a=Q(3+j%3,5+j%3);fp=Q(2+j//3,5+j//3)/(2 if x else 1);v=1/(1+fp/a)
            elif k==1:
                n=6+j;slack=2+j%3;d=n+slack;t=(slack+2)*(1-x)
                v=Q(sum(max(0,min(n,d-u)) if d-u>=t else 0 for u in range(1,n+1)),n*n)
            elif k==2:
                n=2+j//2;threshold=1+j%2+n//2
                v=Q(sum(comb(n,r) for r in range(n+1) if r>=threshold-x),2**n)
            elif k==3:
                n=8+j;threshold=3+j%4;v=Q(1,2)*(1 if x else Q(n-threshold+1,n))
            elif k==4:
                n=6+j;stock=12+j;target=3+j%3;limit=Q(stock*(1+x),target)-(1+x)
                v=Q(max(0,min(n,limit.numerator//limit.denominator)),n)
            elif k==5:
                n=2+j%4;other=1+j//4;v=1-(1-Q(1,2)**n)*(1-x*Q(1,2)**other)
            elif k==6:
                n=6+j;corr=2+j%3;v=Q(sum(max(0,min(n,n-max(0,u-corr*x))) for u in range(1,n+1)),n*n)
            else:
                a=Q(3+j%3,5+j%3);count=2+j//3;v=1/(1+((1-a)/a)**(count if x else 1))
            expected.append(v)
        assert got==expected,(k,j,got,expected)


def test_full_joint_outcome_and_strict_signs():
    for k,j in product(range(8),range(10)):
        f=m.make_family(k,j);lo,hi=map(Q,f['mechanism']['conditional_fire']);p0,p1=Q(f['p0']),Q(f['p1']);a=lo+(hi-lo)*p0;new=lo+(hi-lo)*p1
        for role in ('ordinary','cancellation','archive'):
            rates=[Q(4,5),Q(1,5),new if role=='ordinary' else a,new if role=='cancellation' else a]
            total=Q(0)
            for bits in product((0,1),repeat=4):
                prob=Q(1)
                for bit,rate in zip(bits,rates):prob*=rate if bit else 1-rate
                outcome=bits[0] and (bits[1] or (bits[2] and not bits[3]))
                if bool(outcome)!=f['negated']:total+=prob
            assert total==Q(f['oracle_by_role'][role])
            assert Q(4,25)<=total<=Q(21,25)


def test_visible_controls_and_inventories():
    units=c.read_units(m.OUT/'probability_plan.jsonl');assert len(units)==960
    for f in sorted({u['parent_family_id'] for u in units}):
        subset=[u for u in units if u['parent_family_id']==f]
        for v in m.VARIANTS:
            rows=[u for u in subset if u['variant']==v]
            assert len({u['messages']['new_news'] for u in rows})==1
            inv=[Counter(re.findall(r'\w+|[^\w\s]',u['scenario_text'].lower())) for u in rows]
            assert inv[0]==inv[1]==inv[2]
            for u in rows:
                assert u['messages']['repeated_news'] in u['scenario_text']
                assert all(t.count(c.PRIOR_TOKEN)==1 for t in u['update_templates'].values())
                assert u['parent_family_id'] not in u['baseline_prompt']
        for ctx in m.CONTEXTS:
            a=next(u for u in subset if u['variant']=='base' and u['context_id']==ctx)
            b=next(u for u in subset if u['variant']=='resample' and u['context_id']==ctx)
            assert a['baseline_prompt']==b['baseline_prompt'] and a['update_templates']==b['update_templates']
            assert c.request_seed(20260922,a['trial_id'],'baseline','baseline')!=c.request_seed(20260922,b['trial_id'],'baseline','baseline')


def test_blind_packet_and_paid_balance():
    packets=[json.loads(s) for s in (m.OUT/'blind_review_packets.jsonl').read_text().splitlines()]
    assert len(packets)==720 and all(set(x)=={'review_id','prompt'} for x in packets)
    paid=c.read_units(m.OUT/'frontier_plan.jsonl');assert len(paid)==144
    fs=[m.make_family(k,j) for k,j in product(range(8),range(10))];ids={u['parent_family_id'] for u in paid};selected=[f for f in fs if f['family_id'] in ids]
    assert len(selected)==24 and sum(f['negated'] for f in selected)==12 and sum(f['falling'] for f in selected)==12
    assert set(Counter(f['stratum'] for f in selected).values())=={3}
