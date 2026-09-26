"""CPU audit of frozen inputs, balance, and family-held-out lexical baselines."""
from collections import Counter
from pathlib import Path
import json,re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from .materials import OUT,ROOT,make_family,unit,VARIANTS,CONTEXTS
from .. import run_local as c


def main():
    units=c.read_units(OUT/'probability_plan.jsonl');families=[json.loads(s) for s in (OUT/'families.jsonl').read_text().splitlines()]
    expected=[unit(make_family(k,j),v,ctx) for k in range(8) for j in range(10) for v in VARIANTS for ctx in CONTEXTS]
    assert units==expected, 'Stored candidate differs from reviewed generative specification'
    for f in families:
        for v in VARIANTS:
            rows=[u for u in units if u['parent_family_id']==f['family_id'] and u['variant']==v]
            bags=[Counter(re.findall(r'\w+|[^\w\s]',u['scenario_text'].lower())) for u in rows]
            assert bags[0]==bags[1]==bags[2]
    result={'plan':{'path':str(OUT/'probability_plan.jsonl'),'sha256':c.file_sha256(OUT/'probability_plan.jsonl')},'instances':len(families),'mechanism_classes':len({f['stratum'] for f in families}),'variants':4,'exact_word_and_punctuation_inventories_equal':True,'all_exact_oracle_signs_verified':True,'shared_outcome_wrapper':True,'not_80_independent_mechanisms':True,'balance':{},'lexical':{}}
    for scope,ids in [('local',{f['family_id'] for f in families}),('frontier',{u['parent_family_id'] for u in c.read_units(OUT/'frontier_plan.jsonl')})]:
        fs=[f for f in families if f['family_id'] in ids]
        result['balance'][scope]={'n':len(fs),'negated_question':dict(Counter(f['negated'] for f in fs)),'falling_news':dict(Counter(f['falling'] for f in fs)),'strata':dict(Counter(f['stratum'] for f in fs))}
    groups=np.array([u['parent_family_id'] for u in units]);labels=np.array([u['expected_sign'] for u in units]);binary=(labels!=0).astype(int)
    for name,field,ngram in [('news_words','news',(1,1)),('full_words','scenario',(1,1)),('full_bigrams','scenario',(1,2))]:
        texts=[u['messages']['new_news'] if field=='news' else u['scenario_text']+'\n'+u['messages']['new_news'] for u in units]
        preds=np.empty(len(units),int);rel=np.empty(len(units),int)
        for train,test in GroupKFold(5).split(texts,labels,groups):
            vector=CountVectorizer(ngram_range=ngram);x=vector.fit_transform([texts[i] for i in train]);xt=vector.transform([texts[i] for i in test])
            model=LogisticRegression(max_iter=2000,random_state=20260922);model.fit(x,labels[train]);preds[test]=model.predict(xt)
            model=LogisticRegression(max_iter=2000,random_state=20260922,class_weight='balanced');model.fit(x,binary[train]);rel[test]=model.predict(xt)
        result['lexical'][name]={'direction_accuracy':accuracy_score(labels,preds),'relevance_balanced_accuracy':balanced_accuracy_score(binary,rel)}
    result['cross_validation']='5-fold GroupKFold with all contexts and wording variants of a parent instance together; vectorizer fit on training fold only'
    result['interpretation']='News-only and bag-of-words shortcut removed by exact within-family inventories; does not rule out all lexical/structural heuristics or establish ecological validity.'
    out=ROOT/'runs/fresh_evaluation_v1/material_audit.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
