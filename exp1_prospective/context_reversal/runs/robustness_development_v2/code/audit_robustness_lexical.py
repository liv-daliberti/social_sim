"""CPU-only lexical checks with all edit variants grouped by parent family."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import numpy as np
from . import direction_design, run_local as common


def audit(units, folds=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    direction_design.validate_units(units)
    scored=[u for u in units if u['context_id']!='masked']
    groups=np.array([u['parent_family_id'] for u in scored])
    labels=np.array([{'positive':'increase','negative':'decrease','broken':'unchanged'}[u['context_id']] for u in scored])
    parents=sorted(set(groups));variants=sorted({u['variant'] for u in scored})
    if len(parents)<2:raise ValueError('Need multiple held-out parent families')
    inputs={'news_only':[u['messages']['new_news'] for u in scored], 'full_text':[u['scenario_text']+'\n'+u['messages']['new_news'] for u in scored]}
    comparisons={}
    family_variants=defaultdict(list)
    for u in scored:family_variants[u['parent_family_id'],u['variant']].append(u)
    for key,items in family_variants.items():
        if len(items)!=3:raise ValueError('Each parent/variant needs all three scored contexts')
        words=[Counter(re.findall(r'\b\w+\b',(u['scenario_text']+' '+u['messages']['new_news']).lower())) for u in items]
        comparisons[':'.join(key)]={'identical_news':len({u['messages']['new_news'] for u in items})==1,'identical_unigram_inventory':all(w==words[0] for w in words[1:])}
    models=[('news_unigram','news_only',{'ngram_range':(1,1)}),('full_unigram','full_text',{'ngram_range':(1,1)}),('full_bigram','full_text',{'ngram_range':(1,2)}),('full_character','full_text',{'analyzer':'char','ngram_range':(3,5)})]
    result={}
    splits=list(GroupKFold(n_splits=min(folds,len(parents))).split(scored,labels,groups))
    for name,field,kwargs in models:
        texts=np.array(inputs[field]);predicted=np.empty(len(labels),dtype=object);predicted_relevant=np.empty(len(labels),dtype=bool);fold_info=[]
        for train,test in splits:
            assert not set(groups[train])&set(groups[test])
            model=make_pipeline(TfidfVectorizer(lowercase=True,max_features=30000,**kwargs),LogisticRegression(C=1.0,max_iter=2000,random_state=20260921))
            model.fit(texts[train],labels[train]);predicted[test]=model.predict(texts[test])
            relevance_model=make_pipeline(TfidfVectorizer(lowercase=True,max_features=30000,**kwargs),LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=20260921))
            relevance_model.fit(texts[train],labels[train]!='unchanged');predicted_relevant[test]=relevance_model.predict(texts[test])
            fold_info.append({'train_parent_families':sorted(set(groups[train])),'test_parent_families':sorted(set(groups[test]))})
        expected_relevant=labels!='unchanged'
        result[name]={'direction_accuracy':float(accuracy_score(labels,predicted)),'direction_balanced_accuracy':float(balanced_accuracy_score(labels,predicted)),'relevance_classifier':'separately fitted class-balanced binary logistic regression','relevance_accuracy':float(accuracy_score(expected_relevant,predicted_relevant)),'relevance_balanced_accuracy':float(balanced_accuracy_score(expected_relevant,predicted_relevant)),'confusion_labels':['increase','decrease','unchanged'],'confusion_matrix':confusion_matrix(labels,predicted,labels=['increase','decrease','unchanged']).tolist(),'folds':fold_info,'predictions':[{'trial_id':u['trial_id'],'parent_family_id':u['parent_family_id'],'variant':u['variant'],'expected':str(y),'predicted':str(p),'relevance_expected':bool(yr),'relevance_predicted':bool(pr)} for u,y,p,yr,pr in zip(scored,labels,predicted,expected_relevant,predicted_relevant)]}
    return {'schema_version':'controlled_robustness_lexical_audit_v1','n_parent_families':len(parents),'n_variants':len(variants),'n_scored_rows':len(scored),'split_unit':'parent_family_id_all_variants_together','chance_reference':{'direction_accuracy':1/3,'relevance_majority_accuracy':2/3,'relevance_balanced_accuracy':.5},'context_inventory_checks':comparisons,'all_scored_contexts_have_identical_news':all(x['identical_news'] for x in comparisons.values()),'all_scored_contexts_have_identical_unigram_inventory':all(x['identical_unigram_inventory'] for x in comparisons.values()),'held_out_baselines':result,'interpretation':'Identical word inventories block unigram context-type shortcuts. Bigram/character models can still encode relationship bindings; report their performance without selecting or dropping target failures.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--design',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    units=direction_design.read_units(a.design);r=audit(units)
    r['provenance']={'design':{'path':str(a.design.resolve()),'sha256':common.file_sha256(a.design)},'code_sha256':common.file_sha256(Path(__file__))}
    common.atomic_write(a.output/'lexical_audit.json',json.dumps(r,indent=2)+'\n')
    lines=['# Controlled-edit lexical audit','',f"{r['n_parent_families']} held-out parent groups; {r['n_scored_rows']} rows. All variants remain in their parent fold.",'',f"Identical news across contexts: {r['all_scored_contexts_have_identical_news']}. Identical unigram inventories: {r['all_scored_contexts_have_identical_unigram_inventory']}.",'','| Baseline | Direction accuracy | Relevance balanced accuracy |','|---|---:|---:|']
    for name,v in r['held_out_baselines'].items():lines.append(f"| {name} | {v['direction_accuracy']:.3f} | {v['relevance_balanced_accuracy']:.3f} |")
    lines+=['',r['interpretation'],''];common.atomic_write(a.output/'lexical_audit.md','\n'.join(lines));print(json.dumps({'output':str(a.output),'identical_unigrams':r['all_scored_contexts_have_identical_unigram_inventory']}))

if __name__=='__main__':main()
