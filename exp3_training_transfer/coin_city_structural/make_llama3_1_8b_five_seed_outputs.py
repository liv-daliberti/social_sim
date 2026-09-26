"""Llama-3.1-8B five-seed (42-46) transfer cells with the eight-seed estimator set, plus cue contrasts."""
import sys, json
sys.path.insert(0, "/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural")
import numpy as np
import make_qwen3_8b_eight_seed_outputs as e8
import make_paper_outputs as core
from pathlib import Path
e8.MODEL = "llama3_1_8b"
R = e8.REPORTS
parent = e8.parent_ledger()
jobs = {(r["arm"], int(r["seed"])): str(r["job_id"]) for r in parent["full_training"] if r.get("model")=="llama3_1_8b" and r.get("kind")=="confirmatory"}
ext = json.load(open("/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/runs/llama3_1_8b_eight_seed_submission_20260919T215132Z.json"))
for r in ext["training_jobs"]:
    jobs[(r["arm"], int(r["seed"]))] = str(r["job_id"])
base = [r for r in parent["base_evaluations"] if r["model"]=="llama3_1_8b"][0]["job_id"]
def roster(seeds):
    ro = [{"model":"llama3_1_8b","arm":a,"seed":s,"kind":"confirmatory","job_id":jobs[(a,s)]} for a in e8.ARMS for s in seeds]
    ro.append({"model":"llama3_1_8b","arm":None,"seed":None,"kind":"base","job_id":str(base)})
    return ro
print("roster jobs:", jobs, "base", base)
r3 = roster((42,43,44)); rows3, p3 = e8.load_rows(r3)
e8.self_test(rows3)
r5 = roster((42,43,44,45,46)); rows5, p5 = e8.load_rows(r5)
print("five-seed files:", [str(p.relative_to(e8.REPO)) for p in p5])
print("parse rates:")
for p in p5:
    rr = core.read_jsonl(p); print(f"  {p.parent.name}: {sum(bool(x['parsed']) for x in rr)}/{len(rr)}")
def show(res, label):
    print(f"\n=== {label} ===")
    for c in res:
        hb=c["hierarchical_bootstrap"]; t=c["student_t"]; w=c["wild_cluster_bootstrap"]; s=c["sign_test"]
        print(f"{c['key']:10s} base={c['base']:.3f} prior={c['population_prior']:.3f} matched={c['causal']:.3f} | HB {hb['estimate']:+.3f} [{hb['ci95_low']:+.3f},{hb['ci95_high']:+.3f}] | t{t['df']} [{t['ci95_low']:+.3f},{t['ci95_high']:+.3f}] | WCB p={w['p_two_sided']:.4f} t={w['t']:.2f} | sign {s['positive']}/{s['seeds']} p={s['p_one_sided']:.3f} | seeds { {k: round(v,3) for k,v in c['seed_level_contrasts'].items()} }")
res3 = e8.analyze(rows3,(42,43,44)); show(res3,"Llama three-seed (paper)")
res5 = e8.analyze(rows5,(42,43,44,45,46)); show(res5,"Llama FIVE-seed")

# cue contrasts, generalized seeds (same algorithm as core.paired_interval / paired_override_interval)
def paired(rows, left, right, seeds, key, seed, reps=5000):
    lv = core.episode_values(rows,left,key); rv = core.episode_values(rows,right,key)
    d = {}
    for s in seeds:
        com = sorted(set(lv.get(s,{})) & set(rv.get(s,{}))); assert com
        d[s] = np.asarray([lv[s][e]-rv[s][e] for e in com])
    hb = e8.hierarchical_interval(d, seeds, repetitions=reps, seed=seed)
    sm = np.asarray([float(np.mean(d[s])) for s in seeds])
    return hb, e8.student_t_interval(sm), sm
def override(rows, common, seeds, seed, reps=5000):
    d={}
    for s in seeds:
        pen={}
        for k in (0,8):
            mis = core.episode_values(rows,{**common,"cue":"misleading","k":k},"pair_id",core.pair_family).get(s,{})
            cor = core.episode_values(rows,{**common,"cue":"correct","k":k},"pair_id",core.pair_family).get(s,{})
            sh = sorted(set(mis)&set(cor)); pen[k]={e: mis[e]-cor[e] for e in sh}
        fam = sorted(set(pen[0])&set(pen[8])); d[s]=np.asarray([pen[8][e]-pen[0][e] for e in fam])
    hb = e8.hierarchical_interval(d, seeds, repetitions=reps, seed=seed)
    sm = np.asarray([float(np.mean(d[s])) for s in seeds])
    return hb, e8.student_t_interval(sm), sm
common={"model":"llama3_1_8b","arm":"causal","decode":"stochastic","domain":"coin_harbor","target_structure":"mediated_b"}
BS=20260818
for seeds, rows, label in (((42,43,44),rows3,"3-seed check vs paper"),((42,43,44,45,46),rows5,"5-seed")):
    print(f"\n=== cue contrasts, episode-matched Llama, joint cell, {label} ===")
    a = paired(rows,{**common,"cue":"none"},{**common,"cue":"correct"},seeds,"pair_id",BS+3000)
    m = paired(rows,{**common,"cue":"misleading"},{**common,"cue":"correct"},seeds,"pair_id",BS+4000)
    o = override(rows,common,seeds,BS+5000)
    for name,(hb,t,sm) in (("absent-correct",a),("misleading-correct",m),("override k8-k0",o)):
        print(f"{name:20s} HB {hb[0]:+.3f} [{hb[1]:+.3f},{hb[2]:+.3f}] | t [{t[1]:+.3f},{t[2]:+.3f}] | seeds {[round(v,3) for v in sm]}")
    for k in (0,2,4,8):
        vals={cue: core.mean_cell(rows,{**common,"cue":cue,"k":k},"pair_id") for cue in ("correct","none","misleading")}
        ak=paired(rows,{**common,"cue":"none","k":k},{**common,"cue":"correct","k":k},seeds,"pair_id",BS+100+k)[0]
        mk=paired(rows,{**common,"cue":"misleading","k":k},{**common,"cue":"correct","k":k},seeds,"pair_id",BS+k)[0]
        print(f"  k={k}: MAE correct={vals['correct']:.3f} none={vals['none']:.3f} mis={vals['misleading']:.3f} | absent-correct {ak[0]:+.3f} [{ak[1]:+.3f},{ak[2]:+.3f}] | mis-correct {mk[0]:+.3f} [{mk[1]:+.3f},{mk[2]:+.3f}]")
json.dump({"three_seed":res3,"five_seed":res5,"roster":r5,"score_files":[str(p.relative_to(e8.REPO)) for p in p5]}, open("/tmp/claude-363432/-n-fs-similarity-social-sim/54e64e28-f3d1-4f39-8e9a-d9ce92e871cd/scratchpad/out/llama3_1_8b_step300_stochastic_five_seed.json","w"), indent=1, default=float)
