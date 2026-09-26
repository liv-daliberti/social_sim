"""14B seed-level estimators, parse rates, old-interval reproduction, synthesis."""
import sys, json, math
sys.path.insert(0, "/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural")
import numpy as np
import make_paper_outputs as core
import make_qwen3_8b_eight_seed_outputs as e8
from pathlib import Path
R = Path("/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/reports")
ROOTS = {
 ("causal",42):"scale_causal_qwen3_14b_s42_20260826_025010_j30880155",
 ("causal",43):"scale_causal_qwen3_14b_s43_20260924_011636_j31481143",
 ("causal",44):"scale_causal_qwen3_14b_s44_20260826_183443_j30880157",
 ("population_prior",42):"scale_population_prior_qwen3_14b_s42_20260826_183724_j30880158",
 ("population_prior",43):"scale_population_prior_qwen3_14b_s43_20260826_183724_j30880159",
 ("population_prior",44):"scale_population_prior_qwen3_14b_s44_20260826_204654_j30880160",
}
RECOVERY2 = "scale_causal_qwen3_14b_s43_recovery2_20260828_204430_j30948628"
BASE = "base_qwen3_14b_j30879415"
CELLS = e8.CELLS
SEEDS = (42,43,44)

def load(roots):
    rows = []
    for (arm, seed), name in roots.items():
        r = core.read_jsonl(R/name/"stochastic_n5.scores.jsonl")
        core.validate_rows(r, {"kind":"confirmatory","model":"qwen3_14b","arm":arm,"seed":seed,"job_id":name}, "stochastic")
        rows += r
    b = core.read_jsonl(R/BASE/"stochastic_n5.scores.jsonl")
    core.validate_rows(b, {"kind":"base","model":"qwen3_14b","job_id":BASE}, "stochastic")
    return rows + b

def cells(rows, seed_base, label):
    out = []
    for i,(lab,dom,st) in enumerate(CELLS):
        common={"model":"qwen3_14b","decode":"stochastic","domain":dom,"target_structure":st,"cue":"correct"}
        means={arm: core.mean_cell(rows,{**common,"arm":arm}) for arm in ("base","population_prior","causal")}
        diffs = e8.seed_differences(rows,{**common,"arm":"population_prior"},{**common,"arm":"causal"},SEEDS)
        sm = np.asarray([float(np.mean(diffs[s])) for s in SEEDS])
        hb = e8.hierarchical_interval(diffs, SEEDS, repetitions=5000, seed=seed_base+i)
        st_ = e8.student_t_interval(sm)
        wcb = e8.wild_cluster_p(sm, repetitions=9999, seed=seed_base+i)
        sg = e8.sign_test(sm)
        out.append({"cell":lab,"key":e8.KEYS[lab],**means,"seed_means":{str(s):float(v) for s,v in zip(SEEDS,sm)},
                    "hb":hb,"student_t":st_,"wcb":wcb,"sign":sg})
    print(f"\n=== {label} ===")
    for c in out:
        print(f"{c['key']:10s} base={c['base']:.3f} prior={c['population_prior']:.3f} matched={c['causal']:.3f} | "
              f"HB {c['hb'][0]:+.3f} [{c['hb'][1]:+.3f},{c['hb'][2]:+.3f}] | t2 [{c['student_t'][1]:+.3f},{c['student_t'][2]:+.3f}] | "
              f"WCB p={c['wcb']['p_two_sided']:.4f} t={c['wcb']['t']:.2f} | sign {c['sign']['positive']}/3 p={c['sign']['p_one_sided']:.3f} | "
              f"seeds {c['seed_means']}")
    return out

rows = load(ROOTS)
# parse rates per endpoint
print("=== parse rates (stochastic, 7200 draws each) ===")
for name in list(ROOTS.values())+[BASE]:
    r = core.read_jsonl(R/name/"stochastic_n5.scores.jsonl")
    g = core.read_jsonl(R/name/"greedy.scores.jsonl")
    print(f"{name}: stoch {sum(bool(x['parsed']) for x in r)}/{len(r)} = {np.mean([bool(x['parsed']) for x in r]):.4f}; greedy {sum(bool(x['parsed']) for x in g)}/{len(g)}")
    k = list(r[0].keys())
print("row keys:", k)

new = cells(rows, 20260825, "REGISTERED s43=31481143, renderer bootstrap seed 20260825+cell")
new18 = cells(rows, 20260818, "REGISTERED s43=31481143, eight-seed-script bootstrap seed 20260818+cell")
old_roots = dict(ROOTS); old_roots[("causal",43)] = RECOVERY2
old = cells(load(old_roots), 20260825, "OLD s43=recovery2 30948628, seed 20260825+cell")
old18 = cells(load(old_roots), 20260818, "OLD s43=recovery2 30948628, seed 20260818+cell")

# greedy joint means for completeness
def synth(q14_est, q14_lo, q14_hi, label):
    ests = [(0.31934000, 0.13376, 0.48857), (0.2512940320833334, 0.058208621503472203, 0.46998328662500005), (q14_est,q14_lo,q14_hi)]
    m = np.mean([e for e,_,_ in ests])
    se = math.sqrt(sum(((h-l)/2/1.959964)**2 for _,l,h in ests))/3
    print(f"{label}: synthesis {m:.4f} [{m-1.959964*se:.4f},{m+1.959964*se:.4f}]  (SE={se:.4f})")
    return m, se
# pull exact 4B numbers from registered_results
reg = json.load(open(R/"registered_results.json"))
q4 = next(r for r in reg["registered_estimates"]["primary"] if r["model"]=="qwen3_4b" and r["decode"]=="stochastic")["population_prior_minus_causal"]
q8 = next(r for r in reg["registered_estimates"]["primary"] if r["model"]=="qwen3_8b" and r["decode"]=="stochastic")["population_prior_minus_causal"]
print("\nexact 4B:", q4, "\nexact 8B:", q8)
def synth2(q14, label):
    ests=[(q4["estimate"],q4["ci95_low"],q4["ci95_high"]),(q8["estimate"],q8["ci95_low"],q8["ci95_high"]),q14]
    m=np.mean([e for e,_,_ in ests]); se=math.sqrt(sum(((h-l)/2/1.959964)**2 for _,l,h in ests))/3
    print(f"{label}: synthesis {m:.4f} [{m-1.959964*se:.4f},{m+1.959964*se:.4f}] SE={se:.4f}")
    # alt: mean of bounds
    print(f"   alt mean-of-bounds [{np.mean([l for _,l,_ in ests]):.3f},{np.mean([h for _,_,h in ests]):.3f}]")
    return m
mo = synth2((0.11980725666666665, -0.027, 0.261), "OLD paper (.120 [-.027,.261])")
mo2 = synth2(old[3]["hb"], "OLD recomputed HB")
mn = synth2(new[3]["hb"], "NEW registered HB")
# Student-t over 9 size-seed contrasts as alt
q4s = json.load(open(R/"qwen3_4b_step300_stochastic_eight_seed.json"))["transfer_cells"][3]["seed_level_contrasts"]
q8s = json.load(open(R/"qwen3_8b_step300_stochastic_eight_seed.json"))["transfer_cells"][3]["seed_level_contrasts"]
nine = [q4s[s] for s in ("42","43","44")]+[q8s[s] for s in ("42","43","44")]+list(new[3]["seed_means"].values())
print("nine size-seed contrasts:", [round(v,3) for v in nine], "positive:", sum(v>0 for v in nine))
print("Student-t(8df) over 9:", e8.student_t_interval(np.asarray(nine)))
# size-cell agreement: 12 cells
print("14B cells positive:", [ (c["key"], c["hb"][0]>0) for c in new])
# MAE reduction pct: synthesis / prior MAE
priors = {"4b": next(r for r in reg["registered_estimates"]["primary"] if r["model"]=="qwen3_4b" and r["decode"]=="stochastic")["mean_response_mae"],
          "8b": next(r for r in reg["registered_estimates"]["primary"] if r["model"]=="qwen3_8b" and r["decode"]=="stochastic")["mean_response_mae"]}
print("prior MAEs 4b/8b:", priors, "14b prior:", new[3]["population_prior"], "matched:", new[3]["causal"])
pm = np.mean([priors["4b"]["population_prior"], priors["8b"]["population_prior"], new[3]["population_prior"]])
print(f"pct reduction: new/8Bprior={100*mn/priors['8b']['population_prior']:.2f}%  new/meanprior={100*mn/pm:.2f}%  old/8Bprior={100*mo/priors['8b']['population_prior']:.2f}%  old/meanprior={100*mo/np.mean([priors['4b']['population_prior'], priors['8b']['population_prior'], old[3]['population_prior']]):.2f}%")
json.dump({"new_20260825":new,"new_20260818":new18,"old_20260825":old}, open("/tmp/claude-363432/-n-fs-similarity-social-sim/54e64e28-f3d1-4f39-8e9a-d9ce92e871cd/scratchpad/out/qwen14_extra.json","w"), indent=1, default=float)
