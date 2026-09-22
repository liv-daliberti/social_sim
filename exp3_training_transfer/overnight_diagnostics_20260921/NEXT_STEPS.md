# Live handoff (03:47 UTC September22)

User goal remains active. Do not mark complete while the fixed training/evaluation
roster is unfinished. No paper edits or external messages have been made.

1. Structure component diagnosis DONE. `structure/completion_receipt.json`,
   `FINDINGS.md`, `REPORT.md`, `BASELINE_AUDIT.md`.8 models,6144 generations,
   all parsed; reference identification100% allcells. Matched forecast cue-use
   remains nearzero in City; oracle summaries fail8B.32B partially computes
   response deviations but omitsbaseline; preserve frozen clipped metrics and
   labeled posthoc raw-response/offset sensitivity. No further promptsearch.
2. Evidence-use jobs31444965/66 onA5000 nearing completion. Agent evidence_use
   owns analysis and optional summaryfigure.10 conditions =both disclosures×
   base+matched45/46+prior45/46. Old42–44 adapterweights absent. Fresh heldout720
   perdisclosure and192 coherentgainpairs perdisclosure frozen, noise replayed.
   Known-mechanism baseline indicates strong gaininformation except noisy
   extrapolationworld. Expect `analysis/summary.json` complete=true; retain
   two-seed inferential caveat. Extra3world sensitivity explicitly posthoc.
3. Mechanism controls: fixed20new trainings =10shuffled+10matched, all2A5000,
   numerical config identical, actor memoryfraction.76 to match originalA6000
   absolute budget(.38). Ten controls have exact targetvector multiset preserved
   withinworld×k, no selfassignments, no unchangednumericvectors, sameprompts.
   Sixmatched restoremissing42–44; fourmatched45/46 removehardwareconfound.
   Historical45/46 fresh stochastic endpoints complete, supplementary only.
   Agent shuffled_controls owns gate/release scripts. Successful canary31445281_2
   has actual learnerupdates, releasehelper waiting buffered metric then submits
   remaining19. PriorcanaryOOM and volatilecachehash failure preserved; samecell
   retried,no outcome-based exclusion. Route10controls+5matched allcs/cs and
   other5matched mltheory for40GPU capacity; scheduler decides availability.
   Maximum30h perjob; originalbudget4800prompts/300rounds unchanged. Monitor,
   recover infrastructure failures transparently withoutchangingbudget. Primary
   analyzer requires all40endpoint/decodecells; don'tsubstituteoldtestresults.
4. SFTjobs31445286/87 running1A6000 each, two fixed learningrates1e-6/2e-5,
   same4800original8Btrainingprompts, seed42, oneepoch300updates, completionloss.
   Startsbase8B pinnedb968826..., LoRA32/64. Runtime~16seconds/update so~80min.
   Runner pauses after40min and selfsubmits up to6one-hour allocations, resumes
   exactoptimizer/LoRA/RNG and skipscompletedbatches. Finalcomponent inference
   selfsubmits. Inspect `learnability/runs/*/train.jsonl`,complete.json,
   continuation_jobs.txt,evaluation_jobs.txt. Bothoutcomes reported; no moreSFT
   expansion. OriginalSFTgate CLOSED in gate_initial_closed.json; subsequent
   explicitly outcome-informed amendment launchesunchanged2configs because32B
   shows conditionalnumericalcapacity+wrongabsolutesemantics. No claimthat the
   initialgatepassed or fulltasksolved. SelectionSFTunnecessary(identity100%).
   Agent structure_diagnostics available to analyze both SFTendpoints with same
   frozen scoring plusraw/offset sensitivity. OriginalheldoutstochasticSFT eval
   optional separatejob; not yet launched. Do not call one-seed result general
   method superiority; presentationmatched300SFTupdatesvs~2400RLupdates.

Scripts all live in this directory. Most shell tools require require_escalated
because default bwrap namespace creation fails ENOSPC; auto-review has allowed
research actions. Source/data freezes must remain intact; record amendments.
Cluster rewrites >1h all jobs toaccountowned partitions.<=1h mltheory/all/none
A6000 jobs run node208; longA5000 allcs→cs,mltheory→mltheory. Never cancel other
existing campaigns. Sharedrepo has many preexisting userchanges, leave them.
