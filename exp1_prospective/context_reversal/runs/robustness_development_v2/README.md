# Controlled robustness development, version 2

All 20 original parent families retained. Four variants: repaired base, exact
renaming, meaning-preserving paraphrase, and identical-text resample. All three
large local models run probability forecasting and a separate direction task.
Each model has 1,280 numeric and 960 direction records; all variants remain clustered
within 20 parents. Original material and output files are unchanged.

Inputs, scoring and execution code were frozen before inference. Thirty material
and scoring tests passed, six CPU runner dry runs passed, and an independent
scoring review checked provenance, unconditional denominators and control arms.
Lexical audits held out entire parent families including every variant. Unigram
and bigram direction accuracy was 1/3 and direct relevance balanced accuracy 1/2;
character-ngram direction accuracy 0.325 and relevance balanced accuracy 0.478.
These are development diagnostics, not proof that every possible shortcut is absent.

Jobs:
- qwen2_5_72b_instruct: 31444437
- llama3_1_70b_instruct: 31444557
- qwen3_32b: 31444439
- automatic CPU finalizer: 31444440

Check raw response manifests and completion.json for actual completion.
No frontier calls or new human-review round is part of this development run.
The separate prospective fresh evaluation has a $25 authorized cap; see
../fresh_evaluation_v1/REVIEW_AND_FREEZE_PLAN.md. It is not yet frozen or run.

Llama was resubmitted while pending with zero responses: eight A5000 GPUs
replace four A6000 GPUs to avoid the estimated two-hour queue. Tensor parallelism
changes before collection; checkpoint, prompts, sampling and scoring are fixed.
Exact original and replacement commands remain in submission.json.

Conditional checkpoint continuations are queued for the two 70B-class models
because initial weight loading is slow. They only resume missing records, and
verify/skip completed arms; saved outputs are never regenerated.
- qwen2_5_72b_instruct: 31444437 -> 31444787
- llama3_1_70b_instruct: 31444557 -> 31444788

Completed Qwen3 diagnostic: all 1,280 numerical and 960 categorical records are
valid. Independent audit: 43,851 checks, zero discrepancies. Repaired-base numeric
paired reversal is 7/20; categorical paired reversal is 8/20, and broken-link
news is classified unchanged in 2/20. See
../../results/robustness_development_v2/qwen3_32b/review.md.

New-news numerical discrepancies versus repaired base average 11.19 points for
renaming, 14.31 for paraphrase, and 11.49 for identical-text resampling. Excess
edit discrepancies have intervals spanning zero; these results do not identify
an edit-specific causal penalty or establish invariance. These remain unvalidated
development materials, and the audit verifies execution/scoring rather than
independent material validation.

Automatic audit and paper-completion job: **31445467**, after finalizer31444440.
Its reviewed code and canonical paper snapshot are fixed in
`finish_pipeline/config.json`. It audits all three models, generates all-variant
tables, builds an isolated paper copy, verifies the nine-page main-paper boundary,
and publishes only against the unchanged canonical source snapshot. Persistent
backups and a transaction journal preserve displaced files. A changed paper,
failed audit, incomplete cohort or failed build stops publication; the detailed
outcome is recorded in `finish_pipeline/status.json`. This stage makes no model
or frontier API calls. The overnight job stopped at an audit metadata assumption. The repaired
completion stage subsequently passed all three audits and published both sections;
see `finish_pipeline_metadata_repair/status.json` and
`../../results/EXP1_STATUS_2026-09-22.md`. All6,720 responses are complete and valid.
