# Small GPT-5.6 follow-up

This extension uses GPT-5.6 Sol (`gpt-5.6-sol`), the canonical model associated
with the requested `gpt-5.6` alias. Availability was verified through the official
OpenAI API using the provided project credential. No credential is saved here.

The 80-unit plan keeps every original family and context but uses one repeat:
320 forecast calls. The local three-repeat experiment remains the separate
completed `development_v1` run. Read `PROTOCOL.md` for settings, budget and limits.

The response output is `responses/frontier_gpt56_pilot_v1/gpt_5_6_frontier.jsonl`,
relative to the context-reversal directory. Its companion manifest records usage,
returned model IDs, configuration, code/input hashes and planned response coverage.
The completed summary is written separately under `results/frontier_gpt56_pilot_v1/`.
The run is complete: **320/320 valid records**, 320 attempted forecast calls,
and an estimated token cost of **$0.712128**. `completion.json` records final
coverage, model identity, usage, and provenance. The estimate is not a billing
invoice.

GPT-5.6 has **37/40 correct directions** and **17/20 paired reversals**. Mean
absolute broken-link movement is **0.5 percentage points** and meets the original
2-point stability criterion on this development set. All 80 no-news and all 80
repeated-news controls have zero movement. See
[the complete analysis](../../results/frontier_gpt56_pilot_v1/summary.md) and
[the matched local comparison](../../results/frontier_gpt56_pilot_v1/comparison.md).
The latter selects repeat zero from each original local response file without
altering the completed three-repeat local analysis. This is a descriptive
comparison of differently configured deployments on unvalidated materials.

An [independent raw-response audit](../../results/frontier_gpt56_pilot_v1/independent_audit.md)
passed 2,902 checks of the plan, prompts, context-specific priors, records, hashes,
usage, cost, and recomputed outcomes. All failed cases remain in the reported
denominators; destination ambiguities are documented for future material review.
