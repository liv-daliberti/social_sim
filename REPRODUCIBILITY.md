# Reproducibility and validity audit

Last audited: **2026-08-24**. This document identifies the numerical authority,
inference unit, and fail-closed check for each paper experiment. `make review`
runs maintained lint and unit/invariant tests; heavyweight model training and
full bootstrap recomputation are intentionally separate.

## Evidence map

| Experiment | Frozen numerical authority | Inference/validation boundary | Audit result |
|---|---|---|---|
| 1: prospective updates | `exp1_prospective/data/results/consistency_report_2026-06-17.json` | Markets are bootstrap clusters; all packets and repeated initial runs stay together | 20,000-draw artifact exactly reproduced; source checksum and frozen summaries cross-checked |
| 2: Coin City selection | `exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/analysis/multimodel_results.json` `analysis/symbol_context_model_comparison_20260824.json`, and `analysis/reference_selection_results.json` under the same data root | Episodes are paired within model; the primary analysis has six deployments and the arbitrary-symbol extension requires the complete 15-model roster | The primary design validator and analysis reproduce; the final symbol builder rejects missing arms or non-1,250-record files before writing the paper authority |
| 3: Coin City training transfer | `exp3_training_transfer/coin_city_structural/reports/registered_results.json` | Training seed is the top-level unit; five stochastic draws are averaged within task | Exact 24-job/48-file roster validates 207,360 scored rows; Qwen3-8B is the main result and stochastic Qwen3-4B/Llama-8B are appendix comparators |
| 4: historical markets | `exp3_training_transfer/polymarket/reports/exp3b_model_roster_summary.json` and `exp3_training_transfer/polymarket/PROTOCOL.md` | Three training seeds per model; connected-component family bootstrap over event/series/template links | Qwen3-8B main result and Qwen3-4B/Llama-8B appendix comparators validate on the exact 1,024-task locked test |

The Polymarket Qwen3-8B/Llama-8B replacement roster completed all six training
jobs and both dependency-locked tests. Qwen3-8B is the main paper result;
Qwen3-4B and Llama-8B are appendix comparators. All three improve significantly
over their own untrained checkpoints. No trained model conclusively beats the raw
market price, and the trained Qwen3-8B and Llama-8B means have higher Brier error
than the train-only Platt-calibrated market baseline.

## Important reconstruction rules

1. The Experiment 1 row-level collection directories contain later retries and
   top-ups. Re-evaluating their current contents is a new analysis, not a
   bit-for-bit paper reconstruction. Start from the frozen June 17 report.
2. Do not treat packets, repeated decodes, or multiple rows from one market as
   independent top-level replications. The maintained analyses encode the
   clustering described in the manuscript.
3. Partial model rosters do not silently render confirmatory tables. Current
   analyzers reject missing models/arms/seeds, duplicate jobs, task-universe
   drift, malformed identities, and insufficient bootstrap draws.
4. Experiment 2's 56,250 total counts response records, not matched parseable
   cases. Tables report complete-case `n`; Llama-3.1-8B has `n=242` at `k=0`.
   Arithmetic-only reparsing makes no new model calls, and unresolved outputs
   remain missing.
5. Experiment 2's probe targets differ in what the displayed evidence can
   determine. `exp2_v2/biased_news/analysis/compute_bayes_ceilings.py`
   derives the Bayes-optimal ceiling for each: regime AUC 1.000 under a correct
   semantic cue and .979 under an arbitrary label, full-slope R^2 .992, and
   within-regime residual R^2 .000 at k=0 and .009 at k=4. The residual is a
   probe-specificity control, not a target a model could hit; a null there is a
   property of the design and must not be reported as a limit on a checkpoint.
6. The complete registered Coin City roster includes Llama-3.1-8B. Its
   stochastic point estimate favors episode-matched training but its interval
   crosses zero. Its cue-use and evidence-override intervals exclude zero. The
   manuscript keeps Qwen3-8B in the main body and reports stochastic Qwen3-4B
   and Llama-8B comparators in the appendix; temperature-zero endpoints remain
   provenance rather than manuscript results.
7. There is one physical manuscript tree, `paper/`; all source and output
   workflows target it directly.

## Reviewer commands

```bash
# Static checks and all lightweight maintained suites
make review

# Experiment 1 deterministic 20,000-market-bootstrap artifact
python exp1_prospective/agent/evaluate_clustered_uncertainty.py

# Experiment 2 frozen design, without model API calls
cd exp2_v2/biased_news
python eval/validate_coin_city_stable_relationship_claude_n250.py
cd ../..

# Canonical anonymous ICLR PDF and exact compiled-input source archive
make paper
make submission OUTPUT=/tmp/iclr-submission.zip
```

The complete Experiment 3/4 workspace includes ignored checkpoints, score files,
and raw corpora. `hf_release/` is the anonymous distribution boundary; its
builder records checksums and strips local absolute paths. Git intentionally
retains source, manifests, compact frozen aggregates, paper tables, and tests
rather than multi-gigabyte raw/runtime payloads.
