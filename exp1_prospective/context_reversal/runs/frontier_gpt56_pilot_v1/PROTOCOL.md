# GPT-5.6 exploratory extension

This small extension was authorized after inspection of the three local-model
pilot results. It is not a new confirmatory sample. The original 20 authored,
unvalidated families remain unchanged, including previously identified ambiguities.
All four contexts are retained. Only original repeat 0 is used: 80 separate
context-specific baselines and 240 independently branched updates, 320 calls total.
No family is selected or discarded based on local or frontier performance.

The requested GPT-5.6 alias resolves to GPT-5.6 Sol in official documentation.
The project lists and exposes `gpt-5.6-sol`; its alias metadata lookup returned
404, so requests use that canonical model ID. Calls use the official OpenAI
Responses API, low reasoning effort, at most 2,048 output tokens including
reasoning, no tools, no web retrieval, and `store=false`. Strict JSON output has
one numeric probability field in [0,1]. No temperature or seed is supplied.
These API settings differ from local-model decoding; comparisons are exploratory
and do not isolate architecture or parameter scale.

Each update repeats exactly its context-specific prior and the frozen context.
New-news, no-news, and repeated-baseline-information branches are independent.
The omitted-context condition has no prescribed sign or zero-effect target.
Scoring uses the existing analyzer: 40 directional trials, 20 paired reversals,
20 broken-link trials; zeros and missing/invalid responses fail unconditional
directional scoring. Magnitude summaries report valid coverage without zero
imputation. Uncertainty uses 2,000 whole-family resamples with the existing seed.
The original +/-2 percentage-point broken-link stability diagnostic is retained.

The cap is 320 logical inference requests and an estimated $15 total ceiling.
No model substitutions or performance-based retries are allowed. Raw probabilities,
exact prompts, response model IDs, usage, and failures are preserved. Credentials
remain in process memory and are excluded from manifests and output records.

Run metadata and the frozen plan are in this directory. The frontier report is
separate from the fixed three-local-model aggregate. All results remain development
diagnostics until a fresh independent material-validation and confirmation study.
