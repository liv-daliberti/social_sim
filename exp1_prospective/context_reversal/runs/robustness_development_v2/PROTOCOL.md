# Controlled robustness development, version 2

Authorized after the original 20-family pilot. No new frontier inference is part
of this development run. Original materials, labels, outputs and reports remain
immutable. Repairs and all derived variants are a separately identified version.
The existing human review is not repeated or relabeled as covering new material.

## Materials and local runs

Retain all 20 parent families, including prior successes and failures. Each has
four versions: repaired base, consistent entity renaming, meaning-preserving
paraphrase, and an identical-text resample. Each version has positive, negative,
broken and masked contexts. Each context receives its own numerical baseline and
three independent update branches: new evidence, no news, and repeated evidence.
Identical evidence is shared across the relationship contexts within a version.
A separate direction companion shows no numerical forecast and asks increase,
decrease, unchanged or unclear for each context/message. The three original local
checkpoints (Qwen2.5-72B, Llama-3.1-70B and Qwen3-32B) are retained; development
uses their existing constrained, thinking-disabled settings at temperature 0.7,
top-p 1 and 128 output tokens. No model is selected by development performance.

There are 320 context units, 1,280 numerical records and 960 direction records
per model. Variants are observations within 20 parent families, not 80 independent
families. The identical-text resample uses distinct request IDs/seeds under the
existing deterministic seed derivation. Its discrepancy from base estimates
sampling variability at the same temperature; it is not an additional family.

Explicit routes repair previously unspecified transfer destinations. Every signed
and broken context contains the same positive, negative and side-channel roles;
only actor-to-role assignments change. Row order is counterbalanced. Broken-link
items must not be uniquely marked by a recurring phrase absent from signed items.
Every repair, naming map and paraphrase is retained in material provenance.

## Frozen scoring

Primary numeric sign: new-news minus own-context baseline is strictly positive
or negative as appropriate. Zero movement fails. Paired reversal requires both
signed contexts correct. Broken-link stability retains the original requirement:
signed-mean 90% interval strictly within +/-2 probability points AND the upper
95% mean-absolute-movement bound below 2 points, with complete measurements.
No-news and repeated-news controls are scored separately.

Naming/paraphrase invariance compares the update (not just the final probability)
against repaired base. The descriptive per-trial tolerance is 2 probability
points. Report baseline differences separately, exact categorical agreement,
and joint categorical correctness; stable wrong answers are not correct answers.
Also report absolute edit discrepancy minus absolute identical-text-resample
discrepancy on observed common measurements. This reference does not isolate a
causal edit effect with a single resample.

All binary denominators retain planned trials, including missing/invalid outputs
and unclear judgments against determinate targets. Magnitudes never impute missing
values. Masked new-news directions have no target. Descriptive intervals use
2,000 whole-parent-family bootstrap draws, seed 20260921, carrying every variant,
context and branch together. Report all models and all versions. Do not relabel,
exclude, repair or regenerate outputs in response to their correctness.

## Lexical and execution checks

Check identical evidence across relationship contexts, exact renaming maps,
base/resample visible equality, preserved paraphrase quantities, prompt-label
separation and word-inventory counterbalancing. Report news-only, full-text
unigram/bigram and character-ngram baselines with whole parent families held out;
variants of the same parent never cross folds. Report balanced relevance accuracy
and three-class direction accuracy. Stronger lexical baselines remain visible.

The criteria for freezing the subsequent evaluation concern material clarity,
absence of accidental label leakage, complete controls, faithful execution and
validated scoring. Target-model success rates are not a selection criterion.
