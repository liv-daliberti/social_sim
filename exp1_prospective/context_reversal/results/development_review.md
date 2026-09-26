# Development pilot review after the first two completed models

Llama-3.1-70B and Qwen3-32B each produced 960 distinct, valid probability records.
An independent read-only audit matched raw JSON probabilities and planned response
keys to the analysis summaries. No parse or truncation discrepancy was found.
The Qwen2.5-72B attempt produced no responses because engine startup encountered
an Inductor/Triton cache race and stalled until the scheduler timeout. Its missing
outcomes are infrastructure missingness, not evidence of poor model behavior.

| Model | Correct sign | Both directions correct | Broken-link mean absolute movement |
|---|---:|---:|---:|
| Llama-3.1-70B | 78/120 (65.0%) | 22/60 (36.7%) | 10.5 pp |
| Qwen3-32B | 77/120 (64.2%) | 19/60 (31.7%) | 13.1 pp |

Both completed models fail the prespecified 2 pp broken-link stability criterion.
All 240 no-news branches per model exactly reproduce the context-specific prior;
repeated baseline information changes 1/240 Llama responses and 9/240 Qwen
responses. These news-induced movements are not explained by ordinary
re-elicitation drift under this prompting protocol.

There are 20 unvalidated authored families with three repeats. The 60 paired
trials are not 60 independent families. Bootstrap uncertainty is over families.
The full descriptive intervals and valid denominators are in each model summary.

## Development concerns

Baseline saturation mechanically prevents the expected signed movement in
19/120 Llama trials and 12/120 Qwen trials. These remain failures in unconditional
scoring. Inspecting why fictional uncertain events receive initial probabilities
of exactly zero or one is an instrument-development task.

Independent review is still required. In dev_06, cartons transferred out of the
buffer have no specified destination; a reader could infer delivery to the
packing line. In dev_07, water leaving usable storage is not explicitly described
as bypassing the turbines. These ambiguities can alter the intended direction.
Do not retrospectively exclude or relabel these families based on model outcomes.
Any material revision belongs in a separately versioned development cohort.

The clinic-queue family (dev_13, repeat 0) illustrates a selective-updating concern:
Qwen moves from 50% to 75% under the helpful link, 70% under the harmful link,
and 75% under the unrelated link; no-news and repeat controls remain at 50%.
This is a useful diagnostic case, not a basis for a general model claim.

These exploratory results do not yet support a stronger confirmatory paper claim.
Complete the prescribed model roster, investigate instrument limitations, and
independently validate a fresh frozen family pool before confirmatory inference.
