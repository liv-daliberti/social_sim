# Reviewing fictional forecasting scenarios

These are assistant-authored **development candidates**, not independently
validated materials. Generating this packet does not validate them. All named
people, organizations, quantities, dates, and causal relationships belong to
fictional exercises. Your judgments should use the supplied information, not
outside facts or real event outcomes.

You must not have seen target-model responses, the author's intended answers,
private mappings, or another reviewer's answers before completing your review.
Do not open the experiment's source materials or implementation while reviewing.
If you have already seen any of those, tell the study coordinator before starting.

Work independently. You should receive only one `review_batch`, containing one
version of each scenario. Do not obtain other batches or compare related versions.
There is no requirement to use every answer choice equally often, and there is no
presumed correct answer in the reviewer file. It is useful to identify uncertain,
implausible, or underspecified cases.

For each row, read `scenario_text` as the information available before the message.
Then read `new_message` as newly supplied information. Judge the effect of the
message relative to that same scenario before the message. Do not compare the row
to other scenarios. No forecast from a target model is needed or provided.

Fill the following blank columns; preserve `review_id` and all scenario text.

- `direction_judgment`: Enter `increase`, `decrease`, `no_material_effect`, or
  `unclear`. Judge how the message changes the probability that the question
  resolves YES. Use `unclear` if you cannot determine a defensible direction.
  `no_material_effect` is a judgment of negligible effect, not an expression of
  uncertainty. A small effect with a defensible direction may still be an increase
  or decrease. No numerical effect-size threshold is imposed by this review.
- `relevance_judgment`: Enter `relevant`, `irrelevant`, or `unclear`. Does the
  message bear on the resolution through a relationship supported by the supplied
  information? A relevant message need not settle the outcome.
- `confidence_1_to_5`: Rate confidence in your direction judgment: 1 = very low,
  2 = low, 3 = moderate, 4 = high, 5 = very high.
- `ambiguity_judgment`: Enter `none_identified`, `possible`, or `substantial`.
  Consider missing links, conflicting statements, unclear comparisons with the
  baseline, uncertainty about what the new number measures, and multiple
  reasonable interpretations. Explain identified ambiguity in `reviewer_notes`.
- `alternative_causal_paths`: Describe any plausible additional route from the
  message to the outcome that is supported by the supplied scenario and could
  change your answer. Write `none identified` if you find none. Do not invent
  unstated facts; if a route needs an additional assumption, name that assumption.
- `reviewer_notes`: Briefly explain your direction judgment, the intermediate
  connection(s) it uses, and any concerns about plausibility, directness, or clarity.
  This is a free-text field; a concise explanation is sufficient.

Return your completed file to the coordinator through the agreed study process.
Do not discuss candidate judgments with other reviewers before submitting.

## Coordinator instructions

The complete CSV/JSON contains 80 rows in four batches of 20. Each batch contains
one version of every family, with versions mixed across families. Assign each
reviewer to only one batch. For a prospective three-rater review, recruit at least
three independent reviewers per batch; do not have one person complete multiple
batches. Remove rows from all other batches before distributing a reviewer file.
An assignment to a batch does not imply that annotation has been collected.

Distribute this README and the selected public rows only. Do not distribute
`human_review_candidates.private.json`, raw development materials, code, target
outputs, author-proposed labels, or annotation summaries before ratings are final.
The private mapping joins anonymous review IDs to source families and variants for
analysis after independent collection. It includes unvalidated author labels.

Record a reviewer ID separately, obtain any study consent required by your actual
recruitment process, and preserve raw submissions before any discussion or
adjudication. Specify compensation, retention rules, and analysis before collecting
ratings. No retention gate or human-validation status is implemented by this
export. In particular, do not retain candidates based on target-model performance.

These 20 families remain development items after review. For a confirmatory
experiment, create and independently review a fresh held-out family pool, freeze
its retained text and protocol before target-model responses, and keep it separate
from this pilot. This packet alone cannot support a claim of independent
validation or confirmatory evidence.

The public JSON and coordinator mapping record the source material SHA-256. The
export uses seed 20260921 for deterministic anonymous IDs, batch assignment, and
row order. Regenerate it from the repository root with:

```bash
python -m exp1_prospective.context_reversal.review_packet
```

Re-exporting identical content is safe. A changed source or export seed requires a
new output filename so that the previous review packet is preserved.
