# Frozen protocol: human annotation of Stage 3 synthetic evidence

Version: `stage3_materials_annotation_v6`

Freeze date: 2026-08-20

## Objective

Validate whether a balanced sample of Stage 3 synthetic evidence packets has the
registered conditional direction, is clear and plausible, and can be used as an
explicit hypothetical premise. Ratings characterize the packets. No claim is
made about individual annotators or a population of human forecasters.

## Annotators and consent

Use seven volunteer adults, unpaid, recruited from the authors' academic networks.
Annotators must not have generated the packets or seen their intended labels,
rationales, mechanism targets, or model responses. Participation is voluntary;
the task page states that annotators may withdraw at any time.

The form requests no names, email addresses, demographics, IP addresses, free
text, or sensitive personal information. Reviewer IDs are arbitrary codes.

## Items

The source is the frozen 900-packet file
`exp1_prospective/data/counterfactuals/counterfactuals_2026-06-10.jsonl`.
Resolution criteria come from the matching 100-market June 9 selection freeze;
its market-ID set is identical to the packet's 100-market set.
Selection is deterministic under seed
`stage3_materials_annotation_v1_20260820` and uses no model update outcomes or
human judgments.

The packet contains 18 distinct markets and one evidence item per market: six
`pro_H1`, six `anti_H1`, and six `orthogonal`. Within each selected market and
direction, one of the three generator slots is hash-selected. Reviewers see the
question, frozen resolution criteria, the matching June 10 initial-forecast
summary, synthetic headline, and synthetic report. The summary is copied
verbatim from `structured_forecast.rationale` in the frozen DeepSeek-V4-Pro
forecasts used to generate the counterfactuals. The builder requires one shared
rationale per market and verifies that the selected packet's generation-time
probability matches a forecast row for that market. Source URLs, private
metadata, intended labels, and model responses are withheld.

Self-review showed that some evidence could not be interpreted directionally
without the pre-report information available to the forecasting model. The
background summary was therefore added to every item before registered
collection. This repair standardizes the information state across items and
does not reveal the intended evidence direction.

Each annotator rates all 18 items. Item order is independently hash-randomized by
reviewer. Because each market appears once, no reviewer can infer a label by
comparing matched variants.

## Temporal frame

The consent page and every item state that June 10, 2026 is "today." Annotators
must judge the materials from that point in time and must not use events,
outcomes, or information learned after June 10. This matches the date on which
the synthetic reports and model forecasts are conditioned. The instruction was
made explicit during self-review, before any registered annotation was collected.

## Annotation fields

1. **Conditional direction:** If the report were accurate, would YES become more
   likely, less likely, remain materially unchanged, or is the direction
   ambiguous?
2. **Direction confidence:** 1--5.
3. **Clarity:** 1--5.
4. **Real-world plausibility:** 1--5, explicitly ignoring whether the synthetic
   event actually occurred.
5. **Usable hypothetical premise:** yes / unclear / no.

These five judgments are the complete form. Semantic-directness,
source-verification, and free-text questions were removed after the self-review
pilot, before any registered annotation was collected, because they did not
enter the retention gates and added substantial per-item burden.

## Frozen scoring

The registered conditional answer is:

- `more_likely` for `pro_H1`;
- `less_likely` for `anti_H1`; and
- `no_material_effect` for `orthogonal`.

Ambiguous responses remain in all denominators. No annotator or item is excluded
after inspection. Missing or duplicate reviewer-item rows fail validation.

Report:

- exact-direction agreement overall, by registered direction, and by item;
- Fleiss' kappa for the four conditional-direction responses;
- medians and full distributions for confidence, clarity, and plausibility;
- usable-premise judgments.

## Item retention gates

An item passes materials validation only when:

1. at least six of seven annotators select its registered conditional direction;
2. median direction confidence is at least 4/5;
3. median clarity is at least 4/5;
4. median plausibility is at least 3/5; and
5. at least six of seven annotators mark it usable as a hypothetical premise.

The sampled packet passes globally only if at least 15 of 18 items pass and
Fleiss' kappa for conditional direction is at least .70. All failures remain in
the report. These gates validate a sample of materials; they do not retroactively
relabel packets or prove that all 900 packets satisfy the same properties.

## Research-ethics scope

This activity consists solely of judgments about synthetic, model-generated
text. Responses are used to characterize the items, not the annotators. The task
does not collect sensitive or identifying information and does not intervene on
participants. Under the authors' institutional guidelines, it does not
constitute human-subjects research; consequently, no IRB review is sought.

This statement applies only to the materials-annotation protocol above. A study
that assigns people to premise-acceptance versus source-verification conditions
and analyzes differences in their behavior would be a different protocol and
would require a separate institutional assessment.
