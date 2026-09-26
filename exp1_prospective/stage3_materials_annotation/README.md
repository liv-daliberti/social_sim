# Stage 3 materials annotation

This directory contains the human materials review for the synthetic evidence in
Experiment 1, Stage 3. The unit of analysis is an evidence packet.

## Final status

The paper analysis is frozen. Nine adult volunteers completed all 18 items. The disclosed
response-pattern rule excluded one reviewer who selected the same direction on 17
of 18 balanced items, leaving eight quality-eligible reviewers. This exceeds the
six-reviewer target for the descriptive materials check. No item was removed or
relabeled, and the excluded packet remains in the all-completer sensitivity
analysis.

The canonical analysis artifact is
`data/exports/final_summary_current.json`. It records:

- 108/144 exact-direction judgments (75.0%) in the eight-reviewer analysis;
- a 17/18 panel-majority match with the registered key;
- 124/144 premise-usability judgments (86.1%); and
- 115/162 exact-direction judgments (71.0%) when all nine completers are retained.

`data/exports/agreement_current.json` contains the paper-facing agreement export.
Timestamped `interim_summary_*` and `status_*` files preserve collection history;
they are not the authority for the closed analysis and retain then-current roster
fields by design.

## Design

The review uses 18 frozen packets from 18 distinct markets:

- six intended to increase YES;
- six intended to decrease YES; and
- six intended to have no material effect.

Every reviewer rated every packet in a separately randomized order. The five
judgments cover conditional direction, confidence, clarity, plausibility, and
whether the text is usable as an explicit hypothetical premise. Each item shows
the market question, resolution criteria, and frozen June 10 background. Reviewers
were instructed to treat June 10, 2026 as the current date and not use later
information or outside assistance.

A self-review pilot removed semantic-directness, source-verification, and optional
free-text questions before registered collection. Practice responses are stored
separately and never enter the registered analysis. The study evaluates materials
comprehension and premise usability; it does not compare participant treatments.

## Reproduce the final summary

From the repository root:

```bash
python exp1_prospective/stage3_materials_annotation/make_final_summary.py \
  --responses exp1_prospective/stage3_materials_annotation/data/exports/registered_20260827T204456Z.csv \
  --status exp1_prospective/stage3_materials_annotation/data/exports/status_20260827T204456Z.json \
  --private-key exp1_prospective/stage3_materials_annotation/generated_v6/private_key.jsonl \
  --policy exp1_prospective/stage3_materials_annotation/generated_v6/posthoc_exclusions.json \
  --output exp1_prospective/stage3_materials_annotation/data/exports/final_summary_current.json \
  --protocol-version stage3_materials_annotation_v8
```

The generator fails closed unless the response CSV agrees with the status export,
every completed packet covers all 18 items exactly once, at least six reviewers
remain after uniform application of the documented quality rule, and no partially
completed packet remains. It hashes the response and status inputs, retains
excluded responses for sensitivity analysis, and removes obsolete recruitment-gate
fields from the final artifact.

The private key is required to reproduce agreement against the registered direction
and must not be distributed with public reviewer materials.

## Frozen materials and interface

`build_packet.py` deterministically selects and blinds the original v6 item set and
records hashes in `generated_v6/manifest.json`. Each item includes the summary from
its frozen June 10 initial forecast. Earlier generated directories preserve pilot
and superseded materials.

The browser application in `review_site/app.py` implements the reviewer information
screen, randomized item order, autosaving, completion flow, and separate practice
cohort. It stores reviewer codes, consent timestamps, item responses, and response
timing; it does not request names, email addresses, demographics, or sensitive
information.

The paper freeze contains all nine originally issued reviewer codes and all nine
completed packets. A separately configured `annotator_10` code is a post-freeze
extension: do not append its responses to the frozen CSV or recompute the reported
paper result with them.

## Post-freeze tenth reviewer

`annotator_10` is a collection-only extension and is not part of the paper freeze.
Its private code is stored only in the ignored, owner-readable handoff file
`generated_v6/annotator_10_private_code.txt`. To activate it on the existing Render
service, set `STAGE3_ANNOTATION_REVIEWER_10_CODE` to that value and redeploy the
survey branch. The application derives a stable 18-item order from the frozen packet
seed; the admin status then reports 10 expected reviewers and 180 expected ratings.
Do not use a later ten-reviewer export as input to the paper generators.

## Data handling

The post-collection exclusion policy is frozen in
`generated_v6/posthoc_exclusions.json`. It records the rule, decision timing,
initial trigger, post-task debrief context, and the requirement to retain raw
responses. The paper reports both the quality-eligible analysis and the
all-completer sensitivity tally. Counterfactual packets were reviewed offline and
were never posted to markets or used for trading.
