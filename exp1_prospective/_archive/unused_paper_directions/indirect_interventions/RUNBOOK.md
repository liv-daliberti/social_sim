# Indirect-intervention execution runbook

## Current checkpoint (2026-08-13)

- The frozen direct-packet Experiment 1 data are untouched.
- The original six-reviewer UI pilot is retained only for application mechanics.
  Its repeated-exposure assignment is explicitly invalid and now fails the
  assignment audit; it must not be used as evidence.
- The production schemas, prompts, generator, lexical audit, distributed blinded
  assignments, development/core validation gates, shortcut-baseline gate,
  instrument freezer, target runner, and market-clustered analysis are implemented.
- The frozen 100-market core and its deterministic run-0 canonical event models
  resolve completely. The 400 direct/orthogonal/null controls have been built.
- Twenty-five unit tests pass.
- GPT-5.6 Sol generated the complete disjoint readable-v3 development pool: 60
  families for 20 markets. The fail-closed audit passes with zero errors and zero
  warnings; see `reports/development_candidate_audit_readable_v3.json`.
- The original pool is preserved but superseded after a readability review found
  73 warnings across 29 families. Readable v3 limits evidence to 20--45 words,
  directional connections to 20--40 words, and broken connections to 15--28
  words. Connections have at most two short sentences, no semicolons, and no
  all-caps abbreviations.
- During development calibration, one market twice produced the forbidden cue
  `no` in evidence. Both drafts were rejected, the prompt gained an explicit
  mechanical self-scan, and the regenerated family passed without weakening the
  lexical gate.
- Human editing is 0/60, so the blinded rating packet is intentionally blocked.
  No core prompt has been sent, no production human rating has been collected,
  and no target model has been called.

The next operation is independent human editing of all 60 development families.
After that provenance is complete, blinded development ratings can begin. Core
generation later sends each core question, rules, and stored canonical initial
event model. Neither stage sends target-model update outputs. The current
generator configuration is `gpt-5.6-sol`, medium reasoning effort, and a
12,000-token output ceiling.

Authentication deliberately requires `GPT56_AZURE_API_KEY`, which must be scoped
to the `liv` project hosting `gpt-5.6-sol`. The forecasting-agent project key is
not interchangeable. Requests set `store=false`.

## Stop/go sequence

Every command is fail closed. Do not create a gate report by hand.

### 1. Generate the disjoint 20-market development pool

After approval for the external Azure transfer:

```bash
python exp1_prospective/indirect_interventions/scripts/build_development_candidates.py
```

This makes 20 generator calls and writes 60 families. It is append-only and
resumes only at complete three-family market boundaries.

The completed pool can be re-audited with:

```bash
python exp1_prospective/indirect_interventions/scripts/audit_lexical_leakage.py \
  --development
```

### 2. Human edit before independent ratings

Because only one generator deployment is currently available, a human editor
must inspect every family, correct bridge minimality and plausibility problems,
and set:

```json
{
  "status": "human_edited",
  "editor_id": "deidentified-editor-code",
  "edited_at": "ISO-8601 timestamp"
}
```

The editor cannot serve on either validation panel. The editor must not inspect
target-model output. Give the editor `DEVELOPMENT_EDITOR_HANDOFF.md`, which
contains the frozen input hash, per-family checklist, allowed fields, and audit
conditions.

### 3. Blind and rate the development pool

```bash
python exp1_prospective/indirect_interventions/scripts/build_review_packet.py \
  --development \
  --candidates exp1_prospective/indirect_interventions/data/development/development_candidates_readable_v3.jsonl \
  --packet exp1_prospective/indirect_interventions/data/review/development_review_packet.json \
  --reviewers exp1_prospective/indirect_interventions/data/review/development_reviewers.json \
  --private-key exp1_prospective/indirect_interventions/data/review/development_review_key.jsonl \
  --manifest exp1_prospective/indirect_interventions/data/review/development_review_manifest.json \
  --full-reviewers 27 --shortcut-reviewers 9
```

Point the existing Flask app at those packet, assignment, and database paths via
`INDIRECT_REVIEW_PACKET`, `INDIRECT_REVIEWERS`, and `INDIRECT_REVIEW_DB`.
The browser shows only the question, short evidence, and a section labeled “How
it connects.” The private event model is never shown, and full resolution rules
are collapsed by default.
The packet requires 720 independent ratings: 540 full-context and 180 masked.
It uses 27 full-context plus 9 shortcut reviewers, with exactly 20 markets per
person. The allocator fails closed unless every task has three raters and no
reviewer sees two families, controls, or bridge variants from the same market.

```bash
python exp1_prospective/indirect_interventions/scripts/validate_development_annotations.py
```

Core generation remains locked unless this produces
`ready_for_core_generation=true`.

### 4. Generate, human-edit, and audit the core

```bash
python exp1_prospective/indirect_interventions/scripts/build_candidates.py
python exp1_prospective/indirect_interventions/scripts/audit_lexical_leakage.py
```

Core generation makes 100 calls and writes 300 families. A non-rating editor must
again inspect and mark all families. The audit then requires exact bridge framing,
valid signed chains, forbidden-cue absence, balanced surface valence, complete
editor provenance, 300 families, and 400 controls.

### 5. Collect the core human panels

```bash
python exp1_prospective/indirect_interventions/scripts/build_review_packet.py
```

Default assignments spread 4,800 ratings over 120 reviewers:

- 3,900 full-context ratings (1,300 cases times three);
- 900 evidence-only masked ratings (300 cases times three).

Record actual compensation and institutional review/consent handling. Export the
de-identified annotations, then run:

```bash
python exp1_prospective/indirect_interventions/scripts/validate_core_annotations.py \
  --compensation 'RECORD ACTUAL RATE'
```

The gate requires exact coverage, Fleiss' kappa at least .70, and at least 80
markets with all three families and all four controls retained.

### 6. Run all bridge-masked baselines

```bash
python exp1_prospective/indirect_interventions/scripts/run_shortcut_baselines.py \
  --embedding-predictions PATH \
  --evidence-instruction-predictions PATH \
  --question-evidence-predictions PATH
```

Grouped sentiment, TF--IDF, and frozen hashing baselines run locally. The required
frozen sentence-embedding and instruction-classifier predictions must be supplied
for exactly the retained families. The freeze remains locked if a required
baseline is absent or any bridge-masked macro-F1 exceeds .40.

### 7. Freeze before opening any target output

```bash
python exp1_prospective/indirect_interventions/scripts/freeze_instrument.py
```

The freezer hashes candidates, controls, prompts, schemas, human annotations,
validation, baselines, canonical forecasts, and the randomized 16-prompt battery.
It refuses to overwrite a different frozen artifact.

### 8. Run targets and analyze

```bash
python exp1_prospective/indirect_interventions/scripts/run_updates.py --model MODEL
python exp1_prospective/indirect_interventions/scripts/analyze.py
```

The runner rejects mixed 0--1/0--100 scales, unequal H1/headline probabilities,
missing canonical forecasts, and any inconsistent freeze hash. The analysis uses
all trials for triplet accuracy, conservatively scores invalid pairs as zero in
the reversal contrast, resamples markets in 10,000 bootstrap replicates, and
Holm-adjusts model-level reversal tests.

## Manuscript rule

Do not add a mechanism-conditioned result to the ICLR draft until the frozen
target report exists. If the gates fail, retain the current narrow
direction-selective-updating claim and report the failed instrument transparently.
