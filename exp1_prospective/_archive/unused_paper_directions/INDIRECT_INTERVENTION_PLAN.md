# Experiment 1 hardening: indirect, mechanism-reversal interventions

## Purpose

The frozen Experiment 1 instrument establishes direction-selective forecast
revision under mostly direct synthetic evidence. It rules out indiscriminate
reaction to topically related text and tests propagation from the stated
hypothesis to the headline forecast. It does **not** rule out a capable
entailment/relevance policy that classifies a packet as favorable, unfavorable,
or irrelevant and applies a learned probability shift.

This plan closes that identification gap as far as a behavioral experiment can:

1. narrow the current claim to exactly what the existing instrument identifies;
2. add indirect interventions whose sign requires a validated multi-edge causal
   bridge;
3. hold evidence text literally fixed while the bridge is reversed or broken;
4. obtain independent, blinded human labels for direction, plausibility,
   strength, directness, and causal-chain validity;
5. benchmark explicit lexical and stateless sign-classification shortcuts; and
6. use fail-closed gates so weak materials or ambiguous results cannot support a
   causal-mechanism claim.

Even a successful behavioral result will not prove a particular internal
representation. The defensible conclusion is **mechanism-conditioned causal
composition in forecasting updates**, not direct observation of a latent causal
model.

## Threat model

The replacement instrument must address all of the following alternatives.

| Threat | Why the frozen instrument is vulnerable | Required fix |
|---|---|---|
| Lexical polarity | Many pro/anti headlines directly sound favorable or unfavorable | Identical evidence under opposite bridge contexts |
| Direct entailment | Question plus packet may directly imply YES/NO | At least two signed causal edges; forbid outcome paraphrases |
| Generic fixed shifting | A classifier can attach a learned update magnitude to a sign | Reversal, broken-bridge, and literal-null conditions |
| Topical reactivity | Any same-domain text may induce movement | Matched broken-bridge and orthogonal controls |
| Prompt-induced drift | Merely asking for an update can move the forecast | Format-identical no-information control |
| Generator self-validation | All 900 existing plausibility values are the prompted constant 4 | Independent blinded human annotation; generator ratings discarded |
| Selective scoring | EHC/HFC currently condition on movement of at least 3 pp | New primary metrics score every trial, including abstention |
| Packet dependence | Packets are nested within markets | Market-clustered inference with mechanism families nested inside |
| Post-hoc item selection | Difficult items could be chosen after observing target models | Human-only gates and frozen hashes before target-model calls |

## Claim correction before new model calls

The manuscript should immediately describe the frozen result as
**direction-selective, internally coherent updating under semantically direct
packets**.

The existing evidence supports:

- relevant packets induce larger revisions than topical orthogonal packets;
- the stated H1 posterior and headline forecast usually move together; and
- frontier systems rarely react to the orthogonal controls.

It does not by itself support:

- exclusion of a strong entailment/relevance classifier;
- recovery of the correct real-world causal structure; or
- existence of a particular internal representation.

Accordingly, “disfavors surface matching” should be limited to **undifferentiated
topic matching**, and the stronger mechanism language should be reserved for the
hard intervention result. Experiment 2 remains the paper's known-ground-truth
test of latent-structure recovery.

## Frozen sample and development boundary

- **Evaluation markets:** the existing 100-market frozen core.
- **Development markets:** 20 non-core extension or newly sampled markets.
- **Initial contexts:** one canonical, parseable initial forecast per
  model–market, selected by a deterministic rule before packet generation.
- **No web access during updating.**
- Development markets may be used to refine generation prompts, the annotation
  interface, parsing, and token limits. No target-model output on a core market
  may be inspected before the instrument is frozen.
- Candidate rejection or regeneration on core markets may use only human
  annotation and automated text audits, never target-model behavior.

## Hard intervention battery

Each evaluation market receives three indirect mechanism families plus direct
and null controls. A full model run contains 16 prompts per market.

### 1. Three mechanism-reversal families (12 prompts)

For each family, construct one evidence item `E` and four contexts:

1. **Positive bridge (`B+`):** `E -> M1 -> M2 -> H1`, so the correct update
   is pro-YES.
2. **Negative bridge (`B-`):** the same `E`, with one minimal bridge clause
   reversed, so `E -> M1 -> M2 -> not-H1`.
3. **Broken bridge (`B0`):** the same `E`, with the decisive link removed,
   making it causally irrelevant to resolution.
4. **Bridge-masked ablation:** the same question, prior probability, and `E`,
   but without the bridge addendum.

The evidence headline and body are byte-identical across all four contexts.
Actors, entities, numbers, style, length, sentiment, and topical vocabulary are
therefore exactly matched. Only the causal bridge changes.

Every valid directional chain must contain at least two signed edges between the
observed development and market resolution. The evidence must not contain:

- YES/NO, “more/less likely,” odds, probability, forecast, on-track, setback,
  boost, hurt, win, lose, approve/reject when those directly paraphrase the
  resolution criterion;
- a direct statement that the market outcome will or will not occur; or
- the packet's intended label, rationale, or mechanism metadata.

The three families should target different actors and mechanisms. Surface
sentiment is assigned by a Latin-square schedule so positive, negative, and
neutral language is independent of the correct sign.

The bridge-masked condition is an ablation, not an assumed null. Its purpose is
to measure what the same model does when the information needed to determine the
registered sign is absent.

### 2. Direct calibration controls (3 prompts)

Create one independently validated direct pro-YES packet, one direct anti-YES
packet, and one topical orthogonal packet. These preserve continuity with the
frozen instrument and quantify the difficulty increase from direct to indirect
interventions. They are not part of the hard primary endpoint.

### 3. Literal no-information control (1 prompt)

Use the identical update wrapper with a neutral statement that supplies no new
event information. This estimates generic prompt-induced drift separately from
topical orthogonal evidence.

## Candidate generation

- Generate each four-context family jointly, rather than prompting separately
  for pro and anti text.
- Use at least two generator models or one generator plus human editing; record
  provenance.
- Generation output may propose bridges and rationales, but those fields are
  stripped before annotation and evaluation.
- The generator's plausibility self-rating is not collected or reported as
  validation.
- Run schema checks for exact evidence identity within a family, forbidden
  lexical cues, balanced labels, distinct mechanisms, word-count bounds, and
  absence of target labels in evaluator prompts.
- Version the protocol, prompt, schema, and candidate pool; hash every frozen
  artifact.

## Independent human validation

Use paid annotators who did not generate the packets and do not see intended
labels or model outputs. Randomize records and remove direction, family role,
rationale, and generator metadata.

### Full-context panel

At least three independent raters see:

- market question and resolution rules;
- canonical initial event model;
- the assigned bridge context; and
- the evidence packet.

For each item they report:

1. direction: increases YES / decreases YES / no material effect / ambiguous;
2. confidence in direction (1–5).

The reviewer instrument deliberately stops there. Structural chain properties
remain part of the private packet audit, not an additional burden placed on
reviewers.

### Shortcut panel

A separate three-rater panel sees only the question, rules, and evidence, with
the bridge removed. It reports the same direction and confidence judgments.
This measures whether a confident fixed sign leaks without the registered
causal bridge.

### Item-level retention gates

A mechanism family is retained only if:

- all three full-context raters agree on the registered label for (B^+),
  (B^-), and (B^0); “ambiguous” fails;
- median directional confidence is at least 4/5;
- the private packet audit confirms at least two signed edges for each
  directional bridge; and
- shortcut-panel raters do not unanimously assign the same increase/decrease
  direction with median confidence of at least 4/5.

Direct and orthogonal controls require unanimous direction labels and median
confidence of at least 4/5. Null controls require unanimous “no material
effect” with median confidence of at least 4/5.

Across the locked pool require:

- at least 80 retained markets and 240 retained mechanism families;
- Fleiss' kappa of at least .70 for direction before adjudication;
- balanced retention of positive and negative contexts; and
- no bridge-masked family with a unanimous high-confidence fixed sign.

If a gate fails, revise or regenerate using annotations only, then obtain fresh
ratings. If fewer than 80 markets pass after two rounds, stop and report that a
clean instrument could not be constructed.

Report rater counts, compensation, instructions, exclusions, agreement,
distributions, and the complete de-identified annotation file.

## Explicit shortcut baselines

Evaluate all baselines with grouped cross-validation by market. Hyperparameters
and any learned update magnitude are fit only on development markets.

1. headline sentiment/polarity heuristic;
2. TF–IDF logistic classifier;
3. frozen sentence-embedding linear classifier;
4. evidence-only instruction-model sign classifier;
5. question-plus-evidence classifier with the bridge masked;
6. full-context stateless sign-only classifier; and
7. each sign classifier coupled to one learned fixed update magnitude, scored
   on the same forecast endpoints as the agents.

The first five operationalize the critique directly. The sixth tests whether a
one-shot textual causal-composition policy is sufficient even when lexical
shortcuts fail. If it succeeds, the paper must describe the result as behavioral
causal composition, not maintenance of a privileged internal representation.

Instrument freeze requires bridge-masked headline/evidence baselines to remain
near the balanced three-class chance level (macro-F1 no greater than .40 on the
locked human labels). This audit is performed before target-model calls.

## Model execution

- Run every model shown in the Experiment 1 main result table if the exact
  deployment or frozen local weights remain available.
- Use one deterministic canonical initial forecast per model–market.
- Present each prompt in an independent update context at greedy decoding where
  supported.
- Randomize prompt order by a frozen seed.
- Use identical substantive prompts and record unavoidable provider wrappers.
- Normalize every probability field to `[0, 1]` before writing deltas; values
  outside the range fail closed rather than being silently rescaled.
- Preserve invalid outputs in denominators.
- Record exact deployment IDs, timestamps, prompts, token limits, hashes, raw
  responses, parser versions, and parse coverage.

At 100 markets and 16 prompts per market, the full battery is 1,600 calls per
model (16,000 for ten models), smaller than the existing 38,730-record update
collection.

## Preregistered endpoints

Let `delta+`, `delta-`, and `delta0` be the headline forecast changes for a
model and market/mechanism family under the positive, negative, and broken
bridges. Set the movement threshold `tau = .03` before model calls.

### Co-primary hard endpoints

1. **Paired reversal contrast**

   `C_m = mean_f(delta+ - delta-)`

   A lexical evidence policy sees identical `E` and predicts no systematic
   contrast. The model passes this endpoint only if the multiplicity-adjusted
   95% lower confidence bound is above zero.

2. **Unconditional triplet accuracy**

   Score every trial:

   - `B+` is correct only if `delta >= tau`;
   - `B-` is correct only if `delta <= -tau`; and
   - `B0` is correct only if `abs(delta) < tau`.

   Macro-average across the three labels within market, then equally across
   markets. No record is excluded for failing to move.

### Secondary endpoints

- complete-triplet success: all three members correct;
- positive and negative sign accuracy separately;
- broken-bridge specificity and literal-null drift;
- direct-control accuracy;
- direct-to-indirect accuracy drop;
- bridge-present versus bridge-masked contrast;
- H1-to-headline propagation and ICS;
- response magnitude versus human strength rating; and
- paired differences from every shortcut baseline.

The old threshold-conditioned EHC/HFC and sensitivity are retained only for
continuity and are not primary evidence for the hard causal claim.

## Inference

- Resample markets, keeping all nested mechanism families and contexts together.
- Use 10,000 market-cluster bootstrap replicates.
- Report model-specific estimates and intervals; do not pool records as
  independent.
- Holm-adjust the ten model-level reversal tests.
- For baseline comparisons, bootstrap paired market differences.
- Publish exact denominators, invalid-output rates, and all registered endpoints.
- Do not replace failed registered endpoints with more favorable thresholds or
  subsets.

No target-model pilot is used for power. With 100 market clusters and three
families per market, precision is reported from the registered clustered design;
if fewer than 80 markets pass instrument validation, the experiment does not run.

## Claim gates

Claims are determined mechanically.

### A model earns “mechanism-conditioned updating” only if

- its adjusted lower confidence bound for `C_m` is above zero;
- its unconditional triplet accuracy exceeds the best bridge-masked shortcut
  baseline by a paired interval excluding zero;
- directional movement exceeds broken-bridge and literal-null movement; and
- parse coverage is at least 95%.

### The paper may use a broad cross-model statement only if

- every predeclared frontier model passes the model-level gate; and
- a majority of the predeclared open-weight models pass.

Otherwise, name only the models that pass.

### Interpretation regardless of success

- Passing rules out evidence-only polarity, undifferentiated topical response,
  and a fixed sign-plus-shift policy that ignores the bridge.
- Passing does not uniquely identify an internal causal representation; a
  stateless full-context causal reasoner may also pass.
- If full-context sign-only baselines match updating agents, describe the shared
  capability as textual causal composition.
- If target agents do not beat bridge-masked baselines, retain the original
  Experiment 1 result as direct direction-selective updating and abandon the
  stronger mechanism claim.

## Implementation artifacts

Create a separate, fail-closed subtree so the frozen 900-packet experiment is
never overwritten:

```
exp1_prospective/indirect_interventions/
├── PROTOCOL.md
├── schema/
│   ├── candidate.schema.json
│   ├── annotation.schema.json
│   └── evaluation.schema.json
├── prompts/
│   ├── generate_family.txt
│   ├── full_context_annotation.txt
│   ├── shortcut_annotation.txt
│   └── update.txt
├── scripts/
│   ├── build_candidates.py
│   ├── blind_annotation_export.py
│   ├── validate_annotations.py
│   ├── audit_lexical_leakage.py
│   ├── freeze_instrument.py
│   ├── run_updates.py
│   ├── run_shortcut_baselines.py
│   └── analyze.py
├── tests/
├── data/
│   ├── development/
│   ├── candidates/
│   ├── annotations/
│   └── frozen/
├── runs/
└── reports/
```

Required tests include label stripping, evidence byte identity, bridge-only
minimality, probability range validation, balance, grouped splits, metric edge
cases, nested bootstrap reproducibility, and manifest hash verification.

## Manuscript integration

### Main body

- Narrow the current Experiment 1 paragraph before new results are available.
- Replace the direct example with one complete indirect triplet once frozen.
- Show one compact direct-versus-indirect/reversal table or panel.
- Lead with the paired reversal contrast and unconditional triplet accuracy.
- State explicitly what lexical baselines can and cannot do.
- Reserve “mechanism-conditioned” for models passing the registered gates.

### Appendix

Report:

- complete protocol and freeze hashes;
- generation prompts and forbidden-term rules;
- rater instructions, compensation, agreement, and full rating distributions;
- retained and rejected counts with reasons;
- examples of positive, negative, broken, and masked contexts;
- lexical leakage audits and all shortcut baselines;
- all model-by-condition estimates and clustered intervals;
- invalid outputs and sensitivity analyses; and
- the unchanged frozen direct-packet results as the original diagnostic.

The constant generator field `plausibility_rating=4` must never again be
described as evidence of validity.

## Execution order and stop/go decisions

1. **Claim surgery:** narrow current prose and draft the registered protocol.
2. **Development:** build and annotate families only on development markets.
3. **Code freeze:** finish schemas, audits, runners, metrics, and tests.
4. **Core candidates:** generate without target-model calls.
5. **Human validation:** run both panels, filter only by registered gates.
6. **Instrument audit:** verify minimum sample, agreement, balance, strength,
   plausibility, lexical leakage, and exact evidence identity.
7. **Freeze:** write manifests and SHA-256 hashes; archive candidate and rejection
   logs.
8. **Shortcut baselines:** run and publish before target-model results are opened.
9. **Target runs:** execute every predeclared deployment.
10. **Locked analysis:** run the registered clustered analysis once.
11. **Paper update:** apply the claim gates verbatim, whether positive or negative.
12. **Reproduction pass:** rebuild all tables/figures from frozen JSON and verify
    the PDF against the result manifest.

## Definition of “solid”

The critique is fully addressed only when all of the following are true:

- the existing direct instrument is no longer presented as excluding strong
  entailment/relevance alternatives;
- correct signs for the new instrument come from independent humans, not the
  generator;
- evidence is identical across sign-reversal contexts;
- registered direction, confidence, and null status satisfy predeclared human
  gates;
- literal null, broken bridge, orthogonal, and bridge-masked controls are present;
- lexical and sign-plus-fixed-shift baselines are actually run;
- every trial enters unconditional primary scoring;
- uncertainty is clustered by market and multiplicity is controlled;
- materials, annotations, code, outputs, and exclusions are fully auditable; and
- manuscript language follows the registered result gates rather than the desired
  interpretation.
