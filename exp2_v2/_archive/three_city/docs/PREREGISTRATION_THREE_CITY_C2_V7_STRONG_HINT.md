# Three-city C2 v7 strong-structure positive control

This appendix arm was requested and frozen on 2026-07-28 after partial results
from the original two-arm v7 confirmatory run had been viewed. It is therefore a
post-specified positive control, not a third confirmatory arm and not evidence
for the original preregistered hint contrast.

## Purpose

The arm tests whether named frontier models can execute the intended
cross-city inference once the latent two-pattern structure is explicitly
disclosed. It removes the need to discover that structure but does not disclose
City C's hidden type, which reference city it matches, either response value,
the data-generating equation, a prior, or an answer.

## Prompt manipulation

Every frozen number, table, background, common noise statement, reasoning
instruction, token budget, JSON schema, and rationale limit is copied from the
v7 structure-blind task. Immediately before the common reasoning instruction,
the positive-control arm inserts exactly:

> Cities in this task follow one of two recurring response patterns,
> demonstrated by Cities A and B. City C follows the same response pattern as
> one of those cities. Use the background descriptions and City C's completed
> cases to infer which pattern applies, then combine that structure with City
> C's evidence when forecasting.

The original structure-blind and minimal relevance-hint task files and running
jobs are not modified.

## Design and reporting

- The arm uses the same 120 episodes, three background conditions, target
  prefixes \(k\in\{0,1,2,4\}\), answer key, endpoints, model roster,
  temperature zero, and 512-token limit as v7.
- One received completion is retained per model-task pair. Unparseable received
  completions are terminal; transport failures before a completion may be
  retried.
- The strong arm is paired to the earlier arms by task and episode, but it is
  collected later. Comparisons therefore retain a possible execution-time
  confound and are labeled post-specified.
- The focal positive-control readout is truthful mechanism-relevant background
  at \(k=2\): forecast MAE and the same frozen structure-use coefficient
  \(\beta\). Episode-paired percentile bootstrap intervals are descriptive.
- The paper presentation may show the three arms side by side, but the original
  blind-versus-minimal-hint estimands remain the only confirmatory outcomes.
