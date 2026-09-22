# Completed diagnostic integration into the manuscript

The canonical `paper/main.pdf` now reports the completed overnight diagnostics.
Main text: `paper/transfer_and_grounding.tex`; methods and tables:
`paper/experiment3_diagnostics_appendix.tex` (Appendix D.1.8–D.1.10;
Tables 48–51 in this build). Abstract, introduction, limitations, conclusion,
appendix guide, and reproducibility statement are aligned.

## Framing

- Perfect cue-to-reference identification does not imply successful forecasting.
  Exact-pattern assistance does not rescue the 8B models; stronger-model
  conditional numerical ability is qualified by the post hoc baseline audit.
- A single supervised recipe reaches 47/48 in-domain forecast structure choices,
  against 24/48 for the primary same-seed RL comparison. Both fixed supervised
  rates and every transfer cell are reported. This establishes in-domain
  learnability under one recipe, not an isolated or general SFT advantage.
- The mechanism-family accuracy advantage remains a comparison against a fixed
  population prior. Coherent gain tracking improves on seen mechanisms, but the
  two-seed intervention does not establish held-out evidence use.
- No result from the unfinished distribution-matched shuffled training campaign
  enters the manuscript. Its contrast remains unresolved.

## Verification

- `make paper` succeeds: 79 PDF pages including references and appendix; scientific
  main text ends on page 9. Abstract: 140 words. No overfull boxes or undefined
  references/citations. Main pages 8–9 and the supervised/evidence-use tables
  were visually inspected.
- `python paper/generate_exp3_diagnostic_tables.py` regenerates four tables and
  their source-hash manifest exactly. All recorded source hashes were checked.
- Both supervised recipes' original-task structure accuracy, response MAE, and
  cue calibration were independently recomputed from the raw generations in
  all four domain/label cells; all agree with the published summaries.
- Relevant existing layout checks pass (main-page limit, wrapped transfer grid,
  artifact links, input existence, balanced groups). These tests were updated
  to locate the active manuscript wording rather than retired phrases.
- Running `paper/tests/test_result_claims.py` and `paper/tests/test_layout.py`
  yields 14 passes and 8 failures. Remaining failures assert older exact prose,
  a two-line title, or content in retired section wrappers. They are not failures
  of the new diagnostic computations or PDF build. The broader legacy test
  migration was not included in this scientific revision.
