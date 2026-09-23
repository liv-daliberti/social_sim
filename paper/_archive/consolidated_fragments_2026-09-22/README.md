# Consolidated manuscript fragments (retired 2026-09-22)

These files were the per-section `\input` fragments of the ICLR manuscript. On
2026-09-22 the manuscript was consolidated into two hand-edited sources so the
project can be worked on directly in Overleaf:

- `paper/main.tex` — preamble and the whole main body.
- `paper/appendix.tex` — every appendix section.

The consolidation was mechanical: each fragment's text was inlined verbatim at
its `\input` site, and the rebuilt PDF is text-identical to the build these
fragments produced. They are kept only for provenance. Nothing active reads
them; edit `main.tex` and `appendix.tex` instead.

Inline order, main body: frontmatter, inductive_forecasting, evidence_ladder,
transfer_and_grounding (with fig_exp3_transfer_grid, fig_exp3_results_grid),
related_work, limitations, conclusion, submission_statements, acknowledgements.

Inline order, appendix: appendix_guide, teaser_probe_appendix,
experiment1_appendix, experiment1_context_reversal_appendix,
experiment2_appendix (with experiment2_mechanistic_probe_appendix),
experiment3_appendix (with experiment3_diagnostics_appendix,
experiment3_training_setup_appendix).

`tables/` fragments were deliberately *not* inlined: their generators rewrite
them in place.
