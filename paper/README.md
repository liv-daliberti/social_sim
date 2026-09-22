# ICLR manuscript

`paper/` is the one canonical manuscript tree. `main.tex` is the anonymous ICLR
submission; there is no separately maintained author or non-ICLR version.

## Build and review

From the repository root:

```bash
make paper
make review
make submission
```

The resulting PDF is `paper/main.pdf`. `make submission` rebuilds the paper,
derives `.archive_manifest` from LaTeX's recorder output, rejects unresolved
references or citations, and packages only files used by the build.

The `paper/ICLR` path is a compatibility symlink back to this directory for
legacy tooling. It is not a second manuscript tree. Maintained jobs and new code
must write to `paper/` directly.

## What to edit

- `main.tex` owns the ICLR style, anonymity, and section order.
- `main.tex` lists the active prose inputs, including `frontmatter.tex`,
  `evidence_ladder.tex`, and `transfer_and_grounding.tex`. Some older
  `experiment*_section.tex` wrappers are not inputs to the current manuscript.
- `experiment*_appendix.tex` and `appendix_guide.tex` contain supplementary
  material.
- `tables/` contains checked-in analysis outputs. A generated table should name
  its generator in its first comment.
- `figures/` contains compiled manuscript figures plus source material needed by
  the maintained figure generators.
- `make_submission_zip.sh` is the submission-boundary check, not a general
  backup script.

The maintained generators are:

| Command | Output |
|---|---|
| `python paper/generate_figures.py` | Experiment 1 directional-updating figure |
| `python paper/generate_paper_figures.py` | Experiment 1 scale/uncertainty figure |
| `python paper/generate_coin_city_split_figures.py` | Experiment 2 prompt and result figures |
| `python paper/generate_coin_city_appendix_tables.py --help` | Experiment 2 appendix table bodies |
| `python paper/generate_coin_city_symbol_context_tables.py` | Experiment 2 13-system arbitrary-symbol and matched-family scaling tables |
| `python paper/generate_coin_city_mechanistic_comparison.py` | Experiment 2 behavioral-screen table, hidden-state table, and probe figure |
| `python paper/generate_coin_city_robustness_tables.py` | Experiment 2 generator-population and GPT repeat tables |
| `python paper/generate_exp3_diagnostic_tables.py` | Completed component, supervised learnability, and gain-intervention tables with source hashes |
| `python paper/generate_exp4_scale_figure.py` | Experiment 4 Qwen scale figure and appendix table |

Run generators from the repository root. They read frozen experiment artifacts
from the corresponding experiment directory and write only to this paper tree.

## Archive policy

Historical drafts, unused wrappers, editor recovery files, and superseded
figures belong in `_archive/`. Active LaTeX, generators, scheduler finalizers,
tests, and deployment configurations must never read from `_archive/`.
`tests/test_repository_boundaries.py` enforces that rule.

The former named-author `main.tex` is retained under
`_archive/manuscript_snapshots/` for provenance. `paper/ICLR` is now only the
compatibility symlink described above; the retired duplicate tree remains
recoverable from Git history rather than as a second live manuscript.
