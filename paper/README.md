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

The manuscript is two hand-edited LaTeX sources, so it can be worked on directly
in Overleaf:

- `main.tex` owns the ICLR style, anonymity, and the whole main body.
- `appendix.tex` owns every appendix section, in reading order.
- `tables/` contains checked-in analysis outputs, pulled in by `\input`. A
  generated table should name its generator in its first comment; edit the
  generator, never the table or a copy of it in `appendix.tex`.
- `figures/` contains compiled manuscript figures plus source material needed by
  the maintained figure generators.
- `make_submission_zip.sh` is the submission-boundary check, not a general
  backup script.

Everything else a build touches is generated or third-party. `%= BEGIN/END =%`
banners inside both files record which retired fragment each span came from;
those fragments are kept for provenance in
`_archive/consolidated_fragments_2026-09-22/` and nothing reads them. The older
`experiment*_section.tex`, `methods_section.tex`, and `results_section.tex`
wrappers were already not inputs to the manuscript and remain unused.

`appendix.tex` carries a `%%% FRESH-CONTEXT-APPENDIX %%%` marker. The guarded
publication pipeline in `exp1_prospective/context_reversal/fresh/publish.py`
inserts an `\input` below it in an isolated staged copy; keep the marker
verbatim.

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
