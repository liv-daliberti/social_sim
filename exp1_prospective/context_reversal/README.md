# Experiment 1: paired context reversal

This maintained workflow implements the context-reversal development pilot for
the ICLR manuscript. It leaves the original frozen Experiment 1 data untouched.
Read `PROTOCOL.md` before interpreting results. The first cohort is 20 fictional,
assistant-authored families without independent human validation; it is
instrument-development evidence, not a confirmatory study on real markets.

## Local follow-up diagnostics

Direction-only classification on the three existing local models and a Qwen3-32B
thinking versus matched non-thinking probability comparison were submitted under
[runs/local_diagnostics_v1](runs/local_diagnostics_v1/README.md). These use all 20
existing families, one repeat, and preserve prior results. No new human-review
round or frontier API calls are part of this follow-up. See the run's
`submission.json`, response manifests, and eventual `completion.json` for actual
execution and result status.

## Completed runs

The original three-model local pilot is complete: **2,880/2,880 valid records**.
See [the full local summary](results/development_summary.md) and
[completion record](runs/development_v1/completion.json). Its three-repeat model
roster remains fixed.

A subsequent **GPT-5.6 Sol** extension is also complete: **320/320 valid records**,
all 20 families and four contexts, repeat zero. It achieved 37/40 directional
updates and 17/20 paired reversals; broken-link absolute movement averaged 0.5
percentage points, meeting the original stability criterion. See
[the frontier protocol and completion](runs/frontier_gpt56_pilot_v1/README.md),
[full analysis](results/frontier_gpt56_pilot_v1/summary.md), and
[matched single-repeat comparison](results/frontier_gpt56_pilot_v1/comparison.md).
This extension was added after inspecting local results, with different inference
settings; neither run has independent material validation.

The historical launch/recovery notes below document execution. All listed pilot
jobs have ended. The ICLR main text and context-reversal appendix report these
results explicitly as development evidence.

## Design

Each family has positive, negative, broken, and omitted relational contexts.
The same new evidence is reused verbatim. Every context gets its own initial
forecast. Three independently forked updates then receive new news, no new
information, or a fact already in the initial information. Three repeats produce
960 calls per model. The roster is Qwen2.5-72B-Instruct, Llama-3.1-70B-Instruct,
and Qwen3-32B, all cached locally and evaluated offline.

## Build and inspect the development instrument

From the repository root:

```bash
python -m exp1_prospective.context_reversal.materials
python -m exp1_prospective.context_reversal.design
python -m pytest -q exp1_prospective/context_reversal/tests
```

`data/development_families.jsonl` contains visible text and separate private
scoring metadata. `data/development_plan.jsonl` contains the exact baseline
prompts and update templates; `data/development_plan.manifest.json` records the
family roster and hashes. The prompt compiler uses a visible-field whitelist.
The `masked` context is an ablation, not an assumed zero-effect condition.

`run_local.py --help` documents the offline runner. `run_local.sbatch` allocates
GPUs and invokes it; no hosted-provider credentials are needed. Submission
records live in `runs/`, scheduler output in `logs/`, raw responses under
`responses/`, and diagnostic summaries in `results/`. Preserve the raw responses
and manifests when sharing a result: they record every planned call and failure.
The runner refuses incompatible resumes and never substitutes another model.

## Interpret results

Analysis requires the design roster, including when an interrupted job omits
entire families. Directional failures and missing calls remain in unconditional
denominators. Numeric summaries state valid coverage. Resampling is over whole
families. No current ICLR table is automatically changed by this pipeline.

A later confirmatory cohort requires new families, independent blinded human
validation, a fixed meaningful effect and sample size, and a frozen protocol
before target outputs are examined. Development revisions remain versioned and
must not be described as preregistered confirmation.

## Launched development run

The frozen run is `runs/development_v1/`; `submission.json` records checkpoint
snapshots, resources, commands, and Slurm job IDs. The original submissions are
31433798 (Qwen2.5-72B), 31433799 (Llama-3.1-70B), and 31433800 (Qwen3-32B).
All three run the same frozen design and automatically analyze their outputs.

```bash
squeue -j 31433798,31433799,31433855
python exp1_prospective/context_reversal/aggregate.py
```

Aggregation requires all three prescribed model summaries and verifies their
design hashes. It produces `results/development_summary.md` and `.json`.
Job completion and response completeness must be checked separately: an error
status is never evidence of a valid forecast.

## Independent material review

`data/HUMAN_REVIEW_README.md` explains the blinded candidate packet.
`data/human_review_candidates.csv` and `.json` contain 80 anonymous versions
in four balanced batches, one version per family in each batch. Send a reviewer
only their assigned batch and the reviewer instructions. The separate `.private.json`
mapping is coordinator-only and contains the author's intended answers.
Generating a packet does not constitute collecting independent validation.

Qwen3-32B job 31433800 was cancelled while pending and replaced by 31433855
on four idle A5000 GPUs; no model responses existed before this resource change.

## Lexical diagnostic

`baseline_audit.py` evaluates fixed word-based classifiers with entire families
held out, using each material variant once rather than duplicating decode repeats.
See `results/lexical_audit/summary.md`. News-only and question-plus-news inputs
are identical across contexts within a family; their inability to separate the
conditions is structural. Full-prompt lexical classifiers remain useful diagnostics.
These 20 vignettes reuse five mechanism classes; varied domains do not establish
generalization to 20 distinct types of mechanism.

CPU finalizer job 31434033 waits for all three current model jobs
(`afterany`), then verifies scheduler state, manifests and response coverage,
reanalyzes every prescribed model, and writes `results/development_summary.md`
plus `runs/development_v1/completion.json`. Failed runs remain visible with all
planned denominators. The latter file records `plan_complete` explicitly.

## Qwen72 infrastructure recovery

Qwen72 job 31433798 loaded its weights but encountered an Inductor/Triton
cache directory race during startup, then stalled until the one-hour timeout.
It saved zero model responses. The original submission, incomplete report and
completion record are preserved under `runs/development_v1/attempts/31433798_timeout/`.
Retry job 31440348 uses the same frozen runner, checkpoint, prompts and decoding
settings, with eager execution, TorchDynamo disabled, isolated compiler/Outlines
caches, eight A5000 GPUs, and a two-hour limit. Its response path is separate.
CPU finalizer 31440349 regenerated all three summaries after the retry
completed. The existing two completed model response files are retained.
