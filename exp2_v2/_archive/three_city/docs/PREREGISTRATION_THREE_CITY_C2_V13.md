# Preregistration: matched two-regime structural choice (C2 v13)

## Design history

V12 was stopped after partial collection and is retained as a post-hoc pilot.
It will not be pooled with v13. Inspection of the frozen v12 design revealed
that its supposedly neutral A/B/C benchmark already interpolated A and B using
City C's profile index. That benchmark therefore contained the same structural
choice that the clue was intended to supply.

V13 corrects the estimand before collection. Its final confirmatory episode
seeds begin at 105,000 and do not overlap any earlier design.

An initial no-model-call freeze displayed eight cases for each reference city
and used 2.5 effective reference cases. Before any model responses were
collected, the reference tables were shortened at the researcher's request.
That intermediate freeze displayed four cases for City A and four for City B.

A subsequent pre-model parameter search used development seeds 200,000--200,799
to seek larger adjacent-arm gaps while preserving five-round convergence. A
moderate candidate was locked before a single evaluation on holdout seeds
103,000--103,119: case-noise SD 9.5, regime separation [1.70, 2.10], and 2.0
effective reference cases. No model response was collected during either
design revision.

After that benchmark check and still before model collection, the researcher
requested exactly two displayed reference polls from City A and two from City
B. This final reference-count revision retains the locked numerical parameters
and 2.0-case matched pooling strength, uses fresh confirmatory seeds
105,000--105,119, and does not require neutral pooling to beat City C-only after
the first two evidence rounds. Here “two reference polls” is distinct from the
City C evidence-round index k=1,...,5.

## Intended causal contrast

The three estimators represented by the prompt arms are:

1. **City C only:** estimate City C's response using only its 1, 2, 4, 8, or 16
   observed cases.
2. **A/B/C without structure:** partially pool City C toward the symmetric
   average of Cities A and B. This reduces early variance but is biased toward
   the midpoint because A and B represent different regimes.
3. **A/B/C with structural choice:** use exactly the same pooling strength, but
   center the prior on the reference city selected by profile proximity.

Both pooled estimators use a prior worth 2.0 City C cases. Its weight is
therefore 2.0/(2.0+n_C), so every estimator converges toward the City-C estimate
as City C evidence accumulates. The neutral and clue-aware benchmarks differ
only in prior location, never in prior strength.

## Data-generating process

Each episode contains two distinct news-response regimes separated by 1.70 to
2.10 response units. Cities A and B represent opposite regimes, with their
labels counterbalanced. City C is generated from one regime with independent
city-level deviation. Continuous profile indices place City C near the
reference from its regime without naming that reference in the clue. Each
observed case has independent poll noise with SD 9.5 points.

Cities A and B each contribute two displayed cases. Thus, at the first round,
the prompt contains four reference cases but only one direct City C case; at
the final round, the 16 direct City C cases dominate the reference information.

The exact clue is:

> **Structural clue:** The cities come from two distinct response regimes,
> represented by Cities A and B. City C is drawn from the same response regime
> as whichever reference city has the closer regional-profile index. Use the
> closer-profile reference as the relevant analogue rather than averaging
> Cities A and B. This clue does not say whether A or B is closer; infer that
> from the displayed indices. City-specific differences and case noise remain
> possible.

Arm B and arm C receive identical records. The clue above is the only textual
difference. It states the structural selection rule but not which city is
selected; the model must compare the displayed profile indices.

## Pre-collection gates

Before model collection, the frozen 120-episode design must satisfy all of the
following:

- exactly 600 tasks per arm at rounds k=1,2,3,4,5;
- exact A/B-label and target-regime balance;
- profile proximity selects the true regime reference in every episode;
- median displayed A/B separation of at least 14 poll points;
- round-1 deterministic MAE ordered as City C only > neutral A/B/C >
  structural A/B/C;
- at round 1, structural choice improves over City C only by at least 2.5
  points;
- neutral pooling beats City C only in the first two evidence rounds, and
  structural choice beats neutral pooling at every round;
- by round 5, the spread among all three deterministic MAEs is below 0.35 poll
  points; and
- the two pooled estimators have identical prior weights at every task.

There are 600 tasks per arm, 1,800 calls per model, and 5,400 calls across the
same three models. No v13 model calls may occur before the design is frozen and
all validation gates pass.

## Analysis implementation amendment during collection

After collection began and the researcher questioned the interim comparison,
an audit found that an inherited analysis helper still requested the old
neutral-benchmark field name `abc_shrinkage`; the frozen v13 answer key names
the same quantity `abc_no_structure`. The v13 analyzer was amended to create
that legacy alias in memory for inherited scoring and plotting helpers. This
compatibility repair does not change any prompt, task ID, prompt hash, response,
gold outcome, estimator value, or model call. The task and answer-key hashes
remain unchanged. Interim results had already been viewed when the repair was
made, so this is documented as an analysis-code amendment rather than a
pre-collection change.
