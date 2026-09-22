# Paired context reversal: local-model development pilot

Protocol: `exp1_context_reversal_local_development_v1`, recorded 2026-09-21
before target inference. This is an exploratory instrument-development pilot,
separate from the frozen Experiment 1 records and from confirmatory paper results.
No frontier-model API, web retrieval, fine-tuning, or external generation service
is used. All inference uses locally cached open-weight checkpoints.

## Scope and materials

Twenty assistant-authored fictional forecasting families cover several domains.
These are conditional vignettes for development, not forecasts on real markets.
They are marked `authored_development_unvalidated`; independent human editing and
blinded validation have not occurred. They must not be presented as validated
materials or used as a substitute for the planned independent confirmatory cohort.
Any revision after seeing model behavior creates a new development version.
The future confirmatory set must use new, independently validated families and
must be frozen before target outputs are inspected.

Every family has identical news under three relational contexts: a positive
connection to the focal outcome, a reversed connection, and a broken connection.
A fourth condition omits the relational context. The masked condition has no
prespecified direction and is not presumed irrelevant. Gold directions, private
mechanism descriptions, provenance, and condition identifiers never enter prompts.

## Correct prior and controls

The model first forecasts separately in each context, with all contextual
information already present. From that exact probability, three independent
update branches receive new news, no information, or an exact repeat of information
already present in the baseline. Updates never see another branch's response.
Thus context changes cannot be mistaken for responses to new news. All models
receive the same instruction about novelty. Repeated news is a baseline fact,
not a second presentation of the genuinely new evidence.

## Model and decode roster

Pilot roster: Qwen2.5-72B-Instruct, Llama-3.1-70B-Instruct, and Qwen3-32B,
using complete local snapshot paths recorded in submission manifests. Standard
chat templates; Qwen3 thinking disabled. No tool calls or explanations are
requested. Syntax-constrained output is `{"probability": <number in [0,1]>}`;
the grammar specifies neither a probability value nor an update direction.
Temperature 0.7, top-p 1.0, maximum 128 output tokens, maximum context 4096,
three independent repeats, deterministic per-call seeds from a fixed base seed.
The unit is a family, not a repeated decode.

Twenty families × four contexts × three repeats = 240 initial calls plus 720
independent updates per model: 960 calls/model, 2,880 calls for the three models.
The first two families may be used for a separately logged infrastructure smoke
run before the complete pilot. Smoke results are development data, not exclusions.

## Outcomes and missingness

Primary development descriptions: unconditional correctness of the direction of
new-news movement, paired reversal correctness (both signs correct), and the
corresponding drift-adjusted change `p(new news) - p(no news)`. Exactly zero
movement is incorrect for either directional context. No 3-percentage-point
filter is applied. Also report continuous signed movements and absolute movement
in broken, no-news, repeated-news, and masked conditions.

Intervals resample complete families, retaining contexts, branches and repeats;
2,000 resamples with a fixed seed. A ±2-percentage-point tolerance describes
broken-context stability, fixed before outcomes. A signed mean near zero is
insufficient: report absolute movement too, and require its upper 95% interval
below 2 pp before labeling absolute stability. These development intervals do not
license a confirmatory population claim or determine which models are reported.

Every planned response is accounted for from the compiled design. Missing or
invalid directional calls fail unconditional correctness. Missing baselines block
all three updates rather than receiving an imputed probability. Numeric summaries
state their valid denominator, and stability is indeterminate if planned broken
observations are missing. Retries may address infrastructure failure, never replace
a valid model response. Record raw output, probability, error status, exact prompt,
prompt and input hashes, local snapshot, code versions, decoding and Slurm job ID.

## Development decision

Inspect all three models, all families, and all controls, regardless of direction.
Use the pilot to diagnose ambiguity, saturation, formatting, and revision drift.
A new frozen cohort and independent human validation are needed before a stronger
Experiment 1 claim enters the ICLR manuscript. No current paper result is replaced
by this pilot. There is no automatic launch of an 80–100-family confirmatory run.
