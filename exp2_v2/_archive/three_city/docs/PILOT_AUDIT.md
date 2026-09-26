# Three-city C2 v1 small-pilot audit

## Decision

**Do not scale the current frozen v1 task yet.** The four-episode pilot confirms
that the structure-blind setup can elicit the intended mechanism inference, but
it also exposes a temporal ambiguity that contaminates the numerical forecast.
The design is headed in the right conceptual direction and needs one targeted
revision before a larger run.

This pilot used four balanced numerical episodes, four paired context variants,
and prefixes `k=0,1,2`: 48 prompts per model. Claude Opus 4.8, DeepSeek V4 Pro,
and GPT-5.4 each completed the same 48 prompts with no parse failures. Four
episodes are not enough for model ranking or paper claims.

## What worked

1. **The models inferred rather than received the structure.** Prompts never
   stated a type count, matching rule, coefficient, prior, equation, noise
   level, or context reliability. Nevertheless, rationales spontaneously
   described one demonstrated city as highly responsive and the other as
   buffered, then compared City C with those examples.
2. **Mechanism-relevant context moved behavior in the intended direction.**
   For every model and every prefix, the forecast-implied response under the
   high-transmission background exceeded the response under the buffered
   background. At `k=0`, the mean high-minus-buffered differences were 0.540
   for Claude, 0.328 for DeepSeek, and 0.253 for GPT.
3. **Diagnostic City C evidence was detectable.** Under no context, behavioral
   pattern accuracy at `k=2` was 1.00 for Claude, 1.00 for DeepSeek, and 0.75
   for GPT. DeepSeek's no-context forecast MAE fell to 0.70 poll points and
   GPT's to 3.20.

These are the behavioral ingredients the experiment was intended to make
visible.

## What did not yet work

### 1. The reference histories do not identify the temporal transition clearly

The reference news schedule alternates large positive and negative shocks. It
therefore repeatedly returns the poll near 50. A forecaster can fit the examples
either as:

```text
next poll = current poll + response to this week's news
```

or approximately as:

```text
next poll = baseline 50 + response to this week's news
```

without being contradicted clearly.

This mattered in the pilot. Some rationales correctly inferred the high-response
pattern but then forecast from 50 rather than from City C's current poll. For
example, after City C had reached 61.9, Claude described a roughly 0.9 response
to the next `-10` shock but predicted 41.0. The inferred pattern was right; the
state transition was not. That produces the misleading combination of perfect
behavioral pattern classification and 5.525-point no-context MAE at `k=2`.

This is not evidence against latent recovery. It is an avoidable
identification/readout confound.

Across all context variants, the implied response fell outside the interval
spanned by the two demonstrations in 21/48 Claude forecasts, 10/48 DeepSeek
forecasts, and 14/48 GPT forecasts. The models were never told that this
interval was a hard bound, so these are not automatically invalid predictions.
The concentration at `k=2`, however, is further evidence that the readout is
mixing temporal-state interpretation with mechanism recovery.

### 2. Orthogonal context moved forecasts too much

Mean absolute changes in forecast-implied response between the no-context and
orthogonal-context prompts ranged from 0.141 to 0.463 at `k=0` and from 0.220
to 0.458 at `k=2`. For a ten-point held-out news shock, those correspond to
roughly 1.4–4.6 poll points.

The intended C2b signature is selective use of mechanism-relevant context, not
generic sensitivity to any city description. A larger sample may reduce some
of this movement, but the control should be strengthened before scaling.

### 3. Misleading-context override was not uniform

From `k=0` to `k=2`, misleading-context response error fell from 0.755 to 0.607
for DeepSeek and from 0.703 to 0.413 for GPT, but rose from 0.740 to 1.203 for
Claude. Claude's failure is closely related to the baseline-versus-current-poll
ambiguity above, so it should be retested only after that ambiguity is removed.

## Required v2 revision

Preserve the structure-blind prompt and one-forecast readout, but:

1. replace the alternating reference shocks with a balanced schedule containing
   consecutive same-sign shocks, such as `[+10, +8, -6, -12, -8, +8]`;
2. use the table heading “poll at end of week” so chronology is unambiguous
   without stating a transition equation;
3. verify offline that a baseline-reset model and a cumulative-transition model
   make sharply different predictions on the displayed reference examples;
4. length-match the orthogonal backgrounds more closely to the
   mechanism-relevant backgrounds; and
5. rerun this same four-episode diagnostic before authorizing any larger sample.

The v1 prompts, answer key, hashes, responses, and plot should remain frozen as
an audit trail rather than being overwritten.

## Artifacts

- `biased_news/data/three_city_c2/pilot_v1/pilot_diagnostics.png`
- `biased_news/data/three_city_c2/pilot_v1/pilot_summary.md`
- `biased_news/data/three_city_c2/pilot_v1/responses_*.jsonl`
- `biased_news/eval/run_three_city_c2_pilot.py`
- `biased_news/analysis/analyze_three_city_c2_pilot.py`
