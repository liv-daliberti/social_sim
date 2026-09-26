# Does model capacity explain the arbitrary-label boundary?

## Question

Experiment 3 finds that the higher-rate supervised recipe reaches 100.0% structure
accuracy when trained natively in Coin Harbor with semantic labels, but stays at
chance in both arbitrary-label cells (41.7% City, 53.5% Harbor) at three seeds.
That was read as a property of the label mapping.

Experiment 2 independently measures something that may explain it: on 88 held-out
episodes, checkpoints of 14-72B encode the changing episode-local KIV/ZOR mapping
in their hidden states, while 4-8B checkpoints do not reliably encode it.

The native-cell runs were all Qwen3-8B -- on the wrong side of that threshold.
So the boundary may not be "this recipe cannot learn arbitrary labels" but
"this model cannot represent the mapping the task requires."

## Test

Repeat the native-cell design at Qwen3-14B, which Experiment 2 places above the
threshold. Everything else is held fixed: same training data, same recipe
(lr 2e-5, LoRA r32/alpha64 on seven projection modules, one epoch, batch 16,
constant schedule), same three seeds, same 768-prompt component diagnostic.

Four cells, so the comparison is interpretable in both directions:

  coin_city_semantic     positive control -- 8B reaches 97.9-100.0%
  coin_harbor_semantic   positive control -- 8B reaches 100.0%
  coin_city_arbitrary    the failing cell -- 8B reaches 41.7%
  coin_harbor_arbitrary  the failing cell -- 8B reaches 53.5%

## Outcomes and what each would mean

- Arbitrary cells clear at 14B while semantic cells also clear: the boundary is
  model capacity, not the recipe, and Exp 2's encoding threshold predicts it.
  The Exp 3 claim becomes a capacity account rather than a barrier.
- Arbitrary cells stay at chance while semantic cells clear: capacity at this
  scale does not explain it, and the label-mapping reading survives a direct
  challenge -- a stronger result than the current one.
- Semantic cells do not clear: the 14B recipe did not train; the run is
  uninformative about capacity and must not be read either way.

Both of the first two outcomes are publishable and are reported. This is
declared before the runs.

## Status

Exploratory. Not a registered endpoint. No result from it may replace the
three-seed native-cell result, which stands as reported.
