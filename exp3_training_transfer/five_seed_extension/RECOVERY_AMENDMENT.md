# Five-seed submission allocation amendment

Recorded on 2026-08-25 UTC before either added architecture-sweep seed began.

The incremental launch accepted 78 of 84 training jobs, then Slurm rejected the
first added Qwen3-1.7B request on the parent `mltheory/all` route with
`Requested node configuration is not available`. No Qwen3-1.7B,
Llama-3.2-3B, or Qwen3-32B added-seed job had been accepted at that point.

The Qwen3-1.7B and Llama-3.2-3B seed-45/46 jobs are therefore resubmitted on
`allcs/cs` with medium QOS, using the exact pinned local snapshots already
used by seeds 42--44. This changes queue eligibility only. Data, prompts,
reward, seeds, rank/alpha, learning rate, optimizer, batch, rollouts, 4,800
presentations, 300 rounds, decoder, and endpoint are unchanged. Qwen3-32B
already used the same `allcs/cs` bootstrap route in its parent extension.
