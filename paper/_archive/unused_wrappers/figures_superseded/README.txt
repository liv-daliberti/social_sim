Superseded Experiment-3 figures (moved 2026-07-09).

exp3_curves.{pdf,png}   -- model-scale learning curves
exp3_heatmap.{pdf,png}  -- model x step heatmap

Both were generated from reports/curves.csv, i.e. the Jul-7 scale-ladder runs, which:
  * trained the DEGENERATE one-step probe (a news-ignoring constant collects ~70% of the
    oracle's reward), and
  * ran on the OLD catalog, whose held-out `two_news_feedback` had spectral radius 1.297
    (an explosive world) and whose training mix still contained `skip` (resting poll ~100).

Their generators (plot_curves_lines.py, plot_curves_heatmap.py) also parse `predicted_poll`,
which multi-shock completions no longer emit, so they cannot be re-run against current dumps.

There is currently NO h* scale ladder: only Qwen3-4B family+control. Regenerating these
figures requires (a) porting the generators to the multi-shock schema and (b) new ladder runs.
Nothing in the paper includes them.
