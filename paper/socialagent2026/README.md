# SocialAgent @ NeurIPS 2026 workshop cut

This directory contains a workshop-specific manuscript. It preserves the
canonical ICLR paper in the parent directory and reorganizes the main text around three questions about evidence use,
relationship selection, and transfer.

- Venue: SocialAgent, Second Workshop on Large Language Models for Social
  Reasoning and Simulation, NeurIPS 2026.
- Submission mode: double blind, non-archival long paper.
- Limit: 9 content pages, with unlimited references and appendix.
- CFP: https://social-llm-workshop.github.io/
- Build: run make workshop from the repository root, or run
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex here.

neurips_2026.sty is the unmodified official style file from the NeurIPS 2026
author kit:
https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

The workshop CFP does not request the main-conference paper checklist, so this
version includes the paper, references, and technical appendix without that
checklist.
