# A5000 actor memory allocation amendment

Recorded2026-09-22 03:30UTC after the first A5000 startup failure and before any new rollout/update. Full disclosed shuffled seed42 attempt31445124_2 did not complete initialization: original actor fraction.38 on24GB gave no cache blocks, then only464 token capacity, below the frozen3072 context. It was cancelled, with failure logs retained. No outcome informed this change.

Set `VLLM_RATIO=0.76` on24GB A5000s, giving the same nominal18.24GB actor allocation as.38 on48GB A6000s. The actor and learner have separate GPUs. All model, data, reward, optimizer, LoRA, context, prompt truncation, decoding, rollout/update count and batch settings remain frozen. Apply the same memory allocation to all ten new shuffled controls and six matched recovery runs. This is an operational hardware/memory difference, and must not be described as exact runtime identity to historical runs.

Retry the same disclosed seed42 control. Require one completed actor rollout and learner update before moving remaining nine controls and six matched recoveries to this hardware. Keep the full4800 training budget and30h scheduler limit. The failure counts as one infrastructure attempt; it does not remove a seed. Effective environment and NVIDIA device information are emitted in each new job's log.
