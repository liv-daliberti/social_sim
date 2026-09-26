# Experiment 3/4 maintained code

The current paper uses four maintained roots:

- `mechanism_family/`: structural-OOD data, training, evaluation, and reporting;
- `coin_city_structural/`: Coin City A-to-B transfer;
- `polymarket/`: historical-market training and sealed evaluation;
- `biased_news/`: shared RL runtime used by Polymarket launchers.

The structural-OOD, Coin City, and three-model Polymarket scientific rosters are
complete. `monitor_campaign.py` reads the completed Polymarket replacement ledger
alongside the mechanism ledger and reports both 8B locked-test summaries.
Historical studies, stale status snapshots, and one-off scheduler repair scripts
live under `_archive/`.
