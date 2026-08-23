# Experiment 3/4 maintained code

The current paper and running campaign use four roots:

- `mechanism_family/`: structural-OOD data, training, evaluation, and reporting;
- `coin_city_structural/`: Coin City A-to-B transfer;
- `polymarket/`: historical-market training and sealed evaluation;
- `biased_news/`: shared RL runtime used by Polymarket launchers.

`monitor_campaign.py` reads the registered mechanism and Polymarket-extension
ledgers. The root finalizer and verifier scripts are live scheduler dependencies;
do not rename them while jobs are queued or running. Historical studies and
one-off scheduler repair scripts live under `_archive/`.
