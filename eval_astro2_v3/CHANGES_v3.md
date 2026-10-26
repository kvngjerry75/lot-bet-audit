# v3 changes (independent audit, 2026-10-24)

New modules
* clean_data.py: removes out-of-pool rows, exact duplicates, copied-forward draws (same numbers in the same order on a
  later date, or the same set within 30 days) and cross-game copies; keeps and flags make-up draws that share a date.
  Writes *_audited.csv + cleaning_log.csv. 18 rows removed across the four files.
* randomness_battery.py: 104 tests (20 per game + 24 cross-game), each calibrated on simulated fair-machine
  histories, BH and Bonferroni across the whole family.
* rule_miner.py: the data-mined rule engine (about 45,000 rules for DayWA, 108,000 for the 5/90 games), sequential
  likelihood-ratio evidence per rule, PROVEN threshold = rules / 0.05, honest walk-forward scoring of every draw after
  draw 100, fair-machine controls, positive control (--plant). Also a stage4 expert ("rulemine").
* test_v3.py: no-look-ahead, quiet-on-fair-data, finds-a-planted-rule, cleaning, full layout, tracker same-day.

Changed
* nla_config.py: expanding training windows (train_window and hazard window 10**9 instead of 1,500 / 1,200);
  eval_draws 0 = score every draw after the warm-up; ensemble hedge_eta 2.0 -> 0.5; rulemine settings.
* eval_harness.py layout(): full mode (eval_draws <= 0).
* experts.py, eval_harness.py, stage4_astronomical.py: "rulemine" expert; old "rulemem" off unless --rulemem;
  --last300 restores the v2 layout; --feature-windows, --tft-seq-len, --markov-half-life for window tests.
* tracker.py: a pick logged on the draw day itself is not counted unless --allow-same-day; dates with two different
  draws are not scored; anytime-valid e-value for the core-hit metric (confirm at >= 80).
* rule_memory.py: calendar rules now see the next draw's date (v2 scored them in the backtest but they never voted
  on the live pick); default next draw date skips weekdays the game does not use.
