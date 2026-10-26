# eval_astro2 v3: run in Google Colab

v3 = the audited v2 system plus: data cleaning, a randomness battery, the data-mined rule engine, full-history
training and scoring, a stricter tracker. Everything in v2 still runs.

## 1. Set up (once per session)
```
!unzip -o -q eval_astro2_v3.zip
%cd eval_astro2_v3
!pip -q install -r requirements.txt
!python test_pipeline.py          # must end with ALL TESTS PASSED
!python test_v3.py                # must end with ALL V3 TESTS PASSED
from google.colab import drive; drive.mount('/content/drive')
```

## 2. After every result: add it, then re-clean
Add results to the *_clean.csv files (numbers in drawn order), then rebuild the audited files that every v3 tool reads:
```
!python tracker.py add-draw --csv daywa_clean.csv --date 2026-10-21 --numbers 23 32 26 11 22
!python clean_data.py             # writes daywa_audited.csv, main_draw_audited.csv, noonrush_audited.csv, vag_audited.csv
```
Read the removal list it prints. A new "copy of the draw of ..." line means a result was entered twice.

## 3. Before every draw: the rule engine's vote, logged to the tracker
Seconds per game with --nulls 0 (the controls are only needed for the monthly check below):
```
!python rule_miner.py --game all --nulls 0 --tracker /content/drive/MyDrive/tracker_v3.csv
```
Log picks on the day BEFORE the draw. v3 counts a pick only if it was logged on an earlier calendar day than its draw
(draw times are not in the data). Use --allow-same-day when scoring only if you know the pick was made before the draw.

## 4. Score the live record
```
!python tracker.py score --csv daywa_audited.csv    --tracker /content/drive/MyDrive/tracker_v3.csv
!python tracker.py score --csv main_draw_audited.csv --tracker /content/drive/MyDrive/tracker_v3.csv
!python tracker.py score --csv noonrush_audited.csv --tracker /content/drive/MyDrive/tracker_v3.csv
!python tracker.py score --csv vag_audited.csv      --tracker /content/drive/MyDrive/tracker_v3.csv
```
The pre-registered metric is the core hit (>= 2 of the 5 drawn balls inside the 10 / 12 core). The "evidence"
line is an anytime-valid e-value: you may look every day. An edge is CONFIRMED only when it reaches 80.

## 5. Monthly: is there any pattern at all?
```
!python randomness_battery.py --sims 3000          # 20-40 min on a Colab CPU; 104 tests, BH-corrected
!python rule_miner.py --game all --nulls 50        # 2-3 h for all four: the honest scorecard vs 50 fair-machine histories
!python rule_miner.py --game Main --plant 0.03 --nulls 10   # positive control: proves the engine can find a real rule
```

## 6. Optional: the ML ensemble (stage4) in v3 mode
v3 defaults: expanding training window from the first usable draw and EVERY draw after the warm-up is scored
(v2: last 300 only, at most 1,500 training draws). The data-mined rule engine is an expert ("rulemine"); the old
fixed-library rule memory is off unless you pass --rulemem.
```
!python stage4_astronomical.py --csv daywa_audited.csv --null-reps 100 --anchor-always-on --null-workers 0 --no-tracker
!python stage4_astronomical.py --csv daywa_audited.csv --last300 ...          # the v2 layout, for comparison
!python stage4_astronomical.py --csv daywa_audited.csv --feature-windows 5,10,20,50,100,200,400 --markov-half-life 200 --tft-seq-len 100
```
A full-history null test takes several times longer than v2's (every draw is scored in every replicate). On a T4
start with --null-reps 40 and add more only if the observed result is near significance.
