# eval_astro2 v2: run in Google Colab

## 1. Upload and set up (once per session)
Upload eval_astro2_v2.zip to Colab (Files panel), then:

!unzip -o -q eval_astro2_v2.zip
%cd eval_astro2_v2
!pip -q install -r requirements.txt
!python test_pipeline.py                    # must end with ALL TESTS PASSED

Keep the tracker on Google Drive so it survives the session:
from google.colab import drive; drive.mount('/content/drive')

## 2. Is any machine biased? (seconds; run this first)
!python bias_check.py --csv main_draw_clean.csv noonrush_clean.csv vag_clean.csv daywa_clean.csv

## 3. Daily routine (one game; repeat for the others)
After each draw, add the result (numbers in drawn order). The CSV keeps its own layout:
!python tracker.py add-draw --csv daywa_clean.csv --date 2026-10-21 --numbers 23 32 26 11 22

BEFORE the next draw, run the prediction. It logs the pick to the tracker automatically:
!python stage4_astronomical.py --csv daywa_clean.csv --null-reps 100 --anchor-always-on --null-workers 0 --tracker /content/drive/MyDrive/tracker.csv

Score every logged pick so far against what was actually drawn:
!python tracker.py score --csv daywa_clean.csv --tracker /content/drive/MyDrive/tracker.csv

## 4. All four games, then the combined verdict
!python stage4_astronomical.py --csv main_draw_clean.csv --null-reps 100 --anchor-always-on --null-workers 0 --tracker /content/drive/MyDrive/tracker.csv
!python stage4_astronomical.py --csv noonrush_clean.csv  --null-reps 100 --anchor-always-on --null-workers 0 --tracker /content/drive/MyDrive/tracker.csv
!python stage4_astronomical.py --csv vag_clean.csv       --null-reps 100 --anchor-always-on --null-workers 0 --tracker /content/drive/MyDrive/tracker.csv
!python stage4_astronomical.py --aggregate outputs/

## 5. Rule-memory engine (learns draw by draw, like a person)
Walks every draw in order: guesses ~2,300 rules (repeats, shifts, sums, differences, last digits, mirrors,
overdue/hot/cold, calendar, each under conditions such as "when yesterday's sum was low"), tests each on the next
draw, promotes rules that keep working, demotes them when they stop, and votes with the rules in memory.
It then runs the SAME engine on fair-machine draws so you can see whether it learned anything real.
!python rule_memory.py --csv daywa_clean.csv    --null-reps 100
!python rule_memory.py --csv noonrush_clean.csv --null-reps 100
!python rule_memory.py --csv main_draw_clean.csv --null-reps 50
!python rule_memory.py --csv vag_clean.csv      --null-reps 50
Read the scorecard: a "p" below 0.05 on a hedge or memory vote row means it beat the fair machine on history.
The next-draw TOP 2 / 5 / 12 lists are printed at the end; the rule memory is saved to outputs/rule_memory_<game>.json.
Add the latest draw first (tracker.py add-draw), otherwise it predicts a draw that already happened.

## What is new in this version
* Rule memory is now an expert inside the main system (stage4_astronomical.py): its hedge vote over ~2,300 rules
  joins the TFT, ranker, hazard, markov, anchor, MLP and booster, and gets weight only when it earns it on the
  validation window, like every other expert. --no-rulemem leaves it out. Nothing changes in the commands above.
* Null test stops early: once 10 null replicates match or beat your result, more cannot make it significant, so
  --null-reps 100 usually finishes after 20-40 replicates. A result that is heading towards significance keeps
  running to the full 100. (--no-early-stop runs all of them.)
* Repeatable GPU runs: the same command on the same data now gives the same picks on a T4.
* A loud warning when the draw being predicted is already in the past (add the latest result first).
* tracker.py add-draw: adds a result without editing the CSV by hand; refuses duplicates and out-of-pool numbers.
* No more "X does not have valid feature names" warning flood.
* bias_check.py: frequency, hot-ball stability, repeats and first-ball tests for every game, corrected for the
  number of tests.
* From before: track-record rule for vote weights, --anchor-always-on, --top2-mode consensus, MLP and booster experts
  (--no-mlp / --no-booster), parallel null test (--null-workers 0 = auto), live tracker.

## Notes
* --null-workers 0 = auto: 2 workers sharing the GPU on a T4, one per 2 CPU cores on a CPU/TPU runtime.
  If Colab runs out of memory, use a fixed number.
* After a disconnect, rerun the same command: the null test resumes from its checkpoint.
* Picks logged after their draw date are marked LATE by the tracker and never counted.
* Results: outputs/report_*.md, outputs/stage4_results_*.json, logs in outputs/logs/.
