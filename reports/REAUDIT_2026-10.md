# Re-audit of the updated system (v2), 26 Oct 2026

This re-audit follows your original brief and the audit skill's checklist, run on the updated bundle from a fresh
unzip. Question asked: **can the hit rates go any higher?**

## Verdict

**No. The updated system is correct, and its hit rates are already at the ceiling the machines allow.** I tried
every remaining idea. Each one was tuned on older draws and then judged on the newest 750 draws it had never seen.
None beat the current model; the ones that looked best on old draws did worse on new ones. Main, Noonrush and VAG
are still at chance.

| Game | Fair machine | Current model (all draws scored) | Current model (newest 750) | Ceiling if the bias were known exactly |
|---|---|---|---|---|
| Afriluck, 2+ in 10-core | 28.1% | 31.5% | 32.0% | **30.6%** |
| DayWA, 2+ in 10-core | 38.1% | 40.3% | 40.1% | **39.3%** |
| Main / Noonrush / VAG, 2+ in 10-core | 9.4% | 9.5% / 9.2% / 9.1% | – | at chance (p = 0.45 / 0.62 / 0.73) |

The model ran slightly above its ceiling over the scored draws. That gap is luck, so expect about 30.6% and 39.3%
in future.

## 1. Reproduction (fresh unzip)

- `test_v3.py`, `test_pipeline.py` and `test_audit.py`: all pass (the 2 TFT tests are skipped here because torch is
  not installed).
- `main_beast.py`: Afriluck 31.5%, Perm-2 back 0.644 / 0.720 (core 10 / core 3); DayWA 40.3%.
- `sharp.py` with 50 controls: Afriluck p = 0.020 (the smallest p 50 controls can give); DayWA p = 0.078.
- `profit_history.py`: 1,761 draws, GH₵79,245 staked, 64.4% back, −GH₵28,245; beats 99.7% of random players.
- All of these match the first audit exactly.

## 2. Code review of the changes

I checked every changed line against your original bundle:

- `forecaster.py`: the bad machine row fix and the real next-draw date are correct.
- `sharp.py`: the "current core" now uses the latest draw.
- `profit_history.py`: `--core` and `--component` work as described.
- `features_recent.py`: the long-run columns use draws strictly before t. They stay banned unless the Afriluck
  switch is on, and the ban list is copied, never edited in place.
- `main_beast.py`: `ASTRO_K`, the pruned combiner, the schedule and the money lines are correct.
- The new tools (`bet_lab.py`, `dependence.py`, `forward_test.py`, `test_audit.py`) are correct.

**No new bugs, and no leakage:** the future-scramble test passes on every component.

## 3. Attempts to raise the rate (nested: chosen on older draws, scored on the newest 750)

| Idea | Afriluck, newest 750 | DayWA, newest 750 | Verdict |
|---|---|---|---|
| **Current model** | **32.0%** | **40.1%** | keep |
| Best of 124 tuned variants (groupings, pull strength, memory length), chosen by hit rate | 30.9% | 38.1% | worse: overfits |
| The same, chosen by log-score | 30.5% | 35.3% | worse |
| Average of all 124 | 30.7% | 39.1% | worse |
| Average of the top 5 on older draws | 28.8% | 40.0% | worse |
| Cores spread across decades (draws are slightly anti-clustered) | 31.9% | 36.3% | worse |
| Ball strengths from the exit order (Plackett–Luce), setting chosen on older draws | 33.1% | – | +1.1 points, 95% CI −2.5 to +4.7; it was *worse* on older draws (28.6% vs 32.0%); fair machines produce this gap routinely (0.6 sd). **Noise** |
| 12-exit model (winning + machine numbers) | 30.9% | – | worse: the machine numbers carry the opposite colour tilt |
| DayWA ball-set change / drift | – | CUSUM p = 0.41, single split p = 0.87, ball × period p = 0.66, per-ball spread p = 0.55 | no change to exploit |

## 4. The audit skill's checklist

1. **"The stats are near luck."** True for Main, Noonrush and VAG (re-tested against 100 fair machines each). Not
   true for Afriluck: the pink edge is real (p = 0.005 with 200 controls), but it is capped at about 30.6%. DayWA is
   borderline and not proven.
2. **"Training on only 300 draws."** Not the case in this code. Stage 4 trains on all history from the first draw
   (`train_window` 10⁹), and every draw after warm-up is scored (full layout). Sharp and the MAIN BEAST also use the
   whole history, weighted by record.
3. **"Only 20 past draws."** The TFT reads a 20-draw sequence, and some features use 5/10/20/50-draw rates. All-history
   features sit alongside them: the anchor's 250 / 500 / all windows, sharp's full history, and Afriluck's new
   long-run columns. Testing memory lengths of 250 / 600 / 1,500 / all draws found that **using all draws without
   forgetting is best**.
4. **"The rule memory is dumb."** It was replaced in v3 by `rule_miner.py`, which mines rules from the data with
   sequential tests and planted-rule controls. That is the design you described. It finds planted rules and finds
   nothing on the real draws.
5. **"There is a real mathematical pattern."** The only patterns found are physical and small: Afriluck's pink-first
   ordering, the yellow-heavy machine numbers, slightly anti-clustered draws, and DayWA's +2% low-number tilt. None
   raises the hit rate above the ceilings in the table at the top.

## 5. What still raises your chance of winning a given draw

Covering more numbers. Afriluck Perm-2 on the model's top N, over 1,761 draws:

| Numbers covered | Stake per draw | Wins something | Average loss per draw |
|---|---|---|---|
| 3 | GH₵3 | 1 draw in 31 | GH₵0.90 |
| 6 | GH₵15 | 1 in 8 | GH₵5.10 |
| 10 | GH₵45 | 1 in 3 | GH₵15.60 |
| 15 | GH₵105 | 1 in 2 | GH₵36.20 |

The model's numbers win more often than random ones at every size, but the cost grows faster than the winnings.

## Reproduce

```
!python test_audit.py
!python tuning_lab.py --game Afriluck
!python tuning_lab.py --game DayWA
!python tuning_lab.py --game Main
```
