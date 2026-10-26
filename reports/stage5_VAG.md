# STAGE 5 MAIN BEAST: VAG

2039 draws (2019-08-13 to 2026-10-24); 1939 forecast blind (each from earlier draws only); machine numbers used; one pass 7s.

| Measure | Forecaster | Chance |
|---|---|---|
| hits in its top 5 | 0.288 | 0.278 |
| core-12 hit (2+ drawn) | 244 of 1939 = 12.6% | 13.0% |
| both of its top 2 drawn | 6 = 0.31% | 0.25% |
| log-loss skill vs chance | +0.00001 | 0 |

Average lens weights over the scored draws (how much each way of thinking was trusted): chance 0.22, laws 0.13, precedents 0.18, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.24, decades 0.12, odd/even 0.09, ML brain 0.01
Each lens's total log-score gain over chance (above 0 = it beat chance on the scored draws): laws -0.7, precedents -0.0, partners -57.1, analogues -111.5, base rates -86.8, low/high +2.0, decades +0.5, odd/even -1.1, ML brain -8.3

Against 20 fair-machine histories run through the same engine:

| Measure | Forecaster | Fair controls (mean) | p (controls >= forecaster) |
|---|---|---|---|
| hits in top 5 | 0.2883 | 0.2765 | 0.143 |
| core-12 hit | 0.1258 | 0.1284 | 0.619 |
| log-loss skill | 0.0000 | -0.0000 | 0.143 |

## Reasoning for the next draw (after 2026-10-24)

Top 5: [47, 49, 48, 46, 50] | core 12: [47, 49, 48, 46, 50, 55, 59, 57, 53, 51, 85, 58] | strongest pick is x1.020 chance

Lens weights now: chance 0.25, laws 0.03, precedents 0.07, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.50, decades 0.11, odd/even 0.02, ML brain 0.00

```
  47: forecast 0.057 vs chance 0.056
      - law '-6' from winning #5, 1 draw(s) ago (53): 146 hits in 2038 tries (luck 113), belief 1.5e-03
      - law '-5' from machine #3, 1 draw(s) ago (52): 44 hits in 603 tries (luck 34), belief 3.1e-04
      - precedent winning 60 -> 47: 9 of 139 times (luck 7.7)
      - precedent winning 53 -> 47: 9 of 109 times (luck 6.1)
      - base rate x1.17
  49: forecast 0.057 vs chance 0.056
      - law '+5' from machine #2, 7 draw(s) ago (44): 45 hits in 597 tries (luck 33), belief 5.7e-04
      - law '-8' from machine #3, 2 draw(s) ago (57): 40 hits in 602 tries (luck 33), belief 5.5e-05
      - precedent winning 60 -> 49: 9 of 139 times (luck 7.7)
      - precedent winning 25 -> 49: 8 of 103 times (luck 5.7)
      - base rate x1.10
  48: forecast 0.057 vs chance 0.056
      - law '-6' from Noonrush latest winning #2 (54): 71 hits in 938 tries (luck 52), belief 1.8e-03
      - law '+4' from machine #2, 7 draw(s) ago (44): 47 hits in 597 tries (luck 33), belief 1.4e-03
      - precedent winning 77 -> 48: 15 of 127 times (luck 7.1)
      - precedent winning 60 -> 48: 9 of 139 times (luck 7.7)
      - base rate x1.12
  46: forecast 0.057 vs chance 0.056
      - law '+5' from machine #1, 1 draw(s) ago (41): 46 hits in 603 tries (luck 34), belief 7.4e-04
      - law '+8' from Noonrush latest machine #1 (38): 64 hits in 921 tries (luck 51), belief 1.4e-04
      - precedent winning 25 -> 46: 10 of 103 times (luck 5.7)
      - precedent winning 53 -> 46: 10 of 109 times (luck 6.1)
      - base rate x1.11
  50: forecast 0.057 vs chance 0.056
      - law '+1' from machine #2, 5 draw(s) ago (49): 44 hits in 599 tries (luck 33), belief 3.5e-04
      - law '-5' from machine #2, 4 draw(s) ago (55): 42 hits in 600 tries (luck 33), belief 1.4e-04
      - precedent winning 60 -> 50: 14 of 139 times (luck 7.7)
      - precedent winning 77 -> 50: 8 of 127 times (luck 7.1)
      - base rate x0.97
  55: forecast 0.057 vs chance 0.056
      - law '+2' from machine #4, 7 draw(s) ago (53): 50 hits in 597 tries (luck 33), belief 5.4e-03
      - law '+4' from machine #5, 4 draw(s) ago (51): 37 hits in 600 tries (luck 33), belief 1.6e-05
      - precedent winning 53 -> 55: 12 of 109 times (luck 6.1)
      - precedent winning 66 -> 55: 10 of 116 times (luck 6.4)
      - base rate x1.12
```