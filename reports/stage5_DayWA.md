# STAGE 5 MAIN BEAST: DayWA

2076 draws (2019-11-26 to 2026-10-24); 1976 forecast blind (each from earlier draws only); machine numbers used; one pass 5s.

| Measure | Forecaster | Chance |
|---|---|---|
| hits in its top 5 | 0.640 | 0.641 |
| core-10 hit (2+ drawn) | 776 of 1976 = 39.3% | 38.1% |
| both of its top 2 drawn | 22 = 1.11% | 1.35% |
| log-loss skill vs chance | -0.00001 | 0 |

Average lens weights over the scored draws (how much each way of thinking was trusted): chance 0.23, laws 0.18, precedents 0.22, partners 0.01, analogues 0.00, base rates 0.00, low/high 0.18, decades 0.06, odd/even 0.12, ML brain 0.01
Each lens's total log-score gain over chance (above 0 = it beat chance on the scored draws): laws -0.2, precedents -0.0, partners -16.3, analogues -138.7, base rates -40.6, low/high +1.0, decades -0.9, odd/even -1.0, ML brain -24.2

Against 20 fair-machine histories run through the same engine:

| Measure | Forecaster | Fair controls (mean) | p (controls >= forecaster) |
|---|---|---|---|
| hits in top 5 | 0.6397 | 0.6472 | 0.714 |
| core-10 hit | 0.3927 | 0.3847 | 0.333 |
| log-loss skill | -0.0000 | -0.0000 | 0.238 |

## Reasoning for the next draw (after 2026-10-24)

Top 5: [7, 4, 10, 6, 8] | core 10: [7, 4, 10, 6, 8, 2, 5, 1, 3, 9] | strongest pick is x1.009 chance

Lens weights now: chance 0.20, laws 0.13, precedents 0.15, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.40, decades 0.06, odd/even 0.06, ML brain 0.00

```
  7: forecast 0.129 vs chance 0.128
      - law 'digit sum' from VAG latest winning #4 (25): 117 hits in 693 tries (luck 89), belief 3.0e-03
      - law 'digit sum' from Main latest winning #1 (16): 109 hits in 698 tries (luck 89), belief 4.4e-05
      - precedent winning 19 -> 7: 40 of 267 times (luck 34.2)
      - precedent winning 17 -> 7: 37 of 274 times (luck 35.1)
      - base rate x1.05
  4: forecast 0.129 vs chance 0.128
      - law '+1' from VAG latest machine #4 (3): 51 hits in 294 tries (luck 38), belief 8.0e-04
      - law '+4' from Noonrush latest winning #5 (39): 63 hits in 459 tries (luck 59), belief 8.8e-07
      - precedent winning 22 -> 4: 41 of 276 times (luck 35.4)
      - precedent winning 19 -> 4: 41 of 267 times (luck 34.2)
      - base rate x1.06
  10: forecast 0.129 vs chance 0.128
      - law '+7' from VAG latest machine #4 (3): 50 hits in 294 tries (luck 38), belief 5.0e-04
      - law '-6' from Noonrush latest winning #3 (16): 72 hits in 471 tries (luck 60), belief 2.7e-05
      - precedent winning 19 -> 10: 40 of 267 times (luck 34.2)
      - precedent winning 17 -> 10: 34 of 274 times (luck 35.1)
      - base rate x1.02
  6: forecast 0.129 vs chance 0.128
      - law '+3' from VAG latest machine #4 (3): 47 hits in 294 tries (luck 38), belief 1.2e-04
      - law '-10' from Noonrush latest winning #3 (16): 63 hits in 471 tries (luck 60), belief 3.5e-07
      - precedent winning 22 -> 6: 37 of 276 times (luck 35.4)
      - precedent winning 17 -> 6: 36 of 274 times (luck 35.1)
      - base rate x1.04
  8: forecast 0.129 vs chance 0.128
      - law '-8' from Noonrush latest winning #3 (16): 62 hits in 471 tries (luck 60), belief 2.2e-07
      - law '+9' from Noonrush latest machine #1 (38): 51 hits in 414 tries (luck 53), belief 8.4e-08
      - precedent winning 19 -> 8: 39 of 267 times (luck 34.2)
      - precedent winning 22 -> 8: 37 of 276 times (luck 35.4)
      - base rate x1.05
  2: forecast 0.129 vs chance 0.128
      - law 'complement 40-x' from Noonrush latest machine #1 (38): 62 hits in 414 tries (luck 53), belief 1.7e-05
      - law '+3' from Noonrush latest machine #1 (38): 61 hits in 414 tries (luck 53), belief 1.0e-05
      - precedent winning 22 -> 2: 41 of 276 times (luck 35.4)
      - precedent winning 19 -> 2: 40 of 267 times (luck 34.2)
      - base rate x1.06
```