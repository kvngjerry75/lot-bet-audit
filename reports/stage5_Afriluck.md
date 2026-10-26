# STAGE 5 MAIN BEAST: Afriluck (Anopa+Midday+evening)

2059 draws (2024-06-29 to 2026-10-24); 1959 forecast blind (each from earlier draws only); machine numbers used; one pass 7s.

| Measure | Forecaster | Chance |
|---|---|---|
| hits in its top 6 | 0.660 | 0.632 |
| core-10 hit (2+ drawn) | 587 of 1959 = 30.0% | 28.1% |
| both of its top 2 drawn | 15 = 0.77% | 0.94% |
| log-loss skill vs chance | +0.00029 | 0 |

Average lens weights over the scored draws (how much each way of thinking was trusted): chance 0.26, laws 0.03, precedents 0.03, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.51, decades 0.10, odd/even 0.06, ML brain 0.01
Each lens's total log-score gain over chance (above 0 = it beat chance on the scored draws): laws -0.6, precedents +0.0, partners -16.6, analogues -130.9, base rates -31.8, low/high +14.0, decades +11.4, odd/even +0.1, ML brain -3.6

Against 20 fair-machine histories run through the same engine:

| Measure | Forecaster | Fair controls (mean) | p (controls >= forecaster) |
|---|---|---|---|
| hits in top 6 | 0.6595 | 0.6330 | 0.143 |
| core-10 hit | 0.2996 | 0.2791 | 0.095 |
| log-loss skill | 0.0003 | -0.0000 | 0.048 |

## Reasoning for the next draw (after 2026-10-24)

Top 6: [23, 27, 21, 25, 26, 24] | core 10: [23, 27, 21, 25, 26, 24, 28, 22, 1, 5] | strongest pick is x1.032 chance

Lens weights now: chance 0.32, laws 0.00, precedents 0.00, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.64, decades 0.05, odd/even 0.00, ML brain 0.00

```
  23: forecast 0.109 vs chance 0.105
      - law '+1' from machine #1, 6 draw(s) ago (22): 59 hits in 485 tries (luck 51), belief 1.7e-05
      - law '-1' from machine #1, 3 draw(s) ago (24): 56 hits in 487 tries (luck 51), belief 3.7e-06
      - precedent winning 44 -> 23: 36 of 240 times (luck 25.3)
      - precedent winning 1 -> 23: 34 of 250 times (luck 26.3)
      - base rate x1.15
  27: forecast 0.109 vs chance 0.105
      - law '+5' from machine #1, 6 draw(s) ago (22): 63 hits in 485 tries (luck 51), belief 1.1e-04
      - law '+7' from machine #5, 6 draw(s) ago (20): 61 hits in 485 tries (luck 51), belief 4.3e-05
      - precedent winning 1 -> 27: 36 of 250 times (luck 26.3)
      - precedent winning 6 -> 27: 26 of 221 times (luck 23.3)
      - base rate x1.05
  21: forecast 0.109 vs chance 0.105
      - law 'complement 57-x' from machine #4, 6 draw(s) ago (36): 61 hits in 474 tries (luck 50), belief 8.4e-05
      - law '-6' from machine #6, 5 draw(s) ago (27): 60 hits in 486 tries (luck 51), belief 2.6e-05
      - precedent winning 1 -> 21: 33 of 250 times (luck 26.3)
      - precedent winning 5 -> 21: 29 of 216 times (luck 22.7)
      - base rate x1.14
  25: forecast 0.109 vs chance 0.105
      - law 'double' from machine #1, 5 draw(s) ago (41): 66 hits in 486 tries (luck 51), belief 4.2e-04
      - law '+1' from machine #1, 3 draw(s) ago (24): 62 hits in 487 tries (luck 51), belief 6.1e-05
      - precedent winning 24 -> 25: 27 of 226 times (luck 23.8)
      - precedent winning 6 -> 25: 25 of 221 times (luck 23.3)
      - base rate x1.02
  26: forecast 0.109 vs chance 0.105
      - law '-4' from machine #6, 2 draw(s) ago (30): 60 hits in 488 tries (luck 51), belief 2.3e-05
      - law '+2' from machine #1, 3 draw(s) ago (24): 57 hits in 487 tries (luck 51), belief 5.9e-06
      - precedent winning 1 -> 26: 33 of 250 times (luck 26.3)
      - precedent winning 44 -> 26: 30 of 240 times (luck 25.3)
      - base rate x0.97
  24: forecast 0.109 vs chance 0.105
      - law '+2' from machine #5, 3 draw(s) ago (22): 63 hits in 487 tries (luck 51), belief 9.7e-05
      - law '-6' from machine #6, 2 draw(s) ago (30): 60 hits in 488 tries (luck 51), belief 2.3e-05
      - precedent winning 1 -> 24: 33 of 250 times (luck 26.3)
      - precedent winning 44 -> 24: 32 of 240 times (luck 25.3)
      - base rate x1.05
```