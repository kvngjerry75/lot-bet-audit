# STAGE 5 MAIN BEAST: Noonrush

933 draws (2023-09-11 to 2026-10-24); 833 forecast blind (each from earlier draws only); machine numbers used; one pass 4s.

| Measure | Forecaster | Chance |
|---|---|---|
| hits in its top 5 | 0.247 | 0.278 |
| core-12 hit (2+ drawn) | 83 of 833 = 10.0% | 13.0% |
| both of its top 2 drawn | 3 = 0.36% | 0.25% |
| log-loss skill vs chance | -0.00006 | 0 |

Average lens weights over the scored draws (how much each way of thinking was trusted): chance 0.22, laws 0.16, precedents 0.22, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.17, decades 0.04, odd/even 0.18, ML brain 0.02
Each lens's total log-score gain over chance (above 0 = it beat chance on the scored draws): laws -0.5, precedents -0.0, partners -49.6, analogues -73.1, base rates -56.8, low/high -0.7, decades -4.4, odd/even -0.8, ML brain -8.4

Against 20 fair-machine histories run through the same engine:

| Measure | Forecaster | Fair controls (mean) | p (controls >= forecaster) |
|---|---|---|---|
| hits in top 5 | 0.2473 | 0.2727 | 0.952 |
| core-12 hit | 0.0996 | 0.1308 | 1.000 |
| log-loss skill | -0.0001 | -0.0000 | 0.619 |

## Reasoning for the next draw (after 2026-10-24)

Top 5: [52, 46, 26, 8, 2] | core 12: [52, 46, 26, 8, 2, 10, 14, 4, 6, 36, 32, 12] | strongest pick is x1.003 chance

Lens weights now: chance 0.28, laws 0.17, precedents 0.28, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.14, decades 0.00, odd/even 0.13, ML brain 0.00

```
  52: forecast 0.056 vs chance 0.056
      - law '+6' from machine #2, 1 draw(s) ago (46): 76 hits in 916 tries (luck 51), belief 3.0e-02
      - law 'turning' from VAG latest winning #4 (25): 53 hits in 777 tries (luck 43), belief 8.5e-05
      - precedent winning 54 -> 52: 4 of 45 times (luck 2.5)
      - precedent winning 77 -> 52: 3 of 51 times (luck 2.8)
      - base rate x1.02
  46: forecast 0.056 vs chance 0.056
      - law '-3' from winning #3, 5 draw(s) ago (49): 72 hits in 928 tries (luck 52), belief 3.7e-03
      - law '-7' from VAG latest winning #5 (53): 71 hits in 933 tries (luck 52), belief 2.0e-03
      - precedent winning 39 -> 46: 5 of 42 times (luck 2.3)
      - precedent winning 54 -> 46: 3 of 45 times (luck 2.5)
      - base rate x0.95
  26: forecast 0.056 vs chance 0.056
      - law 'complement 90-x' from winning #4, 4 draw(s) ago (64): 71 hits in 919 tries (luck 51), belief 3.1e-03
      - law '-2' from DayWA latest winning #4 (28): 64 hits in 933 tries (luck 52), belief 9.7e-05
      - precedent winning 62 -> 26: 4 of 58 times (luck 3.2)
      - precedent winning 16 -> 26: 4 of 55 times (luck 3.1)
      - base rate x1.04
  8: forecast 0.056 vs chance 0.056
      - law 'digit sum' from VAG latest winning #5 (53): 61 hits in 836 tries (luck 46), belief 4.8e-04
      - law '+6' from machine #5, 4 draw(s) ago (2): 66 hits in 913 tries (luck 51), belief 4.2e-04
      - precedent machine 76 -> 8: 5 of 49 times (luck 2.7)
      - precedent machine 72 -> 8: 5 of 36 times (luck 2.0)
      - base rate x1.11
  2: forecast 0.056 vs chance 0.056
      - law 'digit sum' from machine #2, 6 draw(s) ago (20): 61 hits in 819 tries (luck 46), belief 7.9e-04
      - law 'double' from machine #2, 1 draw(s) ago (46): 63 hits in 916 tries (luck 51), belief 1.0e-04
      - precedent machine 76 -> 2: 6 of 49 times (luck 2.7)
      - precedent winning 77 -> 2: 4 of 51 times (luck 2.8)
      - base rate x0.93
  10: forecast 0.056 vs chance 0.056
      - law 'digit sum' from machine #3, 2 draw(s) ago (19): 59 hits in 823 tries (luck 46), belief 2.9e-04
      - law '+5' from machine #1, 2 draw(s) ago (5): 64 hits in 915 tries (luck 51), belief 1.7e-04
      - precedent winning 77 -> 10: 6 of 51 times (luck 2.8)
      - precedent machine 38 -> 10: 6 of 51 times (luck 2.8)
      - base rate x1.00
```