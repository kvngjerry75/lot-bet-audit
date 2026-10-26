# STAGE 5 MAIN BEAST: Main

2833 draws (2017-12-15 to 2026-10-24); 2733 forecast blind (each from earlier draws only); machine numbers used; one pass 13s.

| Measure | Forecaster | Chance |
|---|---|---|
| hits in its top 5 | 0.274 | 0.278 |
| core-12 hit (2+ drawn) | 383 of 2733 = 14.0% | 13.0% |
| both of its top 2 drawn | 6 = 0.22% | 0.25% |
| log-loss skill vs chance | -0.00000 | 0 |

Average lens weights over the scored draws (how much each way of thinking was trusted): chance 0.27, laws 0.04, precedents 0.13, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.47, decades 0.04, odd/even 0.04, ML brain 0.01
Each lens's total log-score gain over chance (above 0 = it beat chance on the scored draws): laws -1.5, precedents -0.0, partners -67.4, analogues -155.7, base rates -101.5, low/high -0.8, decades -8.5, odd/even -1.6, ML brain -5.6

Against 20 fair-machine histories run through the same engine:

| Measure | Forecaster | Fair controls (mean) | p (controls >= forecaster) |
|---|---|---|---|
| hits in top 5 | 0.2741 | 0.2791 | 0.810 |
| core-12 hit | 0.1401 | 0.1308 | 0.095 |
| log-loss skill | -0.0000 | -0.0000 | 0.143 |

## Reasoning for the next draw (after 2026-10-24)

Top 5: [51, 49, 71, 67, 55] | core 12: [51, 49, 71, 67, 55, 69, 59, 83, 73, 63, 61, 77] | strongest pick is x1.002 chance

Lens weights now: chance 0.35, laws 0.08, precedents 0.35, partners 0.00, analogues 0.00, base rates 0.00, low/high 0.16, decades 0.00, odd/even 0.07, ML brain 0.00

```
  51: forecast 0.056 vs chance 0.056
      - law 'complement 91-x' from machine #3, 1 draw(s) ago (40): 185 hits in 2581 tries (luck 143), belief 3.1e-03
      - law '+3' from Noonrush latest machine #3 (48): 71 hits in 1078 tries (luck 60), belief 2.7e-05
      - precedent winning 63 -> 51: 13 of 178 times (luck 9.9)
      - precedent machine 40 -> 51: 12 of 161 times (luck 8.9)
      - base rate x0.99
  49: forecast 0.056 vs chance 0.056
      - law '+1' from Noonrush latest machine #3 (48): 79 hits in 1078 tries (luck 60), belief 8.7e-04
      - law '-3' from VAG latest machine #3 (52): 46 hits in 693 tries (luck 38), belief 5.0e-05
      - precedent winning 69 -> 49: 11 of 177 times (luck 9.8)
      - precedent winning 60 -> 49: 11 of 174 times (luck 9.7)
      - base rate x0.96
  71: forecast 0.056 vs chance 0.056
      - law '-6' from Noonrush latest winning #1 (77): 80 hits in 1101 tries (luck 61), belief 6.8e-04
      - law '-1' from Noonrush latest machine #5 (72): 64 hits in 1078 tries (luck 60), belief 1.3e-06
      - precedent winning 16 -> 71: 14 of 172 times (luck 9.6)
      - precedent winning 60 -> 71: 13 of 174 times (luck 9.7)
      - base rate x0.97
  67: forecast 0.056 vs chance 0.056
      - law '-5' from Noonrush latest machine #5 (72): 68 hits in 1078 tries (luck 60), belief 7.3e-06
      - law '-10' from Noonrush latest winning #1 (77): 68 hits in 1101 tries (luck 61), belief 3.7e-06
      - precedent winning 69 -> 67: 22 of 177 times (luck 9.8)
      - precedent winning 16 -> 67: 14 of 172 times (luck 9.6)
      - base rate x1.19
  55: forecast 0.056 vs chance 0.056
      - law '+7' from Noonrush latest machine #3 (48): 70 hits in 1078 tries (luck 60), belief 1.7e-05
      - law '+3' from VAG latest machine #3 (52): 42 hits in 693 tries (luck 38), belief 8.7e-06
      - precedent winning 16 -> 55: 20 of 172 times (luck 9.6)
      - precedent winning 63 -> 55: 13 of 178 times (luck 9.9)
      - base rate x1.11
  69: forecast 0.056 vs chance 0.056
      - law '-7' from Noonrush latest machine #4 (76): 68 hits in 1078 tries (luck 60), belief 7.3e-06
      - law '-3' from Noonrush latest machine #5 (72): 64 hits in 1078 tries (luck 60), belief 1.3e-06
      - precedent winning 63 -> 69: 15 of 178 times (luck 9.9)
      - precedent machine 39 -> 69: 14 of 141 times (luck 7.8)
      - base rate x1.13
```