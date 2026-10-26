# Afriluck lab: every model, walk-forward, vs fair-machine controls

2059 draws pooled (Anopa, Midday, evening), 2024-06-29 to 2026-10-24. Scored on the last 1759 draws; each predicted from earlier draws only.

| Model | Hits in top 6 | Core-12 hit (2+) | Log-loss skill | Return: Direct-1 first ball (30x) | Return: Direct-2 (60x) | Return: Direct-3 (600x) | Return: Direct-4 (10000x) | p (top-6 vs fair controls) |
|---|---|---|---|---|---|---|---|---|
| chance (theory) | 0.632 | 0.372 | 0 | 53% | 56% | 41% | 38% | - |
| colour | 0.671 | 0.420 | +0.0004 | 49% (29 wins) | 51% (15 wins) | 136% (4 wins) | 0% (0 wins) | 0.091 |
| frequency | 0.674 | 0.401 | -0.0003 | 65% (38 wins) | 61% (18 wins) | 68% (2 wins) | 0% (0 wins) | 0.091 |
| recent50 | 0.625 | 0.395 | -0.0280 | 41% (24 wins) | 65% (19 wins) | 34% (1 wins) | 0% (0 wins) | 0.545 |
| overdue | 0.587 | 0.336 | -0.0002 | 60% (35 wins) | 51% (15 wins) | 0% (0 wins) | 0% (0 wins) | 1.000 |
| markov | 0.638 | 0.378 | -0.0014 | 51% (30 wins) | 44% (13 wins) | 0% (0 wins) | 0% (0 wins) | 0.091 |
| boosting | 0.681 | 0.410 | +0.0002 | 55% (32 wins) | 65% (19 wins) | 0% (0 wins) | 0% (0 wins) | 0.091 |
| rule_engine | 0.633 | 0.383 | -0.0000 | 68% (40 wins) | 75% (22 wins) | 68% (2 wins) | 0% (0 wins) | 0.273 |
| ensemble | 0.687 | 0.396 | +0.0004 | 56% (33 wins) | 61% (18 wins) | 68% (2 wins) | 0% (0 wins) | 0.091 |

Return = money back per 1 staked, every draw, on the model's top balls. 100% = break even. Direct-3 and Direct-4 win so rarely that a single lucky hit swings their return; read them as noise unless the win count is large.
Fair-machine controls: 10. With few controls the smallest possible p is 0.091.