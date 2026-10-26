# Can you beat a lottery with data science? A rigorous forecasting & audit pipeline

> I built a forecasting system to test whether two real betting markets — Ghana's lotteries and
> international football — can be beaten with data. This repo is the honest answer, with the code and the
> statistics behind it. Spoiler: one of the two has a real, measurable edge. The other does not, and I can
> prove it.

**Why this project is worth your time (if you're hiring or curious):** most "prediction systems" online are
sold on hope. This one is built to *disprove itself* if the signal isn't real — walk-forward evaluation,
fair-machine controls, planted-signal tests, multiple-testing correction, and confidence intervals on every
claim. That discipline is the whole point.

---

## The question

Two markets, one method:

1. **Ghana lottery draws** (Afriluck 6/57, three draws a day; NLA DayWA 5/39 and three 5/90 games). ~2,000
   draws each, scraped from the operators.
2. **International football** — 238,854 club matches across 38 divisions and 27 countries (2000–2026), with
   real bookmaker odds.

For each: is there a pattern you can actually make money on, and how sure can we be?

---

## Headline findings

| Market | Is there a real signal? | Can you profit from it? |
|---|---|---|
| **Afriluck lottery** | **Yes** — a physical ball bias (one colour of ball comes out ~5% more than fair, z ≈ 6, confirmed against 200 simulated fair machines, p = 0.005) | **No.** Even with the bias known *exactly*, the best bet returns ~0.62 per GH₵1. The payout is bigger than the edge. |
| **NLA DayWA / Main / Noonrush / VAG** | No (at chance after correcting for the number of models tried) | No |
| **Football — predicting winners** | The bookmaker's prices predict better than my models | No |
| **Football — shopping for the best price** | **Yes** | **Yes:** a pre-registered rule returned **1.12 per GH₵1** over 832 bets on a 10-year out-of-sample test (p = 0.0003, survives multiple-testing correction) — see the caveats before you get excited |

The interesting result isn't "I found a way to win." It's that the **same rigorous method** says *no* to the
lottery and *yes* (with heavy caveats) to one narrow football strategy — because the method is built to tell
the difference honestly.

---

## How it's built

```
scrape  →  clean & audit  →  walk-forward models  →  honest evaluation  →  money, in real currency
```

- **Scrapers** that pull live results/odds and handle each operator's quirks (delayed draws, duplicate
  variants, missing rows, bad data).
- **Audited cleaning**: removes copies, out-of-pool numbers, cross-contamination between games, and logs
  every removal.
- **Models** (all walk-forward — every forecast for draw/match *t* uses only data before *t*):
  - a self-tuning hierarchical empirical-Bayes rate model (the lottery colour bias lives here),
  - Elo ratings and a Poisson goals model for football,
  - logistic stacking of a model with the market's own prices,
  - a Bayesian model-averaging combiner.
- **Evaluation that tries to kill the result:**
  - *Walk-forward only* — no peeking at the future (verified by a leakage test that scrambles future data and
    checks nothing earlier moves),
  - *fair-machine / fair-market controls* — the same pipeline run on simulated unbiased histories,
  - *planted-signal positive controls* — a known effect is injected to prove the detector finds it,
  - *nested tuning* — every setting chosen on older data, scored once on the newest, untouched slice,
  - *Benjamini–Hochberg correction* across every strategy tried,
  - *bootstrap confidence intervals* and realised profit/loss in cedis.

---

## Selected numbers

- Afriluck colour bias: **3.10 "pink" balls per draw vs 2.95 expected** (2,061 draws, z ≈ 5.9); Plackett–Luce
  exit strength **×1.112** (95% CI 1.07–1.15).
- Break-even needs a pair of balls to be **×1.77** more likely than fair; the bias delivers ~×1.11. Hence
  ~0.62 back per GH₵1 at best — a structural loss, confirmed over 1,761 walk-forward draws (64.4% returned).
- Football, systematic strategies at a single bookmaker all lose (favourites 0.955, accumulators 0.88 and
  falling with each leg, longshots 0.75). Only cross-bookmaker **price shopping on short-priced bets** cleared
  break-even out-of-sample.

---

## Repo layout

| Path | What it does |
|---|---|
| `main_beast.py` | one runner: every model on one lottery game, walk-forward, with the next-draw forecast |
| `bet_lab.py` | every lottery bet type replayed in GH₵ with CIs, drawdowns, controls |
| `dependence.py` | searches for structure beyond the colour bias (pairs, regime changes, machine numbers) |
| `sports_lab.py` | the football study: 238k matches, every strategy, nested + out-of-sample |
| `tuning_lab.py` | proves more tuning doesn't raise the hit rate (124 variants, scored out-of-sample) |
| `forward_test.py` | logs picks *before* each draw and scores them, with a pre-registered stop rule |
| `test_audit.py` | the checks: leakage, payout maths, positive/negative controls |
| `reports/` | full write-ups and every run's output |

## Run it

```bash
pip install -r requirements.txt
python main_beast.py --game Afriluck          # lottery forecast + honest money verdict
python sports_lab.py --matches matches.csv    # the football study
python test_audit.py                          # the evidence checks
```

## Tech

Python · NumPy · pandas · scikit-learn · SciPy. Walk-forward evaluation, empirical-Bayes, Elo, Poisson,
Plackett–Luce, bootstrap, Benjamini–Hochberg.

## What I'd stress in an interview

- I set out to win and the data said *you can't* for the lottery — and I reported that, loudly, instead of
  burying it. Knowing when **not** to bet is the result.
- Every positive claim here is one a sceptic can reproduce and attack. That's deliberate.

## Honest limitations

- Lottery payouts are inferred from the operator's public winners feed, not an official table.
- The football price-shopping edge is small, shrinking, needs accounts at many bookmakers, and bookmakers
  restrict winners — it is **not** easy money, and the repo says so in full.
- This is research, not financial advice. Betting loses money on average by design.

---

*Author: **Jeremiah Asiedu** · [github.com/kvngjerry75](https://github.com/kvngjerry75) · kvngjerry75@gmail.com · [jamsolutionshq.com](https://jamsolutionshq.com) · Accra, Ghana*
