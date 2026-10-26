# I spent months trying to beat Ghana's lottery with data science. Here's the honest answer.

*A story about a physical bias that's real, a payout that's bigger than the bias, and the one betting market
where the maths actually works.*

---

I want to tell you what happens when you point real data science at the games people around me bet on every
day — not to sell you a system, but because the honest answer turned out to be more interesting than a fake
one.

I built a forecasting pipeline: scrapers, cleaning, walk-forward models, and the kind of evaluation that's
designed to catch *me* fooling myself. Then I aimed it at two things: Ghana's lotteries, and international
football betting.

## Part 1: The lottery has a real flaw. You still can't win.

Afriluck is a 6/57 gravity-machine draw, three times a day. I scraped 2,061 draws and found something real:
one colour of ball comes out more often than it should. Pink balls (1–28) show up **3.10 times per draw**
when a fair machine would give **2.95**. Over 2,000 draws that's a z-score of about 6 — not luck. I confirmed
it by running the whole analysis on 200 simulated *fair* machines; the real machine beat them with p = 0.005.

So I'd found a genuine physical bias. The dream, right?

Here's the wall. To break even on the cheapest bet (Perm-2, where you back pairs of numbers), a pair of balls
has to come up **1.77 times** more often than fair. The pink bias delivers about **1.11 times**. I simulated a
*perfect* forecaster that knew the bias exactly, with zero error — it still returned about **62 pesewas for
every cedi staked**. The payout is simply set bigger than the edge. I replayed 1,761 real draws betting every
one: staked GH₵79,245, got back GH₵51,000. A 36% loss, exactly as the maths predicted.

I tried everything to find more: do certain numbers come out together (no), did the ball set change over time
(no), can the machine's "machine numbers" tell us anything (they're real, and they're useless for winning). I
tested 124 tuned versions of the model; none beat the simple one on data it hadn't seen. The honest ceiling is
the ceiling.

**The lottery result:** there's a real bias, it's measurable, and it is not enough. Anyone selling you lottery
numbers is selling you a 36%-a-draw loss with extra steps.

## Part 2: Football — where the maths flips

Then I turned the same machine on 238,854 football matches across 27 countries, with real bookmaker odds.

First, the bad news for tipsters: **predicting match results doesn't beat the bookmaker.** I built rating
models (Elo, Poisson goals models, the works). The bookmaker's own prices were *better* forecasts than my
models, and every popular system lost money at a single bookmaker — backing favourites (0.955 per cedi),
accumulators (worse with every leg), longshots (0.75).

But one thing worked. Not predicting — *price shopping*. Different bookmakers post different odds on the same
match. If you take the fair price, then only bet when some bookmaker is offering clearly more than that, on
short-priced games, a rule I fixed *before* looking at the test data returned **1.12 per cedi over 832 bets in
a 10-year out-of-sample test** (p = 0.0003, and it survives correction for everything else I tried).

The caveats are real and I won't hide them: it's a thin edge, it's shrinking as bookmakers get sharper, it
needs accounts at many bookmakers, and bookmakers limit people who win this way. It's a grind, not a jackpot.
But it's the one place where the evidence said *yes*.

## The point

The same method said **no** to the lottery and a careful **yes** to one football strategy. That's the whole
value of doing it honestly: a method built to disprove itself can actually tell the difference between a real
edge and wishful thinking.

Most people never run the fair-machine controls, never test out-of-sample, never correct for the twenty things
they tried. So they find "patterns" everywhere and lose money everywhere. I went looking for an edge, found
that the obvious ones are mirages, and found the one small real one hiding somewhere unglamorous.

If you take one thing from this: **knowing when not to bet is a result worth having.** It's the difference
between data science and fortune-telling.

---

*Code, data and the full statistical write-up are on my GitHub:
[github.com/kvngjerry75/lot-bet-audit](https://github.com/kvngjerry75/lot-bet-audit). Built in Python with
NumPy, pandas, scikit-learn and SciPy. Questions welcome — kvngjerry75@gmail.com ·
[jamsolutionshq.com](https://jamsolutionshq.com).*

*This is research, not financial advice. On average, betting loses money by design; that's rather the point of
the article.*
