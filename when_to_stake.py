#!/usr/bin/env python3
"""
when_to_stake.py: does Afriluck's pink-ball edge come and go in a way you can see coming? If yes, stake only then.

    python when_to_stake.py --afriluck-dir afriluck_output

1. DIRECT CHECKS: pink balls per draw by session (Anopa / Midday / evening), by weekday, and whether pink-heavy
   draws come in runs (they would if a strongly biased ball set is used on some days and not others).
2. TIMING MODEL: before each draw, a model trained only on earlier draws predicts whether the pink 10-ball core will
   get 2+ winners, from: session, weekday, pink share of the last 1 / 5 / 20 / 100 draws, and days since the
   previous draw. Stake only on the draws it rates highest. Its stake-day hit rate and Perm-2 return are compared
   with (a) staking every draw and (b) the same model run on 50 FAKE histories where the pink edge is exactly as
   strong on every draw. If the real stake days do no better than the fake ones, there is no timing signal.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

NB, K, PINK, CORE = 57, 6, 28, 10


def load(folder):
    parts = []
    for rank, g in enumerate(("anopa", "midday", "evening")):
        r = pd.read_csv(Path(folder) / f"afriluck_{g}.csv")
        num = [f"Number{i}" for i in range(1, 7)]
        r = r[r[num].notna().all(axis=1)].copy()
        r["Date"] = pd.to_datetime(r["Date"])
        r["key"] = r[num].astype(int).apply(lambda v: tuple(sorted(v)), axis=1)
        r = r.drop_duplicates(subset=["Date", "key"])
        r["session"] = rank
        parts.append(r)
    df = pd.concat(parts).sort_values(["Date", "session"], kind="stable").reset_index(drop=True)
    keep, seen = [], {}
    for dt, kk in zip(df["Date"], df["key"]):
        keep.append(kk not in seen or (dt - seen[kk]).days > 3)
        seen[kk] = dt
    df = df[np.array(keep)].reset_index(drop=True)
    W = df[[f"Number{i}" for i in range(1, 7)]].astype(int).to_numpy() - 1
    return W, pd.DatetimeIndex(df["Date"]), df["session"].to_numpy()


def core_for(t, X):
    """The pink core used on draw t: the 10 pink balls with the best record before t (colour model + tie-break)."""
    c = X[:t, :PINK].sum(0)
    return np.argsort(-(c + np.arange(PINK) * 1e-9))[:CORE]


def features(pk, dates, session):
    n = len(pk)
    s = pd.Series(pk.astype(float))
    f = {}
    for w in (1, 5, 20, 100):
        f[f"pink_last{w}"] = s.shift(1).rolling(w, min_periods=1).mean().fillna(K * PINK / NB).to_numpy()
    for j in range(3):
        f[f"session{j}"] = (session == j).astype(float)
    for j in range(7):
        f[f"dow{j}"] = (dates.dayofweek == j).astype(float)
    gap = np.r_[1, np.diff(dates.values).astype("timedelta64[D]").astype(int)]
    f["days_gap"] = np.minimum(gap, 7)
    return np.column_stack(list(f.values()))


def evaluate(W, dates, session, keep=0.3, warm=400, refit=100):
    n = len(W)
    X = np.zeros((n, NB))
    X[np.repeat(np.arange(n), K), W.ravel()] = 1
    pk = X[:, :PINK].sum(1)
    hits = np.array([X[t, core_for(t, X)].sum() for t in range(n)])
    y = (hits >= 2).astype(int)
    F = features(pk, dates, session)
    score = np.full(n, np.nan)
    for a in range(warm, n, refit):
        m = LogisticRegression(C=0.3, max_iter=2000).fit(F[100:a], y[100:a])
        b = min(a + refit, n)
        score[a:b] = m.predict_proba(F[a:b])[:, 1]
    s = slice(warm, n)
    sc, yy, hh = score[s], y[s], hits[s]
    sel = sc >= np.quantile(sc, 1 - keep)
    perm = lambda h: 60 * np.mean([math.comb(int(x), 2) for x in h]) / math.comb(CORE, 2)
    return dict(all=yy.mean(), stake=yy[sel].mean(), n_stake=int(sel.sum()), n=len(yy),
                perm_all=perm(hh), perm_stake=perm(hh[sel]), pk=pk)


def fake(n, ratio, rng):
    """Draws with a constant pink edge: each pink ball `ratio` times as likely to come out as a yellow one."""
    w = np.where(np.arange(NB) < PINK, ratio, 1.0)
    g = np.log(w)[None, :] + rng.gumbel(size=(n, NB))
    return np.argsort(-g, axis=1)[:, :K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--controls", type=int, default=50)
    a = ap.parse_args()
    W, dates, session = load(a.afriluck_dir)
    n = len(W)
    pk = (W < PINK).sum(1)
    print(f"Afriluck: {n} draws. Pink balls per draw: {pk.mean():.3f} (fair machine {K * PINK / NB:.3f})\n")
    print("1. Direct checks")
    for j, nm in enumerate(("Anopa", "Midday", "evening")):
        m = session == j
        print(f"   {nm:8s}: {pk[m].mean():.3f} pink per draw over {m.sum()} draws")
    print("   by weekday: " + ", ".join(f"{d}: {pk[dates.dayofweek == i].mean():.2f}"
                                       for i, d in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                                       if (dates.dayofweek == i).sum() > 30))
    # runs: lag-1 correlation of pink count, and day-level variance vs constant-edge fakes
    ratio = (pk.mean() / PINK) / ((K - pk.mean()) / (NB - PINK))
    rng = np.random.default_rng(3)

    def stats(pkx):
        lag1 = np.corrcoef(pkx[1:], pkx[:-1])[0, 1]
        day = pd.Series(pkx).groupby(dates.normalize().values).mean()
        return lag1, day.var()
    r_lag, r_var = stats(pk)
    sims = np.array([stats((fake(n, ratio, rng) < PINK).sum(1)) for _ in range(500)])
    print(f"   do pink-heavy draws come in runs? lag-1 correlation {r_lag:+.3f} (constant-edge fakes "
          f"{sims[:, 0].mean():+.3f}, p {(1 + (sims[:, 0] >= r_lag).sum()) / 501:.3f})")
    print(f"   do some DAYS run much pinker than others? day-to-day variance {r_var:.3f} (fakes {sims[:, 1].mean():.3f}, "
          f"p {(1 + (sims[:, 1] >= r_var).sum()) / 501:.3f})")
    print(f"   (pink ball edge estimated at x{ratio:.3f} per ball vs yellow)\n")
    print("2. Timing model (stake only on the 30% of draws it rates highest)")
    real = evaluate(W, dates, session)
    ctrl = []
    for i in range(a.controls):
        Wf = fake(n, ratio, rng)
        ctrl.append(evaluate(Wf, dates, session))
    gain = real["stake"] - real["all"]
    cg = np.array([c["stake"] - c["all"] for c in ctrl])
    pg = np.array([c["perm_stake"] - c["perm_all"] for c in ctrl])
    print(f"   real draws: 2+ in core on stake days {real['stake']:.1%} ({real['n_stake']} draws) vs every draw "
          f"{real['all']:.1%} ({real['n']} draws): gain {gain * 100:+.1f} points")
    print(f"   Perm-2 back: stake days {real['perm_stake']:.0%} vs every draw {real['perm_all']:.0%}")
    print(f"   constant-edge fakes (no timing signal by construction): average gain {cg.mean() * 100:+.1f} points; "
          f"p (fakes >= real) {(1 + (cg >= gain).sum()) / (len(cg) + 1):.3f}; Perm-2 gain on fakes {pg.mean() * 100:+.1f} points")
    print("\nVERDICT: " + ("the timing model finds stake days that beat the constant-edge fakes - worth a forward test."
                         if (1 + (cg >= gain).sum()) / (len(cg) + 1) < 0.05 else
                         "no timing signal: the pink edge is the same on every draw, so the best 'when' is any draw "
                         "you choose to play - timing adds nothing."))


if __name__ == "__main__":
    main()
