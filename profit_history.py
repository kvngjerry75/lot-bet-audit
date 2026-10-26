#!/usr/bin/env python3
"""
profit_history.py: what you would have won and lost, draw by draw, playing the MAIN BEAST core on Afriluck Perm-2.

    python main_beast.py --game Afriluck --afriluck-dir afriluck_output --save afl_forecasts.npz
    python profit_history.py afl_forecasts.npz
    python profit_history.py afl_forecasts.npz --core 3      # Perm-2 on the top 3 only (3 pairs, GH₵3 a draw)

Every forecast in the file was made BEFORE its draw, from earlier draws only (walk-forward), so this is what a
player following the system every draw would really have seen. Stake: GH₵1 on each of the 45 pairs of the 10-ball
core (GH₵45 per draw); a winning pair pays 60x. Writes profit_by_draw.csv and profit_by_month.csv next to the input.
Also replays the same history with RANDOM 10-ball cores (pure luck), to show which streaks luck alone produces.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAY, CORE, WARM = 60, 10, 300
STAKE = math.comb(CORE, 2)


def _args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("forecasts", nargs="?", default="afl_forecasts.npz")
    ap.add_argument("--core", type=int, default=10, help="Perm-2 on the top N numbers (3..10)")
    ap.add_argument("--component", default="", help="score one saved component instead of the MAIN BEAST, e.g. sharp")
    return ap.parse_args()


def windows(net, dates, w):
    c = np.r_[0, np.cumsum(net)]
    s = c[w:] - c[:-w]
    b, x = int(np.argmax(s)), int(np.argmin(s))
    return (s[b], dates[b], dates[b + w - 1]), (s[x], dates[x], dates[x + w - 1]), s


def streaks(win):
    best_l = cur = 0
    end = 0
    for i, v in enumerate(win):
        cur = 0 if v else cur + 1
        if cur > best_l:
            best_l, end = cur, i
    return best_l, end


def main():
    global CORE, STAKE
    a = _args()
    CORE, STAKE = a.core, math.comb(a.core, 2)
    f = np.load(a.forecasts, allow_pickle=True)
    p, W, dates, sess = f["p"], f["W"], pd.to_datetime(f["dates"]), f["session"]
    if a.component:
        p = f["c__" + a.component]
    n, nb = W.shape[0], p.shape[1]
    k = W.shape[1]
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    rows = np.arange(WARM, n)
    order = np.argsort(-(p[rows] + np.random.default_rng(1).random((len(rows), nb)) * 1e-12), axis=1)[:, :CORE]
    hits = X[rows[:, None], order].sum(1).astype(int)
    pay = PAY * np.array([math.comb(h, 2) for h in hits])
    net = pay - STAKE
    d = dates[rows]
    df = pd.DataFrame(dict(date=d.date, session=sess[rows], core=[" ".join(str(b + 1) for b in o) for o in order],
                           drawn=[" ".join(str(b + 1) for b in W[t]) for t in rows], hits=hits, staked=STAKE,
                           returned=pay, net=net, bankroll=np.cumsum(net)))
    out = Path(a.forecasts).with_name("profit_by_draw.csv")
    df.to_csv(out, index=False)

    # chance reference
    from math import comb
    ch_pmf = np.array([comb(CORE, j) * comb(nb - CORE, k - j) / comb(nb, k) for j in range(k + 1)])
    ch_back = PAY * sum(ch_pmf[j] * comb(j, 2) for j in range(k + 1)) / STAKE
    who = a.component or "MAIN BEAST"
    print(f"# Afriluck Perm-2 on the {who} top-{CORE} core: {len(rows)} draws, {d[0].date()} to {d[-1].date()}")
    print(f"Every draw: GH₵{STAKE} on the {STAKE} pairs, 60x per winning pair. Forecasts made before each draw.\n")
    tot_s, tot_r = STAKE * len(rows), pay.sum()
    print(f"Total staked GH₵{tot_s:,} | returned GH₵{tot_r:,} | NET GH₵{tot_r - tot_s:+,} | {tot_r / tot_s:.1%} back "
          f"(pure luck would give {ch_back:.1%} back)\n")
    print("| winners in core | draws | share | pure luck | pays |")
    print("|---|---|---|---|---|")
    for j in range(k + 1):
        m = hits == j
        if j <= CORE:
            print(f"| {j} | {m.sum()} | {m.mean():.1%} | {ch_pmf[j]:.1%} | GH₵{PAY * comb(j, 2)} |")

    print("\n## Best and worst stretches (net, after stakes)")
    for w, lab in ((11, "11 draws (~4 days)"), (21, "21 draws (~1 week)"), (90, "90 draws (~1 month)"),
                   (270, "270 draws (~3 months)")):
        (bv, b0, b1), (wv, w0, w1), _ = windows(net, d, w)
        print(f"- {lab}: best GH₵{bv:+,.0f} ({b0.date()} to {b1.date()}) | worst GH₵{wv:+,.0f} ({w0.date()} to {w1.date()})")
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(np.r_[0, cum])[1:]
    dd = cum - peak
    i_t = int(np.argmin(dd))
    i_p = int(np.argmax(cum[:i_t + 1])) if i_t > 0 else 0
    print(f"- deepest fall from a high point: GH₵{dd[i_t]:,.0f} ({d[i_p].date()} to {d[i_t].date()}, "
          f"{i_t - i_p} draws)")
    trough = np.minimum.accumulate(np.r_[0, cum])[1:]
    ru = cum - trough
    j_p = int(np.argmax(ru))
    j_t = int(np.argmin(cum[:j_p + 1]))
    print(f"- biggest climb from a low point: GH₵{ru[j_p]:+,.0f} ({d[j_t].date()} to {d[j_p].date()}, {j_p - j_t} draws)")
    lose, end = streaks(hits >= 2)
    print(f"- longest run with NO winning pair: {lose} draws in a row (GH₵{lose * STAKE:,} lost), ended {d[end].date()}")
    winrun, wend = streaks(hits < 2)
    print(f"- longest run where EVERY draw had a winning pair: {winrun} draws, ended {d[wend].date()}")
    print(f"- profitable draws (3+ in core): {(hits >= 3).sum()} of {len(rows)} = {(hits >= 3).mean():.1%} "
          f"(about 1 in {1 / max((hits >= 3).mean(), 1e-9):.0f})")

    print("\n## Month by month")
    mdf = df.assign(month=pd.to_datetime(df["date"]).dt.to_period("M")).groupby("month").agg(
        draws=("net", "size"), staked=("staked", "sum"), returned=("returned", "sum"), net=("net", "sum"))
    mdf["back"] = mdf["returned"] / mdf["staked"]
    mdf["cumulative"] = mdf["net"].cumsum()
    mdf.to_csv(out.with_name("profit_by_month.csv"))
    print("| month | draws | staked | returned | net | back | running total |")
    print("|---|---|---|---|---|---|---|")
    for m, r in mdf.iterrows():
        print(f"| {m} | {r.draws} | {r.staked:,} | {r.returned:,} | {r.net:+,} | {r.back:.0%} | {r.cumulative:+,} |")
    print(f"\nWinning months: {(mdf['net'] > 0).sum()} of {len(mdf)}")

    print("\n## By session")
    for s in pd.unique(df["session"]):
        m = df["session"] == s
        print(f"- {s}: {m.sum()} draws, {df.loc[m, 'returned'].sum() / df.loc[m, 'staked'].sum():.1%} back, "
              f"net GH₵{df.loc[m, 'net'].sum():+,}")

    # luck: the same draws, random 10-ball cores
    rng = np.random.default_rng(7)
    best11, months_won, totals = [], [], []
    mon = pd.to_datetime(df["date"]).dt.to_period("M").to_numpy()
    for _ in range(1000):
        rc = rng.random((len(rows), nb)).argsort(1)[:, :CORE]
        h = X[rows[:, None], rc].sum(1).astype(int)
        nt = PAY * (h * (h - 1) // 2) - STAKE
        best11.append(windows(nt, d, 11)[0][0])
        months_won.append((pd.Series(nt).groupby(mon).sum() > 0).sum())
        totals.append(nt.sum())
    best11, totals = np.array(best11), np.array(totals)
    print("\n## Is a hot streak skill or luck? (1,000 replays of the same draws with RANDOM cores)")
    print(f"- random cores: total net {np.median(totals):+,.0f} on average (system {net.sum():+,}); the system beat "
          f"{(totals < net.sum()).mean():.1%} of random players")
    print(f"- best 11-draw stretch of a random player: GH₵{np.median(best11):+,.0f} typical, "
          f"GH₵{np.percentile(best11, 90):+,.0f} for the luckiest 10% (system's best: GH₵{windows(net, d, 11)[0][0]:+,.0f})")
    print(f"- winning months of a random player: {np.median(months_won):.0f} of {len(mdf)} typical "
          f"(system {(mdf['net'] > 0).sum()})")
    print(f"\nPer-draw detail: {out}  |  months: {out.with_name('profit_by_month.csv')}")


if __name__ == "__main__":
    main()
