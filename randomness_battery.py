#!/usr/bin/env python3
"""
randomness_battery.py: the strongest direct test of "is there a mathematical pattern behind the draws?".

    !python randomness_battery.py                       # the four *_audited.csv files, 2000 fair-machine simulations each
    !python randomness_battery.py --sims 5000           # finer p-values

Every statistic below is computed on the real history and on `--sims` simulated histories of a FAIR machine (5 balls
drawn uniformly without replacement, in random drawn order, independently every draw, same number of draws). The
p-value is the share of fair histories at least as extreme as the real one (and a z-score against the simulated
mean / SD, which extends below the simulation resolution). One family over every test and every game:
Benjamini-Hochberg q and Bonferroni. A real pattern of the kind the owner describes MUST show up in at least one row.

Within a draw      frequency, positional frequency (ball x drawn position), drawn-order patterns (120 orderings),
                   pair co-occurrence, sum, pairwise differences, odd/even, last digits, consecutive numbers
Between draws      repeats at lags 1..10, ball-to-ball follow-ups (a in draw t-1 -> b in draw t), positional steps
                   (N_j(t) - N_j(t-1) mod pool), lag-1 correlation of sums, overdue/hazard (gap length vs hit rate),
                   hot-ball persistence (first half vs second half)
Calendar           ball x weekday, ball x month, ball = day of month / day + month
Across games       overlap of each game's draw with the other games' previous-day draws, cross-game follow-ups
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

K = 5
NUM = [f"Number{i}" for i in range(1, K + 1)]
GAMES = {"DayWA": ("daywa_audited.csv", 39), "Main": ("main_draw_audited.csv", 90),
         "Noonrush": ("noonrush_audited.csv", 90), "VAG": ("vag_audited.csv", 90)}


def load(path):
    r = pd.read_csv(path)
    r["Date"] = pd.to_datetime(r["Date"])
    r = r.iloc[::-1].reset_index(drop=True)                       # oldest first (files are newest-first)
    r = r.iloc[np.argsort(r["Date"].to_numpy(), kind="stable")].reset_index(drop=True)
    return r[NUM].to_numpy(int) - 1, pd.DatetimeIndex(r["Date"])


def fair(n, nb, rng):
    return rng.random((n, nb)).argsort(1)[:, :K]                   # uniform 5-subset in uniformly random drawn order


def onehot(P, nb):
    X = np.zeros((len(P), nb))
    X[np.repeat(np.arange(len(P)), K), P.ravel()] = 1
    return X


def chi(obs, exp):
    exp = np.asarray(exp, float)
    m = exp > 0
    return float((((obs - exp) ** 2)[m] / exp[m]).sum())


PERM_INDEX = {}


def perm_codes(P):
    """index 0..119 of the relative-rank pattern of each drawn 5-tuple."""
    r = np.argsort(np.argsort(P, 1), 1)
    return r @ np.array([625, 125, 25, 5, 1])


def stats(P, dates, nb, ctx):
    """dict name -> statistic (larger = further from a fair machine, except where marked two-sided)."""
    n = len(P)
    X = onehot(P, nb)
    p = K / nb
    out = {}
    c = X.sum(0)
    out["frequency (ball counts)"] = chi(c, np.full(nb, n * p))
    h = n // 2
    a, b = X[:h].sum(0), X[h:].sum(0)
    out["hot balls stay hot (1st vs 2nd half corr)"] = float(np.corrcoef(a, b)[0, 1])
    pos = np.stack([np.bincount(P[:, j], minlength=nb) for j in range(K)])
    out["positional frequency (ball x position)"] = chi(pos, np.full(pos.shape, n / nb))
    pc = np.bincount(ctx["code_index"][perm_codes(P)], minlength=120)
    out["drawn-order patterns (120 orderings)"] = chi(pc, np.full(120, n / 120))
    pairs = (X.T @ X)[ctx["iu"]]
    out["pair co-occurrence"] = chi(pairs, np.full(len(pairs), n * K * (K - 1) / (nb * (nb - 1))))
    s = P.sum(1) + K
    sh = np.bincount(np.clip((s - ctx["s_lo"]) // ctx["s_w"], 0, ctx["s_bins"] - 1), minlength=ctx["s_bins"])
    out["sum of the draw (distribution)"] = chi(sh, ctx["s_exp"] * n)
    S = np.sort(P, 1)
    d = np.abs(S[:, :, None] - S[:, None, :])[:, ctx["iu5"][0], ctx["iu5"][1]].ravel()
    out["pairwise differences (distribution)"] = chi(np.bincount(d, minlength=nb), ctx["d_exp"] * n)
    odd = ((P + 1) % 2).sum(1)
    out["odd count (distribution)"] = chi(np.bincount(odd, minlength=K + 1), ctx["odd_exp"] * n)
    out["last digits"] = chi(np.bincount(((P + 1) % 10).ravel(), minlength=10), ctx["ld_exp"] * n)
    cons = (np.diff(S, axis=1) == 1).sum(1)
    out["consecutive numbers (distribution)"] = chi(np.bincount(cons, minlength=K), ctx["cons_exp"] * n)
    # between draws
    zs = []
    for lag in range(1, 11):
        ov = (X[lag:] * X[:-lag]).sum(1)
        zs.append((ov.mean() - ctx["ov_m"]) / math.sqrt(ctx["ov_v"] / len(ov)))
    out["repeats at lag 1 (z, two-sided)"] = abs(zs[0])
    out["repeats at lags 1..10 (max |z|)"] = float(np.max(np.abs(zs)))
    T = X[:-1].T @ X[1:]
    out["ball-to-ball follow-ups (a -> b)"] = chi(T, np.full(T.shape, (n - 1) * p * p))
    st = (P[1:] - P[:-1]) % nb
    out["positional steps N_j(t)-N_j(t-1) mod pool"] = sum(chi(np.bincount(st[:, j], minlength=nb),
                                                             np.full(nb, (n - 1) / nb)) for j in range(K))
    out["lag-1 correlation of sums (|r|)"] = abs(float(np.corrcoef(s[1:], s[:-1])[0, 1]))
    last = np.full(nb, -10 ** 6)
    hit_by = np.zeros(ctx["g_bins"])
    try_by = np.zeros(ctx["g_bins"])
    for t in range(n):
        g = np.clip((t - last) * K / nb, 0, None)                  # gap in units of the mean gap nb/5
        bins = np.minimum(np.digitize(g, ctx["g_edges"]), ctx["g_bins"] - 1)
        ok = last >= 0
        np.add.at(try_by, bins[ok], 1)
        np.add.at(hit_by, bins[ok], X[t, ok])
        last[P[t]] = t
    out["overdue / hazard (hit rate by gap length)"] = chi(hit_by, try_by * p) + chi(try_by - hit_by, try_by * (1 - p))
    # calendar
    for name, key in (("ball x weekday", "dow"), ("ball x month", "month")):
        lab = ctx[key]
        tab = np.stack([X[lab == k].sum(0) for k in np.unique(lab)])
        e = np.outer(tab.sum(1), c) / c.sum()
        out[name] = chi(tab, e)
    dd = ctx["day"]
    out["ball = day of month"] = float(X[np.arange(n), np.clip(dd - 1, 0, nb - 1)][dd <= nb].sum())
    dm = ctx["day"] + ctx["month"]
    out["ball = day + month"] = float(X[np.arange(n), np.clip(dm - 1, 0, nb - 1)][dm <= nb].sum())
    return out


TWO_SIDED_RAW = {"ball = day of month", "ball = day + month", "hot balls stay hot (1st vs 2nd half corr)"}


def make_ctx(P, dates, nb, rng):
    ctx = {"iu": np.triu_indices(nb, 1), "iu5": np.triu_indices(K, 1), "dow": dates.dayofweek.to_numpy(),
           "month": dates.month.to_numpy(), "day": dates.day.to_numpy()}
    codes = sorted({int(x) for x in perm_codes(np.array([np.random.default_rng(i).permutation(K) for i in range(4000)]))})
    idx = np.zeros(3125, dtype=int)
    idx[codes] = np.arange(len(codes))
    ctx["code_index"] = idx
    big = fair(200_000, nb, rng)
    s = big.sum(1) + K
    ctx["s_lo"], ctx["s_w"], ctx["s_bins"] = int(np.percentile(s, 0.5)), max(1, int((np.percentile(s, 99.5) - np.percentile(s, 0.5)) / 20)), 22
    sb = np.clip((s - ctx["s_lo"]) // ctx["s_w"], 0, ctx["s_bins"] - 1)
    ctx["s_exp"] = np.bincount(sb, minlength=ctx["s_bins"]) / len(big)
    S = np.sort(big, 1)
    d = np.abs(S[:, :, None] - S[:, None, :])[:, ctx["iu5"][0], ctx["iu5"][1]].ravel()
    ctx["d_exp"] = np.bincount(d, minlength=nb) / len(big)
    ctx["odd_exp"] = np.bincount(((big + 1) % 2).sum(1), minlength=K + 1) / len(big)
    ctx["ld_exp"] = np.bincount(((big + 1) % 10).ravel(), minlength=10) / len(big)
    ctx["cons_exp"] = np.bincount((np.diff(S, axis=1) == 1).sum(1), minlength=K) / len(big)
    pk = [math.comb(K, k) * math.comb(nb - K, K - k) / math.comb(nb, K) for k in range(K + 1)]
    ctx["ov_m"] = sum(k * q for k, q in enumerate(pk))
    ctx["ov_v"] = sum(k * k * q for k, q in enumerate(pk)) - ctx["ov_m"] ** 2
    ctx["g_edges"] = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    ctx["g_bins"] = len(ctx["g_edges"]) + 1
    return ctx


def cross_game(games, sims, rng):
    """Overlap of game B's draw with game A's latest draw on an EARLIER date (A -> B), and the a -> b follow-up table."""
    rows = []
    names = list(games)
    for A in names:
        for B in names:
            if A == B:
                continue
            PA, dA, nbA = games[A]
            PB, dB, nbB = games[B]
            j = np.searchsorted(dA.values, dB.values, side="left") - 1     # last A draw strictly before B's date
            ok = j >= 0
            XA, XB = onehot(PA, max(nbA, nbB)), onehot(PB, max(nbA, nbB))
            src, dst = XA[j[ok]], XB[ok]

            def stat(dst_):
                ov = (src * dst_).sum()
                T = src.T @ dst_
                e = np.outer(src.sum(0), dst_.sum(0)) / len(dst_)
                return ov, chi(T, e)
            o_real, c_real = stat(dst)
            null = np.array([stat(onehot(fair(len(dst), nbB, rng), max(nbA, nbB))) for _ in range(sims)])
            for k, (name, real) in enumerate((("overlap with previous-date draw", o_real),
                                              ("follow-up table a -> b", c_real))):
                arr = null[:, k]
                if k == 0:
                    z = (real - arr.mean()) / arr.std()
                    pv = (1 + (np.abs(arr - arr.mean()) >= abs(real - arr.mean())).sum()) / (sims + 1)
                else:
                    z = (real - arr.mean()) / arr.std()
                    pv = (1 + (arr >= real).sum()) / (sims + 1)
                rows.append(dict(game=f"{A} -> {B}", test=f"cross-game {name}", real=float(real),
                                 fair_mean=float(arr.mean()), z=float(z), p=float(pv)))
    return rows


def run_game(name, P, dates, nb, sims, rng):
    ctx = make_ctx(P, dates, nb, rng)
    real = stats(P, dates, nb, ctx)
    null = {k: [] for k in real}
    for _ in range(sims):
        s = stats(fair(len(P), nb, rng), dates, nb, ctx)
        for k in real:
            null[k].append(s[k])
    rows = []
    for k, r in real.items():
        arr = np.array(null[k])
        mu, sd = arr.mean(), arr.std() or 1.0
        if k in TWO_SIDED_RAW:
            pv = (1 + (np.abs(arr - mu) >= abs(r - mu)).sum()) / (sims + 1)
        else:
            pv = (1 + (arr >= r).sum()) / (sims + 1)
        rows.append(dict(game=name, test=k, real=float(r), fair_mean=float(mu), z=float((r - mu) / sd), p=float(pv)))
    return rows


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="outputs/randomness_battery.csv")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    games, rows = {}, []
    for g, (f, nb) in GAMES.items():
        P, dates = load(Path(a.dir) / f)
        games[g] = (P, dates, nb)
        rows += run_game(g, P, dates, nb, a.sims, rng)
        print(f"  {g}: {len(P)} draws, {a.sims} fair-machine histories done", flush=True)
    rows += cross_game(games, a.sims, rng)
    df = pd.DataFrame(rows)
    df["q_BH"] = bh(df["p"])
    df["p_Bonferroni"] = np.minimum(1, df["p"] * len(df))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(df.to_string(index=False, formatters={"real": "{:.3f}".format, "fair_mean": "{:.3f}".format,
                                                "z": "{:+.2f}".format, "p": "{:.4f}".format, "q_BH": "{:.3f}".format,
                                                "p_Bonferroni": "{:.3f}".format}))
    print(f"\n{len(df)} tests | smallest p {df['p'].min():.4f} | raw p < 0.05: {(df['p'] < 0.05).sum()} "
          f"(expected by luck alone: {0.05 * len(df):.1f}) | BH q < 0.05: {(df['q_BH'] < 0.05).sum()}")


if __name__ == "__main__":
    main()
