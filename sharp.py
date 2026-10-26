#!/usr/bin/env python3
"""
sharp.py: the sharpest honest predictor for a machine with a physical ball bias (Afriluck, DayWA).

    python sharp.py --game Afriluck --afriluck-dir afriluck_output
    python sharp.py --game DayWA --nla-dir nla_output
    python sharp.py --game Afriluck --afriluck-dir afriluck_output --controls 50

Model (hierarchical, self-tuning):
  rate(ball) = rate(its group) x deviation(ball within its group)
    group      low/high halves (on Afriluck exactly the pink 1-28 and yellow 29-57 balls)
    deviation  the ball's own record, pulled towards 1 unless its evidence is strong (strength k)
    recency    older draws count less (half-life h), because worn balls change the bias over time
  16 versions (k in 100/300/1000/no-own-deviation x h in none/1500/800/400) run side by side; before every draw
  each version is trusted according to its log-score on the draws before it (Bayesian model averaging). So the
  amount of pulling and forgetting is learned from the past, never from the draw being forecast.
Scored walk-forward on every draw after draw 300, against fair-machine controls, next to two references:
  colour only (one rate per group) and the ORACLE (cheats: the true full-history rate of every ball).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

KS = [100.0, 300.0, 1000.0, None]
HS = [None, 1500.0, 800.0, 400.0]


def load(game, nla_dir, afl_dir):
    if game == "DayWA":
        r = pd.read_csv(Path(nla_dir) / "daywa_audited.csv")
        if "Promotionday" in r:
            r = r[r["Promotionday"].astype(str) != "Holiday"]
        r["Date"] = pd.to_datetime(r["Date"])
        r = r.iloc[::-1]
        r = r.iloc[np.argsort(r["Date"].to_numpy(), kind="stable")]
        return 39, 5, r[[f"Number{i}" for i in range(1, 6)]].astype(int).to_numpy() - 1, pd.DatetimeIndex(r["Date"])
    parts = []
    for rank, g in enumerate(("anopa", "midday", "evening")):
        r = pd.read_csv(Path(afl_dir) / f"afriluck_{g}.csv")
        num = [f"Number{i}" for i in range(1, 7)]
        r = r[r[num].notna().all(axis=1)].copy()
        r["Date"] = pd.to_datetime(r["Date"])
        r["key"] = r[num].astype(int).apply(lambda v: tuple(sorted(v)), axis=1)
        r = r.drop_duplicates(subset=["Date", "key"])
        r["rank"] = rank
        parts.append(r)
    df = pd.concat(parts).sort_values(["Date", "rank"], kind="stable").reset_index(drop=True)
    keep, seen = [], {}
    for dt, kk in zip(df["Date"], df["key"]):
        keep.append(kk not in seen or (dt - seen[kk]).days > 3)
        seen[kk] = dt
    df = df[np.array(keep)].reset_index(drop=True)
    return 57, 6, df[[f"Number{i}" for i in range(1, 7)]].astype(int).to_numpy() - 1, pd.DatetimeIndex(df["Date"])


def norm(p, k):
    p = np.clip(p, 1e-6, None)
    for _ in range(5):
        p = np.clip(p * (k / p.sum(-1, keepdims=True)), 1e-6, 0.95)
    return p


def predict_all(X, nb, k):
    """(versions, n, nb) walk-forward probabilities + colour-only."""
    n = len(X)
    q = k / nb
    grp = (np.arange(nb) >= nb // 2).astype(int)
    ng = np.bincount(grp)
    out, colour = [], None
    for h in HS:
        # decayed counts before each draw
        decay = 1.0 if h is None else 0.5 ** (1 / h)
        C = np.zeros((n, nb))
        T = np.zeros(n)
        c, tt = np.zeros(nb), 0.0
        for t in range(n):
            C[t], T[t] = c, tt
            c = c * decay + X[t]
            tt = tt * decay + 1
        gh = np.stack([np.bincount(grp, weights=C[t], minlength=2) for t in range(n)])     # group hits
        grate = (gh + 100 * q * ng) / (T[:, None] * ng + 100 * ng)                           # group rate per ball
        gball = grate[:, grp]
        if h is None:
            colour = norm(gball, k)
        for kk in KS:
            if kk is None:
                p = gball
            else:
                dev = (C + kk * gball) / (T[:, None] * gball + kk * gball) if True else None
                p = gball * dev
            out.append(norm(p, k))
    return np.stack(out), colour


def bma(P, X, warm=300, forget=0.999):
    """Before each draw, weight versions by (slowly forgetting) past log-score."""
    V, n, nb = P.shape
    ls = np.zeros(V)
    res = np.empty((n, nb))
    for t in range(n):
        w = np.exp(ls - ls.max())
        w /= w.sum()
        res[t] = (w[:, None] * P[:, t]).sum(0)
        y = X[t]
        ls = forget * ls + (y * np.log(P[:, t]) + (1 - y) * np.log1p(-P[:, t])).sum(1)
    return res, w


def score(p, X, W, k, nb, warm, core=10, pay2=None):
    s = slice(warm, len(X))
    pr, Xs = p[s], X[s]
    o = np.argsort(-(pr + np.random.default_rng(1).random(pr.shape) * 1e-12), axis=1)
    ar = np.arange(len(pr))[:, None]
    hc = Xs[ar, o[:, :core]].sum(1).astype(int)
    q = k / nb
    pc = np.clip(pr, 1e-9, 1 - 1e-9)
    skill = 1 - (-(Xs * np.log(pc) + (1 - Xs) * np.log1p(-pc)).sum(1).mean()) / (-(k * math.log(q) + (nb - k) * math.log(1 - q)))
    r = dict(core2=(hc >= 2).mean(), core3=(hc >= 3).mean(), topk=Xs[ar, o[:, :k]].sum(1).mean(),
             pair=(Xs[ar, o[:, :2]].sum(1) == 2).mean(), skill=skill, n=len(pr), core_n=int((hc >= 2).sum()))
    r["perm2"] = pay2 * np.mean([math.comb(h, 2) for h in hc]) / math.comb(core, 2) if pay2 else float("nan")
    return r


def run(W, nb, k, warm):
    n = len(W)
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    P, colour = predict_all(X, nb, k)
    sharp, w_last = bma(P, X, warm)
    oracle = np.broadcast_to(norm((X.sum(0) / n)[None, :], k), (n, nb))
    return X, dict(sharp=sharp, colour=colour, oracle=oracle), w_last, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Afriluck", choices=["Afriluck", "DayWA"])
    ap.add_argument("--nla-dir", default="nla_output")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--controls", type=int, default=50)
    ap.add_argument("--warm", type=int, default=300)
    a = ap.parse_args()
    nb, k, W, dates = load(a.game, a.nla_dir, a.afriluck_dir)
    pay2 = 60 if a.game == "Afriluck" else None
    core = 10
    X, preds, w_last, P = run(W, nb, k, a.warm)
    real = {m: score(p, X, W, k, nb, a.warm, core, pay2) for m, p in preds.items()}
    rng = np.random.default_rng(5)
    ctrl = []
    for i in range(a.controls):
        Wf = rng.random((len(W), nb)).argsort(1)[:, :k]
        Xf, pf, _, _ = run(Wf, nb, k, a.warm)
        ctrl.append(score(pf["sharp"], Xf, Wf, k, nb, a.warm, core, pay2))
    ch2 = sum(math.comb(core, j) * math.comb(nb - core, k - j) for j in range(2, k + 1)) / math.comb(nb, k)
    ch3 = sum(math.comb(core, j) * math.comb(nb - core, k - j) for j in range(3, k + 1)) / math.comb(nb, k)
    chp = sum(math.comb(j, 2) * math.comb(core, j) * math.comb(nb - core, k - j) for j in range(k + 1)) / math.comb(nb, k)
    print(f"# Sharp predictor: {a.game} ({len(W)} draws, {dates[0].date()} to {dates[-1].date()}); "
          f"scored on {real['sharp']['n']} draws, each from earlier draws only\n")
    print(f"| model | 2+ in 10-core | 3+ in core | hits in top {k} | top-2 both | log-loss skill | Perm-2 back |")
    print("|---|---|---|---|---|---|---|")
    print(f"| chance | {ch2:.1%} | {ch3:.1%} | {k * k / nb:.3f} | {k * (k - 1) / (nb * (nb - 1)):.2%} | 0 | "
          + (f"{pay2 * chp / 45:.0%}" if pay2 else "-") + " |")
    for m, lab in (("colour", "colour / low-high only"), ("sharp", "SHARP (self-tuning hierarchical)"),
                   ("oracle", "ORACLE (cheats, sees the future)")):
        r = real[m]
        print(f"| {lab} | {r['core_n']} = {r['core2']:.1%} | {r['core3']:.1%} | {r['topk']:.3f} | {r['pair']:.2%} | "
              f"{r['skill']:+.5f} | " + (f"{r['perm2']:.0%}" if pay2 else "-") + " |")
    if ctrl:
        print(f"\nSHARP vs {a.controls} fair-machine histories (p = share of fair controls doing at least as well):")
        for key, lab in (("core2", "2+ in core"), ("topk", f"hits in top {k}"), ("skill", "log-loss skill")):
            arr = np.array([c[key] for c in ctrl])
            print(f"  {lab}: {real['sharp'][key]:.4f} vs fair mean {arr.mean():.4f}, p = "
                  f"{(1 + (arr >= real['sharp'][key]).sum()) / (len(arr) + 1):.3f}")
    names = [f"pull {'none' if kk is None else int(kk)} / half-life {'none' if h is None else int(h)}" for h in HS for kk in KS]
    top = np.argsort(-w_last)[:3]
    print("\nVersions it trusts most now: " + "; ".join(f"{names[i]} ({w_last[i]:.2f})" for i in top))
    Pn, _ = predict_all(np.vstack([X, np.zeros((1, nb))]), nb, k)      # row n = the NEXT draw (uses every draw)
    p_next = (w_last[:, None] * Pn[:, -1]).sum(0)
    o = np.argsort(-p_next)
    print(f"Current 10-ball core: {[int(b) + 1 for b in o[:10]]} | strongest ball x{p_next[o[0]] / (k / nb):.3f} chance")


if __name__ == "__main__":
    main()
