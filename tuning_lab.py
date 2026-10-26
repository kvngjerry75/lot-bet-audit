#!/usr/bin/env python3
"""
tuning_lab.py: can more tuning raise the hit rate? Nested test: every setting is chosen on older draws (TUNE), and the
newest 750 draws (TEST) are scored once.

    python tuning_lab.py --game Afriluck
    python tuning_lab.py --game DayWA
    python tuning_lab.py --game Main          # also Noonrush, VAG

124 candidates: 6 ball groupings (halves, thirds, quarters, decades, odd/even, halves x odd/even) x 5 pull strengths
x 4 memory lengths, plus ball-only rates. Pipelines compared on TEST: the current model, the single best candidate on
TUNE (by hit rate and by log-score), the average of every candidate, the average of the TUNE top 5.
Audit result (26 Oct 2026): nothing beat the current model on TEST, in either game.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def onehot(W, nb):
    X = np.zeros((len(W), nb))
    X[np.repeat(np.arange(len(W)), W.shape[1]), W.ravel()] = 1
    return X


def norm(p, k):
    p = np.clip(p, 1e-6, None)
    for _ in range(5):
        p = np.clip(p * (k / p.sum(-1, keepdims=True)), 1e-6, 0.95)
    return p


def groupings(nb):
    b = np.arange(nb)
    return {"halves": (b >= nb // 2).astype(int), "thirds": (b * 3 // nb), "quarters": (b * 4 // nb),
            "decades": (b + 1) // 10, "odd/even": (b % 2), "halves x odd/even": (b >= nb // 2) * 2 + (b % 2)}


def decayed(X, h):
    n, nb = X.shape
    dec = 1.0 if h is None else 0.5 ** (1 / h)
    C = np.zeros((n + 1, nb))
    T = np.zeros(n + 1)
    c, tt = np.zeros(nb), 0.0
    for t in range(n):
        C[t], T[t] = c, tt
        c = c * dec + X[t]
        tt = tt * dec + 1
    C[n], T[n] = c, tt
    return C, T


def candidates(X, k, quick=False):
    """dict name -> (n+1, nb) walk-forward probabilities (row t uses draws < t)."""
    n, nb = X.shape
    q = k / nb
    out = {"chance": np.full((n + 1, nb), q)}
    HS = [None, 1500.0, 600.0, 250.0] if not quick else [None, 600.0]
    KS = [None, 1000.0, 300.0, 100.0, 30.0] if not quick else [None, 300.0]
    grp = groupings(nb)
    for h in HS:
        C, T = decayed(X, h)
        out[f"ball only, hl {h}"] = norm((C + 50 * q) / (T[:, None] + 50), k)
        for gname, g in grp.items():
            G = g.max() + 1
            ng = np.bincount(g, minlength=G)
            gh = np.stack([np.bincount(g, weights=C[t], minlength=G) for t in range(n + 1)])
            grate = (gh + 100 * q * ng) / (T[:, None] * ng + 100 * ng)
            gb = grate[:, g]
            for kk in KS:
                p = gb if kk is None else gb * (C + kk * gb) / (T[:, None] * gb + kk * gb)
                out[f"{gname}, pull {kk}, hl {h}"] = norm(p, k)
    return out


def bma(P, X, names, forget=0.999):
    n = len(X)
    S = np.stack([P[c] for c in names])
    ls = np.zeros(len(names))
    res = np.empty((n + 1, X.shape[1]))
    for t in range(n + 1):
        w = np.exp(ls - ls.max())
        w /= w.sum()
        res[t] = (w[:, None] * S[:, t]).sum(0)
        if t < n:
            y = X[t]
            ls = forget * ls + (y * np.log(S[:, t]) + (1 - y) * np.log1p(-S[:, t])).sum(1)
    return res


def metrics(p, X, rows, k, core=10, ticket=None):
    rng = np.random.default_rng(1)
    o = np.argsort(-(p[rows] + rng.random((len(rows), X.shape[1])) * 1e-12), axis=1)
    ar = np.arange(len(rows))[:, None]
    hc = X[rows][ar, o[:, :core]].sum(1)
    ht = X[rows][ar, o[:, :ticket or k]].sum(1)
    pc = np.clip(p[rows], 1e-9, 1 - 1e-9)
    ll = (X[rows] * np.log(pc) + (1 - X[rows]) * np.log1p(-pc)).sum(1).mean()
    return dict(core2=(hc >= 2).mean(), core3=(hc >= 3).mean(), topk=ht.mean(), tick2=(ht >= 2).mean(), ll=ll,
                perm2=(hc * (hc - 1) / 2).mean() / math.comb(core, 2))


def nested(W, nb, k, tune_rows, test_rows, quick=False):
    """Choose on TUNE, report on TEST. Returns dict of predictions for the pipelines compared."""
    X = onehot(W, nb)
    P = candidates(X, k, quick)
    names = [c for c in P if c != "chance"]
    # (1) single best candidate on TUNE by 2+ in core, (2) by log-score, (3) BMA of all, (4) BMA of TUNE top-5 by log-score
    sc = {c: metrics(P[c], X, tune_rows, k) for c in names}
    best_core = max(names, key=lambda c: sc[c]["core2"])
    best_ll = max(names, key=lambda c: sc[c]["ll"])
    top5 = sorted(names, key=lambda c: -sc[c]["ll"])[:5]
    pipes = {"chance": P["chance"],
             "CURRENT: sharp-style (halves, BMA of 16)": bma(P, X, ["chance"] + [c for c in names if c.startswith("halves,") and "hl 250" not in c and "pull 30" not in c]),
             f"best on TUNE by hit rate [{best_core}]": P[best_core],
             f"best on TUNE by log-score [{best_ll}]": P[best_ll],
             "BMA of every candidate": bma(P, X, ["chance"] + names),
             "BMA of TUNE top-5 by log-score": bma(P, X, ["chance"] + top5)}
    return X, pipes, sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="DayWA", choices=["Afriluck", "DayWA", "Main", "Noonrush", "VAG"])
    ap.add_argument("--nla-dir", default="nla_audited")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    a = ap.parse_args()
    import forecaster as fc
    d = fc.load(argparse.Namespace(game=a.game, nla_dir=a.nla_dir, afriluck_dir=a.afriluck_dir))
    W, nb, k = d["W"], d["nb"], d["k"]
    n = len(W)
    test_rows = np.arange(n - 750, n)
    tune_rows = np.arange(300, n - 750)
    X, pipes, sc = nested(W, nb, k, tune_rows, test_rows)
    from math import comb
    ch = sum(comb(10, j) * comb(nb - 10, k - j) for j in range(2, k + 1)) / comb(nb, k)
    print(f"# {a.game}: TUNE draws 300-{n - 751}, TEST draws {n - 750}-{n - 1} ({d['dates'][n - 750].date()} to "
          f"{d['dates'][-1].date()}); chance 2+ in 10-core = {ch:.1%}\n")
    print("| pipeline | TUNE 2+ in core | TEST 2+ in core | TEST 3+ | TEST hits in top k | TEST ticket 2+ of k | TEST log-score gain |")
    print("|---|---|---|---|---|---|---|")
    base = metrics(pipes["chance"], X, test_rows, k)["ll"]
    for nm, p in pipes.items():
        mt, mv = metrics(p, X, tune_rows, k), metrics(p, X, test_rows, k)
        print(f"| {nm} | {mt['core2']:.1%} | {mv['core2']:.1%} | {mv['core3']:.1%} | {mv['topk']:.3f} | {mv['tick2']:.1%} | {mv['ll'] - base:+.5f} |")
    print(f"\n{len(sc)} candidates searched. If no tuned row beats the CURRENT row on TEST, extra tuning only fits noise.")
