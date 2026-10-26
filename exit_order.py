#!/usr/bin/env python3
"""
exit_order.py: the "which balls does this machine let out first" model (Plackett-Luce, as used to rate horses from
race finishing orders), fitted walk-forward on the FULL exit order of every draw:
Afriluck = 6 winning balls in drawn order, then the 6 machine numbers in drawn order (12 exits per draw);
DayWA = 5 winning balls in drawn order.

    python exit_order.py --game Afriluck --afriluck-dir afriluck_output
    python exit_order.py --game DayWA --nla-dir nla_output

Each ball b has a strength s_b: at every exit, P(ball b comes out next) = s_b / (sum of strengths of balls still in).
Strengths are fitted only on draws BEFORE the one being forecast (refit every 50 draws), shrunk towards "all equal",
optionally with older draws down-weighted (ball wear changes things). The fitted strengths give each ball's chance
to be in the next draw and to be the FIRST ball (for Afriluck's Direct-1).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load(game, nla_dir, afl_dir):
    if game == "DayWA":
        r = pd.read_csv(Path(nla_dir) / "daywa_audited.csv")
        r = r[r.get("Promotionday", "").astype(str) != "Holiday"] if "Promotionday" in r else r
        r["Date"] = pd.to_datetime(r["Date"])
        r = r.iloc[::-1]
        r = r.iloc[np.argsort(r["Date"].to_numpy(), kind="stable")]
        W = r[[f"Number{i}" for i in range(1, 6)]].astype(int).to_numpy() - 1
        return dict(nb=39, k=5, W=W, M=np.full_like(W, -1), dates=pd.DatetimeIndex(r["Date"]), session=np.zeros(len(W), int))
    parts = []
    for rank, g in enumerate(("anopa", "midday", "evening")):
        r = pd.read_csv(Path(afl_dir) / f"afriluck_{g}.csv")
        num = [f"Number{i}" for i in range(1, 7)]
        mac = [f"Machine{i}" for i in range(1, 7)]
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
    W = df[[f"Number{i}" for i in range(1, 7)]].astype(int).to_numpy() - 1
    Mr = df[[f"Machine{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    M = np.where(np.isnan(Mr), 0, Mr).astype(int) - 1
    bad = np.isnan(Mr).any(axis=1) | (M < 0).any(axis=1) | (M >= 57).any(axis=1)
    M[bad] = -1
    for t in range(len(W)):                                   # machine numbers must come from the remaining balls
        if M[t, 0] >= 0 and np.isin(M[t], W[t]).any():
            M[t] = -1
    return dict(nb=57, k=6, W=W, M=M, dates=pd.DatetimeIndex(df["Date"]), session=df["rank"].to_numpy())


def stages(W, M, use_machine):
    """Every exit as (draw index, ball, mask of balls still in the machine)."""
    nb = None
    return W, M, use_machine, nb


def fit(W, M, nb, upto, use_machine, half_life, prior):
    """MAP log-strengths from draws < upto."""
    exits, remain, wts = [], [], []
    for t in range(upto):
        w = 1.0 if not half_life else 0.5 ** ((upto - 1 - t) / half_life)
        seq = list(W[t]) + (list(M[t]) if use_machine and M[t, 0] >= 0 else [])
        avail = np.ones(nb, bool)
        for b in seq:
            exits.append(b)
            remain.append(avail.copy())
            wts.append(w)
            avail[b] = False
    if not exits:
        return np.zeros(nb)
    E = np.array(exits)
    R = np.array(remain, dtype=float)
    Wt = np.array(wts)
    onehot = np.zeros((len(E), nb))
    onehot[np.arange(len(E)), E] = 1
    cnt = (Wt[:, None] * onehot).sum(0)

    def f(th):
        ex = np.exp(th - th.max())
        den = R @ ex
        ll = (cnt * th).sum() - (Wt * (np.log(den) + th.max())).sum()
        p = (R * ex[None, :]) / den[:, None]
        grad = cnt - (Wt[:, None] * p).sum(0)
        return -(ll - 0.5 * prior * (th ** 2).sum()), -(grad - prior * th)
    res = minimize(f, np.zeros(nb), jac=True, method="L-BFGS-B")
    return res.x - res.x.mean()


def inclusion(theta, k, sims=4000, rng=None):
    """P(ball in the first k exits) and P(ball is first), by simulation (Gumbel trick)."""
    rng = rng or np.random.default_rng(0)
    g = theta[None, :] + rng.gumbel(size=(sims, len(theta)))
    order = np.argsort(-g, axis=1)[:, :k]
    inc = np.bincount(order.ravel(), minlength=len(theta)) / sims
    ex = np.exp(theta - theta.max())
    return inc, ex / ex.sum()


def backtest(d, use_machine, half_life, prior, warm=300, refit=50, core=10):
    W, M, nb, k = d["W"], d["M"], d["nb"], d["k"]
    n = len(W)
    inc_all = np.zeros((n, nb))
    first_all = np.zeros((n, nb))
    for a in range(warm, n, refit):
        th = fit(W, M, nb, a, use_machine, half_life, prior)
        inc, first = inclusion(th, k)
        b = min(a + refit, n)
        inc_all[a:b], first_all[a:b] = inc, first
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    s = slice(warm, n)
    pr, Xs = inc_all[s], X[s]
    rng = np.random.default_rng(1)
    o = np.argsort(-(pr + rng.random(pr.shape) * 1e-12), axis=1)
    ar = np.arange(len(pr))[:, None]
    hc = Xs[ar, o[:, :core]].sum(1).astype(int)
    hk = Xs[ar, o[:, :k]].sum(1)
    top2 = Xs[ar, o[:, :2]].sum(1) == 2
    fb = np.argmax(first_all[s], axis=1) == W[s][:, 0]
    perm2 = np.array([math.comb(h, 2) for h in hc])
    q = k / nb
    pc = np.clip(pr, 1e-6, 1 - 1e-6)
    skill = 1 - (-(Xs * np.log(pc) + (1 - Xs) * np.log1p(-pc)).sum(1).mean()) / \
        (-(k * math.log(q) + (nb - k) * math.log(1 - q)))
    return dict(n=n - warm, core2=(hc >= 2).mean(), core3=(hc >= 3).mean(), topk=hk.mean(), pair=top2.mean(),
                first=fb.mean(), perm2_pairs=perm2.mean(), skill=skill, theta_last=fit(W, M, nb, n, use_machine, half_life, prior))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Afriluck", choices=["Afriluck", "DayWA"])
    ap.add_argument("--nla-dir", default="nla_output")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    a = ap.parse_args()
    d = load(a.game, a.nla_dir, a.afriluck_dir)
    nb, k = d["nb"], d["k"]
    core = 10
    ch_core2 = sum(math.comb(core, j) * math.comb(nb - core, k - j) for j in range(2, k + 1)) / math.comb(nb, k)
    ch_pairs = sum(math.comb(j, 2) * math.comb(core, j) * math.comb(nb - core, k - j) for j in range(k + 1)) / math.comb(nb, k)
    pay2 = 60 if a.game == "Afriluck" else None
    print(f"{a.game}: {len(d['W'])} draws, machine numbers on {(d['M'][:, 0] >= 0).sum()} of them\n")
    print(f"| variant | 2+ in 10-core | 3+ | hits in top {k} | top-2 both | first ball right | Perm-2 back | skill |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| chance | {ch_core2:.1%} | - | {k * k / nb:.3f} | {k * (k - 1) / (nb * (nb - 1)):.2%} | {1 / nb:.2%} | "
          + (f"{pay2 * ch_pairs / 45:.0%}" if pay2 else "-") + " | 0 |")
    variants = [("winning order only", False, 0, 30.0), ("winning + machine order", True, 0, 30.0),
                ("winning + machine, recent-weighted (half-life 600)", True, 600, 30.0),
                ("winning + machine, light shrinkage", True, 0, 10.0)]
    if a.game == "DayWA":
        variants = [("winning order", False, 0, 30.0), ("winning order, recent-weighted (half-life 600)", False, 600, 30.0),
                    ("winning order, light shrinkage", False, 0, 10.0)]
    for name, um, hl, pr in variants:
        r = backtest(d, um, hl, pr, core=core)
        print(f"| {name} | {r['core2']:.1%} | {r['core3']:.1%} | {r['topk']:.3f} | {r['pair']:.2%} | {r['first']:.2%} | "
              + (f"{pay2 * r['perm2_pairs'] / 45:.0%}" if pay2 else "-") + f" | {r['skill']:+.5f} |", flush=True)
    th = r["theta_last"]
    o = np.argsort(-th)
    print("\nStrongest balls now (relative chance to come out next, 1.00 = average): "
          + ", ".join(f"{b + 1}:{math.exp(th[b]):.2f}" for b in o[:12]))
    print("Weakest: " + ", ".join(f"{b + 1}:{math.exp(th[b]):.2f}" for b in o[::-1][:8]))


if __name__ == "__main__":
    main()
