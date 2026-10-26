#!/usr/bin/env python3
"""
afriluck_lab.py: every model we have, run honestly on Afriluck (6 balls from 57), with real bet returns.

    python afriluck_lab.py --dir afriluck_output                 # pooled Anopa + Midday + evening, 20 fair controls
    python afriluck_lab.py --dir afriluck_output --controls 50   # more fair-machine controls (slower)

Every model predicts draw t from draws before t only (walk-forward), for every draw after the first 300.
Models
  chance        every ball equally likely
  colour        pink (1-28) vs yellow (29-57): one shared rate per colour, learned from past draws
  frequency     each ball's own long-run rate (Bayesian, shrunk towards chance)
  recent50      each ball's rate over the last 50 draws
  overdue       balls that have not come for longest
  markov        "after a comes b" follow-ups
  boosting      gradient boosting on per-ball features (counts 10/20/50/all, gap, colour, follow-up score)
  rule_engine   the v3 data-mined rule engine (rule_miner.py), adapted to 6 balls
  ensemble      average of colour, frequency and boosting
Scores: hits among the top 6, top-12 core hit (2+ drawn), and money returned per 1 staked on Afriluck's bets
(Direct-1 first ball 30x on the top ball, Direct-2 60x on the top 2, Direct-3 600x on the top 3, Direct-4 10,000x on
the top 4). The same models are run on simulated fair machines with the same dates: p = share of fair controls
doing at least as well.
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
NB, K, PINK = 57, 6, 28
WARM, REFIT = 300, 100
BETS = {"Direct-1 first ball (30x)": (1, 30), "Direct-2 (60x)": (2, 60), "Direct-3 (600x)": (3, 600),
        "Direct-4 (10000x)": (4, 10_000)}
SESSION = {"anopa": 0, "midday": 1, "evening": 2}


def load(folder):
    parts = []
    for g, rank in SESSION.items():
        p = Path(folder) / f"afriluck_{g}.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p)
        num = [f"Number{i}" for i in range(1, K + 1)]
        r = r[r[num].notna().all(axis=1)].copy()
        r["Date"] = pd.to_datetime(r["Date"])
        r["rank"] = rank
        r["key"] = r[num].astype(int).apply(lambda v: tuple(sorted(v)), axis=1)
        r = r.drop_duplicates(subset=["Date", "key"])             # evening B/C/D rows repeat A (in another order)
        parts.append(r[["Date", "rank", "key"] + num])
    df = pd.concat(parts).sort_values(["Date", "rank"], kind="stable").reset_index(drop=True)
    keep, seen = [], {}
    for dt, k in zip(df["Date"], df["key"]):                      # the same 6 numbers again within 3 days = a copy
        keep.append(k not in seen or (dt - seen[k]).days > 3)
        seen[k] = dt
    df = df[np.array(keep)].reset_index(drop=True)
    P = df[[f"Number{i}" for i in range(1, K + 1)]].astype(int).to_numpy() - 1
    return P, pd.DatetimeIndex(df["Date"])


def onehot(P):
    X = np.zeros((len(P), NB))
    X[np.repeat(np.arange(len(P)), K), P.ravel()] = 1
    return X


def norm6(p):
    p = np.clip(p, 1e-6, None)
    for _ in range(5):
        p = np.clip(p * (K / p.sum(-1, keepdims=True)), 1e-6, 0.999)
    return p


# ------------------------------------------------------------------------------------------------ experts
def experts(P, dates, use_rule=True, use_boost=True):
    n = len(P)
    X = onehot(P)
    C = np.vstack([np.zeros(NB), np.cumsum(X, 0)])            # C[t] = counts in draws < t
    base = K / NB
    out = {}
    t = np.arange(n)
    out["chance"] = np.full((n, NB), base)
    pink = np.arange(NB) < PINK
    pk = (C[:-1][:, pink].sum(1) + 50 * base * PINK) / (t * PINK + 50 * PINK)
    yl = (C[:-1][:, ~pink].sum(1) + 50 * base * (NB - PINK)) / (t * (NB - PINK) + 50 * (NB - PINK))
    out["colour"] = norm6(np.where(pink[None, :], pk[:, None], yl[:, None]))
    a0 = 200.0                                                 # shrinkage: 200 draws' worth of "fair" prior
    out["frequency"] = norm6((C[:-1] + a0 * base) / (t[:, None] + a0))
    lo = np.maximum(t - 50, 0)
    out["recent50"] = norm6((C[t] - C[lo] + 5 * base) / ((t - lo)[:, None] + 5))
    last = np.full(NB, -1)
    gap = np.zeros((n, NB))
    for i in range(n):
        gap[i] = i - last
        last[P[i]] = i
    out["overdue"] = norm6(base * (1 + 0.02 * (gap - gap.mean(1, keepdims=True)) / (gap.std(1, keepdims=True) + 1e-9)))
    T = np.zeros((NB, NB))
    Na = np.zeros(NB)
    mk = np.full((n, NB), base)
    for i in range(1, n):
        T += np.outer(X[i - 2], X[i - 1]) if i >= 2 else 0
        Na += X[i - 2] if i >= 2 else 0
        rows = P[i - 1]
        mk[i] = ((T[rows] + 30 * base) / (Na[rows, None] + 30)).mean(0)
    out["markov"] = norm6(mk)
    if use_boost:
        out["boosting"] = boost(P, X, C, gap, mk)
    if use_rule:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "eval_astro2_v3"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rule_miner as rmn
        rmn.K = K
        dates_all = pd.DatetimeIndex(list(dates) + [dates[-1] + pd.Timedelta(days=1)])
        res = rmn.run(np.asarray(P, dtype=np.int64), dates_all, NB, {}, warmup=WARM, return_p=True)
        out["rule_engine"] = norm6(res["p"][:n])
    ens = [k for k in ("colour", "frequency", "boosting") if k in out]
    out["ensemble"] = norm6(np.mean([out[k] for k in ens], axis=0))
    return out


def boost(P, X, C, gap, mk):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(P)
    t = np.arange(n)
    feats = []
    for w in (10, 20, 50):
        lo = np.maximum(t - w, 0)
        feats.append((C[t] - C[lo]) / np.maximum(t - lo, 1)[:, None])
    feats.append(C[:-1] / np.maximum(t, 1)[:, None])
    feats.append(np.minimum(gap, 200) / 200)
    feats.append(np.broadcast_to((np.arange(NB) < PINK).astype(float), (n, NB)))
    feats.append(mk)
    F = np.stack(feats, axis=2)
    pred = np.full((n, NB), K / NB)
    for a in range(WARM, n, REFIT):
        m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=400,
                                           l2_regularization=10.0, random_state=0)
        m.fit(F[50:a].reshape(-1, F.shape[2]), X[50:a].ravel())
        b = min(a + REFIT, n)
        pred[a:b] = m.predict_proba(F[a:b].reshape(-1, F.shape[2]))[:, 1].reshape(b - a, NB)
    return norm6(pred)


# ------------------------------------------------------------------------------------------------ scoring
def score(p, P, seed=0):
    X = onehot(P)
    s = slice(WARM, len(P))
    pr, Xs, Ps = p[s], X[s], P[s]
    rng = np.random.default_rng(seed)
    order = np.argsort(-(pr + rng.random(pr.shape) * 1e-9), axis=1)
    ar = np.arange(len(pr))[:, None]
    h6 = Xs[ar, order[:, :6]].sum(1)
    core = (Xs[ar, order[:, :12]].sum(1) >= 2)
    q = np.clip(pr, 1e-9, 1 - 1e-9)
    bce = -(Xs * np.log(q) + (1 - Xs) * np.log1p(-q)).sum(1).mean()
    b0 = K / NB
    ubce = -(K * math.log(b0) + (NB - K) * math.log(1 - b0))
    res = dict(top6=h6.mean(), core=core.mean(), skill=1 - bce / ubce)
    for bet, (m, pay) in BETS.items():
        if m == 1:
            win = Ps[:, 0] == order[:, 0]
        else:
            win = Xs[ar, order[:, :m]].sum(1) == m
        res[bet] = pay * win.mean()
        res[bet + " wins"] = int(win.sum())
    return res


def chance_levels():
    from math import comb
    return dict(top6=K * 6 / NB, core=sum(comb(12, j) * comb(NB - 12, K - j) for j in range(2, K + 1)) / comb(NB, K),
                skill=0.0, **{b: pay * (1 / NB if m == 1 else comb(NB - m, K - m) / comb(NB, K))
                              for b, (m, pay) in BETS.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="afriluck_output")
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--no-rule", action="store_true", help="skip the rule engine (fast)")
    ap.add_argument("--out", default="afriluck_lab_report.md")
    a = ap.parse_args()
    P, dates = load(a.dir)
    n = len(P)
    print(f"{n} draws ({dates.min().date()} to {dates.max().date()}); scoring draws {WARM + 1}..{n} "
          f"({n - WARM} draws), every one predicted from earlier draws only", flush=True)
    real = {k: score(v, P) for k, v in experts(P, dates, not a.no_rule).items()}
    rng = np.random.default_rng(7)
    ctrl = []
    for i in range(a.controls):
        Pf = rng.random((n, NB)).argsort(1)[:, :K]
        ctrl.append({k: score(v, Pf, i) for k, v in experts(Pf, dates, not a.no_rule).items()})
        print(f"\r  fair-machine control {i + 1}/{a.controls}", end="", flush=True)
    print()
    ch = chance_levels()
    cols = ["top6", "core", "skill"] + list(BETS)
    L = ["# Afriluck lab: every model, walk-forward, vs fair-machine controls", "",
         f"{n} draws pooled (Anopa, Midday, evening), {dates.min().date()} to {dates.max().date()}. "
         f"Scored on the last {n - WARM} draws; each predicted from earlier draws only.", "",
         "| Model | Hits in top 6 | Core-12 hit (2+) | Log-loss skill | " + " | ".join(f"Return: {b}" for b in BETS)
         + " | p (top-6 vs fair controls) |",
         "|---|---|---|---|" + "---|" * len(BETS) + "---|",
         f"| chance (theory) | {ch['top6']:.3f} | {ch['core']:.3f} | 0 | "
         + " | ".join(f"{ch[b]:.0%}" for b in BETS) + " | - |"]
    for k, r in real.items():
        if k == "chance":
            continue
        arr = np.array([c[k]["top6"] for c in ctrl]) if ctrl else np.array([np.nan])
        pv = (1 + (arr >= r["top6"]).sum()) / (len(arr) + 1) if ctrl else float("nan")
        L.append(f"| {k} | {r['top6']:.3f} | {r['core']:.3f} | {r['skill']:+.4f} | "
                 + " | ".join(f"{r[b]:.0%} ({r[b + ' wins']} wins)" for b in BETS) + f" | {pv:.3f} |")
    L += ["", "Return = money back per 1 staked, every draw, on the model's top balls. 100% = break even. "
          "Direct-3 and Direct-4 win so rarely that a single lucky hit swings their return; read them as noise "
          "unless the win count is large.",
          f"Fair-machine controls: {a.controls}. With few controls the smallest possible p is "
          f"{1 / (a.controls + 1):.3f}."]
    text = "\n".join(L)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
