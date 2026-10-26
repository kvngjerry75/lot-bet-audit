#!/usr/bin/env python3
"""
audit_extras.py: the three remaining experiments of the October 2026 audit, reproducible from saved forecasts.

    python main_beast.py --game Afriluck --save afl.npz
    python main_beast.py --game Afriluck --combiner all --no-long-run --save afl_old.npz
    python audit_extras.py --new afl.npz --old afl_old.npz

  1. Combiners, nested: chosen on draws 561-1310, reported on the untouched 1311-2060, each against sharp alone.
  2. The machine-number "early ball" model, walk-forward against sharp, on the draws after machine numbers began.
  3. Old against new: MAIN BEAST and stage 4, Perm-2 on the top 10 / 6 / 3, in GH₵, both spans.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bet_lab as bl   # noqa: E402
import sharp           # noqa: E402

NB, K = 57, 6
Q = K / NB


def perm(p, X, rows, c=10):
    o = bl.rank(p[rows])
    h = X[rows][np.arange(len(rows))[:, None], o[:, :c]].sum(1)
    return 60 * h * (h - 1) / 2 / math.comb(c, 2)


def logscore(p, X, rows):
    pc = np.clip(p[rows], 1e-9, 1 - 1e-9)
    return (X[rows] * np.log(pc) + (1 - X[rows]) * np.log1p(-pc)).sum(1)


def bma(comp, names, X, forget=0.999):
    n = len(X)
    P = np.stack([np.full((n + 1, NB), Q)] + [comp[c] for c in names])
    ls = np.zeros(len(P))
    res = np.empty((n + 1, NB))
    for t in range(n + 1):
        w = np.exp(ls - ls.max())
        w /= w.sum()
        res[t] = (w[:, None] * P[:, t]).sum(0)
        if t < n:
            y = X[t]
            ls = forget * ls + (y * np.log(P[:, t]) + (1 - y) * np.log1p(-P[:, t])).sum(1)
    return sharp.norm(res, K)


def bet_weighted(comp, names, X, eta, forget=0.995, core=10):
    n = len(X)
    P = np.stack([np.full((n + 1, NB), Q)] + [comp[c] for c in names])
    S = np.zeros(len(P))
    res = np.empty((n + 1, NB))
    for t in range(n + 1):
        w = np.exp(eta * (S - S.max()))
        w /= w.sum()
        res[t] = (w[:, None] * P[:, t]).sum(0)
        if t < n:
            o = np.argsort(-P[:, t], 1)[:, :core]
            h = X[t][o].sum(1)
            S = forget * S + (60 * h * (h - 1) / 2 - 45) / 45
    return sharp.norm(res, K)


def combiners(f, X):
    n = len(X)
    comp = {k[3:]: f[k] for k in f.files if k.startswith("c__")}
    allc = [c for c in comp if c != "colour only"]
    cands = {"BMA all 15 (old MAIN BEAST)": bma(comp, allc, X),
             "BMA sharp + stage 5 + chance": bma(comp, ["sharp", "stage5 forecaster"], X),
             "BMA sharp + colour + chance (NEW default)": bma(comp, ["sharp", "colour only"], X),
             "sharp alone": comp["sharp"], "colour only": comp["colour only"]}
    for eta in (1, 3, 10, 30):
        cands[f"weighted by realised Perm-2 profit, eta={eta}"] = bet_weighted(comp, allc, X, eta)
    dev, test = np.arange(561, 1311), np.arange(1311, n)
    base = K * math.log(Q) + (NB - K) * math.log(1 - Q)
    sh = perm(cands["sharp alone"], X, test)
    print("## 1. Combiners (chosen on draws 561-1310 = dev; reported on 1311-2060 = TEST, untouched)\n")
    print("| combiner | dev Perm-2 back | TEST Perm-2 back (top 10) | TEST top 6 | TEST 2+ in core | TEST log-score vs chance "
          "| TEST minus sharp (95% CI) |")
    print("|---|---|---|---|---|---|---|")
    for nm, p in cands.items():
        pt = perm(p, X, test)
        dlt = pt - sh
        bs = bl.block_boot(dlt)
        h = X[test][np.arange(len(test))[:, None], bl.rank(p[test])[:, :10]].sum(1)
        print(f"| {nm} | {perm(p, X, dev).mean():.3f} | {pt.mean():.3f} | {perm(p, X, test, 6).mean():.3f} | "
              f"{(h >= 2).mean():.1%} | {logscore(p, X, test).mean() - base:+.4f} | {dlt.mean():+.3f} "
              f"({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}) |")


def earliness(W, M, X, a0):
    """Walk-forward: a fair 12 is taken out, each ball's share of 'early' (winning) exits is learned from the
    winning numbers AND the machine numbers (late exits), shrunk to its colour's share by a0."""
    n = len(W)
    ok = bl.machine_ok(W, M)
    pink = np.arange(NB) < 28
    p = np.full((n + 1, NB), Q)
    Em, L, Enm, Nnm = np.zeros(NB), np.zeros(NB), np.zeros(NB), 0
    for t in range(n + 1):
        num, den = Em + Enm, Em + L + Nnm * 12 / NB
        pc = np.where(pink, (num[pink].sum() + 1) / (den[pink].sum() + 2), (num[~pink].sum() + 1) / (den[~pink].sum() + 2))
        p[t] = sharp.norm(((num + a0 * pc) / (den + a0) * 12 / NB)[None], K)[0]
        if t < n:
            if ok[t]:
                Em += X[t]
                L += bl.onehot(M[t:t + 1])[0]
            else:
                Enm += X[t]
                Nnm += 1
    return p, ok


def early_ball(f, X, W, M):
    n = len(X)
    _, ok = earliness(W, M, X, 60)
    first = int(np.argmax(ok))
    rows = np.arange(first + 100, n)
    sh = f["c__sharp"]
    base = K * math.log(Q) + (NB - K) * math.log(1 - Q)
    print(f"\n## 2. The machine-number 'early ball' model against sharp (draws {rows[0]}-{n - 1}, {len(rows)} draws)\n")
    print("| shrinkage a0 | log-score vs chance | minus sharp (95% CI) | Perm-2 top 10 | top 6 |")
    print("|---|---|---|---|---|")
    for a0 in (30, 60, 150, 400):
        pe, _ = earliness(W, M, X, a0)
        d = logscore(pe, X, rows) - logscore(sh, X, rows)
        bs = bl.block_boot(d)
        print(f"| {a0} | {logscore(pe, X, rows).mean() - base:+.4f} | {d.mean():+.4f} ({np.percentile(bs, 2.5):+.4f} to "
              f"{np.percentile(bs, 97.5):+.4f}) | {perm(pe, X, rows).mean():.3f} | {perm(pe, X, rows, 6).mean():.3f} |")
    print(f"| sharp | {logscore(sh, X, rows).mean() - base:+.4f} | - | {perm(sh, X, rows).mean():.3f} | "
          f"{perm(sh, X, rows, 6).mean():.3f} |")


def old_new(fo, fn, W):
    n = len(W)
    models = {"OLD MAIN BEAST (15-model average)": fo["p"][:n], "NEW MAIN BEAST (chance + sharp + colour)": fn["p"][:n],
              "OLD stage 4 ensemble (frequency banned)": fo["c__stage4 ORIGINAL ensemble"][:n],
              "NEW stage 4 ensemble (long-run allowed)": fn["c__stage4 ORIGINAL ensemble"][:n]}
    for e in ("mlp", "ranker"):
        if f"c__stage4 {e}" in fo.files:
            models[f"OLD stage 4 {e}"] = fo[f"c__stage4 {e}"][:n]
            models[f"NEW stage 4 {e}"] = fn[f"c__stage4 {e}"][:n]
    chp = 60 * 30 / (57 * 56)
    for lab, rows in (("all scored draws 561-2060", np.arange(561, n)), ("untouched second half 1311-2060", np.arange(1311, n))):
        print(f"\n## 3. Old against new: {lab} ({len(rows)} draws), GH₵1 per line\n")
        print("| forecaster | Perm-2 top 10 back | net GH₵ | top 6 back | net GH₵ | top 3 back | net GH₵ |")
        print("|---|---|---|---|---|---|---|")
        rets = {}
        for nm, p in models.items():
            r = bl.rank(p[rows])
            cells = []
            for c in (10, 6, 3):
                ret = bl.returns(f"perm2_{c}", r, W[rows])
                st = math.comb(c, 2)
                m = bl.money(ret, st)
                rets[(nm, c)] = ret / st
                cells += [f"{m['back']:.3f} ({m['lo']:.2f}-{m['hi']:.2f})", f"{m['net']:+,.0f}"]
            print(f"| {nm} | " + " | ".join(cells) + " |")
        print(f"| pure luck (expected) | {chp:.3f} | {(chp - 1) * 45 * len(rows):+,.0f} | {chp:.3f} | "
              f"{(chp - 1) * 15 * len(rows):+,.0f} | {chp:.3f} | {(chp - 1) * 3 * len(rows):+,.0f} |")
        for c in (10, 6, 3):
            d = rets[("NEW MAIN BEAST (chance + sharp + colour)", c)] - rets[("OLD MAIN BEAST (15-model average)", c)]
            bs = bl.block_boot(d)
            print(f"NEW minus OLD MAIN BEAST, top {c}: {d.mean():+.3f} per GH₵1 (95% CI {np.percentile(bs, 2.5):+.3f} to "
                  f"{np.percentile(bs, 97.5):+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="main_beast.py --game Afriluck --save (new defaults)")
    ap.add_argument("--old", default="", help="main_beast.py --game Afriluck --combiner all --no-long-run --save")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    a = ap.parse_args()
    fn = np.load(a.new, allow_pickle=True)
    W = fn["W"]
    X = bl.onehot(W)
    d = bl.load(a.afriluck_dir)
    assert (d["W"] == W).all(), "the forecast file was made from different data: re-run main_beast"
    combiners(fn, X)
    early_ball(fn, X, W, d["M"])
    if a.old:
        fo = np.load(a.old, allow_pickle=True)
        assert (fo["W"] == W).all()
        old_new(fo, fn, W)


if __name__ == "__main__":
    main()
