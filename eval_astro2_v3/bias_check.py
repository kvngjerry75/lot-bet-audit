#!/usr/bin/env python3
"""
bias_check.py: is any machine biased in a way that LASTS? Run this before building or trusting any predictor.

    !python bias_check.py --csv main_draw_clean.csv noonrush_clean.csv vag_clean.csv daywa_clean.csv

Four tests per game, null hypothesis = a fair machine (5 balls uniformly without replacement, draws independent):
  1. frequency     : are some balls drawn more often overall? (chi-square, corrected for drawing without replacement)
  2. stability     : do the balls that were hot in the older half of the history stay hot in the newer half?
                     A physical bias (heavy / worn ball) must persist, so this correlation would be clearly positive.
  3. repeats       : do numbers from the previous draw come back more often than chance?
  4. first ball    : is the first ball drawn uniform?
All p-values are one family: read the Benjamini-Hochberg q column; a few raw p < 0.05 are expected by chance.
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

K = 5


def load(path):
    r = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    r.columns = [str(c).strip() for c in r.columns]
    v = r[[f"Number{i}" for i in range(1, K + 1)]].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    dates = pd.to_datetime(r["Date"], errors="coerce")
    ok = v.notna().all(axis=1) & dates.notna()
    v, dates = v[ok].astype(int).to_numpy(), dates[ok].to_numpy()
    counts = np.bincount(v[(v >= 1).all(1)].ravel())
    med = np.median(counts[counts > 0])
    nb = int(np.where(counts >= max(5, 0.1 * med))[0].max())          # stray values above the pool are bad rows
    keep = (v >= 1).all(1) & (v <= nb).all(1) & (np.array([len(set(x)) for x in v]) == K)
    v, dates = v[keep], dates[keep]
    v = v[np.argsort(dates, kind="stable")]                            # oldest first
    return v, nb


def tests(v, nb):
    n = len(v)
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), K), v.ravel() - 1] = 1
    p = K / nb
    c = X.sum(0)
    e = n * p
    chi = ((c - e) ** 2 / e).sum() / ((1 - p) * nb / (nb - 1))
    out = {"frequency": (f"chi2 {chi:.1f} on {nb - 1} df", stats.chi2.sf(chi, nb - 1))}
    h = n // 2
    rho, _ = stats.pearsonr(X[:h].sum(0), X[h:].sum(0))
    p_rho = stats.pearsonr(X[:h].sum(0), X[h:].sum(0), alternative="greater").pvalue
    out["stability"] = (f"r = {rho:+.3f}", p_rho)
    ov = (X[1:] * X[:-1]).sum(1)
    pk = [comb(K, k) * comb(nb - K, K - k) / comb(nb, K) for k in range(K + 1)]
    m = sum(k * q for k, q in enumerate(pk))
    var = sum(k * k * q for k, q in enumerate(pk)) - m * m
    z = (ov.mean() - m) / np.sqrt(var / len(ov))
    out["repeats"] = (f"{ov.mean():.3f} per draw vs {m:.3f} (z {z:+.2f})", 2 * stats.norm.sf(abs(z)))
    fb = np.bincount(v[:, 0] - 1, minlength=nb)
    chi1 = ((fb - n / nb) ** 2 / (n / nb)).sum()
    out["first ball"] = (f"chi2 {chi1:.1f} on {nb - 1} df", stats.chi2.sf(chi1, nb - 1))
    return n, out


def bh(pv):
    pv = np.asarray(pv)
    o = np.argsort(pv)
    q = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty_like(q)
    res[o] = np.minimum(q, 1.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.csv:
        v, nb = load(f)
        n, out = tests(v, nb)
        for name, (txt, pv) in out.items():
            rows.append((Path(f).stem, f"5/{nb}", n, name, txt, float(pv)))
    q = bh([r[5] for r in rows])
    print("| game | pool | draws | test | result | raw p | BH q |")
    print("|---|---|---|---|---|---|---|")
    for r, qq in zip(rows, q):
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.3f} | {qq:.3f} |")
    sig = [r for r, qq in zip(rows, q) if qq < 0.05]
    print()
    if sig:
        print("BIAS SIGNAL (q < 0.05): " + "; ".join(f"{r[0]} {r[3]}" for r in sig)
              + ". Confirm it on NEW draws before trusting it.")
    else:
        print("No test is significant after correcting for the number of tests: every game looks like a fair machine. "
              "A predictor built on this history cannot beat chance.")


if __name__ == "__main__":
    main()
