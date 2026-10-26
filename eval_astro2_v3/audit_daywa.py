#!/usr/bin/env python3
"""
Audit a draw CSV for the things that fake a "mechanical bias".

Run:  python audit_daywa.py --csv daywa_blind.csv        (stage2_powerhouse.py must be in the same folder)

It answers four questions in plain terms:
  1. Do whole draws (or long runs of draws) repeat?            -> copied / forward-filled rows
  2. Does each draw share too many balls with the previous ones? -> the only thing recency features can exploit
  3. Is the long-run ball frequency really non-uniform, and is it STABLE over time?
  4. Does the simplest possible model (frozen frequencies from the past) beat random on later draws?
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging

import stage2_powerhouse as s2

N, K = 90, 5
BASE = K / N
OUT = []


def say(msg=""):
    print(msg)
    OUT.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    logging.getLogger("nla").addHandler(logging.NullHandler())
    d = s2.load_data(a.csv, Path(a.out_dir))
    n, P, X = d.n, d.P[:d.n], d.X[:d.n].astype(float)
    rng = np.random.default_rng(a.seed)
    flags = []

    say("=" * 70)
    say(f"AUDIT | {Path(a.csv).name} | {n} draws | {d.info['date_min']} .. {d.info['date_max']}")
    say("=" * 70)

    # 1. repeats -----------------------------------------------------------
    sets = [tuple(sorted(r)) for r in P]
    seen, rep_set = {}, 0
    for i, s in enumerate(sets):
        if s in seen:
            rep_set += 1
        seen[s] = i
    ordered = len({tuple(r) for r in P})
    exp_pairs = n * (n - 1) / 2 / math.comb(N, K)
    say(f"\n1) REPEATED DRAWS")
    say(f"   draws identical (as a set) to an EARLIER draw: {rep_set}  (chance expectation ~{exp_pairs:.2f})")
    say(f"   draws identical including order:              {n - ordered}")
    same = (X[1:] == X[:-1]).all(1)
    run = best = 0
    for v in same:
        run = run + 1 if v else 0
        best = max(best, run)
    say(f"   consecutive identical draws: {int(same.sum())} (longest run {best + 1 if same.any() else 1})")
    if rep_set > max(3, 10 * exp_pairs):
        flags.append(f"{rep_set} whole draws repeat an earlier draw (chance: ~{exp_pairs:.2f}). Rows are likely copied/forward-filled.")

    # 2. overlap with previous draws ---------------------------------------
    say(f"\n2) OVERLAP WITH PREVIOUS DRAWS (chance mean = {K * K / N:.4f} balls)")
    say(f"   {'lag':>4}{'mean shared':>14}{'z':>8}   {'share>=3 balls':>16}")
    sd = math.sqrt(K * BASE * (1 - BASE) * (N - K) / (N - 1))
    worst = 0
    for lag in (1, 2, 3, 7, 14, 30):
        if n <= lag + 50:
            continue
        sh = (X[lag:] * X[:-lag]).sum(1)
        z = (sh.mean() - K * K / N) / (sd / math.sqrt(len(sh)))
        worst = max(worst, abs(z))
        say(f"   {lag:>4}{sh.mean():>14.4f}{z:>8.1f}   {100 * (sh >= 3).mean():>15.2f}%")
    if worst > 5:
        flags.append("Draws share far more balls with recent draws than chance allows (|z|>5). "
                     "This is exactly what recency/rate features exploit; it points to duplicated or overlapping rows, not physics.")

    # 3. frequency: size and stability -------------------------------------
    cnt = X.sum(0)
    exp = n * BASE
    chi = float(((cnt - exp) ** 2 / exp).sum())
    sims = np.empty(300)
    for i in range(300):
        U = np.argsort(rng.random((n, N)), axis=1)[:, :K]
        c = np.bincount(U.ravel(), minlength=N)
        sims[i] = ((c - exp) ** 2 / exp).sum()
    p_chi = (1 + (sims >= chi).sum()) / 301
    h = n // 2
    r_split = stats.spearmanr(X[:h].sum(0), X[h:].sum(0)).statistic
    z = (cnt - exp) / math.sqrt(exp * (1 - BASE))
    top = np.argsort(-z)[:5]
    say(f"\n3) BALL FREQUENCY")
    say(f"   chi-square={chi:.1f} (fair median ~{np.median(sims):.1f}); Monte-Carlo p={p_chi:.3f}")
    say(f"   hottest balls (z): " + ", ".join(f"{b + 1}:{z[b]:+.1f}" for b in top))
    say(f"   split-half correlation of all 90 ball counts: {r_split:+.3f}  (weak test: one hot ball barely moves it)")
    stable = [int(b) + 1 for b in np.argsort(-z)[:10]
              if (X[:h, b].sum() - h * BASE) / math.sqrt(h * BASE * (1 - BASE)) > 2.5
              and (X[h:, b].sum() - (n - h) * BASE) / math.sqrt((n - h) * BASE * (1 - BASE)) > 2.5]
    say(f"   stability of the hottest balls (z in first half | second half; a real bias is hot in BOTH):")
    for b in top[:3]:
        z1 = (X[:h, b].sum() - h * BASE) / math.sqrt(h * BASE * (1 - BASE))
        z2_ = (X[h:, b].sum() - (n - h) * BASE) / math.sqrt((n - h) * BASE * (1 - BASE))
        say(f"      ball {b + 1:>2}: {z1:+.1f} | {z2_:+.1f}")

    # 4. the simplest honest model -----------------------------------------
    say(f"\n4) FROZEN-FREQUENCY TEST (train on the first 70%, never update, score the last 30%)")
    cut = int(0.7 * n)
    p = X[:cut].sum(0)
    top5 = np.argsort(-p)[:5]
    hits = X[cut:][:, top5].sum(1)
    m = len(hits)
    mu = K * K / N
    zz = (hits.mean() - mu) / (sd / math.sqrt(m))
    say(f"   fixed top-5 balls {sorted(int(b) + 1 for b in top5)}: mean hits {hits.mean():.4f} vs chance {mu:.4f} "
        f"(z={zz:+.1f}, p={stats.norm.sf(zz):.4f}) on {m} later draws")
    say(f"   draws with >=2 hits: {(hits >= 2).sum()} (chance ~{m * stats.hypergeom.sf(1, N, K, K):.1f})")
    frozen_ok = zz > 3
    # recency-only model for contrast
    last = np.zeros((n, N))
    hit2 = []
    for t in range(cut, n):
        pr = X[max(0, t - 100):t].mean(0)
        hit2.append(X[t][np.argsort(-pr)[:5]].sum())
    hit2 = np.array(hit2)
    z2 = (hit2.mean() - mu) / (sd / math.sqrt(m))
    say(f"   rolling-100 top-5 (what rate100 does): mean hits {hit2.mean():.4f} (z={z2:+.1f})")

    # verdict ---------------------------------------------------------------
    say("\n" + "-" * 70)
    if flags:
        say("DATA PROBLEMS FOUND. Do not train anything bigger until these are fixed:")
        for f in flags:
            say(f"  * {f}")
    elif z2 > 3 and not frozen_ok:
        say("Rolling frequency works but frozen frequency does not: signal lives in SHORT-TERM structure.")
        say("  Check the date/ordering of the file (are 'blind' rows interleaved or duplicated?) before trusting it.")
    elif stable:
        say(f"Balls {stable} are hot in BOTH halves of history: a stable, specific bias (or a data quirk that is fixed over time).")
        say("  The frozen-frequency line above is the honest measure of how much it is worth. Confirm on genuinely new draws.")
    elif frozen_ok:
        say("A frozen, 70%-old frequency table beats chance on later draws. That is consistent with a stable bias,")
        say("  and is a much stronger and simpler claim than the neural results. Confirm on genuinely new draws.")
    else:
        say("No data problems found and no stable bias visible. The earlier result still needs an explanation:")
        say("  re-run stage2 on this file and send me section 1 and 3 of the report plus the log head.")
    p_out = Path(a.out_dir)
    p_out.mkdir(parents=True, exist_ok=True)
    (p_out / f"audit_{Path(a.csv).stem}.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
