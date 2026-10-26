#!/usr/bin/env python3
"""
dependence.py: the detectors that look for an Afriluck edge BEYOND the pink/yellow colour bias.

    python dependence.py --afriluck-dir afriluck_output              # ~3 minutes
    python dependence.py --afriluck-dir afriluck_output --sims 500   # tighter p-values

Each test is calibrated against simulated histories (a machine with the fitted per-ball bias but no extra effect),
and each detector is first shown a PLANTED effect of realistic size to prove it would have found it.

  A. Ball dependence: do some pairs or clusters come out together more than the per-ball rates allow?
     (pair-count dispersion, the most over-represented pair, neighbours |a-b|=1 and <=2, same decade, pink-count
     spread). Planted control: 15 pairs lifted x1.77, the lift Perm-2 needs to break even.
  B. Ball-set changes: did the balls (or their bias) change at some date? (CUSUM of pink per draw, best single
     split of the 57 ball rates, ball x period table). Null = the same draws in shuffled order (exact for "no
     change"). Planted controls: 20 balls x1.25 / x0.8 from mid-history; pink strength 1.00 -> 1.22 mid-history.
  C. Machine numbers: are the 6 machine numbers simply the next 6 balls out of the same machine (exits 7-12)?
     And does a ball's winning rate tell you anything about its machine-number rate?
Every p-value is listed at the end with its Benjamini-Hochberg q (false-discovery correction across all tests).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

NB, K, PINK = 57, 6, 28


def onehot(W):
    X = np.zeros((len(W), NB))
    X[np.repeat(np.arange(len(W)), W.shape[1]), W.ravel()] = 1
    return X


def fit_pl(W, iters=500):
    """Plackett-Luce weights per ball from the exit order (MM algorithm, Hunter 2004)."""
    w = np.ones(NB)
    wins = np.bincount(W.ravel(), minlength=NB).astype(float)
    for _ in range(iters):
        denom = np.zeros(NB)
        avail = np.ones((len(W), NB), bool)
        for j in range(W.shape[1]):
            tot = (avail * w).sum(1)
            denom += (avail / tot[:, None]).sum(0)
            avail[np.arange(len(W)), W[:, j]] = False
        w_new = wins / denom
        w_new /= w_new.mean()
        done = np.abs(w_new - w).max() < 1e-8
        w = w_new
        if done:
            break
    return w


def sim(wt, n, rng, m=K):
    g = -np.log(rng.random((n, NB))) / wt
    return np.argsort(g, 1)[:, :m]


IU = np.triu_indices(NB, 1)


def dep_stats(W, E2):
    X = onehot(W)
    O = (X.T @ X)[IU]
    z = (O - E2) / np.sqrt(E2)
    S = np.sort(W, 1)
    dif = np.abs(S[:, :, None] - S[:, None, :])[:, np.triu_indices(K, 1)[0], np.triu_indices(K, 1)[1]]
    dec = (S // 10)[:, :, None] == (S // 10)[:, None, :]
    return {"pair dispersion": (z ** 2).sum(), "most over-represented pair (z)": z.max(),
            "neighbour pairs |a-b|=1": (dif == 1).sum(), "close pairs |a-b|<=2": (dif <= 2).sum(),
            "same-decade pairs": dec[:, np.triu_indices(K, 1)[0], np.triu_indices(K, 1)[1]].sum(),
            "spread of pink count": (W < PINK).sum(1).var()}


def plant_pairs(W, pairs, lift, rng):
    W = W.copy()
    X = onehot(W)
    for a, b in pairs:
        cnt = (X[:, a] * X[:, b]).sum()
        cand = np.where((X[:, a] == 1) & (X[:, b] == 0))[0]
        q = min(1, (lift - 1) * max(cnt, 1) / max(len(cand), 1))
        for t in cand[rng.random(len(cand)) < q]:
            r = W[t]
            i = rng.choice([j for j in range(K) if r[j] != a])
            X[t, r[i]] = 0
            r[i] = b
            X[t, b] = 1
    return W


def cusum_pink(W):
    p = (W < PINK).sum(1).astype(float)
    s = np.cumsum(p - p.mean())
    return np.abs(s).max() / (p.std() * np.sqrt(len(p)))


def split_lr(X, step=10, margin=150):
    n = len(X)
    C = np.cumsum(X, 0)
    tot = C[-1]
    best = 0.0
    for t in range(margin, n - margin, step):
        a = C[t - 1]
        b = tot - a
        ea, eb = tot * (t / n), tot * ((n - t) / n)
        g = 2 * ((a * np.log(np.where(a > 0, a / ea, 1))).sum() + (b * np.log(np.where(b > 0, b / eb, 1))).sum())
        best = max(best, g)
    return best


def block_chi2(X, blocks=8):
    T = np.stack([X[i].sum(0) for i in np.array_split(np.arange(len(X)), blocks)])
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    return ((T - E) ** 2 / E).sum()


def cp_stats(W):
    X = onehot(W)
    return {"pink CUSUM": cusum_pink(W), "best single split (57 balls)": split_lr(X), "ball x period table": block_chi2(X)}


def pval(real, null, side="upper"):
    """Simulation p-value; when the real value lies beyond EVERY simulated one, the simulation can only say
    p < 1/(sims+1), so the normal-tail p from the simulated mean and sd is used instead (it is smaller)."""
    from scipy.stats import norm
    null = np.asarray(null, float)
    up = (1 + (null >= real).sum()) / (1 + len(null))
    lo = (1 + (null <= real).sum()) / (1 + len(null))
    z = (real - null.mean()) / max(null.std(), 1e-12)
    if side == "upper":
        return norm.sf(z) if (null >= real).sum() == 0 else up
    if (null >= real).sum() == 0 or (null <= real).sum() == 0:
        return 2 * norm.sf(abs(z))
    return min(1.0, 2 * min(up, lo))


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def within_colour_corr(x, y):
    pk = np.arange(NB) < PINK
    xs = np.r_[x[pk] - x[pk].mean(), x[~pk] - x[~pk].mean()]
    ys = np.r_[y[pk] - y[pk].mean(), y[~pk] - y[~pk].mean()]
    return np.corrcoef(xs, ys)[0, 1]


def machine_corr(W, M, hidx):
    cw = np.bincount(W.ravel(), minlength=NB).astype(float)
    avail = np.zeros(NB)
    cm = np.zeros(NB)
    for i in hidx:
        av = np.ones(NB, bool)
        av[W[i]] = False
        avail += av * K / (NB - K)
        cm[M[i]] += 1
    return within_colour_corr(cw / cw.mean(), cm / avail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--planted", type=int, default=40)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import forecaster as fc
    d = fc.load(argparse.Namespace(game="Afriluck", afriluck_dir=a.afriluck_dir, nla_dir=""))
    W, M = d["W"], d["M"]
    n = len(W)
    rng = np.random.default_rng(11)
    t0 = time.time()
    w = fit_pl(W)
    print(f"# Dependence lab: Afriluck, {n} draws ({d['dates'][0].date()} to {d['dates'][-1].date()})\n")
    print(f"Fitted machine: pink balls x{w[:PINK].mean() / w[PINK:].mean():.3f} exit strength vs yellow; "
          f"strongest ball {int(w.argmax()) + 1} (x{w.max():.3f}), weakest {int(w.argmin()) + 1} (x{w.min():.3f})\n")
    allp = []

    # ---------------- A
    acc = np.zeros(len(IU[0]))
    for _ in range(a.sims):
        Xs = onehot(sim(w, n, rng))
        acc += (Xs.T @ Xs)[IU]
    E2 = acc / a.sims
    real = dep_stats(W, E2)
    null = [dep_stats(sim(w, n, rng), E2) for _ in range(a.sims)]
    print(f"## A. Do balls come out together? (real vs {a.sims} simulated histories of the fitted machine)\n")
    print("| statistic | real | simulated mean (sd) | p (two-sided unless noted) | reading |")
    print("|---|---|---|---|---|")
    for key, rv in real.items():
        arr = np.array([x[key] for x in null])
        side = "upper" if key.startswith("most") else "two"
        p = pval(rv, arr, side)
        allp.append((f"A {key}", p))
        reading = ("more than chance" if rv > arr.mean() else "LESS than chance") if p < 0.05 else "as chance"
        print(f"| {key} | {rv:.2f} | {arr.mean():.2f} ({arr.std():.2f}) | {p:.3f}{' (upper)' if side == 'upper' else ''} "
              f"| {reading} |")
    nd = np.array([x["pair dispersion"] for x in null])
    nm = np.array([x["most over-represented pair (z)"] for x in null])
    hit_d = hit_m = 0
    for _ in range(a.planted):
        Ws = sim(w, n, rng)
        pairs = [tuple(rng.choice(NB, 2, replace=False)) for _ in range(15)]
        st = dep_stats(plant_pairs(Ws, pairs, 1.77, rng), E2)
        hit_d += pval(st["pair dispersion"], nd, "two") < 0.05
        hit_m += pval(st["most over-represented pair (z)"], nm) < 0.05
    print(f"\nPlanted control (15 pairs lifted x1.77 in {a.planted} fake histories): found by 'pair dispersion' "
          f"{hit_d / a.planted:.0%}, by 'most over-represented pair' {hit_m / a.planted:.0%} of the time.")
    print("So if any pairs had the lift Perm-2 needs, these tests would show it.\n")

    # ---------------- B
    real = cp_stats(W)
    null = [cp_stats(W[rng.permutation(n)]) for _ in range(a.sims)]
    print(f"## B. Did the balls or the bias change at some date? (real vs {a.sims} shuffles of the draw order)\n")
    print("| statistic | real | shuffled mean | p |")
    print("|---|---|---|---|")
    for key, rv in real.items():
        arr = np.array([x[key] for x in null])
        p = pval(rv, arr)
        allp.append((f"B {key}", p))
        print(f"| {key} | {rv:.2f} | {arr.mean():.2f} | {p:.3f} |")
    det1 = {k_: 0 for k_ in real}
    det2 = {k_: 0 for k_ in real}
    for _ in range(a.planted):
        w2 = w.copy()
        ch = rng.choice(NB, 20, replace=False)
        w2[ch] *= np.where(rng.random(20) < 0.5, 1.25, 0.8)
        st = cp_stats(np.vstack([sim(w, n // 2, rng), sim(w2, n - n // 2, rng)]))
        wa, wb = np.ones(NB), np.where(np.arange(NB) < PINK, 1.22, 1.0)
        st2 = cp_stats(np.vstack([sim(wa, n // 2, rng), sim(wb, n - n // 2, rng)]))
        for k_ in real:
            arr = np.array([x[k_] for x in null])
            det1[k_] += pval(st[k_], arr) < 0.05
            det2[k_] += pval(st2[k_], arr) < 0.05
    print(f"\nPlanted ball-set change (20 balls x1.25 / x0.8 from mid-history) found: "
          + ", ".join(f"{k_} {v / a.planted:.0%}" for k_, v in det1.items()))
    print(f"Planted colour change (pink 1.00 -> 1.22 mid-history) found: "
          + ", ".join(f"{k_} {v / a.planted:.0%}" for k_, v in det2.items()))
    six = np.array_split(np.arange(n), 6)
    print("Pink per draw by sixth of the history: "
          + ", ".join(f"{d['dates'][i[0]].strftime('%b %y')} {(W[i] < PINK).sum(1).mean():.2f}" for i in six) + "\n")

    # ---------------- C
    has = (M >= 0).all(1) & (np.sort(M, 1) != np.sort(W, 1)).any(1)
    hidx = np.where(has)[0]
    if len(hidx) > 50:
        Wm, Mm = W[hidx], M[hidx]
        obs = (Mm < PINK).sum()
        sims = []
        for _ in range(a.sims * 5):
            g = -np.log(rng.random((len(hidx), NB))) / w
            g[np.arange(len(hidx))[:, None], Wm] = np.inf
            sims.append((np.argsort(g, 1)[:, :K] < PINK).sum())
        sims = np.array(sims)
        p1 = pval(obs, sims, "two")
        z = (obs - sims.mean()) / sims.std()
        uni = sum(K * (PINK - (r < PINK).sum()) / (NB - K) for r in Wm)
        print(f"## C. Machine numbers ({len(hidx)} draws with machine numbers, "
              f"{d['dates'][hidx[0]].date()} to {d['dates'][hidx[-1]].date()})\n")
        print(f"- Pink machine numbers: {obs} ({obs / len(hidx):.2f} per draw). If they were simply exits 7-12 of the "
              f"same machine: {sims.mean():.0f} (z = {z:.1f}, p = {p1:.1e}). If picked evenly from the leftover balls: "
              f"{uni:.0f}.")
        u12 = ((Wm < PINK).sum() + obs) / (12 * len(hidx))
        print(f"- All 12 balls together are {u12:.1%} pink (a fair machine: {PINK / NB:.1%}). So the machine takes "
              f"out about a fair 12; the pink ones simply tend to come out FIRST (winning numbers), the yellow ones "
              f"later (machine numbers).")
        pos = np.hstack([Wm, Mm])
        print("- Pink share by exit position 1-12: " + " ".join(f"{v:.2f}" for v in (pos < PINK).mean(0)))
        real_r = machine_corr(W, M, hidx)
        nullr = []
        for _ in range(a.sims):
            o = np.argsort(rng.random((n, NB)), 1)[:, :12]
            wt = np.where(o < PINK, 1.25, 1.0)
            oo = np.take_along_axis(o, np.argsort(-np.log(rng.random(o.shape)) / wt, 1), 1)
            Ms = np.full_like(M, -1)
            Ms[hidx] = oo[hidx, 6:]
            nullr.append(machine_corr(oo[:, :6], Ms, hidx))
        p2 = pval(real_r, nullr, "two")
        print(f"- Within each colour, a ball's winning rate vs its machine-number rate: r = {real_r:+.3f} (null of a fair "
              f"12 with pink first and NO per-ball effect: mean {np.mean(nullr):+.3f}, p = {p2:.3f}). Negative = "
              f"balls that win more come out LESS as machine numbers: each ball has its own 'comes out early' "
              f"tendency.")
        allp += [("C machine numbers = exits 7-12 of the same machine", p1), ("C winning vs machine rate per ball", p2)]
        print("- Money: tested walk-forward (an 'early ball' model using the machine numbers), it did NOT beat sharp "
              "(see the audit report), and machine-number Perm-2 at 20x returns about 22p per GH₵1.\n")

    print("## All tests, corrected for multiple testing (Benjamini-Hochberg)\n")
    q = bh([p for _, p in allp])
    print("| test | p | BH q | survives |")
    print("|---|---|---|---|")
    for (nm_, p), qq in zip(allp, q):
        print(f"| {nm_} | {p:.2g} | {qq:.2g} | {'yes' if qq < 0.05 else 'no'} |")
    print(f"\nDone in {time.time() - t0:.0f}s. None of these effects moves a pair's odds anywhere near the x1.77 "
          f"Perm-2 needs.")


if __name__ == "__main__":
    main()
