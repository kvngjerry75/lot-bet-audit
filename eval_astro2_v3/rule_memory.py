#!/usr/bin/env python3
"""
rule_memory.py: a rule-memory engine that "thinks like a person", one draw at a time, from the first draw to the latest.

    !python rule_memory.py --csv daywa_clean.csv                      # DayWA (5/39)
    !python rule_memory.py --csv main_draw_clean.csv noonrush_clean.csv
    !python rule_memory.py --csv daywa_clean.csv --null-reps 50        # more fair-machine controls (default 20)

How it thinks
  1. RULES: a library of candidate "maths" rules, each turning the past into a guess for the next draw, e.g.
       "yesterday's 2nd ball + 3", "yesterday's numbers shifted by -1", "numbers drawn 4 draws ago",
       "(yesterday's sum + 2) wrapped into the pool", "same last digit as yesterday", "the 10% most overdue numbers".
  2. CONDITIONS: every rule also exists in "only when ..." versions: a given weekday, yesterday's sum low / mid / high,
       yesterday mostly odd / balanced / mostly even. A conditional rule only guesses when its situation holds.
  3. SEQUENTIAL: before draw t every rule guesses using draws < t only; after draw t every rule is scored.
  4b. HEDGE VOTE: a second, mathematically optimal way to combine all rules sequentially (exponential weights,
       the textbook algorithm with guaranteed regret bounds): every rule keeps a weight that grows with its hits over
       the last --hedge-window draws, and all rules vote in proportion. No hand-set promotion threshold.
  4. MEMORY: a rule is PROMOTED when it beats chance over its recent record (z >= --z-in over the last --window draws)
       and DEMOTED when it stops doing so (z < --z-out). It can come back later. Promoted rules vote on the next draw.
  5. HONESTY: a rule's record BEFORE promotion is how it got picked, so it proves nothing. The engine scores rules only
       on draws AFTER their promotion, and runs the identical engine on simulated fair-machine draws (no pattern exists
       there by construction). If the engine "learns" as well on those, its learning is luck.

Output: a narrative of what it thinks on the latest draws, the memory (rules promoted now), the honest scorecard vs
the fair-machine controls, and the vote for the next draw. Files go to --out-dir (default outputs/).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

K = int(__import__("os").environ.get("ASTRO_K", "5"))   # balls per draw
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(path):
    """(draws (n, 5) drawn order, oldest first; dates; pool size)."""
    r = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    r.columns = [str(c).strip() for c in r.columns]
    v = r[[f"Number{i}" for i in range(1, K + 1)]].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    dates = pd.to_datetime(r["Date"], errors="coerce")
    ok = (v.notna().all(axis=1) & dates.notna()).to_numpy()
    v, dates = v[ok].astype(int).to_numpy(), dates[ok].to_numpy()
    counts = np.bincount(v[(v >= 1).all(axis=1)].ravel())
    med = np.median(counts[counts > 0])
    nb = int(np.where(counts >= max(5, 0.1 * med))[0].max())          # stray values above the pool are bad rows
    keep = (v >= 1).all(axis=1) & (v <= nb).all(axis=1) & (np.array([len(set(x)) for x in v]) == K)
    v, dates = v[keep], dates[keep]
    o = np.argsort(dates, kind="stable")
    return v[o], pd.DatetimeIndex(dates[o]), nb


def simulate(n, nb, rng):
    return np.array([rng.choice(nb, K, replace=False) + 1 for _ in range(n)])


# --------------------------------------------------------------------------- #
# Rule library: every base rule gives a boolean (n+1, nb) "guess" matrix; row t uses draws < t only
# --------------------------------------------------------------------------- #
def _wrap(x, nb):
    return (np.asarray(x) - 1) % nb + 1


def base_rules(v, dates, nb, next_date=None):
    n = len(v)
    N1 = n + 1
    X = np.zeros((N1, nb), dtype=bool)
    X[np.repeat(np.arange(n), K), v.ravel() - 1] = True
    prev = np.vstack([np.zeros((1, K), dtype=int), v])                 # prev[t] = draw t-1 (row 0 = none)
    have = np.arange(N1) >= 1
    rules = []

    def add(name, M):
        M = M.copy()
        M[~have] = False
        rules.append((name, M))

    def from_balls(B):                                                 # B: (N1, m) ball numbers (0 = none)
        M = np.zeros((N1, nb), dtype=bool)
        for j in range(B.shape[1]):
            b = B[:, j]
            ok = b >= 1
            M[np.where(ok)[0], b[ok] - 1] = True
        return M

    # A. positional arithmetic: yesterday's j-th ball + d
    for j in range(K):
        for d in range(-10, 11):
            add(f"yesterday's ball #{j + 1} again" if d == 0 else f"yesterday's ball #{j + 1} {'+' if d > 0 else '-'} {abs(d)}", from_balls(_wrap(prev[:, j:j + 1] + d, nb)
                                                                                     * have[:, None]))
    # B. repeats from k draws ago
    for k in range(1, 11):
        B = np.vstack([np.zeros((k, K), dtype=int), v])[:N1]
        add(f"numbers drawn {k} draw(s) ago", from_balls(B))
    # C. all of yesterday's numbers shifted by d
    for d in list(range(-5, 0)) + list(range(1, 6)):
        add(f"yesterday's numbers {'+' if d > 0 else '-'} {abs(d)}", from_balls(_wrap(prev + d, nb) * have[:, None]))
    # D. sum / difference maths
    s = prev.sum(1)
    for c in range(-5, 6):
        add(f"(yesterday's sum {'+' if c >= 0 else '-'} {abs(c)}) wrapped into 1..{nb}",
            from_balls((_wrap(s + c, nb) * have)[:, None]))
    diffs = np.stack([np.abs(prev[:, i] - prev[:, j]) for i in range(K) for j in range(i + 1, K)], 1)
    add("differences between yesterday's numbers", from_balls(np.where(diffs >= 1, diffs, 0)))
    # E. digit / mirror / complement
    ld = prev % 10
    M = np.zeros((N1, nb), dtype=bool)
    balls = np.arange(1, nb + 1)
    for t in range(1, N1):
        M[t] = np.isin(balls % 10, ld[t])
    add("same last digit as one of yesterday's numbers", M)
    mir = np.array([[int(str(b)[::-1]) if b >= 10 else b * 10 for b in row] for row in prev])
    add("mirror of yesterday's numbers (12 -> 21)", from_balls(np.where((mir >= 1) & (mir <= nb), mir, 0)))
    add(f"complement of yesterday's numbers ({nb + 1} - x)", from_balls(np.where(have[:, None], nb + 1 - prev, 0)))
    # F. gap / frequency (overdue, hot, cold)
    last = np.full(nb, -1)
    gap = np.zeros((N1, nb))
    C = np.vstack([np.zeros((1, nb)), np.cumsum(X[:n], 0)])            # C[t] = counts in draws < t
    for t in range(N1):
        gap[t] = t - last
        if t < n:
            last[v[t] - 1] = t
    rank_gap = np.argsort(np.argsort(-gap, axis=1, kind="stable"), axis=1, kind="stable")   # 0 = most overdue
    for q in range(10):
        lo, hi = q * nb / 10, (q + 1) * nb / 10
        add(f"overdue decile {q + 1} of 10 (1 = most overdue)", (rank_gap >= lo) & (rank_gap < hi))
    for w in (10, 50):
        lo_ = np.maximum(np.arange(N1) - w, 0)
        r = C[np.arange(N1)] - C[lo_]
        rk = np.argsort(np.argsort(-r, axis=1, kind="stable"), axis=1, kind="stable")
        for m in (5, 10):
            add(f"hottest {m} over the last {w} draws", rk < m)
            add(f"coldest {m} over the last {w} draws", rk >= nb - m)
    # G. calendar maths for draw t (its date is known in advance)
    # v3 fix: row n is the NEXT draw, whose date is known; v2 used 0 there, so calendar rules were scored in the
    # backtest but silently never voted on the live pick
    nd = pd.Timestamp(next_date) if next_date is not None else None
    dd = np.concatenate([dates.day.to_numpy(), [nd.day if nd is not None else 0]])
    mm = np.concatenate([dates.month.to_numpy(), [nd.month if nd is not None else 0]])
    add("ball = day of the month", from_balls(np.where((dd >= 1) & (dd <= nb), dd, 0)[:, None]))
    add("ball = day + month", from_balls(np.where(dd + mm <= nb, dd + mm, 0)[:, None]))
    return rules, X


def conditions(v, dates, nb, next_date):
    """Situations known before draw t: weekday of draw t, yesterday's sum band, yesterday's odd count."""
    n = len(v)
    N1 = n + 1
    dow = np.concatenate([dates.dayofweek.to_numpy(), [pd.Timestamp(next_date).dayofweek]])
    prev = np.vstack([np.zeros((1, K), dtype=int), v])
    s = prev.sum(1)
    sims = np.random.default_rng(0).random((20000, nb)).argsort(1)[:, :K].sum(1) + K   # fair-machine sum distribution
    t1, t2 = np.quantile(sims, [1 / 3, 2 / 3])
    odd = (prev % 2).sum(1)
    out = [("always", np.ones(N1, dtype=bool))]
    for k in range(7):
        if (dow == k).sum() > 50:
            out.append((f"when the draw is on a {DOW[k]}", dow == k))
    out += [("when yesterday's sum was low", s <= t1), ("when yesterday's sum was mid", (s > t1) & (s <= t2)),
            ("when yesterday's sum was high", s > t2), ("when yesterday was mostly odd", odd >= 4),
            ("when yesterday was balanced odd/even", (odd >= 2) & (odd <= 3)), ("when yesterday was mostly even", odd <= 1)]
    for _, c in out:
        c[0] = False
    return out


# --------------------------------------------------------------------------- #
# The sequential memory
# --------------------------------------------------------------------------- #
def run_engine(v, dates, nb, args, next_date, narrate=False):
    n = len(v)
    rules, X = base_rules(v, dates, nb, next_date)
    conds = conditions(v, dates, nb, next_date)
    p = K / nb
    Hb = np.stack([(M[:n] & X[:n]).sum(1) for _, M in rules]).astype(float)     # base hits (R0, n)
    Kb = np.stack([M.sum(1) for _, M in rules]).astype(float)                   # base guess sizes (R0, n+1)
    names, H, Kk = [], [], []
    for ci, (cn, cm) in enumerate(conds):
        for ri, (rn, _) in enumerate(rules):
            names.append((rn, cn, ri, ci))
        H.append(Hb * cm[None, :n])
        Kk.append(Kb * cm[None, :])
    H, Kk = np.vstack(H), np.vstack(Kk)                                          # (R, n), (R, n+1)
    R = len(names)
    E = H - Kk[:, :n] * p                                                         # excess hits
    V = Kk[:, :n] * p * (1 - p) * np.clip(nb - Kk[:, :n], 0, None) / (nb - 1)     # variance under a fair machine
    A = (Kk[:, :n] > 0).astype(float)
    cE, cV, cA = (np.hstack([np.zeros((R, 1)), np.cumsum(M_, 1)]) for M_ in (E, V, A))

    W, zin, zout, minact = args.window, args.z_in, args.z_out, args.min_active
    promoted = np.zeros(R, dtype=bool)
    post_e = np.zeros(R)
    post_v = np.zeros(R)
    post_n = np.zeros(R)
    ever = np.zeros(R, dtype=bool)
    n_prom = np.zeros(n + 1, dtype=int)
    start = args.warmup
    ens5, ensK, ens_core = [], [], []
    hed5, hedK, hed_core = [], [], []
    R0, C_ = len(rules), len(conds)
    CM = np.stack([c for _, c in conds])                                          # (C, n+1)
    G_all = np.stack([M for _, M in rules])                                       # (R0, n+1, nb)

    def hedge_order(t):
        lo2 = max(0, t - args.hedge_window)
        g = (cE[:, t] - cE[:, lo2]).reshape(C_, R0)
        w = np.exp(args.eta * (g - g.max()))
        w = w * CM[:, t][:, None] / np.maximum(Kb[:, t][None, :], 1)
        sc = w.sum(0) @ G_all[:, t, :]
        return np.argsort(-sc - 1e-9 * np.arange(nb), kind="stable")
    V_core = 12 if nb >= 80 else 10
    story = []
    zr_t = np.zeros(R)
    for t in range(start, n + 1):
        lo = max(0, t - W)
        e_w, v_w, a_w = cE[:, t] - cE[:, lo], cV[:, t] - cV[:, lo], cA[:, t] - cA[:, lo]
        zr = np.where(v_w > 0, e_w / np.sqrt(np.maximum(v_w, 1e-12)), 0.0)
        new_in = (~promoted) & (zr >= zin) & (a_w >= minact)
        new_out = promoted & (zr < zout)
        promoted = (promoted | new_in) & ~new_out
        ever |= promoted
        n_prom[t] = promoted.sum()
        zr_t = zr
        if t == n:
            break
        act = promoted & (Kk[:, t] > 0)                                           # promoted AND guessing today
        post_e[act] += E[act, t]
        post_v[act] += V[act, t]
        post_n[act] += 1
        # the vote of today's active memory: each rule votes for its guessed balls with weight z
        score = np.zeros(nb)
        if act.any():
            for r in np.where(act)[0]:
                rn, cn, ri, ci = names[r]
                score += rules[ri][1][t] * zr[r] / max(Kk[r, t], 1)
        order = np.argsort(-score - 1e-9 * np.arange(nb), kind="stable") if act.any() else None
        drawn = X[t]
        ho = hedge_order(t)
        hed5.append(drawn[ho[:K]].sum())
        hedK.append(drawn[ho[:V_core]].sum())
        hed_core.append(drawn[ho[:V_core]].sum() >= 2)
        if order is not None:
            ens5.append(drawn[order[:K]].sum())
            ensK.append(drawn[order[:V_core]].sum())
            ens_core.append(drawn[order[:V_core]].sum() >= 2)
        if narrate and t >= n - args.narrate:
            lines = [f"Draw {pd.Timestamp(dates[t]).date()} ({DOW[pd.Timestamp(dates[t]).dayofweek]}), actual "
                     f"{' '.join(map(str, v[t]))}:"]
            if new_in.any():
                for r in np.where(new_in)[0][:3]:
                    lines.append(f"   + I now believe: '{names[r][0]}' {names[r][1]} (recent z {zr[r]:+.2f})")
            if new_out.any():
                for r in np.where(new_out)[0][:3]:
                    lines.append(f"   - No, let me go back: dropping '{names[r][0]}' {names[r][1]} (recent z {zr[r]:+.2f})")
            if order is not None:
                top = [int(b) + 1 for b in order[:K]]
                lines.append(f"   My top 5 from {int(act.sum())} active rules: {top} -> {int(drawn[order[:K]].sum())} hit(s)")
            else:
                lines.append("   Nothing in memory applies today: no guess.")
            story.append("\n".join(lines))
    # next draw vote
    t = n
    act = promoted & (Kk[:, t] > 0)
    score = np.zeros(nb)
    backers = {}
    for r in np.where(act)[0]:
        rn, cn, ri, ci = names[r]
        g = rules[ri][1][t]
        score += g * zr_t[r] / max(Kk[r, t], 1)
        for b in np.where(g)[0]:
            backers.setdefault(int(b) + 1, []).append(f"{rn} {cn}".replace(" always", ""))
    order = np.argsort(-score - 1e-9 * np.arange(nb), kind="stable")
    nxt = [int(b) + 1 for b in order if score[b] > 0]
    hnext = [int(b) + 1 for b in hedge_order(n)]
    memory = sorted([dict(rule=names[r][0], condition=names[r][1], recent_z=round(float(zr_t[r]), 2),
                          draws_used_after_promotion=int(post_n[r]),
                          hits_vs_expected_after_promotion=round(float(post_e[r]), 2)) for r in np.where(promoted)[0]],
                    key=lambda d: -d["recent_z"])
    pe, pv = post_e.sum(), post_v.sum()
    return dict(n=n, nb=nb, rules=R, promoted_now=int(promoted.sum()), ever=int(ever.sum()),
                post_excess=float(pe), post_z=float(pe / math.sqrt(pv)) if pv > 0 else 0.0, post_uses=int(post_n.sum()),
                ens_top5=float(np.mean(ens5)) if ens5 else float("nan"),
                ens_topK=float(np.mean(ensK)) if ensK else float("nan"),
                ens_core=float(np.mean(ens_core)) if ens_core else float("nan"), ens_draws=len(ens5), V=V_core,
                hed_top5=float(np.mean(hed5)), hed_topK=float(np.mean(hedK)), hed_core=float(np.mean(hed_core)),
                hnext=hnext,
                next=nxt, backers=backers, memory=memory, story=story, mean_promoted=float(n_prom[start:].mean()))


# --------------------------------------------------------------------------- #
# Expert interface for the main ensemble (stage4): hedge-vote probabilities, walk-forward
# --------------------------------------------------------------------------- #
def hedge_probs(v, dates_all, nb, start, hedge_window=300, eta=0.5, alpha=0.5):
    """Inclusion probabilities (rows sum to 5) for draws start..n (row n = the next draw). Row t uses draws < t only.
    v: (n, 5) drawn balls 1..nb, oldest first; dates_all: n+1 dates (the last one = the next draw)."""
    dates_all = pd.DatetimeIndex(dates_all)
    n = len(v)
    rules, X = base_rules(v, dates_all[:n], nb, dates_all[n])
    conds = conditions(v, dates_all[:n], nb, dates_all[n])
    p = K / nb
    G = np.stack([M for _, M in rules]).astype(float)                            # (R0, n+1, nb)
    Kb = G.sum(2)                                                                 # (R0, n+1)
    Hb = np.einsum("rtb,tb->rt", G[:, :n], X[:n].astype(float))                  # base hits (R0, n)
    CM = np.stack([c for _, c in conds]).astype(float)                            # (C, n+1)
    E = CM[:, None, :n] * (Hb - Kb[:, :n] * p)[None]                              # (C, R0, n) excess hits
    cE = np.concatenate([np.zeros(E.shape[:2] + (1,)), np.cumsum(E, 2)], 2)       # cE[..., t] = sum over draws < t
    out = np.empty((n + 1 - start, nb))
    for t in range(start, n + 1):
        g = cE[:, :, t] - cE[:, :, max(0, t - hedge_window)]
        w = np.exp(eta * (g - g.max())) * CM[:, t][:, None] / np.maximum(Kb[:, t][None, :], 1)
        sc = w.sum(0) @ G[:, t, :]
        s = sc / sc.mean() if sc.mean() > 0 else np.ones(nb)
        out[t - start] = p * (1 - alpha + alpha * s)                              # shrunk towards chance
    for _ in range(4):
        out = np.clip(out, 1e-6, 1 - 1e-6)
        out *= K / out.sum(1, keepdims=True)
    return out


# --------------------------------------------------------------------------- #
def chance_core(V, nb):
    tot = math.comb(nb, K)
    return sum(math.comb(V, j) * math.comb(nb - V, K - j) for j in range(2, K + 1)) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=100, help="recent record (draws) used to promote / demote")
    ap.add_argument("--z-in", type=float, default=2.5, help="promote when the recent z-score reaches this")
    ap.add_argument("--z-out", type=float, default=0.5, help="demote when the recent z-score falls below this")
    ap.add_argument("--min-active", type=int, default=20, help="a rule must have guessed on this many recent draws")
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--hedge-window", type=int, default=300, help="draws of history behind each hedge weight")
    ap.add_argument("--eta", type=float, default=0.5, help="hedge learning rate")
    ap.add_argument("--null-reps", type=int, default=20, help="fair-machine control runs per game")
    ap.add_argument("--narrate", type=int, default=8, help="print the engine's thinking for the last N draws")
    ap.add_argument("--next-date", default="")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in args.csv:
        v, dates, nb = load(f)
        if args.next_date:
            nd = args.next_date
        else:                                   # v3: next day on which this game actually draws (not always +1 day)
            from rule_miner import next_draw_date
            nd = str(next_draw_date(dates).date())
        game = Path(f).stem
        print("=" * 100)
        print(f"{game}: 5/{nb}, {len(v)} draws {dates[0].date()} .. {dates[-1].date()} | predicting {nd}")
        print("=" * 100)
        res = run_engine(v, dates, nb, args, nd, narrate=True)
        print(f"Rule library: {res['rules']} rules (base rules x conditions). Walked every draw in order from draw "
              f"{args.warmup}.\n")
        print("--- How it thought on the latest draws ---")
        print("\n".join(res["story"]))
        rng = np.random.default_rng(args.seed)
        nulls = []
        for i in range(args.null_reps):
            vs = simulate(len(v), nb, rng)
            nulls.append(run_engine(vs, dates, nb, args, nd))
            print(f"\r  fair-machine control {i + 1}/{args.null_reps}", end="", flush=True)
        print("\n")
        p5, pK, pc = K * K / nb, res["V"] * K / nb, chance_core(res["V"], nb)

        def emp(key, higher=True):
            arr = np.array([r[key] for r in nulls])
            return (1 + (arr >= res[key]).sum()) / (len(arr) + 1), float(np.mean(arr))

        rows = [("rules ever promoted", "ever", ""), ("rules in memory on average", "mean_promoted", ""),
                ("honest z of promoted rules AFTER promotion", "post_z", "0 = chance"),
                ("memory vote: hits in top 5", "ens_top5", f"chance {p5:.3f}"),
                (f"memory vote: hits in top {res['V']}", "ens_topK", f"chance {pK:.3f}"),
                (f"memory vote: core-{res['V']} hit (>= 2 of 5)", "ens_core", f"chance {pc:.3f}"),
                ("hedge vote: hits in top 5", "hed_top5", f"chance {p5:.3f}"),
                (f"hedge vote: hits in top {res['V']}", "hed_topK", f"chance {pK:.3f}"),
                (f"hedge vote: core-{res['V']} hit (>= 2 of 5)", "hed_core", f"chance {pc:.3f}")]
        print("--- Scorecard: real draws vs the same engine on fair-machine draws ---")
        print("| measure | real draws | fair-machine average | p (real beats fair machine) | note |")
        print("|---|---|---|---|---|")
        card = {}
        for label, key, note in rows:
            pval, mean = emp(key)
            card[key] = dict(real=res[key], null_mean=mean, p=pval)
            print(f"| {label} | {res[key]:.3f} | {mean:.3f} | {pval:.3f} | {note} |")
        ok = (card["post_z"]["p"] < 0.05 and card["ens_core"]["p"] < 0.05) or card["hed_core"]["p"] < 0.01
        verdict = ("THE MEMORY LEARNED SOMETHING REAL: promoted rules beat the fair-machine controls after promotion. "
                   "Confirm it on new draws with the tracker." if ok else
                   "NO REAL LEARNING: the engine promotes rules on fair-machine draws just as readily, and after promotion "
                   "its rules hit at the same rate as there. What it 'learned' is the luck any rule library finds.")
        if args.null_reps < 100:
            verdict += (f" (Only {args.null_reps} fair-machine controls: the smallest possible p is "
                        f"{1 / (args.null_reps + 1):.3f}, and with 9 measures one or two will touch it by chance. Use "
                        f"--null-reps 200 before believing any single p.)")
        print(f"\nVERDICT {game}: {verdict}\n")
        print(f"--- Memory now ({res['promoted_now']} rules) ---")
        for m in res["memory"][:12]:
            print(f"  z {m['recent_z']:+.2f} | {m['rule']} {m['condition']} | after promotion: "
                  f"{m['hits_vs_expected_after_promotion']:+.2f} hits vs expected over {m['draws_used_after_promotion']} draws")
        print(f"\n--- Vote for {nd} ---")
        if res["next"]:
            print(f"TOP 2 = {res['next'][:2]} | TOP 5 = {res['next'][:5]} | TOP {res['V']} = {res['next'][:res['V']]}")
            for b in res["next"][:5]:
                print(f"  {b}: backed by {len(res['backers'][b])} rule(s), e.g. {res['backers'][b][0]}")
        else:
            print("No rule in memory applies to this draw: the engine makes no guess.")
        print(f"Hedge vote: TOP 2 = {res['hnext'][:2]} | TOP 5 = {res['hnext'][:5]} | TOP {res['V']} = {res['hnext'][:res['V']]}")
        print("Status: " + ("supported by the scorecard" if ok else "NOT supported by the scorecard: treat as arbitrary picks"))
        payload = dict(game=game, pool=nb, draws=len(v), predicting=nd, scorecard=card, verdict=verdict,
                       memory=res["memory"], next=res["next"][:res["V"]], hedge_next=res["hnext"][:res["V"]], settings=vars(args))
        (out / f"rule_memory_{game}.json").write_text(json.dumps(payload, indent=2, default=float))
        print(f"Saved {out / f'rule_memory_{game}.json'}\n")


if __name__ == "__main__":
    main()
