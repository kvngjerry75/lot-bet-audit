#!/usr/bin/env python3
"""
rule_miner.py: the rule engine the owner described. It reads the actual numbers draw by draw, from the first draw to
the latest, finds the arithmetic that would have produced each new number from what came before, turns every such
relationship into a rule, tests each rule on the next date, keeps the evidence for and against it, and only lets a
rule vote once its evidence is too strong to be luck.

    !python rule_miner.py --game DayWA                          # one game, 100 fair-machine controls
    !python rule_miner.py --game all --nulls 100                # the four games
    !python rule_miner.py --game Main --plant 0.05              # positive control: a planted rule must be found

HOW IT THINKS (one draw at a time)
  1. Relationships it looks for (each with every constant c = 0..pool-1, so "+3", "+17" ... are all separate rules):
       position arithmetic       ball = yesterday's j-th ball + c         (lags 1, 2, 3; any position of today's draw)
       position to position      today's k-th ball = yesterday's j-th ball + c
       mirror                    ball = c - yesterday's j-th ball
       two-ball arithmetic       ball = yesterday's i-th + j-th + c, i-th - j-th + c, j-th - i-th + c
       multiplication            ball = m x yesterday's j-th ball + c       (m = 2..9)
       sum                       ball = sum of yesterday's (or 2 days ago) numbers + c
       date arithmetic           ball = day + c, month + c, day + month + c, day x month + c, day-of-year + c,
                                 yy + mm + dd + c, days since 2000-01-01 + c
       cross-game                ball = the other game's latest (earlier-date) j-th ball + c
       follow-ups                "after a comes b" for every pair of numbers a, b
     every rule (except follow-ups) also exists in "only on <weekday>" versions. All numbers wrap into 1..pool.
  2. DISCOVERY: after each draw, every rule that produced one of the drawn numbers scores a hit; every rule that
     guessed and missed scores a miss. A rule's record is the evidence; nothing is decided in advance.
  3. EVIDENCE (the "is it luck?" check, done properly): each rule carries a sequential likelihood ratio against
     "pure luck" (a test martingale, mixture over lift 1.25x .. 3x). Under pure luck this number stays below X
     forever with probability >= 1 - 1/X, however long you watch. Because ~100,000 rules are watched at once,
     a rule needs evidence above (number of rules / 0.05) to be called PROVEN.
  4. MEMORY + VOTE: before draw t each ball gets   chance + sum over rules pointing at it of
     P(rule is real | its record) x (its estimated lift - 1) x its chance rate.
     P(rule is real) starts at 1 / (number of rules) and grows only with evidence, so lucky streaks of a few draws
     move nothing, while a rule that keeps working takes over the vote. Rules that stop working lose evidence and
     fade out by themselves (REFINEMENT: the neighbouring versions, c +- 1, other lags, other weekdays, are separate
     rules, so the version that actually works is the one that gathers evidence).
  5. HONEST SCORE: every draw after the warm-up is predicted before it is seen. The identical engine runs on
     simulated fair-machine histories (same dates, same other-game draws); the real score must beat them.
     --plant shows that the engine DOES find a real pattern when one exists.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

K = int(__import__("os").environ.get("ASTRO_K", "5"))   # balls per draw
NUM = [f"Number{i}" for i in range(1, K + 1)]
GAMES = {"DayWA": ("daywa_audited.csv", 39), "Main": ("main_draw_audited.csv", 90),
         "Noonrush": ("noonrush_audited.csv", 90), "VAG": ("vag_audited.csv", 90)}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LIFTS = np.array([1.25, 1.5, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(path):
    r = pd.read_csv(path)
    r["Date"] = pd.to_datetime(r["Date"])
    r = r.iloc[::-1].reset_index(drop=True)
    r = r.iloc[np.argsort(r["Date"].to_numpy(), kind="stable")].reset_index(drop=True)
    return r[NUM].to_numpy(int) - 1, pd.DatetimeIndex(r["Date"])


def next_draw_date(dates):
    """Next date on which this game normally draws (skips weekdays it never or almost never uses)."""
    dows = pd.Series(dates.dayofweek).value_counts()
    used = set(dows[dows >= 0.2 * dows.max()].index)
    d = dates[-1] + pd.Timedelta(days=1)
    while d.dayofweek not in used:
        d += pd.Timedelta(days=1)
    return d


# --------------------------------------------------------------------------- #
# Rule families: every family returns pred (R, n+1) int16 ball index or -1, target (R,) (-1 = any position,
# j = position j), extra activity mask or None, and a namer
# --------------------------------------------------------------------------- #
def families(P, dates_all, nb, others):
    """P (n, 5) 0-based in drawn order, dates_all n+1 dates (last = next draw), others {name: (P, dates)}."""
    n = len(P)
    N1 = n + 1
    V = P + 1                                                     # 1-based values for the arithmetic
    fams = []

    def lagged(k):
        L = np.full((N1, K), -1, dtype=np.int64)
        L[k:n + 1] = V[:n + 1 - k]
        return L                                                  # row t = draw t-k (1-based), -1 = none

    cs = np.arange(nb)
    for k in (1, 2, 3):
        L = lagged(k)
        pred = np.where(L[None, :, :] > 0, (L[None, :, :] - 1 + cs[:, None, None]) % nb, -1)   # (nb, N1, 5)
        pred = pred.transpose(2, 0, 1).reshape(K * nb, N1)
        meta = [(j, c) for j in range(K) for c in cs]
        fams.append(("shift", pred, np.full(len(pred), -1), None,
                     lambda r, k=k, meta=meta: f"ball = {src(meta[r][0], k)} + {meta[r][1]}"))
    L = lagged(1)
    base = np.where(L[None, :, :] > 0, (L[None, :, :] - 1 + cs[:, None, None]) % nb, -1)       # (nb, N1, 5)
    pred = np.concatenate([base.transpose(2, 0, 1).reshape(K * nb, N1)] * K)                  # target position outer
    tgt = np.repeat(np.arange(K), K * nb)
    meta = [(jt, j, c) for jt in range(K) for j in range(K) for c in cs]
    fams.append(("position", pred, tgt, None,
                 lambda r, meta=meta: f"{ordinal(meta[r][0])} ball = yesterday's {ordinal(meta[r][1])} ball + {meta[r][2]}"))
    pred = np.where(L[None, :, :] > 0, (cs[:, None, None] - L[None, :, :]) % nb, -1).transpose(2, 0, 1).reshape(K * nb, N1)
    meta = [(j, c) for j in range(K) for c in cs]
    fams.append(("mirror", pred, np.full(len(pred), -1), None,
                 lambda r, meta=meta: f"ball = {meta[r][1]} - yesterday's {ordinal(meta[r][0])} ball"))
    rows, meta = [], []
    for i in range(K):
        for j in range(i + 1, K):
            ok = (L[:, i] > 0)
            for op, v in (("+", L[:, i] + L[:, j]), ("-", L[:, i] - L[:, j]), ("r-", L[:, j] - L[:, i])):
                rows.append(np.where(ok[None, :], (v[None, :] - 1 + cs[:, None]) % nb, -1))
                meta += [(i, j, op, c) for c in cs]
    fams.append(("two-ball", np.concatenate(rows), np.full(len(meta), -1), None,
                 lambda r, meta=meta: pair_name(*meta[r])))
    rows, meta = [], []
    for m in range(2, 10):
        v = np.where(L > 0, L * m, -1)
        for j in range(K):
            rows.append(np.where(v[None, :, j] > 0, (v[None, :, j] - 1 + cs[:, None]) % nb, -1))
            meta += [(m, j, c) for c in cs]
    fams.append(("multiply", np.concatenate(rows), np.full(len(meta), -1), None,
                 lambda r, meta=meta: f"ball = {meta[r][0]} x yesterday's {ordinal(meta[r][1])} ball + {meta[r][2]}"))
    rows, meta = [], []
    for k in (1, 2):
        Lk = lagged(k)
        s = np.where(Lk[:, 0] > 0, Lk.sum(1), -1)
        rows.append(np.where(s[None, :] > 0, (s[None, :] - 1 + cs[:, None]) % nb, -1))
        meta += [(k, c) for c in cs]
    fams.append(("sum", np.concatenate(rows), np.full(len(meta), -1), None,
                 lambda r, meta=meta: f"ball = sum of {ago(meta[r][0])}'s numbers + {meta[r][1]}"))
    dd, mm, yy = dates_all.day.to_numpy(), dates_all.month.to_numpy(), dates_all.year.to_numpy() % 100
    doy = dates_all.dayofyear.to_numpy()
    epoch = ((dates_all - pd.Timestamp("2000-01-01")).days).to_numpy()
    dvals = {"day": dd, "month": mm, "day + month": dd + mm, "day x month": dd * mm, "day of year": doy,
             "yy + mm + dd": yy + mm + dd, "days since 2000-01-01": epoch}
    rows, meta = [], []
    for name, v in dvals.items():
        rows.append(((v[None, :] - 1 + cs[:, None]) % nb))
        meta += [(name, c) for c in cs]
    fams.append(("date", np.concatenate(rows), np.full(len(meta), -1), None,
                 lambda r, meta=meta: f"ball = {meta[r][0]} + {meta[r][1]}"))
    rows, meta = [], []
    for g, (Pg, dg) in others.items():
        j = np.searchsorted(dg.values, dates_all.values, side="left") - 1   # the other game's last draw on an EARLIER date
        Lg = np.where(j[:, None] >= 0, Pg[np.maximum(j, 0)] + 1, -1)
        for q in range(K):
            rows.append(np.where(Lg[None, :, q] > 0, (Lg[None, :, q] - 1 + cs[:, None]) % nb, -1))
            meta += [(g, q, c) for c in cs]
    if rows:
        fams.append(("cross-game", np.concatenate(rows), np.full(len(meta), -1), None,
                     lambda r, meta=meta: f"ball = {meta[r][0]}'s latest {ordinal(meta[r][1])} ball + {meta[r][2]}"))
    # follow-ups: rule (a, b) is active only when a was drawn yesterday
    Xprev = np.zeros((N1, nb), dtype=bool)
    Xprev[np.repeat(np.arange(1, n + 1), K), P.ravel()] = True
    pred = np.repeat(np.arange(nb), 1)[None, :].repeat(nb, 0).reshape(-1)          # rule a*nb+b -> b
    pred = np.repeat(pred[:, None], N1, 1).astype(np.int64)
    act = np.repeat(Xprev.T, nb, axis=0)                                              # rule a*nb+b active if a drawn
    fams.append(("follow-up", pred, np.full(nb * nb, -1), act,
                 lambda r: f"after {r // nb + 1} comes {r % nb + 1}"))
    return [(f, p.astype(np.int16), np.asarray(t), a, nm) for f, p, t, a, nm in fams]


def ordinal(j):
    return ["1st", "2nd", "3rd", "4th", "5th"][j]


def src(j, k):
    return f"yesterday's {ordinal(j)} ball" if k == 1 else f"{ordinal(j)} ball of {k} draws ago"


def ago(k):
    return {1: "yesterday", 2: "2 draws ago", 3: "3 draws ago"}[k]


def pair_name(i, j, op, c):
    a, b = f"yesterday's {ordinal(i)}", f"{ordinal(j)}"
    expr = {"+": f"{a} + {b}", "-": f"{a} - {b}", "r-": f"{b} - {ordinal(i)} (yesterday)"}[op]
    return f"ball = {expr} + {c}"


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def run(P, dates_all, nb, others, warmup=100, alpha=0.05, chunk=2000, tie_seed=12345, keep_memory=False, return_p=False):
    n = len(P)
    N1 = n + 1
    Xp = np.zeros((N1, nb), dtype=bool)
    Xp[np.repeat(np.arange(n), K), P.ravel()] = True
    Pp = np.vstack([P, -np.ones((1, K), dtype=P.dtype)])
    dow = dates_all.dayofweek.to_numpy()
    days = [k for k in range(7) if (dow[:n] == k).sum() >= 50]
    conds = [("always", np.ones(N1, dtype=bool))] + [(f"on {DOW[k]}", dow == k) for k in days]
    fams = families(P, dates_all, nb, others)
    R_total = sum(len(p) * (1 if a is not None else len(conds)) for _, p, _, a, _ in fams)
    log_prior = -math.log(R_total)
    proven_log = math.log(R_total / alpha)
    score = np.zeros((N1, nb))
    n_proven_end = 0
    ever_proven = 0
    best = []                                                     # (logE at end, name, hits, trials, expected)
    tidx = np.arange(N1)
    for fname, pred_all, tgt_all, act_all, namer in fams:
        cl = [("always", conds[0][1])] if act_all is not None else conds
        for s in range(0, len(pred_all), chunk):
            pred = pred_all[s:s + chunk]
            tgt = tgt_all[s:s + chunk]
            base_act = pred >= 0
            if act_all is not None:
                base_act &= act_all[s:s + chunk]
            pc = np.maximum(pred, 0).astype(np.int64)
            hit_any = Xp[tidx[None, :], pc]
            pos_t = np.where(tgt[:, None] >= 0, tgt[:, None], 0)
            hit_pos = Pp[tidx[None, :], np.broadcast_to(pos_t, pred.shape)] == pred
            hit = np.where(tgt[:, None] >= 0, hit_pos, hit_any)
            q = np.where(tgt >= 0, 1.0 / nb, K / nb)[:, None]
            lp = np.log(LIFTS)[None, :, None]
            lm = np.log((1 - np.minimum(LIFTS[None, :, None] * q[:, :, None], 0.99)) / (1 - q[:, :, None]))
            for cname, cm in cl:
                act = base_act & cm[None, :]
                h = (act & hit)
                h[:, n] = False
                a_ = act.copy()
                a_[:, n] = False
                H = np.zeros(act.shape, dtype=np.int32)
                M = np.zeros(act.shape, dtype=np.int32)
                H[:, 1:] = np.cumsum(h, 1)[:, :-1]                # evidence strictly before t
                M[:, 1:] = np.cumsum(a_, 1)[:, :-1]
                le = H[:, None, :] * lp + (M - H)[:, None, :] * lm   # (r, L, t)
                mx = le.max(1)
                logE = mx + np.log(np.exp(le - mx[:, None, :]).mean(1))
                w = 1.0 / (1.0 + np.exp(-(log_prior + logE)))
                post = np.exp(le - mx[:, None, :])
                Lhat = (post * LIFTS[None, :, None]).sum(1) / post.sum(1)
                contrib = w * (Lhat - 1.0) * q
                use = act & (w > 1e-7)
                if use.any():
                    rr, tt = np.nonzero(use)
                    np.add.at(score, (tt, pred[rr, tt].astype(np.int64)), contrib[rr, tt])
                end = logE[:, n]
                n_proven_end += int((end >= proven_log).sum())
                ever_proven += int((logE[:, warmup:] >= proven_log).any(1).sum())
                if keep_memory:
                    top = np.argsort(-end)[:5]
                    for r in top:
                        exp_ = float(M[r, n] * q[r, 0])
                        best.append((float(end[r]), f"{namer(s + r)} ({cname})" if cname != "always" else namer(s + r),
                                     int(H[r, n]), int(M[r, n]), exp_, float(w[r, n]), fname))
    # forecast and scoring
    p = K / nb + score
    p = np.clip(p, 1e-4, None)
    for _ in range(5):
        p = np.clip(p * (K / p.sum(1, keepdims=True)), 1e-4, 0.999)
    rng = np.random.default_rng(tie_seed)
    tie = rng.random((N1, nb)) * 1e-12
    order = np.argsort(-(p + tie), axis=1, kind="stable")
    V = 12 if nb >= 80 else 10
    ar = np.arange(N1)[:, None]
    X = Xp.astype(float)
    T = slice(warmup, n)
    h5 = X[ar, order[:, :K]].sum(1)[T]
    hV = X[ar, order[:, :V]].sum(1)[T]
    h2 = X[ar, order[:, :2]].sum(1)[T]
    pr = np.clip(p[T], 1e-9, 1 - 1e-9)
    bce = -(X[T] * np.log(pr) + (1 - X[T]) * np.log1p(-pr)).sum(1)
    q0 = K / nb
    ubce = -(K * math.log(q0) + (nb - K) * math.log(1 - q0))
    out = dict(n=n, nb=nb, rules=R_total, scored=int(n - warmup), top5=float(h5.mean()), core=float((hV >= 2).mean()),
               core_hits=int((hV >= 2).sum()), both2=float((h2 >= 2).mean()), skill=float(1 - bce.mean() / ubce),
               proven_now=n_proven_end, ever_proven=ever_proven, V=V,
               max_shift=float(np.abs(score[T]).max()), next_order=[int(b) + 1 for b in order[n]],
               next_p=p[n].tolist(), per_draw_core=(hV >= 2).astype(int).tolist(), per_draw_h5=h5.tolist())
    if return_p:
        out["p"] = p
    if keep_memory:
        best.sort(key=lambda x: -x[0])
        out["memory"] = [dict(rule=b[1], family=b[6], evidence_log=round(b[0], 2), hits=b[2], guesses=b[3],
                              expected_hits=round(b[4], 1), belief_real=round(b[5], 6),
                              proven=b[0] >= proven_log) for b in best[:15]]
        out["proven_threshold_log"] = proven_log
    return out


def fair(n, nb, rng):
    return rng.random((n, nb)).argsort(1)[:, :K]


def miner_probs(P, dates_all, nb, start, others=None, alpha=0.05):
    """stage4 expert interface: inclusion probabilities (rows sum to 5) for draws start..n (row n = next draw).
    P: (n, 5) 0-based drawn order, oldest first; dates_all: n+1 dates. Row t uses draws < t only."""
    res = run(np.asarray(P, dtype=np.int64), pd.DatetimeIndex(dates_all), nb, others or {}, warmup=min(start, len(P) - 1),
              alpha=alpha, return_p=True)
    return res["p"][start:]


def plant(n, nb, rng, rho, offset=17, src_pos=0):
    """Fair machine EXCEPT: with probability rho the ball (yesterday's src_pos ball + offset) is forced into the draw."""
    P = fair(n, nb, rng)
    for t in range(1, n):
        if rng.random() < rho:
            b = (P[t - 1, src_pos] + offset) % nb
            if b not in P[t]:
                P[t, rng.integers(K)] = b
    return P


# --------------------------------------------------------------------------- #
def chance(nb):
    V = 12 if nb >= 80 else 10
    tot = math.comb(nb, K)
    core = sum(math.comb(V, j) * math.comb(nb - V, K - j) for j in range(2, K + 1)) / tot
    return dict(top5=K * K / nb, core=core, both2=K * (K - 1) / (nb * (nb - 1)), skill=0.0)


def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-control averages when --nulls 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="all")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--nulls", type=int, default=100, help="fair-machine control histories")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--plant", type=float, default=0.0, help="positive control: plant a rule with this strength")
    ap.add_argument("--no-cross", action="store_true", help="leave out the cross-game rules")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--tracker", default="", help="log the vote for the next draw to this tracker ledger (before the draw)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {g: load(Path(a.dir) / f) for g, (f, _) in GAMES.items()}
    todo = list(GAMES) if a.game == "all" else [a.game]
    rng = np.random.default_rng(a.seed)
    for g in todo:
        nb = GAMES[g][1]
        P, dates = data[g]
        nd = next_draw_date(dates)
        dates_all = pd.DatetimeIndex(list(dates) + [nd])
        others = {} if a.no_cross else {o: data[o] for o in GAMES if o != g}
        t0 = time.time()
        label = g
        if a.plant > 0:
            P = plant(len(P), nb, rng, a.plant)
            label = f"{g} (PLANTED rule 'ball = yesterday's 1st ball + 17' with strength {a.plant})"
        real = run(P, dates_all, nb, others, a.warmup, keep_memory=True)
        el = time.time() - t0
        print("=" * 110)
        print(f"{label}: 5/{nb}, {len(P)} draws, {real['rules']:,} rules watched, {real['scored']} draws scored "
              f"(from draw {a.warmup}); one pass {el:.0f}s")
        print("=" * 110)
        nulls = []
        for i in range(a.nulls):
            nulls.append(run(fair(len(P), nb, rng), dates_all, nb, others, a.warmup))
            print(f"\r  fair-machine control {i + 1}/{a.nulls}", end="", flush=True)
        print()
        ch = chance(nb)
        card = {}
        print("| measure | real draws | chance | fair-machine controls mean | p (controls >= real) |")
        print("|---|---|---|---|---|")
        for key, label_ in (("top5", "mean hits in top 5"), ("core", f"core-{real['V']} hit rate (>= 2 of 5)"),
                            ("both2", "both of top 2 drawn"), ("skill", "log-loss skill vs uniform"),
                            ("proven_now", "rules PROVEN now (evidence > rules/0.05)"),
                            ("ever_proven", "rules that were ever PROVEN")):
            arr = np.array([x[key] for x in nulls]) if nulls else np.array([np.nan])
            pv = (1 + (arr >= real[key]).sum()) / (len(arr) + 1) if nulls else float("nan")
            card[key] = dict(real=real[key], null_mean=float(np.nanmean(arr)), p=float(pv))
            chs = f"{ch[key]:.4f}" if key in ch else "0"
            print(f"| {label_} | {real[key]:.4f} | {chs} | {np.nanmean(arr):.4f} | {pv:.3f} |")
        k = real["core_hits"]
        print(f"\ncore hits: {k} of {real['scored']} draws")
        print("\n--- Memory: the strongest rules after the latest draw (evidence = log likelihood ratio vs luck; "
              f"PROVEN needs > {real['proven_threshold_log']:.1f}) ---")
        for m in real["memory"][:10]:
            print(f"  {m['evidence_log']:+7.2f} | {m['hits']:4d} hits in {m['guesses']:4d} guesses (luck: {m['expected_hits']:6.1f})"
                  f" | belief real {m['belief_real']:.2e} | {'PROVEN' if m['proven'] else 'not proven'} | {m['rule']}")
        print(f"\n--- Vote for {nd.date()} ({DOW[nd.dayofweek]}) ---")
        top = real["next_order"]
        maxlift = max(real["next_p"]) / (K / nb)
        print(f"TOP 2 = {top[:2]} | TOP 5 = {top[:5]} | CORE {real['V']} = {top[:real['V']]}")
        print(f"Largest lift any ball gets from the memory: x{maxlift:.4f} "
              + ("(no rule is believed: these picks are as good as random)" if maxlift < 1.05 else ""))
        payload = dict(game=g, plant=a.plant, scorecard=card, memory=real["memory"], next_date=str(nd.date()),
                       next_top=top[:real["V"]], rules=real["rules"], scored=real["scored"], core_hits=k,
                       null_rows=[{kk: x[kk] for kk in ("top5", "core", "both2", "skill", "proven_now", "ever_proven")}
                                  for x in nulls])
        if a.tracker and not a.plant:
            import tracker
            added = tracker.log_prediction(a.tracker, f"{Path(GAMES[g][0]).stem}", str(nd.date()), nb, top[:2], top[:5],
                                           top[:real["V"]], "rule_miner", "proven" if real["proven_now"] else "no proven rule")
            print(f"Tracker: {'pick LOGGED' if added else 'a pick was already logged for this draw (first one kept)'} in {a.tracker}")
        tag = f"{g}{'_plant' + str(a.plant) if a.plant else ''}"
        (out_dir / f"rule_miner_{tag}.json").write_text(json.dumps(payload, indent=2, default=float))
        np.save(out_dir / f"rule_miner_{tag}_core.npy", np.array(real["per_draw_core"]))
        print(f"Saved {out_dir / f'rule_miner_{tag}.json'}\n")


if __name__ == "__main__":
    main()
