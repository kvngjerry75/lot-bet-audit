#!/usr/bin/env python3
"""
forecaster.py: an "elite forecaster" engine. It reads the history the way an experienced lotto forecaster does,
through several analytical lenses at once, keeps a track record for every idea it has, and only lets an idea
move its forecast as far as that record justifies. Everything is walk-forward: draw t is forecast from draws
before t only, then scored, then learned from.

    python forecaster.py --game Main --nla-dir nla_output                     # 5/90 with machine numbers
    python forecaster.py --game Afriluck --afriluck-dir afriluck_output       # 6/57, three draws a day
    python forecaster.py --game Main --controls 20                            # + 20 fair-machine histories
    python forecaster.py --game Main --plant 0.25                             # proof: a hidden rule must be found

THE LENSES (how it thinks)
  1. LAWS (general rules that hold for every number):
       "a number comes back"            repeat of a recent number
       "turning"                        68 -> 86, 5 -> 50
       "complement"                     x -> 91-x, x -> 90-x  (57 -> 34 / 33 on 5/90)
       "neighbours"                     x -> x +/- 1..10
       "double", "digit sum"            x -> 2x, 68 -> 14
     each read from every winning position AND every machine-number position, 1 to 7 draws back, and from the
     other games' latest draws. A law pools evidence across all 90 numbers, so it can be proven fast if real.
  2. PRECEDENTS (your 68-71 -> 75-81 idea): every single number and every PAIR in today's winning and machine
     numbers is a trigger. For each trigger the engine recalls every past time it appeared, what came in the next
     draw, and whether the same follow-up keeps coming. "The numbers turned" is handled by context: the same
     trigger is also tracked separately by the weekday of the next draw, so a split like "on Fridays 57-18 instead"
     becomes its own idea with its own record.
  3. PARTNERS: numbers that historically come together; today's numbers call their partners.
  4. ANALOGUES: past draws that look most like today's (sharing 2+ numbers, machine numbers included) and what
     followed THEM.
  5. BASE RATES: each ball's long-run rate.
  6. BALL GROUPS: low/high halves (on Afriluck these are the pink and yellow balls), decades, odd/even. A group
     rate pools many balls, so a physical bias shared by a batch of balls shows up long before single-ball rates do.
HOW IT DECIDES
  * Every idea (law, precedent, context split) has a running score of evidence against pure luck. Its belief
    starts at 1 / (number of ideas of that kind) and grows only with evidence, so a coincidence seen twice moves
    nothing, while an idea that keeps working takes over. Ideas that stop working fade out on their own.
  * Each lens also has its own forecasting record; the lenses are mixed by that record (lens weights), with "pure
    chance" always in the mix, so a lens only gets weight if it has beaten chance on past draws.
  * Before every draw it prints its reasoning: the ideas behind each pick and their records.
"""
from __future__ import annotations

import argparse
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

LIFTS = np.array([1.5, 2.0, 3.0, 5.0])
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NLA = {"Main": ("main_draw_audited.csv", 90), "DayWA": ("daywa_audited.csv", 39),
       "VAG": ("vag_audited.csv", 90), "Noonrush": ("noonrush_audited.csv", 90)}


# ------------------------------------------------------------------------------------------- data
def read_game(path, k, date_col="Date"):
    r = pd.read_csv(path)
    num = [f"Number{i}" for i in range(1, k + 1)]
    mac = [f"Machine{i}" for i in range(1, k + 1)]
    if "Promotionday" in r:
        r = r[r["Promotionday"].astype(str) != "Holiday"]
    r = r[r[num].notna().all(axis=1)].copy()
    r[date_col] = pd.to_datetime(r[date_col])
    r = r.iloc[::-1]
    r = r.iloc[np.argsort(r[date_col].to_numpy(), kind="stable")].reset_index(drop=True)
    W = r[num].astype(int).to_numpy() - 1
    if all(c in r for c in mac):
        M = r[mac].to_numpy(dtype=float)
        M = np.where(np.isnan(M), 0, M).astype(int) - 1
        M[(M < 0).any(axis=1)] = -1
        M[(np.sort(M, 1) == np.sort(W, 1)).all(axis=1)] = -1     # data error: machine numbers copied from the winners
    else:
        M = np.full_like(W, -1)
    return pd.DatetimeIndex(r[date_col]), W, M


def load(args):
    if args.game == "Afriluck":
        parts = []
        for rank, g in enumerate(("anopa", "midday", "evening")):
            p = Path(args.afriluck_dir) / f"afriluck_{g}.csv"
            d, W, M = read_game(p, 6)
            parts.append(pd.DataFrame({"d": d, "rank": rank, "W": list(W), "M": list(M)}))
        df = pd.concat(parts).sort_values(["d", "rank"], kind="stable").reset_index(drop=True)
        df["key"] = df["W"].apply(lambda w: tuple(sorted(w)))
        keep, seen = [], {}
        for dt, kk in zip(df["d"], df["key"]):
            keep.append(kk not in seen or (dt - seen[kk]).days > 3)
            seen[kk] = dt
        df = df[np.array(keep)].reset_index(drop=True)
        return dict(name="Afriluck (Anopa+Midday+evening)", nb=57, k=6, dates=pd.DatetimeIndex(df["d"]),
                    W=np.stack(df["W"]), M=np.stack(df["M"]), others={},
                    session=np.array(["Anopa", "Midday", "evening"])[df["rank"].to_numpy()])
    f, nb = NLA[args.game]
    d, W, M = read_game(Path(args.nla_dir) / f, 5)
    others = {}
    for g, (fo, nbo) in NLA.items():
        if g != args.game and (Path(args.nla_dir) / fo).exists():
            others[g] = read_game(Path(args.nla_dir) / fo, 5) + (nbo,)
    return dict(name=args.game, nb=nb, k=5, dates=d, W=W, M=M, others=others)


# ------------------------------------------------------------------------------------------- transforms
def transforms(nb):
    """Laws: value -> value maps on 1..nb (None when undefined)."""
    def turn(v):
        s = str(v)
        t = int(s[::-1]) if len(s) == 2 else v * 10
        return t if 1 <= t <= nb and t != v else None
    def dsum(v):
        t = sum(map(int, str(v)))
        return t if 1 <= t <= nb and t != v else None
    T = {"comes back": lambda v: v, "turning": turn, f"complement {nb + 1}-x": lambda v: nb + 1 - v,
         f"complement {nb}-x": lambda v: (nb - v) if nb - v >= 1 else None,
         "double": lambda v: (2 * v - 1) % nb + 1, "digit sum": dsum}
    for s in range(1, 11):
        T[f"+{s}"] = (lambda s: lambda v: (v - 1 + s) % nb + 1)(s)
        T[f"-{s}"] = (lambda s: lambda v: (v - 1 - s) % nb + 1)(s)
    names = list(T)
    table = np.full((len(names), nb), -1, dtype=np.int64)
    for i, nme in enumerate(names):
        for v in range(1, nb + 1):
            o = T[nme](v)
            if o is not None:
                table[i, v - 1] = o - 1
    return names, table


def lr_log(h, m, q):
    """log mixture likelihood ratio vs luck, and posterior-mean lift (arrays)."""
    q = np.asarray(q, dtype=float)
    lp = np.log(LIFTS)
    lm = np.log((1 - np.minimum(LIFTS[None] * q[..., None], 0.95)) / (1 - q[..., None]))
    le = h[..., None] * lp + (m - h)[..., None] * lm
    mx = le.max(-1)
    w = np.exp(le - mx[..., None])
    return mx + np.log(w.mean(-1)), (w * LIFTS).sum(-1) / w.sum(-1)


# ------------------------------------------------------------------------------------------- engine
class Forecaster:
    def __init__(self, data, max_lag=7, warm=100, use_brain=False):
        self.use_brain = use_brain
        self.d = data
        self.nb, self.k = data["nb"], data["k"]
        self.max_lag, self.warm = max_lag, warm
        self.tnames, self.ttab = transforms(self.nb)
        self.q = self.k / self.nb
        v = np.arange(self.nb)
        self.groups = [(v >= self.nb // 2).astype(int), v // 10, v % 2]   # halves (0-based: 1..floor(nb/2) low), decades, parity

    # sources for the laws lens at draw t: list of (label, ball or -1)
    def law_sources(self, t, W, M, other_latest):
        src = []
        for L in range(1, self.max_lag + 1):
            for j in range(self.k):
                src.append((f"winning #{j + 1}, {L} draw(s) ago", W[t - L, j] if t - L >= 0 else -1))
                src.append((f"machine #{j + 1}, {L} draw(s) ago", M[t - L, j] if t - L >= 0 else -1))
        for g, lat in other_latest.items():
            for j in range(5):
                b = lat[0][j] if lat is not None else -1
                src.append((f"{g} latest winning #{j + 1}", b if (b < self.nb) else -1))
                b = lat[1][j] if lat is not None else -1
                src.append((f"{g} latest machine #{j + 1}", b if (b < self.nb) else -1))
        return src

    def brain(self, X):
        """STAGE 5 brain: gradient boosting on per-ball features, refit every 100 draws on earlier draws only."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        n, nb = X.shape
        C = np.vstack([np.zeros(nb), np.cumsum(X, 0)])
        t = np.arange(n)
        last = np.full(nb, -1)
        gap = np.zeros((n, nb))
        for i in range(n):
            gap[i] = i - last
            last[np.where(X[i] > 0)[0]] = i
        feats = [(C[t] - C[np.maximum(t - w, 0)]) / np.maximum(np.minimum(t, w), 1)[:, None] for w in (10, 20, 50)]
        feats.append(C[:-1] / np.maximum(t, 1)[:, None])
        feats.append(np.minimum(gap, 300) / 300)
        for gid in self.groups:
            ng = np.bincount(gid)
            hits = np.stack([np.bincount(gid, weights=C[i], minlength=len(ng)) for i in range(n)])
            feats.append((hits / np.maximum(t, 1)[:, None] / ng)[:, gid])
        F = np.stack(feats, axis=2)
        out = np.full((n, nb), self.q)
        for a in range(max(self.warm, 200), n, 100):
            m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=8,
                                               min_samples_leaf=400, l2_regularization=10.0, random_state=0)
            m.fit(F[50:a].reshape(-1, F.shape[2]), X[50:a].ravel())
            b = min(a + 100, n)
            out[a:b] = m.predict_proba(F[a:b].reshape(-1, F.shape[2]))[:, 1].reshape(b - a, nb)
        last_row = out[-1]
        return out, last_row

    def run(self, W, M, dates, narrate=False, next_date=None):
        """next_date: the real date of the draw being forecast (row n); default = the day after the last draw."""
        nb, k, q = self.nb, self.k, self.q
        n = len(W)
        M = M.copy()
        M[((M >= nb) | (M < 0)).any(axis=1)] = -1            # impossible machine numbers (bad rows) are ignored
        X = np.zeros((n, nb))
        X[np.repeat(np.arange(n), k), W.ravel()] = 1
        others = self.d["others"]
        oidx = {g: np.searchsorted(o[0].values, dates.values, side="left") - 1 for g, o in others.items()}
        # ---- law lens state
        n_src = len(self.law_sources(self.max_lag, W, M, {g: None for g in others}))
        nT = len(self.tnames)
        Lh = np.zeros((nT, n_src))
        Lm = np.zeros((nT, n_src))
        prior_law = -math.log(nT * n_src)
        # ---- precedent lens state: trigger keys -> outcome counts
        n_pairs = nb * (nb - 1) // 2
        pid = np.full((nb, nb), -1, dtype=np.int64)
        for i, (a, b) in enumerate(combinations(range(nb), 2)):
            pid[a, b] = pid[b, a] = i
        # key layout: [single W, single M, pair W, pair M] x [any day, weekday 0..6]
        n_keys = (2 * nb + 2 * n_pairs) * 8
        Pn = np.zeros(n_keys, dtype=np.float32)
        Ph = np.zeros((n_keys, nb), dtype=np.float32)
        prior_prec = -math.log(n_keys * nb)
        Co = np.zeros((nb, nb))
        lenses = ["chance", "laws", "precedents", "partners", "analogues", "base rates", "low/high", "decades",
                  "odd/even"] + (["ML brain"] if self.use_brain else [])
        if self.use_brain:
            bp, blast = self.brain(X)
        logscore = np.zeros(len(lenses))
        probs_out = np.zeros((n + 1, nb))
        explain = None
        lens_w_hist = []

        def trigger_keys(t, wd):
            """keys active for forecasting draw t (triggers from draw t-1), with labels."""
            keys, labels = [], []
            if t < 1:
                return np.array([], dtype=np.int64), labels
            for src, arr in ((0, W[t - 1]), (1, M[t - 1])):
                if arr[0] < 0:
                    continue
                for x in arr:
                    base = (src * nb + x) * 8
                    for c, tag in ((0, ""), (1 + wd, f" (when the next draw is a {DOW[wd]})")):
                        keys.append(base + c)
                        labels.append(f"{'winning' if src == 0 else 'machine'} {x + 1}{tag}")
                for a, b in combinations(sorted(arr), 2):
                    base = (2 * nb + src * n_pairs + pid[a, b]) * 8
                    for c, tag in ((0, ""), (1 + wd, f" (when the next draw is a {DOW[wd]})")):
                        keys.append(base + c)
                        labels.append(f"{'winning' if src == 0 else 'machine'} pair {a + 1}-{b + 1}{tag}")
            return np.array(keys, dtype=np.int64), labels

        for t in range(n + 1):
            wd = dates[t].dayofweek if t < n else (pd.Timestamp(next_date) if next_date is not None
                                                  else dates[-1] + pd.Timedelta(days=1)).dayofweek
            other_latest = {}
            for g, o in others.items():
                j = oidx[g][t] if t < n else len(o[0]) - 1
                other_latest[g] = (o[1][j], o[2][j]) if j >= 0 else None
            # ---------------- LAWS forecast
            src = self.law_sources(t, W, M, other_latest)
            sb = np.array([b for _, b in src])
            pred = np.where(sb[None, :] >= 0, self.ttab[:, np.maximum(sb, 0)], -1)    # (nT, n_src) predicted ball
            le, lift = lr_log(Lh, Lm, q)
            belief = 1 / (1 + np.exp(-(prior_law + le)))
            contrib = belief * (lift - 1) * q
            p_law = np.full(nb, q)
            ok = pred >= 0
            np.add.at(p_law, pred[ok], contrib[ok])
            # ---------------- PRECEDENTS forecast
            keys, klabels = trigger_keys(t, wd)
            p_prec = np.full(nb, q)
            if len(keys):
                h = Ph[keys].astype(float)
                m = np.repeat(Pn[keys].astype(float)[:, None], nb, 1)
                le_p, lift_p = lr_log(h, m, np.full(h.shape, q))
                bel_p = 1 / (1 + np.exp(-(prior_prec + le_p)))
                p_prec += (bel_p * (lift_p - 1) * q).sum(0)
            # ---------------- PARTNERS
            if t >= 1 and Co.sum() > 0:
                exp_co = Co.sum() / (nb * (nb - 1))
                z = Co[W[t - 1]].sum(0) / max(exp_co * k, 1e-9) - 1
                z[W[t - 1]] = 0
                p_part = q * np.exp(0.3 * np.clip(z, -2, 2))
            else:
                p_part = np.full(nb, q)
            # ---------------- ANALOGUES
            if t >= 30:
                cur = np.zeros(nb)
                cur[W[t - 1]] = 1
                if M[t - 1, 0] >= 0:
                    cur[M[t - 1]] = 1
                past = X[: t - 1]
                if M[0, 0] >= 0 or (M[: t - 1, 0] >= 0).any():
                    Mx = np.zeros((t - 1, nb))
                    okm = M[: t - 1, 0] >= 0
                    Mx[np.repeat(np.where(okm)[0], k), M[: t - 1][okm].ravel()] = 1
                    past = np.clip(past + Mx, 0, 1)
                ov = past @ cur
                wts = np.where(ov >= 2, ov ** 2, 0.0)
                if wts.sum() > 0:
                    v = (wts[:, None] * X[1:t]).sum(0) / wts.sum()
                    p_ana = q * np.exp(1.5 * (v - v.mean()) / (v.std() + 1e-9) * 0.1)
                else:
                    p_ana = np.full(nb, q)
            else:
                p_ana = np.full(nb, q)
            # ---------------- BASE RATES
            cnt = X[:t].sum(0) if t <= n else X.sum(0)
            p_base = (cnt + 100 * q) / (t + 100)
            # ---------------- BALL GROUPS (low/high halves - the Afriluck colours -, decades, odd/even)
            p_grps = []
            for gid in self.groups:
                pg = np.full(nb, q)
                if t > 0:
                    ng = np.bincount(gid, minlength=gid.max() + 1)
                    hits_g = np.bincount(gid, weights=cnt, minlength=gid.max() + 1)
                    pg = q * ((hits_g + 200 * q * ng) / (t * ng + 200 * ng) / q)[gid]
                p_grps.append(pg)
            extra = []
            if self.use_brain:
                extra = [bp[t] if t < n else blast]
            P_l = np.stack([np.full(nb, q), p_law, p_prec, p_part, p_ana, p_base] + p_grps + extra)
            P_l = np.clip(P_l, 1e-5, None)
            P_l = P_l * (k / P_l.sum(1, keepdims=True))
            P_l = np.clip(P_l, 1e-5, 0.95)
            # lens weights from past log-scores (chance always in the mix)
            gain = logscore - logscore[0]
            lw = np.exp(gain - gain.max())
            lw[0] = max(lw[0], 0.5 * lw.max())
            lw /= lw.sum()
            p = (lw[:, None] * P_l).sum(0)
            probs_out[t] = p
            if t == n:
                if narrate:
                    explain = self.reason(p, src, pred, belief, lift, Lh, Lm, keys, klabels, Ph, Pn, lw, lenses, p_base)
                break
            # ---------------- score and learn from draw t
            y = X[t]
            if t >= self.warm:
                logscore += (y * np.log(P_l) + (1 - y) * np.log1p(-P_l)).sum(1)
                lens_w_hist.append(lw)
            hit = np.where(ok, y[np.maximum(pred, 0)], 0)
            Lh += hit * ok
            Lm += ok
            if len(keys):
                Pn[keys] += 1
                Ph[keys] += y.astype(np.float32)
            Co[np.ix_(W[t], W[t])] += 1
            Co[W[t], W[t]] -= 1
        return probs_out, explain, dict(zip(lenses, np.mean(lens_w_hist, 0) if lens_w_hist else lw)), \
            dict(zip(lenses, logscore - logscore[0]))

    def reason(self, p, src, pred, belief, lift, Lh, Lm, keys, klabels, Ph, Pn, lw, lenses, p_base):
        nb, q = self.nb, self.q
        top = np.argsort(-p)[:12]
        lines = []
        for b in top[:6]:
            why = []
            idx = np.argwhere(pred == b)
            cand = sorted(((belief[i, j], i, j) for i, j in idx), reverse=True)[:2]
            for bel, i, j in cand:
                why.append(f"law '{self.tnames[i]}' from {src[j][0]} ({src[j][1] + 1}): {int(Lh[i, j])} hits in "
                           f"{int(Lm[i, j])} tries (luck {Lm[i, j] * q:.0f}), belief {bel:.1e}")
            if len(keys):
                h = Ph[keys, b]
                best = np.argsort(-h)[:2]
                for bi in best:
                    if h[bi] > 0:
                        why.append(f"precedent {klabels[bi]} -> {b + 1}: {int(h[bi])} of {int(Pn[keys[bi]])} times "
                                   f"(luck {Pn[keys[bi]] * q:.1f})")
            why.append(f"base rate x{p_base[b] / q:.2f}")
            lines.append(f"  {b + 1}: forecast {p[b]:.3f} vs chance {q:.3f}\n      - " + "\n      - ".join(why))
        lw_txt = ", ".join(f"{l} {w:.2f}" for l, w in zip(lenses, lw))
        return dict(top=[int(b) + 1 for b in top], text="\n".join(lines), weights=lw_txt,
                    max_lift=float(p.max() / q))


# ------------------------------------------------------------------------------------------- scoring
def score(probs, W, k, nb, warm, core):
    n = len(W)
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    pr, Xs = probs[warm:n], X[warm:n]
    rng = np.random.default_rng(1)
    o = np.argsort(-(pr + rng.random(pr.shape) * 1e-12), axis=1)
    ar = np.arange(len(pr))[:, None]
    hk = Xs[ar, o[:, :k]].sum(1)
    hc = Xs[ar, o[:, :core]].sum(1)
    h2 = Xs[ar, o[:, :2]].sum(1)
    pc = np.clip(pr, 1e-9, 1 - 1e-9)
    bce = -(Xs * np.log(pc) + (1 - Xs) * np.log1p(-pc)).sum(1).mean()
    q = k / nb
    ub = -(k * math.log(q) + (nb - k) * math.log(1 - q))
    return dict(topk=hk.mean(), core=(hc >= 2).mean(), core_hits=int((hc >= 2).sum()), pair=(h2 == 2).mean(),
                pair_wins=int((h2 == 2).sum()), skill=1 - bce / ub, n=len(pr))


def fair_draws(n, nb, k, with_machine, rng):
    W = rng.random((n, nb)).argsort(1)[:, :k]
    if not with_machine:
        return W, np.full_like(W, -1)
    M = np.empty_like(W)
    for t in range(n):
        rest = np.setdiff1d(np.arange(nb), W[t])
        M[t] = rng.choice(rest, k, replace=False)
    return W, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Main", choices=list(NLA) + ["Afriluck"])
    ap.add_argument("--nla-dir", default="nla_output")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--controls", type=int, default=0)
    ap.add_argument("--plant", type=float, default=0.0,
                    help="proof run: fair draws plus a hidden law 'turning of machine #1 comes next draw'")
    ap.add_argument("--plant-kind", default="law", choices=["law", "pair"],
                    help="law: 'turning of machine #1 comes next'; pair: 300 hidden 'after pair a-b comes c' precedents")
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--brain", action="store_true", help="STAGE 5: add the machine-learning brain as a lens")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = load(a)
    nb, k, W, M, dates = data["nb"], data["k"], data["W"], data["M"], data["dates"]
    core = 10 if nb < 60 else 12
    has_m = bool((M[:, 0] >= 0).any())
    rng = np.random.default_rng(11)
    label = data["name"]
    if a.plant:
        W, M = fair_draws(len(W), nb, k, True, rng)
        names, tab = transforms(nb)
        ti = names.index("turning")
        if a.plant_kind == "pair":
            pr = {}
            while len(pr) < 300:
                x, y = sorted(rng.choice(nb, 2, replace=False))
                pr[(x, y)] = int(rng.integers(nb))
            for t in range(1, len(W)):
                for x, y in combinations(sorted(W[t - 1]), 2):
                    c = pr.get((x, y))
                    if c is not None and c not in W[t] and rng.random() < a.plant:
                        j = rng.integers(k)
                        removed = W[t, j]
                        W[t, j] = c
                        M[t] = np.where(M[t] == c, removed, M[t])
            label += f" (PROOF RUN: fair draws + 300 hidden pair precedents 'after a-b comes c', strength {a.plant})"
        for t in range(1 if a.plant_kind == "law" else len(W), len(W)):
            b = tab[ti, M[t - 1, 0]]
            if b >= 0 and b not in W[t] and rng.random() < a.plant:
                j = rng.integers(k)
                removed = W[t, j]
                W[t, j] = b
                M[t] = np.where(M[t] == b, removed, M[t])          # keep winning and machine numbers apart
        if a.plant_kind == "law":
            label += f" (PROOF RUN: fair draws + hidden law 'turning of machine #1 comes next', strength {a.plant})"
    eng = Forecaster(data, warm=a.warm, use_brain=a.brain)
    t0 = time.time()
    probs, explain, lens_w, lens_gain = eng.run(W, M, dates, narrate=True)
    real = score(probs, W, k, nb, a.warm, core)
    from math import comb
    ch_core = sum(comb(core, j) * comb(nb - core, k - j) for j in range(2, k + 1)) / comb(nb, k)
    out = [f"# {'STAGE 5 MAIN BEAST' if a.brain else 'Forecaster'}: {label}", "",
           f"{len(W)} draws ({dates[0].date()} to {dates[-1].date()}); {real['n']} forecast blind (each from earlier "
           f"draws only); machine numbers {'used' if has_m else 'not available'}; one pass {time.time() - t0:.0f}s.", "",
           "| Measure | Forecaster | Chance |", "|---|---|---|",
           f"| hits in its top {k} | {real['topk']:.3f} | {k * k / nb:.3f} |",
           f"| core-{core} hit (2+ drawn) | {real['core_hits']} of {real['n']} = {real['core']:.1%} | {ch_core:.1%} |",
           f"| both of its top 2 drawn | {real['pair_wins']} = {real['pair']:.2%} | {k * (k - 1) / (nb * (nb - 1)):.2%} |",
           f"| log-loss skill vs chance | {real['skill']:+.5f} | 0 |", "",
           "Average lens weights over the scored draws (how much each way of thinking was trusted): "
           + ", ".join(f"{l} {w:.2f}" for l, w in lens_w.items()),
           "Each lens's total log-score gain over chance (above 0 = it beat chance on the scored draws): "
           + ", ".join(f"{l} {g:+.1f}" for l, g in lens_gain.items() if l != "chance")]
    if a.controls:
        ctrl = []
        for i in range(a.controls):
            Wf, Mf = fair_draws(len(W), nb, k, has_m, rng)
            pf, _, _, _ = eng.run(Wf, Mf, dates)
            ctrl.append(score(pf, Wf, k, nb, a.warm, core))
            print(f"\r  fair-machine control {i + 1}/{a.controls}", end="", flush=True)
        print()
        out += ["", f"Against {a.controls} fair-machine histories run through the same engine:", "",
                "| Measure | Forecaster | Fair controls (mean) | p (controls >= forecaster) |", "|---|---|---|---|"]
        for key, nm in (("topk", f"hits in top {k}"), ("core", f"core-{core} hit"), ("skill", "log-loss skill")):
            arr = np.array([c[key] for c in ctrl])
            out.append(f"| {nm} | {real[key]:.4f} | {arr.mean():.4f} | {(1 + (arr >= real[key]).sum()) / (len(arr) + 1):.3f} |")
    nd = dates[-1] + pd.Timedelta(days=1)
    out += ["", f"## Reasoning for the next draw (after {dates[-1].date()})", "",
            f"Top {k}: {explain['top'][:k]} | core {core}: {explain['top'][:core]} | strongest pick is "
            f"x{explain['max_lift']:.3f} chance", "", "Lens weights now: " + explain["weights"], "", "```",
            explain["text"], "```"]
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
