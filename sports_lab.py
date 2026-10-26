#!/usr/bin/env python3
"""
sports_lab.py: can a model beat the football bookmakers? The same rules of evidence as the lottery audit.

    git clone --depth 1 https://github.com/xgabora/Club-Football-Match-Data-2000-2025.git cfm
    python sports_lab.py --matches cfm/data/Matches.csv                 # ~5 minutes
    python sports_lab.py --matches cfm/data/Matches.csv --stake 10      # GH₵ per bet (default 10)

Data: ~238,000 club matches (27 countries, 2000 - Sep 2026) with Bet365's pre-match odds (taken a day or two before
kick-off), the best price across ~17 European bookmakers, over/under 2.5 goals and Asian handicap odds.

RULES
  * Every forecast for a match uses only matches played before that day (ratings updated after each match; the
    models are refitted once a year on all earlier seasons).
  * Every setting (thresholds, which model) is chosen on DEV = matches before 1 July 2016. The TEST period
    (1 July 2016 onwards, ~10 seasons) is scored once, untouched.
  * Money: flat stake per bet, back per GH₵1 with 95% interval (bootstrap by betting week), net GH₵, deepest
    fall, longest losing run, p-value for "beats break-even", Benjamini-Hochberg across every strategy tested.
  * Positive control: a planted edge (outcomes drawn from a world where the rating model knows something the
    market misses) must be found by the same pipeline.

STRATEGIES (each picks at most one bet per match)
  S1 everything at Bet365        S2 popular systems: favourites, home favourites, draws, longshots
  S3 accumulators (3- and 5-fold of favourites, the common Ghana bet slip)
  S4 rating model from results only (own Elo + logistic) - the typical tipster model - value bets at Bet365
  S5 rating model + the market's own prices (does the model add anything to the market?) - value bets at Bet365
  S6 over/under 2.5 goals with a team-strength goals model (Poisson), value bets at Bet365
  S7 "value betting": bet at the BEST price of ~17 bookmakers when it beats the fair price implied by Bet365
  S8 the same, short prices only, with the threshold and price cap chosen on DEV (pre-registered refinement)
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2016-07-01")


# ============================================================================================ data
def load(path):
    m = pd.read_csv(path, low_memory=False)
    m["date"] = pd.to_datetime(m["MatchDate"])
    m = m[m["FTResult"].isin(["H", "D", "A"])].copy()
    m["country"] = m["Division"].str.replace(r"\d+$", "", regex=True).replace({"EC": "E"})
    m["hk"] = m["country"] + ":" + m["HomeTeam"].astype(str)
    m["ak"] = m["country"] + ":" + m["AwayTeam"].astype(str)
    m = m.sort_values(["date", "Division", "HomeTeam"], kind="stable").reset_index(drop=True)
    m["y"] = m["FTResult"].map({"H": 0, "D": 1, "A": 2}).astype(int)
    m["season"] = np.where(m["date"].dt.month >= 7, m["date"].dt.year, m["date"].dt.year - 1)
    m["goals"] = m["FTHome"] + m["FTAway"]
    odds = m[["OddHome", "OddDraw", "OddAway"]].to_numpy(float)
    ok = np.isfinite(odds).all(1) & (odds > 1.0).all(1)
    inv = np.where(ok[:, None], 1 / np.where(ok[:, None], odds, 2.0), np.nan)
    m["margin"] = inv.sum(1) - 1
    fair = inv / inv.sum(1, keepdims=True)
    for j, c in enumerate("HDA"):
        m[f"pm{c}"] = fair[:, j]                                       # market (Bet365) fair probability
    m["ok1x2"] = ok
    return m


# ============================================================================================ ratings
def elo_walk(m, K=20.0, H=60.0):
    """Own Elo, updated after each DAY (so a match never sees a same-day result). Returns pre-match diff."""
    R = {}
    hadv = {}
    diff = np.empty(len(m))
    dates = m["date"].to_numpy()
    hk, ak = m["hk"].to_numpy(), m["ak"].to_numpy()
    gh, ga = m["FTHome"].to_numpy(float), m["FTAway"].to_numpy(float)
    cty = m["country"].to_numpy()
    i, n = 0, len(m)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        upd = []
        for t in range(i, j):
            rh, ra = R.get(hk[t], 1500.0), R.get(ak[t], 1500.0)
            h = hadv.get(cty[t], H)
            diff[t] = rh - ra + h
            e = 1 / (1 + 10 ** (-(rh - ra + h) / 400))
            if np.isfinite(gh[t]) and np.isfinite(ga[t]):
                s = 1.0 if gh[t] > ga[t] else 0.5 if gh[t] == ga[t] else 0.0
                k = K * (1 + math.log1p(abs(gh[t] - ga[t])))
                upd.append((hk[t], ak[t], k * (s - e), cty[t], s - e))
        for a, b, dl, c, err in upd:                                   # apply the day's results after the day
            R[a] = R.get(a, 1500.0) + dl
            R[b] = R.get(b, 1500.0) - dl
            hadv[c] = hadv.get(c, H) + 0.075 * err * K                 # league home advantage learns slowly
        i = j
    return diff


def goals_walk(m, half_life_matches=20.0):
    """Exponentially weighted attack/defence strengths per team (home and away separately pooled), pre-match.
    Returns expected goals (lam_home, lam_away) from strengths x league averages."""
    a = 0.5 ** (1 / half_life_matches)
    att, dfn = {}, {}
    lg_h, lg_a = {}, {}
    lam = np.full((len(m), 2), np.nan)
    dates = m["date"].to_numpy()
    hk, ak = m["hk"].to_numpy(), m["ak"].to_numpy()
    div = m["Division"].to_numpy()
    gh, ga = m["FTHome"].to_numpy(float), m["FTAway"].to_numpy(float)
    i, n = 0, len(m)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        upd = []
        for t in range(i, j):
            mh, ma = lg_h.get(div[t], 1.45), lg_a.get(div[t], 1.1)
            lam[t] = (mh * att.get(hk[t], 1.0) * dfn.get(ak[t], 1.0), ma * att.get(ak[t], 1.0) * dfn.get(hk[t], 1.0))
            if np.isfinite(gh[t]) and np.isfinite(ga[t]):
                upd.append(t)
        for t in upd:
            mh, ma = lg_h.get(div[t], 1.45), lg_a.get(div[t], 1.1)
            att[hk[t]] = a * att.get(hk[t], 1.0) + (1 - a) * (gh[t] / mh)
            dfn[ak[t]] = a * dfn.get(ak[t], 1.0) + (1 - a) * (gh[t] / mh)
            att[ak[t]] = a * att.get(ak[t], 1.0) + (1 - a) * (ga[t] / ma)
            dfn[hk[t]] = a * dfn.get(hk[t], 1.0) + (1 - a) * (ga[t] / ma)
            lg_h[div[t]] = 0.995 * mh + 0.005 * gh[t]
            lg_a[div[t]] = 0.995 * ma + 0.005 * ga[t]
        i = j
    return lam


def pois_over(lh, la, line=2.5):
    lt = lh + la
    k = np.arange(0, int(line) + 1)
    cdf = (np.exp(-lt)[:, None] * lt[:, None] ** k / np.array([math.factorial(x) for x in k])).sum(1)
    return 1 - cdf


# ============================================================================================ models
def yearly_fit(m, X, y, rows_ok, kind="multi"):
    """Walk-forward: for each season s, fit on all rows of seasons < s, predict season s."""
    from sklearn.linear_model import LogisticRegression
    out = np.full((len(m), 3 if kind == "multi" else 2), np.nan)
    seasons = sorted(m["season"].unique())
    for s in seasons:
        tr = rows_ok & (m["season"].to_numpy() < s) & (m["season"].to_numpy() >= s - 8)   # last 8 seasons
        te = rows_ok & (m["season"].to_numpy() == s)
        if tr.sum() < 3000 or te.sum() == 0:
            continue
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])
    return out


def build(m):
    t0 = time.time()
    m["elo"] = elo_walk(m)
    lam = goals_walk(m)
    m["lamH"], m["lamA"] = lam[:, 0], lam[:, 1]
    print(f"  ratings built in {time.time() - t0:.0f}s", flush=True)
    lg = pd.get_dummies(m["Division"]).to_numpy(float)
    d = (m["elo"].to_numpy() / 400)[:, None]
    ok = m["ok1x2"].to_numpy() & np.isfinite(d[:, 0])
    y = m["y"].to_numpy()
    # S4: results only (Elo diff, its square, league dummies x Elo-free draw/home levels)
    XA = np.hstack([d, d ** 2, lg])
    pA = yearly_fit(m, XA, y, ok)
    # S5: market + model (log market probabilities + Elo diff + goals model)
    lm = np.log(np.clip(m[["pmH", "pmD", "pmA"]].to_numpy(float), 1e-4, 1))
    XB = np.hstack([np.nan_to_num(lm), d, np.log(np.nan_to_num(lam, nan=1.2) + 0.05), lg])
    pB = yearly_fit(m, XB, y, ok)
    for j, c in enumerate("HDA"):
        m[f"pA{c}"] = pA[:, j]
        m[f"pB{c}"] = pB[:, j]
    # S6: over 2.5 goals, Poisson + calibration (+ market O/U fair prob in the B version)
    ou = m[["Over25", "Under25"]].to_numpy(float)
    okou = np.isfinite(ou).all(1) & (ou > 1).all(1) & np.isfinite(m["goals"].to_numpy(float)) & np.isfinite(lam).all(1)
    po = pois_over(np.nan_to_num(lam[:, 0], nan=1.4), np.nan_to_num(lam[:, 1], nan=1.1))
    inv = 1 / np.where(okou[:, None], ou, 2.0)
    m["pmO"] = np.where(okou, inv[:, 0] / inv.sum(1), np.nan)
    m["ou_margin"] = np.where(okou, inv.sum(1) - 1, np.nan)
    yo = (m["goals"].to_numpy(float) > 2.5).astype(int)
    lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    XO = np.hstack([lo(po)[:, None], lg])
    m["pO_model"] = yearly_fit(m, XO, yo, okou, kind="bin")[:, 1]
    XO2 = np.hstack([lo(po)[:, None], lo(np.nan_to_num(m["pmO"].to_numpy(), nan=0.5))[:, None], lg])
    m["pO_mix"] = yearly_fit(m, XO2, yo, okou, kind="bin")[:, 1]
    m["okou"] = okou
    m["over"] = yo
    print(f"  models fitted in {time.time() - t0:.0f}s", flush=True)
    return m


# ============================================================================================ bets & money
def week_boot(ret, stake, weeks, B=2000, seed=0):
    """95% interval of back per GH₵1, resampling whole betting weeks (bets in one week are not independent)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"r": ret, "s": stake, "w": weeks}).groupby("w").sum()
    r, s = df["r"].to_numpy(), df["s"].to_numpy()
    idx = rng.integers(0, len(r), size=(B, len(r)))
    bs = r[idx].sum(1) / s[idx].sum(1)
    return np.percentile(bs, [2.5, 97.5]), bs.std()


def summarise(name, odds, win, stake, dates, extra=""):
    from scipy.stats import norm
    odds, win = np.asarray(odds, float), np.asarray(win, bool)
    n = len(odds)
    if n == 0:
        return dict(strategy=name, bets=0)
    st = np.full(n, stake, float)
    ret = np.where(win, odds * stake, 0.0)
    back = ret.sum() / st.sum()
    weeks = pd.DatetimeIndex(dates).to_period("W").astype(str)
    (lo, hi), se = week_boot(ret, st, weeks)
    p = 1 - norm.cdf((back - 1) / max(se, 1e-12))
    net = ret - st
    cum = np.cumsum(net)
    dd = (cum - np.maximum.accumulate(np.r_[0, cum])[1:]).min()
    lose = cur = 0
    for v in net:
        cur = cur + 1 if v < 0 else 0
        lose = max(lose, cur)
    return dict(strategy=name, bets=n, win_rate=win.mean(), avg_odds=odds.mean(), back=back, lo=lo, hi=hi,
                net=net.sum(), staked=st.sum(), dd=dd, lose=lose, p=p, note=extra)


def pick_value(P, O, tau, max_odds=None):
    """one bet per match: the outcome with the largest expected value p*odds-1, if above tau."""
    ev = P * O - 1
    ev = np.where(np.isfinite(ev), ev, -9)
    if max_odds is not None:
        ev = np.where(O <= max_odds, ev, -9)
    j = ev.argmax(1)
    take = ev[np.arange(len(ev)), j] > tau
    return take, j


def strategies(m, part, stake, tau_choice=None, verbose=False):
    """All strategies on a subset (DEV or TEST). tau_choice: dict of thresholds chosen on DEV (None = scan)."""
    rows = []
    s = m[part & m["ok1x2"]].copy()
    O = s[["OddHome", "OddDraw", "OddAway"]].to_numpy(float)
    MX = s[["MaxHome", "MaxDraw", "MaxAway"]].to_numpy(float)
    y = s["y"].to_numpy()
    dt = s["date"].to_numpy()
    ar = np.arange(len(s))

    # S1 everything
    for j, c in enumerate(["home", "draw", "away"]):
        rows.append(summarise(f"S1 every {c} at Bet365", O[:, j], y == j, stake, dt))
    # S2 systems
    fav = O.argmin(1)
    fo = O[ar, fav]
    rows.append(summarise("S2 every favourite", fo, y == fav, stake, dt))
    for lo_, hi_ in ((1.0, 1.3), (1.3, 1.6), (1.6, 2.2), (2.2, 3.5), (3.5, 7.0), (7.0, 1000)):
        sel = np.zeros_like(O, bool)
        sel[(O >= lo_) & (O < hi_)] = True
        rr, cc = np.where(sel)
        rows.append(summarise(f"S2 every price {lo_:g}-{hi_:g}" if hi_ < 1000 else f"S2 every longshot >{lo_:g}",
                              O[rr, cc], y[rr] == cc, stake, dt[rr]))
    hf = (fav == 0) & (fo < 1.5)
    rows.append(summarise("S2 home favourites under 1.50", O[hf, 0], y[hf] == 0, stake, dt[hf]))
    close = np.abs(s["pmH"].to_numpy() - s["pmA"].to_numpy()) < 0.08
    rows.append(summarise("S2 draws in close matches", O[close, 1], y[close] == 1, stake, dt[close]))

    # S3 accumulators of favourites under 1.6, same weekend, random combos
    rng = np.random.default_rng(1)
    for legs in (3, 5):
        cand = np.where(fo < 1.6)[0]
        wk = pd.DatetimeIndex(dt[cand]).to_period("W").astype(str).to_numpy()
        acc_o, acc_w, acc_d = [], [], []
        for w in np.unique(wk):
            c = cand[wk == w]
            rng.shuffle(c)
            for k in range(0, len(c) - legs + 1, legs):
                g = c[k:k + legs]
                acc_o.append(np.prod(fo[g]))
                acc_w.append(bool((y[g] == fav[g]).all()))
                acc_d.append(dt[g[0]])
        rows.append(summarise(f"S3 {legs}-fold accumulators of favourites <1.60", acc_o, acc_w, stake, acc_d))

    # S4/S5 model value bets, S6 O/U, S7 consensus value
    taus = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]
    specs = []
    PA = s[["pAH", "pAD", "pAA"]].to_numpy(float)
    PB = s[["pBH", "pBD", "pBA"]].to_numpy(float)
    PM = s[["pmH", "pmD", "pmA"]].to_numpy(float)
    specs.append(("S4 results-only model, value at Bet365", PA, O))
    specs.append(("S4b results-only model, value at BEST price", PA, MX))
    specs.append(("S5 model + market, value at Bet365", PB, O))
    specs.append(("S5b model + market, value at BEST price", PB, MX))
    specs.append(("S7 BEST price beats Bet365 fair price", PM, MX))
    chosen = {}
    for nm, P, OO in specs:
        okp = np.isfinite(P).all(1) & np.isfinite(OO).all(1)
        scan = taus if tau_choice is None else [tau_choice.get(nm, 0.05)]
        best = None
        for tau in scan:
            take, j = pick_value(np.where(okp[:, None], P, np.nan), OO, tau, max_odds=10.0)
            take &= okp
            r = summarise(nm + f" (EV > {tau:.0%})", OO[ar[take], j[take]], y[take] == j[take], stake, dt[take])
            if tau_choice is None:
                rows.append(r)
                if r["bets"] >= 300 and (best is None or r["back"] > best[1]):
                    best = (tau, r["back"])
            else:
                rows.append(r)
        if tau_choice is None and best:
            chosen[nm] = best[0]
    # S6 over/under
    so = m[part & m["okou"]]
    OU = so[["Over25", "Under25"]].to_numpy(float)
    MOU = so[["MaxOver25", "MaxUnder25"]].to_numpy(float)
    yo = so["over"].to_numpy()
    dto = so["date"].to_numpy()
    for nm, col, OO in (("S6 goals model, over/under at Bet365", "pO_model", OU),
                        ("S6b goals model + market, over/under at Bet365", "pO_mix", OU),
                        ("S6c goals model + market, over/under at BEST price", "pO_mix", MOU),
                        ("S7b BEST over/under price beats Bet365 fair price", "pmO", MOU)):
        p = so[col].to_numpy(float)
        P2 = np.stack([p, 1 - p], 1)
        okp = np.isfinite(P2).all(1) & np.isfinite(OO).all(1)
        scan = taus if tau_choice is None else [tau_choice.get(nm, 0.05)]
        best = None
        for tau in scan:
            take, j = pick_value(np.where(okp[:, None], P2, np.nan), OO, tau)
            take &= okp
            win = np.where(j == 0, yo == 1, yo == 0)
            r = summarise(nm + f" (EV > {tau:.0%})", OO[np.arange(len(so))[take], j[take]], win[take], stake, dto[take])
            rows.append(r)
            if tau_choice is None and r["bets"] >= 300 and (best is None or r["back"] > best[1]):
                best = (tau, r["back"])
        if tau_choice is None and best:
            chosen[nm] = best[0]
    # S8 refined value betting: best price vs Bet365 fair, short prices only. (threshold, price cap) chosen on DEV
    nm = "S8 BEST price beats Bet365 fair, short prices only"
    okp = np.isfinite(PM).all(1) & np.isfinite(MX).all(1)
    grid = [(t, c) for t in (0.02, 0.03, 0.05, 0.07, 0.10) for c in (2.5, 3.5, 5.0, 10.0)]
    scan = grid if tau_choice is None else [tau_choice.get(nm, (0.05, 2.5))]
    best = None
    for tau, cap in scan:
        take, j = pick_value(np.where(okp[:, None], PM, np.nan), MX, tau, max_odds=cap)
        take &= okp
        r = summarise(nm + f" (EV > {tau:.0%}, price <= {cap:g})", MX[ar[take], j[take]], y[take] == j[take], stake,
                      dt[take])
        if tau_choice is None:
            if r["bets"] >= 1000 and (best is None or r["back"] > best[1]):
                best = ((tau, cap), r["back"], r)
        else:
            rows.append(r)
    if tau_choice is None and best:
        chosen[nm] = best[0]
        rows.append(best[2])
    return pd.DataFrame(rows), chosen


def bh(p):
    p = np.asarray(p, float)
    mm = len(p)
    o = np.argsort(p)
    q = np.empty(mm)
    q[o] = np.minimum.accumulate((p[o] * mm / np.arange(1, mm + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def table(df, title):
    out = [f"### {title}", "", "| strategy | bets | win rate | avg odds | back per GH₵1 (95% CI) | net GH₵ | deepest fall | "
           "longest losing run | p (beats break-even) | BH q |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        if r["bets"] == 0:
            continue
        out.append(f"| {r.strategy} | {r.bets:,} | {r.win_rate:.1%} | {r.avg_odds:.2f} | {r.back:.3f} ({r.lo:.3f}-"
                   f"{r.hi:.3f}) | {r.net:+,.0f} | {r.dd:,.0f} | {r.lose} | {r.p:.3f} | {r.q:.3f} |")
    return "\n".join(out)


# ============================================================================================ positive control
def planted(m, strength=0.5, seed=7):
    """A world where the rating model really knows something: true probability = market blended with the
    results-only model (weight `strength`); outcomes are redrawn from it, odds stay the real Bet365 odds.
    The same pipeline (refits included) must then find profitable S5 bets on the TEST period."""
    rng = np.random.default_rng(seed)
    w = m.copy()
    PM = w[["pmH", "pmD", "pmA"]].to_numpy(float)
    PA = w[["pAH", "pAD", "pAA"]].to_numpy(float)
    ok = np.isfinite(PM).all(1) & np.isfinite(PA).all(1)
    PT = np.where(ok[:, None], (1 - strength) * PM + strength * PA, PM)
    u = rng.random(len(w))
    ynew = (u[:, None] > np.cumsum(np.nan_to_num(PT, nan=1 / 3), 1)).sum(1)
    w["y"] = np.where(ok, np.minimum(ynew, 2), w["y"])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="cfm/data/Matches.csv")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--csv", default="sports_lab_results.csv")
    a = ap.parse_args()
    t0 = time.time()
    m = load(a.matches)
    print(f"# Sports lab: {len(m):,} matches, {m['date'].min().date()} to {m['date'].max().date()}, "
          f"{m['Division'].nunique()} divisions\n", flush=True)
    m = build(m)
    dev = (m["date"] < TEST_START) & (m["season"] >= 2005)
    test = m["date"] >= TEST_START
    print(f"\nDEV (settings chosen here): {dev.sum():,} matches 2005-07 to 2016-06 | TEST (scored once): "
          f"{test.sum():,} matches 2016-07 to {m['date'].max().date()}")
    print(f"Bet365 median margin on 1X2 in TEST: {m.loc[test, 'margin'].median():.1%}; over/under 2.5: "
          f"{m.loc[test, 'ou_margin'].median():.1%}\n")

    # calibration check of the models on TEST (log-loss vs the market)
    tt = m[test & m["ok1x2"]].dropna(subset=["pAH", "pBH"])
    Y = np.eye(3)[tt["y"].to_numpy()]
    ll = lambda P: -(Y * np.log(np.clip(P, 1e-9, 1))).sum(1).mean()
    print("## How good are the forecasts? (TEST, log-loss: lower = better)\n")
    print(f"| forecaster | log-loss |\n|---|---|\n| no information (base rates) | "
          f"{ll(np.tile(Y.mean(0), (len(Y), 1))):.4f} |\n| results-only rating model (S4) | "
          f"{ll(tt[['pAH', 'pAD', 'pAA']].to_numpy()):.4f} |\n| Bet365's prices (margin removed) | "
          f"{ll(tt[['pmH', 'pmD', 'pmA']].to_numpy()):.4f} |\n| model + market (S5) | "
          f"{ll(tt[['pBH', 'pBD', 'pBA']].to_numpy()):.4f} |\n")

    dv, chosen = strategies(m, dev, a.stake)
    dv["q"] = np.nan
    print("## Settings chosen on DEV (value threshold with the best DEV return, 300+ bets)\n")
    for k, v in chosen.items():
        print(f"- {k}: " + (f"EV > {v[0]:.0%}, price <= {v[1]:g}" if isinstance(v, tuple) else f"EV > {v:.0%}"))
    ts, _ = strategies(m, test, a.stake, tau_choice=chosen)
    ts["q"] = bh(ts["p"].fillna(1).to_numpy())
    print()
    print(table(ts, f"TEST 2016-07 to {m['date'].max().date()}: every strategy, flat GH₵{a.stake:g} a bet "
                    f"(BH q across all {len(ts)} rows)"))
    ts.to_csv(a.csv, index=False)

    # realism: S7 with odds 2% worse (stale lines, a Ghana bookmaker's wider margin), and S5 at a margin of ~9%
    print("\n## Sensitivity: the same TEST bets if your bookmaker pays 3% less than these prices (a wider margin, "
          "as on many African betting sites)\n")
    worse = ts[ts["bets"] > 0].copy()
    worse["back_3pct_worse"] = worse["back"] * 0.97
    print("| strategy | back per GH₵1 at these prices | at 3% worse prices |\n|---|---|---|")
    for _, r in worse.iterrows():
        print(f"| {r.strategy} | {r.back:.3f} | {r.back_3pct_worse:.3f} |")

    # positive control
    print("\n## Positive control: a planted edge must be found\n")
    for st_ in (0.1, 0.2, 0.3):
        w = planted(m, st_)
        Xd = np.hstack([np.nan_to_num(np.log(np.clip(w[["pmH", "pmD", "pmA"]].to_numpy(float), 1e-4, 1))),
                        (w["elo"].to_numpy() / 400)[:, None],
                        np.log(np.nan_to_num(w[["lamH", "lamA"]].to_numpy(float), nan=1.2) + 0.05),
                        pd.get_dummies(w["Division"]).to_numpy(float)])
        pB = yearly_fit(w, Xd, w["y"].to_numpy(), w["ok1x2"].to_numpy() & np.isfinite(w["elo"].to_numpy()))
        for j, c in enumerate("HDA"):
            w[f"pB{c}"] = pB[:, j]
        _, ch = strategies(w, dev, a.stake)
        tw, _ = strategies(w, test, a.stake, tau_choice=ch)
        r = tw[tw["strategy"].str.startswith("S5 model + market, value at Bet365")].iloc[0]
        print(f"- planted world, true probability = {1 - st_:.0%} market + {st_:.0%} rating model: S5 at Bet365 "
              f"returns {r.back:.3f} per GH₵1 ({r.lo:.3f}-{r.hi:.3f}) on {r.bets:,} bets, p = {r.p:.4f} -> "
              + ("FOUND" if r.back > 1 and r.p < 0.01 else "missed"))
    # leakage check: scramble every result from 2020 on; nothing forecast for an earlier date may change
    print("\n## Leakage check: results from 2020 on replaced by random results\n")
    raw = load(a.matches)
    late = raw["date"] >= pd.Timestamp("2020-01-01")
    rng = np.random.default_rng(3)
    raw.loc[late, "FTHome"] = rng.integers(0, 5, late.sum())
    raw.loc[late, "FTAway"] = rng.integers(0, 5, late.sum())
    raw.loc[late, "y"] = np.where(raw.loc[late, "FTHome"] > raw.loc[late, "FTAway"], 0,
                                  np.where(raw.loc[late, "FTHome"] == raw.loc[late, "FTAway"], 1, 2))
    raw["goals"] = raw["FTHome"] + raw["FTAway"]
    m2 = build(raw)
    early = (m["date"] < pd.Timestamp("2020-01-01")).to_numpy()
    worst = 0.0
    for c in ("elo", "lamH", "lamA", "pAH", "pBH", "pO_model", "pO_mix"):
        x1, x2 = m[c].to_numpy(float)[early], m2[c].to_numpy(float)[early]
        f = np.isfinite(x1) & np.isfinite(x2)
        worst = max(worst, float(np.abs(x1[f] - x2[f]).max()))
    print(f"largest change in any forecast made before 2020: {worst:.2e} -> " + ("no leakage" if worst < 1e-9 else "LEAK"))

    # forward test plan for the S8 rule
    r8 = ts[ts["strategy"].str.startswith("S8")].iloc[0]
    sd = r8.avg_odds * math.sqrt(r8.win_rate * (1 - r8.win_rate))
    yrs = (m["date"].max() - TEST_START).days / 365.25
    print("\n## Forward-test plan for S8 (paper bets logged before kick-off)\n")
    for h1 in (1.05, 1.08, 1.12):
        n = math.ceil(((2.326 + 1.282) * sd / (h1 - 1)) ** 2)
        print(f"- to confirm a true return of {h1:.2f} (1% false alarm, 90% power): about {n:,} bets "
              f"(the historical rule found about {r8.bets / yrs:.0f} a year in these 38 leagues)")
    print("- faster check: 'closing-line value' - the price you got vs the fair price at kick-off; positive on "
          "average over 150-200 bets is the standard early sign that the edge is real and you can actually get it")
    print(f"\nDone in {time.time() - t0:.0f}s. Full TEST table: {a.csv}")


if __name__ == "__main__":
    main()
