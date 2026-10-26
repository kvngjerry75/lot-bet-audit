#!/usr/bin/env python3
"""
bet_lab.py: what every Afriluck bet would really have returned, in GH₵, with honest uncertainty.

    python bet_lab.py --afriluck-dir afriluck_output                          # uses sharp + colour (fast, ~1 min)
    python bet_lab.py --afriluck-dir afriluck_output --forecasts afl.npz      # also every main_beast component
    python bet_lab.py --afriluck-dir afriluck_output --controls 100           # + fair-machine controls

For every forecaster and every bet type it replays the history walk-forward (each draw is bet from forecasts made
before it, from earlier draws only) and reports:
  back per GH₵1 staked, with a 95% interval (moving-block bootstrap, blocks of 21 draws ~ one week),
  total net in GH₵ (GH₵1 per line), deepest drawdown, longest run of losing draws,
  how often 1,000 random-pick players on the SAME draws did at least as well (luck check),
  the p-value for "this bet beats break-even" (one-sided, block bootstrap), and the Benjamini-Hochberg verdict
  across every row in the table (so trying many bets cannot manufacture a winner).
It also prints the CEILING: what each bet returns if the pink/yellow bias were known exactly (Plackett-Luce fit with
its 95% interval). No forecaster can beat that ceiling with colour information alone.

Payouts (from the operator's public winners feed, not an official table; change PAY if they differ):
  Perm-2 60x per winning pair | Direct-1 30x first ball only | Banker = 1 ball with all 56 others, 60x per pair
  Direct-3 600x | Direct-4 10,000x | machine-number Perm-2 20x per pair (pairs among the 6 machine numbers)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

NB, K, PINK = 57, 6, 28
PAY = dict(perm2=60, direct1=30, direct3=600, direct4=10_000, mperm2=20)
WARM = 300


# ------------------------------------------------------------------------------------------- data
def load(afl_dir):
    import forecaster as fc
    d = fc.load(argparse.Namespace(game="Afriluck", afriluck_dir=afl_dir, nla_dir=""))
    return d


def onehot(W, nb=NB):
    X = np.zeros((len(W), nb))
    X[np.repeat(np.arange(len(W)), W.shape[1]), W.ravel()] = 1
    return X


def machine_ok(W, M):
    return (M >= 0).all(1) & (np.sort(M, 1) != np.sort(W, 1)).any(1)


# ------------------------------------------------------------------------------------------- bets
def bet_lines(name):
    """(lines staked per draw, description)"""
    if name.startswith("perm2_"):
        c = int(name.split("_")[1])
        return math.comb(c, 2), f"Perm-2 on top {c} ({math.comb(c, 2)} pairs)"
    if name.startswith("mperm2_"):
        c = int(name.split("_")[1])
        return math.comb(c, 2), f"Machine-number Perm-2 on top {c} ({math.comb(c, 2)} pairs)"
    return {"direct1": (1, "Direct-1 (first ball) on top ball"), "banker": (56, "Banker on top ball (56 lines)"),
            "direct3": (1, "Direct-3 on top 3"), "direct4": (1, "Direct-4 on top 4")}[name]


def returns(name, order, W, M=None, morder=None):
    """GH₵ returned per draw (GH₵1 per line). order: (T, nb) ball ranking; W: (T, k) drawn, in drawn order."""
    T = len(W)
    Xs = onehot(W)
    ar = np.arange(T)[:, None]
    if name.startswith("perm2_"):
        c = int(name.split("_")[1])
        h = Xs[ar, order[:, :c]].sum(1)
        return PAY["perm2"] * h * (h - 1) / 2
    if name == "direct1":
        return PAY["direct1"] * (order[:, 0] == W[:, 0])
    if name == "banker":
        return PAY["perm2"] * (K - 1) * Xs[np.arange(T), order[:, 0]]
    if name == "direct3":
        return PAY["direct3"] * (Xs[ar, order[:, :3]].sum(1) == 3)
    if name == "direct4":
        return PAY["direct4"] * (Xs[ar, order[:, :4]].sum(1) == 4)
    if name.startswith("mperm2_"):
        c = int(name.split("_")[1])
        h = onehot(M)[ar, morder[:, :c]].sum(1)
        return PAY["mperm2"] * h * (h - 1) / 2
    raise ValueError(name)


def block_boot(x, B=2000, block=21, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(x)
    nblk = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 1), size=(B, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return x[idx].mean(1)


def money(ret, stake, dates=None, rng=None, pay=None):
    """pay given -> an all-or-nothing bet (Direct-1/3/4): exact binomial interval and test, because a bootstrap
    cannot handle a bet that wins only a handful of times (one GH₵10,000 Direct-4 hit is luck, not an edge)."""
    from scipy.stats import beta, binom, norm
    net = ret - stake
    back = ret.mean() / stake
    if pay:
        x, m = int((ret > 0).sum()), len(ret)
        lo = (beta.ppf(0.025, x, m - x + 1) if x > 0 else 0.0) * pay / stake
        hi = beta.ppf(0.975, x + 1, m - x) * pay / stake
        p_be = binom.sf(x - 1, m, stake / pay)                       # P(at least x wins | exactly break-even)
    else:
        bs = block_boot(ret / stake, rng=rng)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        se = bs.std()
        p_be = 1 - norm.cdf((back - 1) / max(se, 1e-12))             # H0: back <= 1 (break-even)
    cum = np.cumsum(net)
    dd = (cum - np.maximum.accumulate(np.r_[0, cum])[1:]).min()
    lose = cur = 0
    for v in net:
        cur = cur + 1 if v < 0 else 0
        lose = max(lose, cur)
    return dict(back=back, lo=lo, hi=hi, net=net.sum(), staked=stake * len(ret), dd=dd, lose=lose, p_be=p_be, n=len(ret))


def luck_replays(name, W, M, R=1000, rng=None, mask=None):
    """total return of random-pick players on the same draws"""
    rng = rng or np.random.default_rng(7)
    T = len(W)
    out = np.empty(R)
    for r in range(R):
        o = rng.random((T, NB)).argsort(1)[:, :10]
        out[r] = returns(name, o, W, M, o).sum()
    return out


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ------------------------------------------------------------------------------------------- ceiling
def fit_colour_pl(W):
    """Plackett-Luce pink exit strength w (yellow = 1) and its standard error on log scale."""
    from scipy.optimize import minimize_scalar
    pink = np.arange(NB) < PINK
    npk = (W < PINK)
    # remaining pink / yellow before each exit
    def nll(lw):
        w = math.exp(lw)
        ll = 0.0
        rp = np.full(len(W), PINK, float)
        ry = np.full(len(W), NB - PINK, float)
        for j in range(W.shape[1]):
            isp = npk[:, j]
            tot = rp * w + ry
            ll += np.where(isp, lw, 0.0).sum() - np.log(tot).sum()
            rp -= isp
            ry -= ~isp
        return -ll
    r = minimize_scalar(nll, bounds=(-1, 1), method="bounded")
    h = 1e-3
    se = 1 / math.sqrt((nll(r.x + h) - 2 * nll(r.x) + nll(r.x - h)) / h ** 2)
    del pink
    return r.x, se


def ceiling(w, N=1_000_000, seed=0):
    """EV per GH₵1 of each bet on PINK balls if the pink strength w were known exactly (simulation)."""
    rng = np.random.default_rng(seed)
    wt = np.where(np.arange(NB) < PINK, w, 1.0)
    acc = dict(pp=0.0, first=0.0, inc=0.0, ppp=0.0, pppp=0.0)
    B = 100_000
    for _ in range(N // B):
        g = -np.log(rng.random((B, NB))) / wt
        o = np.argpartition(g, K, axis=1)[:, :K]
        first = g.argmin(1)
        q = (o < PINK).sum(1).astype(float)
        acc["pp"] += (q * (q - 1) / 2).sum() / math.comb(PINK, 2)
        acc["inc"] += q.sum() / PINK
        acc["first"] += (first < PINK).sum() / PINK
        acc["ppp"] += (q * (q - 1) * (q - 2) / 6).sum() / math.comb(PINK, 3)
        acc["pppp"] += (q * (q - 1) * (q - 2) * (q - 3) / 24).sum() / math.comb(PINK, 4)
    a = {k: v / N for k, v in acc.items()}
    return dict(perm2=PAY["perm2"] * a["pp"], direct1=PAY["direct1"] * a["first"],
                banker=PAY["perm2"] * (K - 1) * a["inc"] / 56, direct3=PAY["direct3"] * a["ppp"],
                direct4=PAY["direct4"] * a["pppp"])


def fair_ev():
    pair = K * (K - 1) / (NB * (NB - 1))
    return dict(perm2=PAY["perm2"] * pair, direct1=PAY["direct1"] / NB, banker=PAY["perm2"] * (K - 1) * (K / NB) / 56,
                direct3=PAY["direct3"] * K * (K - 1) * (K - 2) / (NB * (NB - 1) * (NB - 2)),
                direct4=PAY["direct4"] * K * (K - 1) * (K - 2) * (K - 3) / (NB * (NB - 1) * (NB - 2) * (NB - 3)))


# ------------------------------------------------------------------------------------------- machine-number model
def machine_forecast(W, M):
    """Walk-forward machine-number probabilities: a ball must first NOT be a winner, then be picked from the leftover.
    Uses the past machine draws' colour rates relative to what was available (shrunk to 'uniform from leftover')."""
    n = len(W)
    ok = machine_ok(W, M)
    pink = np.arange(NB) < PINK
    pm = np.zeros((n, NB))
    ep = ey = 0.0                 # expected pink / yellow machine numbers if uniform from leftover
    op = oy = 0.0
    import sharp
    X = onehot(W)
    Xp = np.vstack([X, np.zeros((1, NB))])
    P, colour = sharp.predict_all(Xp, NB, K)
    pwin = colour[:n]
    for t in range(n):
        lp = (op + 30) / (ep + 30)
        ly = (oy + 30) / (ey + 30)
        lift = np.where(pink, lp, ly)
        pm[t] = (1 - pwin[t]) * lift * K / (NB - K)
        if ok[t]:
            rp = PINK - (W[t] < PINK).sum()
            ep += K * rp / (NB - K)
            ey += K * (NB - K - rp) / (NB - K)
            op += (M[t] < PINK).sum()
            oy += (M[t] >= PINK).sum()
    return pm, ok


# ------------------------------------------------------------------------------------------- forecasters
def base_forecasts(W):
    """sharp and colour-only (walk-forward), plus the ORACLE (cheats: full-history ball rates) for reference."""
    import sharp
    n = len(W)
    X = onehot(W)
    P, colour = sharp.predict_all(X, NB, K)
    sp, _ = sharp.bma(P, X)
    oracle = np.broadcast_to(X.mean(0), (n, NB))
    return {"sharp": sp, "colour only": colour, "ORACLE (cheats)": oracle}


def rank(p, seed=1):
    return np.argsort(-(p + np.random.default_rng(seed).random(p.shape) * 1e-12), axis=1)


BETS = ["perm2_3", "perm2_4", "perm2_5", "perm2_6", "perm2_8", "perm2_10", "direct1", "banker", "direct3", "direct4"]
MBETS = ["mperm2_6", "mperm2_10"]


def table(models, W, M, rows, replays=1000, quiet=False, bets=BETS):
    rng = np.random.default_rng(3)
    out = []
    Wr = W[rows]
    luck = {b: luck_replays(b, Wr, None, replays, np.random.default_rng(11)) for b in bets} if replays else {}
    for mname, p in models.items():
        o = rank(p[rows])
        for b in bets:
            stake, desc = bet_lines(b)
            ret = returns(b, o, Wr)
            r = money(ret, stake, rng=rng, pay=PAY.get(b) if b in ("direct1", "direct3", "direct4") else None)
            r["wins"] = int((ret > 0).sum())
            r.update(model=mname, bet=b, desc=desc)
            r["luck_pct"] = (luck[b] < ret.sum()).mean() if replays else float("nan")
            out.append(r)
    return pd.DataFrame(out)


def fmt(df, title):
    lines = [f"### {title}", "",
             "| forecaster | bet | staked per draw | back per GH₵1 (95% CI) | net GH₵ | deepest fall | longest losing run | beats luck | p(beats break-even) | BH q |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        stake, _ = bet_lines(r.bet)
        lines.append(f"| {r.model} | {r.desc} | GH₵{stake} | {r.back:.3f} ({r.lo:.3f}-{r.hi:.3f}) | {r.net:+,.0f} | "
                     f"{r.dd:,.0f} | {r.lose} | {r.luck_pct:.1%} | {r.p_be:.3f} | {r.q:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--forecasts", default="", help=".npz from main_beast.py --save (adds every component)")
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--controls", type=int, default=0, help="fair-machine histories run through sharp")
    ap.add_argument("--start", type=int, default=WARM)
    ap.add_argument("--all", action="store_true", help="print every main_beast component (default: the main ones)")
    a = ap.parse_args()
    d = load(a.afriluck_dir)
    W, M = d["W"], d["M"]
    n = len(W)
    models = base_forecasts(W)
    if a.forecasts:
        f = np.load(a.forecasts, allow_pickle=True)
        assert f["W"].shape == W.shape and (f["W"] == W).all(), "forecast file was made from different data: re-run main_beast"
        comp = {k[3:]: f[k][:n] for k in f.files if k.startswith("c__")}
        if not a.all:
            comp = {k: v for k, v in comp.items() if k in ("stage4 ORIGINAL ensemble", "stage5 forecaster", "exit model")}
        models = {"MAIN BEAST": (f["p"] if "p" in f.files else f["BEAST"])[:n], **comp, **models}
    rows = np.arange(a.start, n)
    print(f"# Bet lab: Afriluck, {len(rows)} draws bet walk-forward ({d['dates'][rows[0]].date()} to "
          f"{d['dates'][rows[-1]].date()}); GH₵1 per line\n")

    lw, se = fit_colour_pl(W)
    fe = fair_ev()
    c_mid, c_hi = ceiling(math.exp(lw)), ceiling(math.exp(lw + 1.96 * se))
    print(f"## 1. The ceiling: pink strength w = {math.exp(lw):.3f} (95% CI {math.exp(lw - 1.96 * se):.3f}-"
          f"{math.exp(lw + 1.96 * se):.3f}) per exit, yellow = 1\n")
    print("| bet | fair machine | if the pink bias were known exactly | at the top of its 95% CI | needs to break even |")
    print("|---|---|---|---|---|")
    need = dict(perm2="pair x1.77", direct1="first ball x1.90", banker="ball x1.77", direct3="triple x2.44",
                direct4="four x2.63")
    for b in ("perm2", "direct1", "banker", "direct3", "direct4"):
        print(f"| {b} | {fe[b]:.3f} | {c_mid[b]:.3f} | {c_hi[b]:.3f} | {need[b]} |")
    print("\nEvery number in the 'known exactly' column is below 1.000: with colour as the only edge, every bet loses.\n")

    df = table(models, W, M, rows, a.replays)
    # machine-number Perm-2, on the draws that have machine numbers
    pm, ok = machine_forecast(W, M)
    mrows = rows[ok[rows]]
    mrows = mrows[mrows >= np.argmax(ok) + 50] if ok.any() else mrows[:0]
    if len(mrows):
        mo = rank(pm[mrows])
        mluck = {}
        for b in MBETS:
            stake, desc = bet_lines(b)
            ret = returns(b, None, W[mrows], M[mrows], mo)
            lr = np.array([returns(b, None, W[mrows], M[mrows], np.random.default_rng(i).random((len(mrows), NB)).argsort(1)).sum()
                           for i in range(a.replays)])
            r = money(ret, stake)
            r["wins"] = int((ret > 0).sum())
            r.update(model="machine model", bet=b, desc=desc + f" [{len(mrows)} draws]", luck_pct=(lr < ret.sum()).mean())
            mluck[b] = r
        df = pd.concat([df, pd.DataFrame(list(mluck.values()))], ignore_index=True)
    df["q"] = bh(df["p_be"].to_numpy())
    print("## 2. Walk-forward returns of every forecaster x bet\n")
    print(fmt(df, f"{len(df)} forecaster x bet rows; BH q = false-discovery-adjusted p for 'beats break-even'"))
    print("\n## 3. Verdict\n")
    real = ~df.model.str.contains("ORACLE")
    nwin = int(((df.q < 0.05) & (df.back > 1) & real).sum())
    if nwin == 0:
        print("NO forecaster x bet combination beats break-even after the multiple-testing correction.")
    else:
        print(f"{nwin} row(s) pass BH q < 0.05: confirm them on FUTURE draws before staking (see the forward-test plan).")
    rare = df[real & (df.wins < 30)]
    if len(rare):
        lucky = rare[rare.back > 1]
        if len(lucky):
            print("Rows above 100% that rest on a handful of wins (luck, as their p-values show): "
                  + "; ".join(f"{r.model} {r.bet} ({r.wins} win{'s' if r.wins != 1 else ''})" for _, r in lucky.iterrows()))
    best = df[real & (df.wins >= 30)].sort_values("back", ascending=False).head(3)
    print("Highest realised returns among bets with 30+ wins (picked AFTER the fact, so expect less in future):")
    for _, r in best.iterrows():
        print(f"- {r.model}, {r.desc}: {r.back:.1%} back -> about GH₵{100 * (1 - r.back):.0f} lost per GH₵100 staked")
    print(f"Expected in future (the ceiling above): Perm-2 on pink balls ~{c_mid['perm2']:.0%} back, i.e. about "
          f"GH₵{100 * (1 - c_mid['perm2']):.0f} lost per GH₵100. Not playing loses nothing.")
    if a.controls:
        print(f"\n## 4. Fair-machine controls ({a.controls} unbiased histories through sharp, Perm-2 on top 10)\n")
        rng = np.random.default_rng(5)
        real = df[(df.model == "sharp") & (df.bet == "perm2_10")].back.iloc[0]
        cb = []
        for _ in range(a.controls):
            Wf = rng.random((n, NB)).argsort(1)[:, :K]
            pf = base_forecasts(Wf)["sharp"]
            cb.append(returns("perm2_10", rank(pf[rows]), Wf[rows]).mean() / 45)
        cb = np.array(cb)
        print(f"sharp on the real machine: {real:.3f} back; on fair machines: mean {cb.mean():.3f}, "
              f"p = {(1 + (cb >= real).sum()) / (1 + len(cb)):.3f} (share of fair machines doing at least as well)")
    out = Path(a.forecasts or "bet_lab").with_name("bet_lab_table.csv")
    df.to_csv(out, index=False)
    print(f"\nFull table: {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
