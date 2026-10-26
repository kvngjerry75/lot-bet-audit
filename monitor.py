#!/usr/bin/env python3
"""
monitor.py: the monthly check for a ball set going bad. Run it after re-scraping NLA and Afriluck.

    python monitor.py --nla-dir nla_output --afriluck-dir afriluck_output
    python monitor.py --nla-dir nla_output --afriluck-dir afriluck_output --out monitor_report.md

For every game (and for its machine numbers, where published) it reports:
  1. LOW/HIGH TILT: does the machine favour low or high numbers (the two-colour effect seen on Afriluck)?
     Full history, the last 12 months, and per machine era for NLA (old machines before Jul 2023, new from Feb 2024).
  2. HOT / COLD BALLS: the most and least drawn balls, as a ratio to a fair machine.
  3. HONEST FORWARD TEST: balls picked as hottest on everything BEFORE the last 12 months, then scored ONLY on the
     last 12 months. This is the lift you could actually have bet on.
  4. BREAK-EVEN ALERT: for each bet type, how much more likely than chance each picked ball must be for the bet to
     pay back. ALERT only when the forward-tested lift's 95% lower bound is above that line.
Payouts are in PAYOUTS below: check them against the operator's current rules and edit if they differ.
Nothing here predicts a draw. It watches for the one thing that could make a game beatable: a physical bias that
grows large enough to pay.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# bet: (numbers on the ticket, payout multiple, chance the ticket wins on a fair machine is computed from the pool)
PAYOUTS = {
    "NLA 5/90": {"Direct-2": (2, 240), "Direct-3": (3, 2100)},          # CHECK against NLA's current table
    "DayWA 5/39": {},                                                    # payout table unknown: fill in to enable
    "Afriluck 6/57": {"Direct-1 (first ball)": (1, 30), "Direct-2": (2, 60), "Direct-3": (3, 600),
                      "Direct-4": (4, 10_000)},
}
NLA_GAMES = {"Main": ("main_draw_results.csv", 90, 5), "DayWA": ("daywa_results.csv", 39, 5),
             "VAG": ("vag_results.csv", 90, 5), "Noonrush": ("noonrush_results.csv", 90, 5)}
AFL_GAMES = {"Afriluck Anopa": "afriluck_anopa.csv", "Afriluck Midday": "afriluck_midday.csv",
             "Afriluck evening": "afriluck_evening.csv"}


# ----------------------------------------------------------------------------- loading
def tidy(df, k, date_col="Date"):
    """Oldest first, numbers as int, holiday rows / blanks / exact repeats / copied rows removed."""
    num = [f"Number{i}" for i in range(1, k + 1)]
    mac = [f"Machine{i}" for i in range(1, k + 1)]
    df = df.copy()
    if "Promotionday" in df:
        df = df[df["Promotionday"].astype(str) != "Holiday"]
    df = df[df[num].notna().all(axis=1)]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.iloc[::-1]                                   # files are newest first; keep each day's own order
    df = df.iloc[np.argsort(df[date_col].to_numpy(), kind="stable")]
    key = df[num].astype(int).apply(lambda r: tuple(sorted(r)), axis=1)
    keep, last_seen = [], {}
    for d, kk in zip(df[date_col], key):
        prev = last_seen.get(kk)
        keep.append(prev is None or (d - prev).days > 30)   # same set within 30 days = a copied row
        last_seen[kk] = d
    df = df[np.array(keep)]
    W = df[num].astype(int).to_numpy()
    M = df[mac].to_numpy(dtype=float) if all(c in df for c in mac) else np.full((len(df), k), np.nan)
    return df[date_col].reset_index(drop=True), W, M


def load_all(nla_dir, afl_dir):
    games = []
    if nla_dir:
        for g, (f, nb, k) in NLA_GAMES.items():
            p = Path(nla_dir) / f
            if p.exists():
                d, W, M = tidy(pd.read_csv(p), k)
                games.append(dict(name=g, pool=nb, k=k, dates=d, W=W, M=M,
                                  payouts=PAYOUTS["DayWA 5/39" if g == "DayWA" else "NLA 5/90"], nla=True))
    if afl_dir:
        for g, f in AFL_GAMES.items():
            p = Path(afl_dir) / f
            if p.exists():
                d, W, M = tidy(pd.read_csv(p), 6)
                games.append(dict(name=g, pool=57, k=6, dates=d, W=W, M=M, payouts=PAYOUTS["Afriluck 6/57"],
                                  nla=False))
        parts = [g for g in games if g["name"].startswith("Afriluck")]
        if len(parts) > 1:                                   # one machine and ball set: pooled view too
            d = pd.concat([g["dates"] for g in parts], ignore_index=True)
            o = np.argsort(d.to_numpy(), kind="stable")
            games.append(dict(name="Afriluck (all three pooled)", pool=57, k=6, dates=d.iloc[o].reset_index(drop=True),
                              W=np.vstack([g["W"] for g in parts])[o], M=np.vstack([g["M"] for g in parts])[o],
                              payouts=PAYOUTS["Afriluck 6/57"], nla=False))
    return games


# ----------------------------------------------------------------------------- statistics
def tilt(W, nb, k):
    """Average ball vs a fair machine: z < 0 = low numbers favoured."""
    n = len(W)
    if n < 30:
        return float("nan"), float("nan")
    mu = (nb + 1) / 2
    sd = math.sqrt((nb * nb - 1) / 12 * (nb - k) / (nb - 1) / k) / math.sqrt(n)
    return float(W.mean()), float((W.mean() - mu) / sd)


def ratios(W, nb, k):
    c = np.bincount(W.ravel() - 1, minlength=nb)
    return c / (len(W) * k / nb)


def forward_test(dates, W, nb, k, top, days=365):
    """Pick the `top` hottest balls on draws before the last `days` days; measure their rate after."""
    cut = dates.max() - pd.Timedelta(days=days)
    before, after = W[(dates < cut).to_numpy()], W[(dates >= cut).to_numpy()]
    if len(before) < 200 or len(after) < 100:
        return None
    r = ratios(before, nb, k)
    pick = np.argsort(-r)[:top] + 1
    hits = np.isin(after, pick).sum()
    trials = len(after) * top                               # ball-draw opportunities
    p0 = k / nb
    rate = hits / trials
    se = math.sqrt(p0 * (1 - p0) / trials)
    lift, lo = rate / p0, (rate - 1.96 * se) / p0
    return dict(pick=[int(x) for x in pick], lift=lift, lo=lo, n_after=len(after))


def fair_chance(nb, k, m):
    """P(all m chosen numbers are among the k drawn)."""
    return math.comb(nb - m, k - m) / math.comb(nb, k)


def break_even_lift(nb, k, m, payout, first_ball=False):
    p = 1 / nb if first_ball else fair_chance(nb, k, m)
    return (1 / (payout * p)) ** (1 / m)


# ----------------------------------------------------------------------------- report
def report(games):
    L = ["# Monthly monitor", ""]
    alerts = []
    L += ["## 1. Low/high tilt (average ball vs a fair machine; z below -3 or above +3 is clear)", "",
          "| Game | Draws | Period | Full history | Last 12 months | Old machines (<Jul 2023) | New machines (Feb 2024+) | Machine numbers |",
          "|---|---|---|---|---|---|---|---|"]
    for g in games:
        d, W, nb, k = g["dates"], g["W"], g["pool"], g["k"]
        last = (d >= d.max() - pd.Timedelta(days=365)).to_numpy()

        def cell(mask):
            m, z = tilt(W[mask], nb, k)
            return "n/a" if z != z else f"{m:.2f} (z {z:+.1f})"
        old = (d < "2023-07-01").to_numpy() if g["nla"] else np.zeros(len(d), bool)
        new = (d >= "2024-02-01").to_numpy() if g["nla"] else np.zeros(len(d), bool)
        Mok = ~np.isnan(g["M"]).any(axis=1)
        mcell = "none published"
        if Mok.sum() >= 100:
            mm, mz = tilt(g["M"][Mok].astype(int), nb, k)
            mcell = f"{mm:.2f} (z {mz:+.1f}, {Mok.sum()} draws)"
        L.append(f"| {g['name']} (fair {(nb + 1) / 2:.1f}) | {len(W)} | {d.min().date()} to {d.max().date()} | "
                 f"{cell(np.ones(len(W), bool))} | {cell(last)} | {cell(old) if old.any() else '-'} | "
                 f"{cell(new) if new.any() else '-'} | {mcell} |")
    L += ["", "## 2. Hot and cold balls (full history; 1.00 = fair)", ""]
    for g in games:
        r = ratios(g["W"], g["pool"], g["k"])
        o = np.argsort(-r)
        hot = ", ".join(f"{b + 1} ({r[b]:.2f})" for b in o[:10])
        cold = ", ".join(f"{b + 1} ({r[b]:.2f})" for b in o[::-1][:10])
        L.append(f"- **{g['name']}**: hot {hot} | cold {cold}")
    L += ["", "## 3. Forward test and break-even (hottest balls picked on data before the last 12 months, scored on "
          "the last 12 months only)", "",
          "| Game | Bet | Balls picked | Forward lift per ball [95% lower bound] | Lift needed to break even | Status |",
          "|---|---|---|---|---|---|"]
    for g in games:
        nb, k = g["pool"], g["k"]
        bets = g["payouts"] or {"(no payout table)": (2, None)}
        for bet, (m, pay) in bets.items():
            ft = forward_test(g["dates"], g["W"], nb, k, max(m, 1))
            if ft is None:
                L.append(f"| {g['name']} | {bet} | - | not enough data | - | - |")
                continue
            need = break_even_lift(nb, k, m, pay, first_ball=bet.startswith("Direct-1")) if pay else None
            status = "no payout table" if need is None else (
                "**ALERT: above break-even**" if ft["lo"] > need else
                ("above break-even but not proven" if ft["lift"] > need else "below break-even"))
            if need is not None and ft["lo"] > need:
                alerts.append(f"{g['name']} {bet}: forward lift {ft['lift']:.2f} (lower bound {ft['lo']:.2f}) > {need:.2f}")
            L.append(f"| {g['name']} | {bet} | {', '.join(map(str, ft['pick']))} | {ft['lift']:.2f} [{ft['lo']:.2f}] | "
                     f"{'-' if need is None else f'{need:.2f}'} | {status} |")
    L += ["", "## Verdict", ""]
    L.append("**ALERT** " + "; ".join(alerts) + ". Confirm on new draws with the tracker before staking anything."
             if alerts else "No bet type is above break-even. Every game still loses money on average, whichever "
                            "balls you pick. Re-run next month.")
    L += ["", "Payouts used: " + "; ".join(f"{k}: " + (", ".join(f"{b} {p}x" for b, (_, p) in v.items()) or "unknown")
                                       for k, v in PAYOUTS.items()) + ". Edit PAYOUTS in monitor.py if they differ."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nla-dir", default="nla_output")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--out", default="monitor_report.md")
    a = ap.parse_args()
    games = load_all(a.nla_dir, a.afriluck_dir)
    if not games:
        raise SystemExit("No game files found. Point --nla-dir / --afriluck-dir at the scraper output folders.")
    text = report(games)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
