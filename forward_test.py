#!/usr/bin/env python3
"""
forward_test.py: the honest forward test. Log the pick BEFORE each draw, score it AFTER, stop by a fixed rule.

    python forward_test.py log   --afriluck-dir afriluck_output              # before a draw: logs the next draw's pick
    python forward_test.py score --afriluck-dir afriluck_output              # after re-scraping: scores every logged pick
    python forward_test.py plan                                              # the power calculation behind the rule

What is logged (forward_log.csv): the time you logged it (UTC = Ghana time), the target draw (date + Anopa / Midday /
evening), the bet and its numbers, from the default MAIN BEAST (chance + sharp + colour; the same forecast main_beast
prints, computed in seconds). A pick logged after its draw started is marked LATE and never counted. One pick per draw.

The rule is fixed in advance (pre-registered), per bet:
  H0: the bet returns what the pink bias allows (~0.62 per GH₵1 for Perm-2).  H1: it returns at least 1.00.
  STOP and reject the edge as soon as the running return's upper 99% bound falls below 1.00 (after 30+ draws).
  Call the edge CONFIRMED only after the planned number of draws (printed by `plan`) if the return is >= 1.00 AND
  its lower 99% bound is above 0.62. Anything else: keep logging, stake nothing.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DRAW_TIME = {"Anopa": "10:00", "Midday": "13:30", "evening": "19:45"}     # Ghana time = UTC
H0, H1, Z = 0.62, 1.00, 2.326                                               # one-sided 99%


def bet_spec(bet):
    import bet_lab as bl
    return bl.bet_lines(bet)


def per_draw_sd(bet, w=1.112, N=400_000, seed=0):
    """sd of 'GH₵ back per GH₵1' on one draw for a bet on the top pink balls, at the fitted pink strength."""
    import bet_lab as bl
    rng = np.random.default_rng(seed)
    wt = np.where(np.arange(57) < 28, w, 1.0)
    g = -np.log(rng.random((N, 57))) / wt
    W = np.argsort(g, 1)[:, :6]
    order = np.tile(np.arange(57), (N, 1))                                  # balls 1.. are pink: a pink-first pick
    stake, _ = bl.bet_lines(bet)
    r = bl.returns(bet, order, W) / stake
    return r.std(), r.mean()


def plan_n(sd, alpha_z=Z, power_z=1.282):
    return int(math.ceil(((alpha_z + power_z) * sd / (H1 - H0)) ** 2))


def forecast(afl_dir):
    import main_beast as mb
    import sharp
    d = mb.load("Afriluck", "", afl_dir)
    W = d["W"]
    n = len(W)
    X = np.zeros((n + 1, 57))
    X[np.repeat(np.arange(n), 6), W.ravel()] = 1
    P, colour = sharp.predict_all(X, 57, 6)
    sp, _ = sharp.bma(P, X)
    p, _, _ = mb.beast({"sharp": mb.norm(sp, 6), "colour only": mb.norm(colour, 6)}, W, 57, 6)
    nd, ns = mb.next_slot(d, "Afriluck")
    return d, [int(b) + 1 for b in np.argsort(-p[n])], nd, ns


def cmd_log(a):
    d, order, nd, ns = forecast(a.afriluck_dir)
    stake, desc = bet_spec(a.bet)
    c = int(a.bet.split("_")[1]) if "_" in a.bet else {"direct1": 1, "banker": 1, "direct3": 3, "direct4": 4}[a.bet]
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC").tz_localize(None)
    due = pd.Timestamp(f"{nd.date()} {DRAW_TIME[ns]}")
    log = Path(a.log)
    df = pd.read_csv(log) if log.exists() else pd.DataFrame()
    key = f"{nd.date()} {ns}"
    if len(df) and ((df["target"] == key) & (df["bet"] == a.bet)).any():
        print(f"A pick for {key} ({a.bet}) is already logged; the first one stands.")
        return
    row = dict(logged_utc=now.strftime("%Y-%m-%d %H:%M"), target=key, draw_time=due.strftime("%Y-%m-%d %H:%M"),
               late=bool(now >= due), bet=a.bet, stake=stake, numbers=" ".join(map(str, order[:c])),
               data_up_to=f"{d['dates'][-1].date()} {d['session'][-1]}")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log, index=False)
    print(f"Logged for {key} (draw at {due:%H:%M}): {desc}: {row['numbers']}  [stake GH₵{stake}]"
          + ("  ** LATE: the draw has already started, this pick will NOT be counted **" if row["late"] else ""))


def cmd_score(a):
    import bet_lab as bl
    import main_beast as mb
    log = Path(a.log)
    if not log.exists():
        print("Nothing logged yet: run `python forward_test.py log` before a draw.")
        return
    df = pd.read_csv(log)
    d = mb.load("Afriluck", "", a.afriluck_dir)
    res = {f"{dt.date()} {s}": w for dt, s, w in zip(d["dates"], d["session"], d["W"])}
    out = []
    for _, r in df.iterrows():
        if r["late"] or r["target"] not in res:
            continue
        nums = np.array([int(x) - 1 for x in str(r["numbers"]).split()])
        order = np.concatenate([nums, np.setdiff1d(np.arange(57), nums)])[None]
        ret = float(bl.returns(r["bet"], order, res[r["target"]][None])[0])
        out.append(dict(target=r["target"], bet=r["bet"], stake=r["stake"], returned=ret))
    pending = int((~df["late"] & ~df["target"].isin(res)).sum())
    print(f"# Forward test: {len(df)} picks logged, {int(df['late'].sum())} late (not counted), {pending} waiting for "
          f"results, {len(out)} scored\n")
    if not out:
        return
    s = pd.DataFrame(out)
    for bet, g in s.groupby("bet"):
        sd, mu0 = per_draw_sd(bet)
        need = plan_n(sd)
        back = g["returned"].sum() / g["stake"].sum()
        m = len(g)
        se = sd / math.sqrt(m)
        upper, lower = back + Z * se, back - Z * se
        print(f"## {bet_spec(bet)[1]}: {m} draws, staked GH₵{g['stake'].sum():,.0f}, returned GH₵{g['returned'].sum():,.0f}, "
              f"net GH₵{g['returned'].sum() - g['stake'].sum():+,.0f}, {back:.1%} back (99% band {max(lower, 0):.2f} to "
              f"{upper:.2f})")
        if m >= 30 and upper < H1:
            print(f"   STOP: a break-even edge is ruled out (the upper bound {upper:.2f} is below 1.00). Stake nothing.")
        elif m >= need and back >= H1 and lower > H0:
            print(f"   CONFIRMED after the planned {need} draws: return >= 1.00 and clearly above the 0.62 colour level. "
                  f"Size stakes with bet_lab's numbers (fractional Kelly), and keep logging.")
        else:
            print(f"   UNDECIDED: keep logging with no stake. Planned length {need} draws ({max(need - m, 0)} to go).")


def cmd_plan(a):
    print("# Forward-test plan (pre-registered): H0 = 0.62 back per GH₵1 (pink bias only), H1 = 1.00 (break-even)")
    print("One-sided 1% false alarm, 90% power. sd = spread of one draw's return per GH₵1 for a pink-first pick.\n")
    print("| bet | GH₵ per draw | expected back now | sd per draw | draws needed | about |")
    print("|---|---|---|---|---|---|")
    for bet in ("perm2_3", "perm2_6", "perm2_10", "banker", "direct1", "direct3"):
        sd, mu = per_draw_sd(bet)
        n = plan_n(sd)
        stake, desc = bet_spec(bet)
        print(f"| {desc} | {stake} | {mu:.2f} | {sd:.2f} | {n} | {n / 20:.0f} weeks at 20 draws a week |")
    sd, _ = per_draw_sd("perm2_10")
    n_small = int(math.ceil(((1.645 + 0.842) * sd / (0.62 - 0.564)) ** 2))
    print(f"\nFor comparison, just showing the pink edge itself (0.62 vs 0.564 for a random pick, 5% / 80%) takes "
          f"about {n_small:,} draws of Perm-2 on 10: that is why it took the whole history to prove it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["log", "score", "plan"])
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--log", default="forward_log.csv")
    ap.add_argument("--bet", default="perm2_10", help="perm2_3 .. perm2_10, banker, direct1, direct3, direct4")
    ap.add_argument("--now", default="", help=argparse.SUPPRESS)          # tests only: pretend the clock says this
    a = ap.parse_args()
    {"log": cmd_log, "score": cmd_score, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    main()
