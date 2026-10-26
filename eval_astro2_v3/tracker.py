#!/usr/bin/env python3
"""
tracker.py: LIVE forward test. Every stage4 run appends its picks to a ledger BEFORE the draw; once the draw is in the
CSV, `score` compares each logged pick with what was actually drawn and tests the running totals against chance.

This is the one test no backtest setting can flatter: the picks are frozen in the ledger before the numbers exist.

    !python stage4_astronomical.py --csv daywa_clean.csv --null-reps 10      # logs the pick to outputs/tracker.csv
    ... after the draw, add it to daywa_clean.csv ...
    !python tracker.py add-draw --csv daywa_clean.csv --date 2026-10-21 --numbers 23 32 26 11 22   # record a result
    !python tracker.py score --csv daywa_clean.csv                          # scores every logged pick so far

Keep outputs/tracker.csv safe between Colab sessions (download it, or point --tracker at Google Drive).
A pick logged AFTER its draw date is marked LATE and never counted.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nla_config import N_DRAWN, chance_core, chance_pair

FIELDS = ["game", "draw_date", "logged_at", "n_balls", "top2", "top5", "core", "top2_mode", "verdict"]


def _fmt(nums):
    return " ".join(str(int(x)) for x in nums)


def _parse(s):
    return [int(x) for x in str(s).split()] if str(s).strip() else []


def log_prediction(path, game, draw_date, n_balls, top2, top5, core, top2_mode="vote", verdict=""):
    """Append one pick; an existing row for the same game + draw date is kept (the FIRST pick counts, not the best)."""
    path = Path(path)
    rows = read_ledger(path)
    if any(r["game"] == game and r["draw_date"] == str(draw_date) for r in rows):
        return False
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(game=game, draw_date=str(draw_date), logged_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        n_balls=int(n_balls), top2=_fmt(top2), top5=_fmt(top5), core=_fmt(core), top2_mode=top2_mode,
                        verdict=verdict))
    return True


def read_ledger(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def actual_draws(csv_path):
    """{date 'YYYY-MM-DD': set(numbers)} from a game CSV (Date, Number1..Number5)."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig", skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = [f"Number{i}" for i in range(1, N_DRAWN + 1)]
    out, dup = {}, set()
    for _, r in raw.iterrows():
        try:
            d = pd.to_datetime(r["Date"], dayfirst=False).date().isoformat()
            nums = {int(float(r[c])) for c in cols}
        except (ValueError, TypeError):
            continue
        if len(nums) == N_DRAWN:
            if d in out and out[d] != nums:
                dup.add(d)                  # v3: two different draws on one date: scoring against either is a guess
            out[d] = nums
    actual_draws.ambiguous = dup
    return out


def add_draw(csv_path, date, numbers):
    """Insert one result into a game CSV in the file's own layout (newest-first, Number1..5 in drawn order).
    Refuses duplicates and anything that is not 5 distinct whole numbers inside the file's pool."""
    date = pd.Timestamp(date).date().isoformat()
    nums = [int(x) for x in numbers]
    if len(nums) != N_DRAWN or len(set(nums)) != N_DRAWN or min(nums) < 1:
        raise SystemExit(f"need {N_DRAWN} distinct positive numbers, got {nums}")
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = [f"Number{i}" for i in range(1, N_DRAWN + 1)]
    pool = int(pd.to_numeric(raw[cols].stack(), errors="coerce").quantile(0.999))
    if max(nums) > pool:
        raise SystemExit(f"{max(nums)} is outside this game's pool (1..{pool})")
    dates = pd.to_datetime(raw["Date"], errors="coerce")
    if (dates.dt.date.astype(str) == date).any():
        raise SystemExit(f"{csv_path} already has a row for {date}; nothing changed")
    day = pd.Timestamp(date).day_name()
    row = {c: "" for c in raw.columns}
    row.update(Date=date, **{c: f"{float(v):.1f}" for c, v in zip(cols, nums)})
    for c, v in (("Promotionday", day), ("RealDay", day), ("WasShifted", "0")):
        if c in row:
            row[c] = v
    newest_first = len(dates) > 1 and dates.iloc[0] > dates.iloc[-1]
    new_ts = pd.Timestamp(date)                                       # insert in place; every other row keeps its order
    if newest_first:
        later = np.where(dates.to_numpy() > np.datetime64(new_ts))[0]
        pos = int(later.max()) + 1 if len(later) else 0
    else:
        earlier = np.where(dates.to_numpy() < np.datetime64(new_ts))[0]
        pos = int(earlier.max()) + 1 if len(earlier) else 0
    out = pd.concat([raw.iloc[:pos], pd.DataFrame([row]), raw.iloc[pos:]], ignore_index=True)
    out.to_csv(csv_path, index=False)
    print(f"Added {date} ({day}): {' '.join(map(str, nums))} to {csv_path} ({len(out)} rows)")


def _z_p(obs_sum, mean, var, n):
    """One-sided p-value (better than chance) for a sum of n independent per-draw scores."""
    if n == 0 or var <= 0:
        return float("nan")
    z = (obs_sum - n * mean) / math.sqrt(n * var)
    return float(stats.norm.sf(z))


def _hits_moments(k, nb):
    """Mean / variance of the number of drawn balls among k fixed picks (hypergeometric)."""
    p = N_DRAWN / nb
    return k * p, k * p * (1 - p) * (nb - k) / (nb - 1)


def score(ledger_path, csv_path, game=None, allow_same_day=False):
    rows = read_ledger(ledger_path)
    draws = actual_draws(csv_path)
    ambiguous = actual_draws.ambiguous
    game = game or Path(csv_path).stem
    rows = [r for r in rows if r["game"] == game]
    if not rows:
        print(f"No logged picks for game '{game}' in {ledger_path}.")
        return
    print(f"Live tracker | game {game} | ledger {ledger_path}\n")
    print("| draw date | logged at | drawn | top-2 hits | top-5 hits | core hits | counted |")
    print("|---|---|---|---|---|---|---|")
    sc = []
    nb = int(rows[0]["n_balls"])
    for r in sorted(rows, key=lambda r: r["draw_date"]):
        drawn = draws.get(r["draw_date"])
        # v3: a pick must be logged on an EARLIER calendar day than its draw (draw times are not in the data, so a
        # same-day log could come after a noon or evening draw). Same-day logs are shown but not counted unless
        # --allow-same-day is given.
        late = r["logged_at"][:10] > r["draw_date"] or (r["logged_at"][:10] == r["draw_date"] and not allow_same_day)
        if (r["draw_date"] in ambiguous):
            print(f"| {r['draw_date']} | {r['logged_at']} | two different draws share this date | | | | not counted |")
            continue
        if drawn is None:
            print(f"| {r['draw_date']} | {r['logged_at']} | (not in CSV yet) | | | | pending |")
            continue
        t2, t5, co = (len(drawn & set(_parse(r[k]))) for k in ("top2", "top5", "core"))
        status = "LATE or SAME-DAY: not counted" if late else "yes"
        print(f"| {r['draw_date']} | {r['logged_at']} | {_fmt(sorted(drawn))} | {t2} | {t5} | {co} | {status} |")
        if not late:
            sc.append((t2, t5, co, len(_parse(r["core"]))))
    n = len(sc)
    print(f"\nCounted picks: {n}")
    if n == 0:
        return
    t2 = np.array([s[0] for s in sc]); t5 = np.array([s[1] for s in sc]); co = np.array([s[2] for s in sc])
    V = sc[0][3]
    m5, v5 = _hits_moments(5, nb)
    m2, v2 = _hits_moments(2, nb)
    pc = chance_core(V, nb)
    core_hits = int((co >= 2).sum())
    p_core = float(stats.binom.sf(core_hits - 1, n, pc))
    both = int((t2 >= 2).sum())
    p_both = float(stats.binom.sf(both - 1, n, chance_pair(nb)))
    print("\n| metric | live result | chance | one-sided p (better than chance) |")
    print("|---|---|---|---|")
    print(f"| mean hits in top-5 | {t5.mean():.3f} | {m5:.3f} | {_z_p(t5.sum(), m5, v5, n):.3f} |")
    print(f"| mean hits in top-2 | {t2.mean():.3f} | {m2:.3f} | {_z_p(t2.sum(), m2, v2, n):.3f} |")
    print(f"| core-{V} hit (>= 2 of 5) | {core_hits}/{n} = {core_hits / n:.1%} | {pc:.1%} | {p_core:.3f} |")
    print(f"| both top-2 drawn | {both}/{n} | {chance_pair(nb):.2%} | {p_both:.3f} |")
    # v3: anytime-valid evidence (test martingale) for the pre-registered metric, the core hit. You may look at it after
    # every draw without inflating false alarms: under pure luck it exceeds 80 with probability < 1/80 EVER.
    lifts = np.array([1.1, 1.25, 1.5, 2.0])
    h = (co >= 2).astype(float)
    le = h.sum() * np.log(lifts) + (len(h) - h.sum()) * np.log((1 - np.minimum(lifts * pc, 0.99)) / (1 - pc))
    E = float(np.exp(le).mean())
    print(f"\nEvidence for a core-hit edge (anytime-valid e-value): {E:.2f}  |  CONFIRMED only if >= 80 "
          f"(= 4 games / alpha 0.05). Below 1 = the record so far is worse than luck.")
    print("\nRead it honestly: p < 0.05 on a metric chosen in advance (core hit) after 100+ counted picks would be real "
          "evidence. A few lucky draws early on are expected and mean nothing.")
    if n < 100:
        print(f"Only {n} counted picks so far: far too few for a verdict (about 100-300 are needed to see a real edge).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="live forward-test ledger for stage4 picks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True, help="the game CSV, updated with the latest draws")
    s.add_argument("--tracker", default="outputs/tracker.csv")
    s.add_argument("--game", default="", help="game label (default: CSV name)")
    s.add_argument("--allow-same-day", action="store_true",
                   help="count picks logged on the draw day itself (only if you KNOW they were logged before the draw)")
    ad = sub.add_parser("add-draw")
    ad.add_argument("--csv", required=True)
    ad.add_argument("--date", required=True, help="YYYY-MM-DD")
    ad.add_argument("--numbers", nargs=N_DRAWN, required=True, type=int, help="the 5 numbers in drawn order")
    a = ap.parse_args(argv)
    if a.cmd == "add-draw":
        add_draw(a.csv, a.date, a.numbers)
    elif a.cmd == "score":
        score(a.tracker, a.csv, a.game or None, a.allow_same_day)


if __name__ == "__main__":
    main()
