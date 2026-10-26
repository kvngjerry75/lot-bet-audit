#!/usr/bin/env python3
"""
clean_data.py: audited cleaning of the four game files. Run it once after every batch of new results; every other
v3 tool reads the *_audited.csv files it writes.

    !python clean_data.py                                   # the four default files, writes *_audited.csv + cleaning_log.csv
    !python clean_data.py --files daywa_clean.csv:39 main_draw_clean.csv:90 noonrush_clean.csv:90 vag_clean.csv:90

What it removes (each removal is written to cleaning_log.csv with the reason):
  1. rows with a missing / non-integer number, a number outside the game's pool (DayWA had 87, 83, 77 in a 5/39 game)
     or the same number twice;
  2. exact duplicate rows (same date, same numbers);
  3. COPIES inside a game: the same 5 numbers in the same drawn order at any distance (chance about 1 in 5 billion for
     5/90), or the same set within 30 days. The later row is dropped (a copied-forward row);
  4. CROSS-GAME COPIES: the same set in two games within 3 days. It is dropped from the game where it does not belong:
     the game that already has another row on that date, or the 5/90 game when every number is <= 39 (a DayWA draw
     pasted into a 5/90 file). If neither rule decides, it is dropped from both.
What it keeps but flags: two different draws on the same date (same_date=1). These dates line up across games
(e.g. 2024-01-02, 2024-03-07, 2024-04-02, 2024-05-02 in both Noonrush and VAG), so they look like real make-up
draws, but their order within the day is unknown. Sequence rules that read "the previous draw" treat them with care.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

K = 5
NUM = [f"Number{i}" for i in range(1, K + 1)]
DEFAULT = {"DayWA": ("daywa_clean.csv", 39), "Main": ("main_draw_clean.csv", 90),
           "Noonrush": ("noonrush_clean.csv", 90), "VAG": ("vag_clean.csv", 90)}


def _read(path, nb):
    r = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    r.columns = [str(c).strip() for c in r.columns]
    r["_row"] = np.arange(len(r))
    r["_date"] = pd.to_datetime(r["Date"].str.strip(), errors="coerce")
    v = r[NUM].apply(lambda c: pd.to_numeric(c.str.strip(), errors="coerce"))
    reason = pd.Series("", index=r.index, dtype=object)
    # Holiday rows written by nla_scraper.py (no numbers): not a draw, dropped quietly
    label = r.get("Promotionday", r.get("Promotionorday", pd.Series("", index=r.index))).astype(str)
    reason[(label.str.strip().str.lower() == "holiday") & v.isna().all(axis=1)] = "holiday"
    reason[(reason == "") & r["_date"].isna()] = "bad date"
    bad = v.isna().any(axis=1) | ((v % 1) != 0).any(axis=1)
    reason[(reason == "") & bad] = "missing or non-integer number"
    vi = v.fillna(0).astype(int)
    reason[(reason == "") & ((vi < 1) | (vi > nb)).any(axis=1)] = f"number outside the pool 1..{nb}"
    reason[(reason == "") & (vi.apply(lambda x: len(set(x)), axis=1) != K)] = "same number twice in one draw"
    for c in NUM:
        r[c] = vi[c]
    return r, reason


def clean(files: dict, out_dir: Path, window_same=30, window_cross=3):
    games, reasons = {}, {}
    for g, (path, nb) in files.items():
        r, reason = _read(path, nb)
        # oldest first; the files are newest-first, so reverse before the stable sort to keep the day's own order
        if len(r) > 1 and r["_date"].iloc[0] > r["_date"].iloc[-1]:
            r, reason = r.iloc[::-1], reason.iloc[::-1]
        o = np.argsort(r["_date"].to_numpy(), kind="stable")
        games[g], reasons[g] = r.iloc[o].copy(), reason.iloc[o].copy()

    # 2 + 3: duplicates and copies inside a game
    for g, r in games.items():
        rs = reasons[g]
        seen_t, seen_s = {}, {}
        for i, row in r.iterrows():
            if rs[i]:
                continue
            tup, st, d = tuple(row[NUM]), frozenset(row[NUM]), row["_date"]
            if tup in seen_t:
                d0 = seen_t[tup]
                rs[i] = ("exact duplicate row" if d0 == d else f"copy of the draw of {d0.date()} (same numbers, same order)")
                continue
            if st in seen_s and (d - seen_s[st]).days <= window_same:
                rs[i] = f"copy of the draw of {seen_s[st].date()} (same set within {window_same} days)"
                continue
            seen_t[tup], seen_s[st] = d, d

    # 4: cross-game copies
    names = list(games)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            ga, gb = names[a_i], names[b_i]
            A, B = games[ga], games[gb]
            okA, okB = reasons[ga] == "", reasons[gb] == ""
            idxB = {}
            for j, row in B[okB].iterrows():
                idxB.setdefault(frozenset(row[NUM]), []).append(j)
            for i, row in A[okA].iterrows():
                for j in idxB.get(frozenset(row[NUM]), []):
                    if abs((B.at[j, "_date"] - row["_date"]).days) > window_cross:
                        continue
                    dA = (A["_date"] == row["_date"]).sum() > 1
                    dB = (B["_date"] == B.at[j, "_date"]).sum() > 1
                    small = max(row[NUM]) <= 39
                    nbA, nbB = files[ga][1], files[gb][1]
                    drop = []
                    if dA != dB:
                        drop = [ga] if dA else [gb]
                    elif small and nbA != nbB:
                        drop = [ga] if nbA > nbB else [gb]
                    else:
                        drop = [ga, gb]
                    if ga in drop:
                        reasons[ga][i] = f"cross-game copy of {gb} {B.at[j, '_date'].date()}"
                    if gb in drop:
                        reasons[gb][j] = f"cross-game copy of {ga} {row['_date'].date()}"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows, summary = [], []
    for g, r in games.items():
        rs = reasons[g]
        keep = r[rs == ""].copy()
        keep["same_date"] = keep["_date"].duplicated(keep=False).astype(int)
        hol = rs == "holiday"
        rs = rs.where(~hol, "")
        r = r[~hol]
        rs = rs[~hol]
        for i in r.index[rs != ""]:
            log_rows.append(dict(game=g, date=str(r.at[i, "Date"]), numbers=" ".join(str(r.at[i, c]) for c in NUM),
                                 reason=rs[i]))
        cols = [c for c in r.columns if not c.startswith("_")] + ["same_date"]
        keep = keep[cols].iloc[::-1]                     # newest-first like the source files
        stem = Path(files[g][0]).stem.replace("_clean", "")
        keep.to_csv(out_dir / f"{stem}_audited.csv", index=False)
        summary.append((g, len(r), int((rs != "").sum()), len(keep), int(keep["same_date"].sum())))
    pd.DataFrame(log_rows).to_csv(out_dir / "cleaning_log.csv", index=False)
    print("| game | rows in | removed | rows out | kept rows on a shared date |")
    print("|---|---|---|---|---|")
    for s in summary:
        print("| " + " | ".join(map(str, s)) + " |")
    print()
    for r in log_rows:
        print(f"  {r['game']:9s} {r['date']}  {r['numbers']:20s}  {r['reason']}")
    return summary, log_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[], help="file.csv:pool pairs (default: the four game files)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--from-scraper", default="", help="folder written by nla_scraper.py (nla_output): cleans its four "
                                                         "*_results.csv files into <out-dir>/<game>_audited.csv")
    a = ap.parse_args()
    if a.from_scraper:
        src = Path(a.from_scraper)
        files = {"DayWA": (str(src / "daywa_results.csv"), 39), "Main": (str(src / "main_draw_results.csv"), 90),
                 "Noonrush": (str(src / "noonrush_results.csv"), 90), "VAG": (str(src / "vag_results.csv"), 90)}
        out = Path(a.out_dir)
        clean(files, out)
        for stem in ("daywa", "main_draw", "noonrush", "vag"):     # the analysis tools read <game>_audited.csv
            p = out / f"{stem}_results_audited.csv"
            if p.exists():
                p.replace(out / f"{stem}_audited.csv")
        return
    if a.files:
        files = {Path(f.split(":")[0]).stem: (f.split(":")[0], int(f.split(":")[1])) for f in a.files}
    else:
        files = DEFAULT
    clean(files, Path(a.out_dir))


if __name__ == "__main__":
    main()
