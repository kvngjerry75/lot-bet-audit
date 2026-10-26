"""Data pipeline: load -> strict clean -> calendar features -> (N, 90) binary tensor.

Every step logs what it did and how many rows it touched. Rejected rows are
written to disk so the cleaning can be audited.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import GAMES, N_BALLS, N_DRAWN, NUMBER_COLS, REQUIRED_COLS

log = logging.getLogger("nla.data")

_ARTIFACT_RE = re.compile(r"col\s*\d+\s*:", re.IGNORECASE)  # e.g. "Col 2: Promotionday"
_HEADER_TOKENS = {c.lower() for c in REQUIRED_COLS}  # repeated header rows
_BLANK_RE = r"(?i)^\s*(nan|none|null|na|n/a|-|--|holiday)?\s*$"


@dataclass
class GameData:
    key: str
    name: str
    path: Path
    df: pd.DataFrame  # clean draws, sorted by date, with feature columns
    X: np.ndarray  # (N, 90) uint8 binary tensor, row = draw, col = ball-1
    holidays: pd.DataFrame  # dropped blank "Holiday" rows (kept as a feature source)
    report: dict


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _read_raw(path: Path) -> tuple[pd.DataFrame, list]:
    log.info("Reading %s", path)
    bad_lines: list = []

    def _on_bad(line):
        bad_lines.append(line)
        return None  # skip the line

    raw = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines=_on_bad,
        encoding="utf-8-sig",
        skipinitialspace=True,
    )
    raw.columns = [str(c).strip() for c in raw.columns]
    lower = {c.lower(): c for c in raw.columns}
    missing = [c for c in REQUIRED_COLS if c.lower() not in lower]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}; found {list(raw.columns)}")
    raw = raw.rename(columns={lower[c.lower()]: c for c in REQUIRED_COLS})[REQUIRED_COLS]

    log.info("Raw rows read: %d | lines skipped by parser (too many fields): %d", len(raw), len(bad_lines))
    for line in bad_lines[:5]:
        log.debug("Parser-skipped line: %s", line)
    return raw, bad_lines


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Returns (clean_draws, holidays, rejected, report)."""
    df = raw.copy()
    for c in REQUIRED_COLS:
        df[c] = df[c].astype(str).str.strip()
    df["_line"] = np.arange(2, len(df) + 2)  # approx. CSV line number (header = line 1)

    reason = pd.Series("", index=df.index, dtype=object)
    counts: dict[str, int] = {}

    def flag(mask: pd.Series, label: str) -> None:
        m = mask & (reason == "")  # first reason wins
        reason[m] = label
        counts[label] = int(m.sum())
        log.info("  clean | %-34s %6d rows", label, counts[label])

    log.info("Cleaning %d raw rows", len(df))

    # 1. text/header artifacts mixed into data
    art = pd.Series(False, index=df.index)
    for c in REQUIRED_COLS:
        s = df[c]
        art |= s.str.contains(_ARTIFACT_RE, na=False) | s.str.lower().isin(_HEADER_TOKENS)
    flag(art, "malformed_header_artifact")

    # 2. dates (ISO first, then a tolerant fallback for the leftovers)
    d = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    retry = d.isna() & (reason == "")
    if retry.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d.loc[retry] = pd.to_datetime(df.loc[retry, "Date"], errors="coerce")
        recovered = int((retry & d.notna()).sum())
        log.warning("  clean | %d non-ISO dates: %d recovered by fallback parser", int(retry.sum()), recovered)
    df["Date"] = d
    flag(d.isna(), "malformed_date")

    # 3. holiday rows: all five numbers blank (kept separately as a feature source)
    nums_s = df[NUMBER_COLS].replace(_BLANK_RE, "", regex=True)
    flag((nums_s == "").all(axis=1), "holiday_blank")
    hol_mask = reason == "holiday_blank"

    # 4. numeric validity
    nums = nums_s.apply(pd.to_numeric, errors="coerce")
    flag(nums.isna().any(axis=1), "malformed_partial_or_nonnumeric")
    nv = nums.fillna(0)
    flag(((nv % 1) != 0).any(axis=1), "non_integer_value")
    flag(((nv < 1) | (nv > N_BALLS)).any(axis=1), "out_of_range_1_90")
    srt = np.sort(nv.to_numpy(), axis=1)
    flag(pd.Series((np.diff(srt, axis=1) == 0).any(axis=1), index=df.index), "duplicate_numbers_in_draw")

    good = reason == ""
    holidays = df.loc[hol_mask, ["Date", "Promotionday"]].sort_values("Date").reset_index(drop=True)
    rejected = df.loc[(reason != "") & ~hol_mask].assign(reason=reason)

    out = df.loc[good, ["Date", "Promotionday"]].copy()
    for c in NUMBER_COLS:
        out[c] = nums.loc[good, c].astype(int)

    # 5. exact duplicates (same date and same numbers)
    dup = out.duplicated(subset=["Date"] + NUMBER_COLS, keep="first")
    n_dup = int(dup.sum())
    if n_dup:
        log.warning("  clean | exact duplicate rows dropped: %d", n_dup)
    out = out[~dup]
    conflicts = int(out.duplicated(subset="Date", keep=False).sum())
    if conflicts:
        log.warning("  clean | %d rows share a date with a DIFFERENT result (kept; inspect manually)", conflicts)

    out = out.sort_values("Date", kind="mergesort").reset_index(drop=True)

    report = {
        "rows_raw": int(len(raw)),
        "dropped": counts,
        "exact_duplicates_dropped": n_dup,
        "same_date_conflicts": conflicts,
        "holiday_rows": int(len(holidays)),
        "rows_clean": int(len(out)),
        "date_min": str(out["Date"].min().date()) if len(out) else None,
        "date_max": str(out["Date"].max().date()) if len(out) else None,
    }
    log.info(
        "Cleaning done: %d raw -> %d clean draws | %d holiday rows set aside | range %s .. %s",
        report["rows_raw"], report["rows_clean"], report["holiday_rows"], report["date_min"], report["date_max"],
    )
    return out, holidays, rejected, report


# --------------------------------------------------------------------------- #
# Features (calendar + holiday shift is ONE feature among several)
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DrawIndex"] = np.arange(len(df))
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # Mon=0
    df["DayName"] = df["Date"].dt.day_name()
    df["DaysSincePrev"] = df["Date"].diff().dt.days.fillna(0).astype(int)

    # HolidayShift = 1 if a blank holiday row falls between the previous draw and this one
    hol = np.sort(holidays["Date"].to_numpy(dtype="datetime64[ns]"))
    dates = df["Date"].to_numpy(dtype="datetime64[ns]")
    prev = np.concatenate([dates[:1], dates[:-1]])
    n_between = np.searchsorted(hol, dates, side="right") - np.searchsorted(hol, prev, side="right")
    df["HolidayShift"] = (n_between > 0).astype(int)
    df.loc[0, "HolidayShift"] = 0

    log.info(
        "Calendar features added: %s | HolidayShift=1 on %d draws (%.1f%%)",
        [c for c in df.columns if c not in REQUIRED_COLS],
        int(df["HolidayShift"].sum()), 100 * df["HolidayShift"].mean(),
    )
    return df


# --------------------------------------------------------------------------- #
# Tensor
# --------------------------------------------------------------------------- #
def build_tensor(df: pd.DataFrame) -> np.ndarray:
    """(N, 90) uint8; X[t, b-1] = 1 iff ball b was drawn at draw t."""
    nums = df[NUMBER_COLS].to_numpy(dtype=int)
    X = np.zeros((len(df), N_BALLS), dtype=np.uint8)
    X[np.repeat(np.arange(len(df)), N_DRAWN), (nums - 1).ravel()] = 1
    assert (X.sum(axis=1) == N_DRAWN).all(), "tensor rows must each contain exactly 5 ones"
    log.info("Tensor built: shape=%s dtype=%s (%.2f MB)", X.shape, X.dtype, X.nbytes / 1e6)
    return X


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def load_game(key: str, data_dir, out_dir) -> GameData:
    name, fname = GAMES[key]
    path = Path(data_dir) / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. CSVs in {Path(data_dir).resolve()}: "
                                f"{sorted(p.name for p in Path(data_dir).glob('*.csv'))}")
    log.info("=== Loading game %s: %s (%s) ===", key, name, fname)

    raw, _ = _read_raw(path)
    out, holidays, rejected, report = clean(raw)
    if len(out) == 0:
        raise ValueError("No valid draws left after cleaning.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(rejected):
        rp = out_dir / f"rejected_rows_{name.lower()}.csv"
        rejected.to_csv(rp, index=False)
        log.info("Rejected rows saved for audit: %s", rp)

    df = add_calendar_features(out, holidays)
    X = build_tensor(df)

    per_dow = df["DayName"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).fillna(0).astype(int)
    log.info("Draws per weekday: %s", per_dow.to_dict())
    counts = X.sum(axis=0)
    log.info("Ball frequency: min=%d (ball %d) max=%d (ball %d) expected=%.1f",
             counts.min(), counts.argmin() + 1, counts.max(), counts.argmax() + 1, len(df) * N_DRAWN / N_BALLS)
    if len(df) < 500:
        log.warning("Only %d clean draws: most diagnostics will have very low power.", len(df))

    return GameData(key, name, path, df, X, holidays, report)
