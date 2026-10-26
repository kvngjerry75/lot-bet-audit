#!/usr/bin/env python3
"""
Stage 2: walk-forward backtest, models, calibration "digital twin", report.

Input : a clean CSV with Date, Promotionday, Number1..Number5 (drawn order),
        optional RealDay, WasShifted.  Rows with blank numbers are treated as
        holidays (calendar feature only).  Newest-first files are fine.
Run   : python stage2_powerhouse.py --csv main_clean.csv
Colab : !pip -q install lightgbm scipy ; !python stage2_powerhouse.py --csv main_clean.csv
        (torch is preinstalled on Colab; choose a GPU runtime for the LSTM)
Quick : add --quick for a smoke test (small backtest, few epochs)

Design rules (so the result can be trusted):
  * Every prediction for draw t uses draws < t only (a leakage self-test runs first).
  * Models are scored against a uniform reference and a seeded random-pick control.
  * Exact binomial tests + Benjamini-Hochberg q-values over ALL tests reported.
  * The twin is a calibration check, not a "sure numbers" generator.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_BALLS, N_DRAWN = 90, int(__import__("os").environ.get("ASTRO_K", "5"))
NUM_COLS = [f"Number{i}" for i in range(1, N_DRAWN + 1)]
BASE = N_DRAWN / N_BALLS
log = logging.getLogger("nla.stage2")
T0 = time.time()

CAL_NAMES = ["dow0", "dow1", "dow2", "dow3", "dow4", "dow5", "dow6", "month_sin", "month_cos",
             "day_of_month", "days_since_prev", "was_shifted", "holiday_shift"]
FEATURE_NAMES = ["gap", "last_pos", "rate10", "rate30", "rate100", "rate_all", "rate_same_dow",
                 "pos1_rate", "pos5_rate", "ball_id", "last_digit",
                 "dow", "month", "day_of_month", "days_since_prev", "was_shifted", "holiday_shift"]


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def setup_logging(out_dir: Path, tag: str) -> Path:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"stage2_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    root = logging.getLogger("nla")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(ch)
    root.addHandler(fh)
    root.propagate = False
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2: models + walk-forward backtest + calibration twin")
    ap.add_argument("--csv", required=True, help="clean CSV (e.g. main_clean.csv)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--backtest", type=int, default=1000, help="number of final draws to walk-forward test")
    ap.add_argument("--min-train", type=int, default=500, help="minimum draws before the first test draw")
    ap.add_argument("--gbm-every", type=int, default=100, help="refit GBM every N test draws")
    ap.add_argument("--lstm-every", type=int, default=250, help="refit LSTM every N test draws")
    ap.add_argument("--train-window", type=int, default=2000, help="max recent draws used per refit")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--half-life", type=float, default=500.0, help="decay half-life (draws) of frequency experts")
    ap.add_argument("--eta", type=float, default=1.0, help="Hedge learning rate (1.0 = Bayesian mixture)")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--twin-draws", type=int, default=100_000)
    ap.add_argument("--twin-reps", type=int, default=200, help="replicate histories for twin calibration p-values")
    ap.add_argument("--next-date", default=None, help="date of the draw to predict (default: last date + 1 day)")
    ap.add_argument("--next-shifted", type=int, default=0, help="1 if the next draw is a shifted draw")
    ap.add_argument("--next-holiday-shift", type=int, default=0)
    ap.add_argument("--time-cap-hours", type=float, default=3.0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="smoke-test settings")
    args, _ = ap.parse_known_args(argv)  # tolerate Jupyter/Colab extra args
    if args.quick:
        args.backtest = min(args.backtest, 200)
        args.min_train = min(args.min_train, 300)
        args.gbm_every, args.lstm_every = 100, 100
        args.epochs, args.train_window = 3, 800
        args.twin_draws, args.twin_reps = 20_000, 30
    args.warmup = 100
    return args


def over_cap(args) -> bool:
    return (time.time() - T0) > args.time_cap_hours * 3600


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Data:
    n: int                    # number of REAL draws; arrays have n+1 rows (last row = the draw to predict)
    dates: pd.DatetimeIndex
    P: np.ndarray             # (n+1, 5) ball ids 0..89 in DRAWN order (last row unused)
    X: np.ndarray             # (n+1, 90) float32 inclusion matrix (last row zeros)
    cal: np.ndarray           # (n+1, 13) float32 calendar features
    dow: np.ndarray
    month: np.ndarray
    dom: np.ndarray
    dsp: np.ndarray
    shifted: np.ndarray
    hshift: np.ndarray
    info: dict


_BLANK_RE = r"(?i)^\s*(nan|none|null|na|n/a|-|--|holiday)?\s*$"


def load_data(path: str, out_dir: Path, next_date=None, next_shifted=0, next_hshift=0) -> Data:
    log.info("Reading %s", path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in ["Date"] + NUM_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {list(raw.columns)}")
    for c in raw.columns:
        raw[c] = raw[c].astype(str).str.strip()
    n_raw = len(raw)
    log.info("Raw rows: %d | columns: %s", n_raw, list(raw.columns))

    d = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    retry = d.isna()
    if retry.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d.loc[retry] = pd.to_datetime(raw.loc[retry, "Date"], errors="coerce")
    nums_s = raw[NUM_COLS].replace(_BLANK_RE, "", regex=True)
    nums = nums_s.apply(pd.to_numeric, errors="coerce")
    all_blank = (nums_s == "").all(axis=1)

    reason = pd.Series("", index=raw.index, dtype=object)

    def flag(mask, label):
        m = mask & (reason == "")
        reason[m] = label
        log.info("  clean | %-30s %6d rows", label, int(m.sum()))

    flag(d.isna(), "bad_date")
    hol = all_blank & (reason == "")
    flag(hol, "holiday_blank")
    flag(nums.isna().any(axis=1), "partial_or_nonnumeric")
    nv = nums.fillna(1)
    flag(((nv % 1) != 0).any(axis=1), "non_integer")
    flag(((nv < 1) | (nv > N_BALLS)).any(axis=1), "out_of_range")
    flag(pd.Series((np.diff(np.sort(nv.to_numpy(), axis=1), axis=1) == 0).any(axis=1), index=raw.index),
         "duplicate_in_draw")

    good = reason == ""
    rejected = raw.loc[(reason != "") & (reason != "holiday_blank")].assign(reason=reason)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(rejected):
        rp = out_dir / "stage2_rejected_rows.csv"
        rejected.to_csv(rp, index=False)
        log.info("Rejected rows saved: %s", rp)
    hol_dates = np.sort(d[hol].to_numpy(dtype="datetime64[ns]"))

    df = raw.loc[good].copy()
    df["Date"] = d[good]
    for c in NUM_COLS:
        df[c] = nums.loc[good, c].astype(int)
    if len(df) == 0:
        raise ValueError("no valid draws")
    if df["Date"].iloc[0] > df["Date"].iloc[-1]:
        log.info("File is newest-first: reversing before the stable date sort")
        df = df.iloc[::-1]
    df = df.sort_values("Date", kind="mergesort")
    dup = df.duplicated(subset=["Date"] + NUM_COLS, keep="first")
    if dup.any():
        log.warning("  clean | exact duplicate rows dropped: %d", int(dup.sum()))
    df = df[~dup].reset_index(drop=True)
    n = len(df)

    # shifted flag
    if "WasShifted" in df.columns:
        shifted = pd.to_numeric(df["WasShifted"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
        src = "WasShifted column"
    elif "Promotionday" in df.columns:
        shifted = (df["Date"].dt.day_name() != df["Promotionday"].str.title()).astype(int).to_numpy()
        src = "derived (weekday of Date != Promotionday)"
    else:
        shifted = np.zeros(n, dtype=int)
        src = "absent (all zeros)"
    log.info("Shifted draws (%s): %d of %d (%.2f%%)", src, int(shifted.sum()), n, 100 * shifted.mean())
    if shifted.sum() == 0:
        log.warning("WasShifted is constant 0: the feature carries no information on this file")
    if "RealDay" in df.columns:
        mism = int((df["Date"].dt.day_name() != df["RealDay"].str.title()).sum())
        log.info("RealDay vs weekday(Date) mismatches: %d", mism)
    if "Promotionday" in df.columns:
        diff = int((df["Date"].dt.day_name() != df["Promotionday"].str.title()).sum())
        log.info("Rows where Promotionday != weekday(Date): %d", diff)

    P = df[NUM_COLS].to_numpy(dtype=int) - 1
    asc = float((np.diff(P, axis=1) > 0).all(axis=1).mean())
    log.info("Rows with numbers already ascending: %.2f%% (random drawn order would be ~0.83%%)", 100 * asc)
    if asc > 0.5:
        log.warning("Numbers look SORTED, so positional (drawn-order) features carry no information")

    # extended arrays: one extra row for the draw we want to predict
    last = df["Date"].iloc[-1]
    nd = pd.Timestamp(next_date) if next_date else last + pd.Timedelta(days=1)
    dates = pd.DatetimeIndex(list(df["Date"]) + [nd])
    N1 = n + 1
    P1 = np.vstack([P, np.zeros((1, N_DRAWN), dtype=int)])
    X = np.zeros((N1, N_BALLS), dtype=np.float32)
    X[np.repeat(np.arange(n), N_DRAWN), P.ravel()] = 1
    assert (X[:n].sum(1) == N_DRAWN).all()
    dsp = np.concatenate([[1], np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int)])
    hv = hol_dates.astype("datetime64[ns]")
    dn = dates.to_numpy(dtype="datetime64[ns]")
    prev = np.concatenate([dn[:1], dn[:-1]])
    hshift = ((np.searchsorted(hv, dn, side="right") - np.searchsorted(hv, prev, side="right")) > 0).astype(int)
    hshift[0] = 0
    hshift[-1] = int(next_hshift)
    sh = np.concatenate([shifted, [int(next_shifted)]])
    dow, month, dom = dates.dayofweek.to_numpy(), dates.month.to_numpy(), dates.day.to_numpy()
    cal = np.zeros((N1, len(CAL_NAMES)), dtype=np.float32)
    cal[np.arange(N1), dow] = 1
    cal[:, 7] = np.sin(2 * np.pi * (month - 1) / 12)
    cal[:, 8] = np.cos(2 * np.pi * (month - 1) / 12)
    cal[:, 9] = dom / 31
    cal[:, 10] = np.clip(dsp, 0, 7) / 7
    cal[:, 11] = sh
    cal[:, 12] = hshift
    info = {"rows_raw": n_raw, "rows_clean": n, "holiday_rows": int(hol.sum()),
            "date_min": str(df["Date"].min().date()), "date_max": str(last.date()),
            "next_date": str(nd.date()), "shifted_rows": int(shifted.sum()),
            "holiday_shift_draws": int(hshift[:n].sum()), "ascending_row_share": asc}
    log.info("Clean draws: %d | %s .. %s | holiday rows set aside: %d | HolidayShift=1 on %d draws",
             n, info["date_min"], info["date_max"], info["holiday_rows"], info["holiday_shift_draws"])
    log.info("Predicting draw dated %s (%s)", info["next_date"], nd.day_name())
    c = X[:n].sum(0)
    log.info("Ball frequency: min=%d (ball %d) max=%d (ball %d) expected=%.1f",
             c.min(), c.argmin() + 1, c.max(), c.argmax() + 1, n * BASE)
    return Data(n, dates, P1, X, cal, dow, month, dom, dsp, sh, hshift, info)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_ball_features(d: Data) -> np.ndarray:
    """(n+1, 90, F). Row t uses draws < t only, plus the known calendar of draw t."""
    N1, n, X, P = len(d.X), d.n, d.X, d.P
    t_idx = np.arange(N1)
    F = np.zeros((N1, N_BALLS, len(FEATURE_NAMES)), dtype=np.float32)
    C = np.vstack([np.zeros((1, N_BALLS)), np.cumsum(X, axis=0)])  # C[t] = sum of X[:t]
    last = np.full(N_BALLS, -1)
    lp = np.zeros(N_BALLS)
    for t in range(N1):
        F[t, :, 0] = np.where(last < 0, 300, np.minimum(t - last, 300))
        F[t, :, 1] = lp
        if t < n:
            last[P[t]] = t
            lp[P[t]] = np.arange(1, N_DRAWN + 1)
    for col, w in ((2, 10), (3, 30), (4, 100)):
        lo = np.maximum(0, t_idx - w)
        r = (C[t_idx] - C[lo]) / np.maximum(t_idx - lo, 1)[:, None]
        r[0] = BASE
        F[:, :, col] = r
    r = C[t_idx] / np.maximum(t_idx, 1)[:, None]
    r[0] = BASE
    F[:, :, 5] = r
    for k in range(7):
        m = d.dow == k
        Ck = np.vstack([np.zeros((1, N_BALLS)), np.cumsum(X * m[:, None], axis=0)])
        cnt = np.concatenate([[0], np.cumsum(m)])
        sel = np.where(m)[0]
        F[sel, :, 6] = Ck[sel] / np.maximum(cnt[sel], 1)[:, None]
    for col, j in ((7, 0), (8, 4)):
        Q = np.zeros((N1, N_BALLS))
        Q[np.arange(n), P[:n, j]] = 1
        Cq = np.vstack([np.zeros((1, N_BALLS)), np.cumsum(Q, axis=0)])
        rr = Cq[t_idx] / np.maximum(t_idx, 1)[:, None]
        rr[0] = 1 / N_BALLS
        F[:, :, col] = rr
    F[:, :, 9] = (np.arange(N_BALLS) / 89.0)[None, :]
    F[:, :, 10] = (np.arange(N_BALLS) % 10)[None, :]
    for col, arr in ((11, d.dow), (12, d.month), (13, d.dom), (14, d.dsp), (15, d.shifted), (16, d.hshift)):
        F[:, :, col] = np.asarray(arr, dtype=np.float32)[:, None]
    return F


def freq_experts(d: Data, half_life: float, k_macro=200.0, k_micro=100.0):
    """Leak-free smoothed frequency experts. Returns macro (N1,90), micro (N1,90), pos (N1,5,90)."""
    N1, n = len(d.X), d.n
    dec = 0.5 ** (1.0 / half_life)
    dec_d = 0.5 ** (7.0 / half_life)
    E, W = np.zeros(N_BALLS), 0.0
    Ed, Wd = np.zeros((7, N_BALLS)), np.zeros(7)
    Ep = np.zeros((N_DRAWN, N_BALLS))
    macro = np.empty((N1, N_BALLS))
    micro = np.empty((N1, N_BALLS))
    pos = np.empty((N1, N_DRAWN, N_BALLS))
    for t in range(N1):
        pm = (E + k_macro * BASE) / (W + k_macro)
        pm *= N_DRAWN / pm.sum()
        k = d.dow[t]
        pd_ = (Ed[k] + k_micro * pm) / (Wd[k] + k_micro)
        pd_ *= N_DRAWN / pd_.sum()
        macro[t], micro[t] = pm, pd_
        pp = (Ep + 20.0 / N_BALLS) / (W + 20.0)
        pos[t] = pp / pp.sum(1, keepdims=True)
        if t < n:
            E = dec * E + d.X[t]
            W = dec * W + 1
            Ed[k] = dec_d * Ed[k] + d.X[t]
            Wd[k] = dec_d * Wd[k] + 1
            Ep = dec * Ep
            Ep[np.arange(N_DRAWN), d.P[t]] += 1
    return macro, micro, pos


def leakage_check(d: Data, half_life: float):
    """Features/experts for rows <= t0 must not change when all draws >= t0 are replaced by noise."""
    t0 = d.n // 2
    rng = np.random.default_rng(0)
    d2 = Data(**{**d.__dict__})
    d2.P, d2.X = d.P.copy(), d.X.copy()
    for t in range(t0, d.n):
        d2.P[t] = rng.permutation(N_BALLS)[:N_DRAWN]
        d2.X[t] = 0
        d2.X[t, d2.P[t]] = 1
    F1, F2 = build_ball_features(d), build_ball_features(d2)
    a = freq_experts(d, half_life)
    b = freq_experts(d2, half_life)
    ok = np.allclose(F1[:t0 + 1], F2[:t0 + 1]) and all(np.allclose(x[:t0 + 1], y[:t0 + 1]) for x, y in zip(a, b))
    log.info("Leakage self-test (features/experts at t<=%d unchanged by future draws): %s", t0,
             "PASSED" if ok else "FAILED")
    if not ok:
        raise RuntimeError("leakage detected in feature construction")


# --------------------------------------------------------------------------- #
# Gradient boosting expert
# --------------------------------------------------------------------------- #
def _norm_inc(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    p = p * (N_DRAWN / p.sum(1, keepdims=True))
    return np.clip(p, 1e-6, 1 - 1e-6)


def fit_gbm(F, X, t_end, args):
    nf = F.shape[2]
    t0 = max(args.warmup, t_end - args.train_window)
    n_val = max(int(0.15 * (t_end - t0)), 50)
    split = t_end - n_val
    Xa, ya = F[t0:split].reshape(-1, nf), X[t0:split].reshape(-1).astype(int)
    Xv, yv = F[split:t_end].reshape(-1, nf), X[split:t_end].reshape(-1).astype(int)
    base_ll = -(yv * np.log(BASE) + (1 - yv) * np.log(1 - BASE)).mean()
    t_fit = time.time()
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                                   random_state=args.seed, n_jobs=-1, verbose=-1)
        model.fit(Xa, ya, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
        best = int(model.best_iteration_ or model.n_estimators)
        pv = np.clip(model.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6)
        vll = -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean()
        imp = model.booster_.feature_importance("gain")
        imp = imp / max(imp.sum(), 1e-12)
        kind = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=10.0, min_samples_leaf=300,
                                               early_stopping=True, validation_fraction=0.15,
                                               random_state=args.seed)
        model.fit(np.vstack([Xa, Xv]), np.concatenate([ya, yv]))
        best, vll, imp, kind = int(model.n_iter_), float("nan"), None, "sklearn-hgb"
    log.info("  GBM[%s] fit t_end=%d rows=%d best_iter=%d val_logloss=%.5f (constant-rate=%.5f, %s) %.1fs",
             kind, t_end, len(ya), best, vll, base_ll,
             "better" if vll < base_ll else "NOT better than constant", time.time() - t_fit)
    return model, imp


def gbm_predict(model, F, a, b):
    nf = F.shape[2]
    p = model.predict_proba(F[a:b].reshape(-1, nf))[:, 1].reshape(b - a, N_BALLS)
    return _norm_inc(p)


def walk_forward_gbm(F, X, a0, n, args):
    out = np.empty((n - a0, N_BALLS))
    model, imp = None, None
    blocks = list(range(a0, n, args.gbm_every))
    log.info("GBM walk-forward: %d refits, %d test draws", len(blocks), n - a0)
    for i, a in enumerate(blocks):
        b = min(a + args.gbm_every, n)
        if model is None or not over_cap(args):
            model, imp = fit_gbm(F, X, a, args)
        else:
            log.warning("  time cap reached: reusing previous GBM")
        out[a - a0:b - a0] = gbm_predict(model, F, a, b)
        log.info("  GBM block %d/%d done (draws %d..%d) elapsed %.0fs", i + 1, len(blocks), a, b - 1, time.time() - T0)
    return out


# --------------------------------------------------------------------------- #
# LSTM expert (inclusion + per-position heads, ordered draws)
# --------------------------------------------------------------------------- #
def build_seq_inputs(d: Data) -> np.ndarray:
    N1, n = len(d.X), d.n
    S = np.zeros((N1, 3 * N_BALLS + d.cal.shape[1]), dtype=np.float32)
    S[:, :N_BALLS] = d.X
    r = np.arange(n)
    S[r, N_BALLS + d.P[:n, 0]] = 1            # first ball out
    S[r, 2 * N_BALLS + d.P[:n, 4]] = 1        # last ball out
    S[:, 3 * N_BALLS:] = d.cal
    return S


def torch_device(args):
    import torch
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def fit_lstm(S, d, t_end, args, dev):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.manual_seed(args.seed)
    L = args.seq_len
    t_lo = max(L, t_end - args.train_window)
    idx = np.arange(t_lo, t_end)
    n_val = max(int(0.15 * len(idx)), 50)
    tr, va = idx[:-n_val], idx[-n_val:]
    St = torch.from_numpy(S).to(dev)
    Ct = torch.from_numpy(d.cal).to(dev)
    Xt = torch.from_numpy(d.X).to(dev)
    Pt = torch.from_numpy(d.P).long().to(dev)
    ar = torch.arange(-L, 0, device=dev)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(S.shape[1], args.hidden, batch_first=True)
            s.drop = nn.Dropout(0.3)
            s.mid = nn.Sequential(nn.Linear(args.hidden + d.cal.shape[1], args.hidden), nn.GELU(), nn.Dropout(0.3))
            s.inc = nn.Linear(args.hidden, N_BALLS)
            s.pos = nn.Linear(args.hidden, N_DRAWN * N_BALLS)
            for m in (s.inc, s.pos):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            with torch.no_grad():
                s.inc.bias.fill_(math.log(BASE / (1 - BASE)))  # start exactly at uniform

        def forward(s, seq, cal):
            o, _ = s.lstm(seq)
            h = s.mid(torch.cat([s.drop(o[:, -1]), cal], 1))
            return s.inc(h), s.pos(h).view(-1, N_DRAWN, N_BALLS)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    def batch(ix):
        ix = torch.as_tensor(ix, device=dev)
        return St[ix[:, None] + ar], Ct[ix], Xt[ix], Pt[ix]

    def losses(zi, zp, y, p):
        li = Fn.binary_cross_entropy_with_logits(zi, y, reduction="none").sum(1).mean()
        lpos = Fn.cross_entropy(zp.reshape(-1, N_BALLS), p.reshape(-1)) * N_DRAWN
        return li, lpos

    base_i = -(N_DRAWN * math.log(BASE) + (N_BALLS - N_DRAWN) * math.log(1 - BASE))
    base_p = N_DRAWN * math.log(N_BALLS)
    best, best_state, bad = float("inf"), None, 0
    t_fit = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = np.random.default_rng(args.seed + ep).permutation(tr)
        tl = 0.0
        for s in range(0, len(perm), 128):
            seq, c, y, p = batch(perm[s:s + 128])
            zi, zp = net(seq, c)
            li, lpos = losses(zi, zp, y, p)
            loss = li + 0.5 * lpos
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(perm[s:s + 128])
        net.eval()
        with torch.no_grad():
            seq, c, y, p = batch(va)
            zi, zp = net(seq, c)
            vi, vp = (float(x) for x in losses(zi, zp, y, p))
        vt = vi + 0.5 * vp
        log.info("    LSTM epoch %2d/%d train=%.4f | val inc=%.4f (uniform %.4f) pos=%.4f (uniform %.4f)",
                 ep + 1, args.epochs, tl / len(perm), vi, base_i, vp, base_p)
        if vt < best - 1e-5:
            best, bad = vt, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                log.info("    LSTM early stop at epoch %d", ep + 1)
                break
    net.load_state_dict(best_state)
    net.eval()
    log.info("  LSTM fit t_end=%d samples=%d best_val_total=%.4f (uniform %.4f) device=%s %.1fs",
             t_end, len(tr), best, base_i + 0.5 * base_p, dev, time.time() - t_fit)
    return net, St, Ct, ar


def lstm_predict(bundle, a, b, dev):
    import torch
    net, St, Ct, ar = bundle
    pi, pp = [], []
    with torch.no_grad():
        for s in range(a, b, 512):
            ix = torch.arange(s, min(s + 512, b), device=dev)
            zi, zp = net(St[ix[:, None] + ar], Ct[ix])
            pi.append(torch.sigmoid(zi).cpu().numpy())
            pp.append(torch.softmax(zp, -1).cpu().numpy())
    return _norm_inc(np.vstack(pi).astype(float)), np.vstack(pp).astype(float)


def walk_forward_lstm(S, d, a0, n, args, dev):
    inc = np.empty((n - a0, N_BALLS))
    pos = np.empty((n - a0, N_DRAWN, N_BALLS))
    blocks = list(range(a0, n, args.lstm_every))
    log.info("LSTM walk-forward: %d refits, %d test draws, device=%s", len(blocks), n - a0, dev)
    bundle = None
    for i, a in enumerate(blocks):
        b = min(a + args.lstm_every, n)
        if bundle is None or not over_cap(args):
            bundle = fit_lstm(S, d, a, args, dev)
        else:
            log.warning("  time cap reached: reusing previous LSTM")
        inc[a - a0:b - a0], pos[a - a0:b - a0] = lstm_predict(bundle, a, b, dev)
        log.info("  LSTM block %d/%d done (draws %d..%d) elapsed %.0fs", i + 1, len(blocks), a, b - 1, time.time() - T0)
    return inc, pos


# --------------------------------------------------------------------------- #
# Ensemble + statistics
# --------------------------------------------------------------------------- #
def hedge(LL: np.ndarray, eta: float):
    """LL: (E, n) per-draw log-likelihood per expert. Returns weights used BEFORE each draw and the final weights."""
    E, n = LL.shape
    cum, W = np.zeros(E), np.empty((n, E))
    for t in range(n):
        w = np.exp(eta * (cum - cum.max()))
        W[t] = w / w.sum()
        cum += LL[:, t]
    w = np.exp(eta * (cum - cum.max()))
    return W, w / w.sum()


def bh_qvalues(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def evaluate(d: Data, a0, inc, posd, args, rng):
    n = d.n
    Pbt, Xbt = d.P[a0:n], d.X[a0:n].astype(float)
    nbt = n - a0
    ar = np.arange(nbt)[:, None]
    res = {"n_backtest": nbt, "tests": [], "experts": {}, "hits": {}, "position": {}}
    ll, bce = {}, {}

    def score(name, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        q = p / p.sum(1, keepdims=True)
        ll[name] = np.log(q[ar, Pbt]).sum(1)
        bce[name] = -(Xbt * np.log(p) + (1 - Xbt) * np.log1p(-p)).sum(1)

    for k, v in inc.items():
        score(k, v)
    mix_names = list(inc)
    LL = np.vstack([ll[k] for k in mix_names])
    Wp, wf = hedge(LL, args.eta)
    inc["ensemble"] = np.einsum("te,etb->tb", Wp, np.stack([inc[k] for k in mix_names]))
    score("ensemble", inc["ensemble"])
    res["ensemble_weights_final"] = dict(zip(mix_names, map(float, wf)))
    res["ensemble_weights_mean"] = dict(zip(mix_names, map(float, Wp.mean(0))))
    log.info("Ensemble (Hedge) mean weights over backtest: %s",
             {k: round(v, 3) for k, v in res["ensemble_weights_mean"].items()})

    def add_test(name, p, **extra):
        res["tests"].append({"name": name, "p": float(p), **extra})

    # log-score gain & 90-ball BCE vs uniform
    for k in inc:
        g = ll[k] - ll["uniform"]
        bd = bce["uniform"] - bce[k]
        row = {"gain_per_draw": float(g.mean()), "bce_mean": float(bce[k].mean()),
               "bce_uniform": float(bce["uniform"].mean()), "bce_gain_per_draw": float(bd.mean())}
        if k != "uniform":
            t1 = stats.ttest_1samp(g, 0.0, alternative="greater")
            t2 = stats.ttest_1samp(bd, 0.0, alternative="greater")
            row.update({"gain_se": float(g.std(ddof=1) / math.sqrt(nbt)), "gain_p": float(t1.pvalue),
                        "bce_gain_p": float(t2.pvalue)})
            add_test(f"{k}: log-score gain vs uniform", t1.pvalue, value=row["gain_per_draw"])
            add_test(f"{k}: 90-ball BCE gain vs uniform", t2.pvalue, value=row["bce_gain_per_draw"])
        res["experts"][k] = row

    # hits (5-ball pick = top-5), mean-hit z-test
    mu0 = N_DRAWN * N_DRAWN / N_BALLS
    var0 = N_DRAWN * BASE * (1 - BASE) * (N_BALLS - N_DRAWN) / (N_BALLS - 1)

    def hits_for(p):
        top = np.argsort(-p, axis=1)[:, :5]
        return Xbt[ar, top].sum(1).astype(int)

    rnd = np.argsort(rng.random((nbt, N_BALLS)), axis=1)[:, :5]
    h_rand = Xbt[ar, rnd].sum(1).astype(int)
    hit_sets = {k: hits_for(inc[k]) for k in inc if k != "uniform"}
    hit_sets["random"] = h_rand
    pmf = stats.hypergeom.pmf(np.arange(6), N_BALLS, N_DRAWN, N_DRAWN)
    for k, h in hit_sets.items():
        z = (h.mean() - mu0) / math.sqrt(var0 / nbt)
        pz = float(stats.norm.sf(z))
        res["hits"][k] = {"mean_hits": float(h.mean()), "expected_mean": mu0, "z": float(z), "p": pz,
                          "exact": {str(j): int((h == j).sum()) for j in range(6)},
                          "at_least": {str(j): int((h >= j).sum()) for j in range(2, 6)}}
        if k != "random":
            add_test(f"{k}: mean hits (top-5 pick)", pz, value=float(h.mean()))
    res["hits"]["expected_exact"] = {str(j): float(nbt * pmf[j]) for j in range(6)}
    ens_h = hit_sets["ensemble"]
    for j in range(2, 6):
        p0 = float(stats.hypergeom.sf(j - 1, N_BALLS, N_DRAWN, N_DRAWN))
        for who, h in (("ensemble", ens_h), ("random", h_rand)):
            k = int((h >= j).sum())
            pv = float(stats.binomtest(k, nbt, p0, alternative="greater").pvalue)
            res["hits"][who].setdefault("binom_p_at_least", {})[str(j)] = pv
            res["hits"][who].setdefault("p0_at_least", {})[str(j)] = p0
            if who == "ensemble":
                add_test(f"ensemble: P(hit>={j}) exact binomial", pv, observed=k, expected=nbt * p0)

    # top-2 pair (both numbers appear among the 5 drawn)
    p0 = N_DRAWN * (N_DRAWN - 1) / (N_BALLS * (N_BALLS - 1))
    top2 = np.argsort(-inc["ensemble"], axis=1)[:, :2]
    rnd2 = rnd[:, :2]
    pair = {}
    for who, tp in (("ensemble", top2), ("random", rnd2)):
        k = int((Xbt[ar, tp].sum(1) == 2).sum())
        pair[who] = {"both_hit": k, "expected": nbt * p0,
                     "p": float(stats.binomtest(k, nbt, p0, alternative="greater").pvalue)}
    res["top2_pair"] = pair
    add_test("ensemble: top-2 pair both drawn (exact binomial)", pair["ensemble"]["p"],
             observed=pair["ensemble"]["both_hit"], expected=nbt * p0)

    # positional heads (drawn order)
    pll = {}
    for k, v in posd.items():
        pll[k] = np.log(np.clip(v, 1e-12, None)[ar, np.arange(N_DRAWN)[None, :], Pbt]).sum(1)
    pnames = list(posd)
    _, pwf = hedge(np.vstack([pll[k] for k in pnames]), args.eta)
    res["position_weights_final"] = dict(zip(pnames, map(float, pwf)))
    for k in pnames:
        if k == "uniform":
            continue
        g = pll[k] - pll["uniform"]
        t = stats.ttest_1samp(g, 0.0, alternative="greater")
        top1 = [int((posd[k][:, j, :].argmax(1) == Pbt[:, j]).sum()) for j in range(N_DRAWN)]
        res["position"][k] = {"gain_per_draw": float(g.mean()), "p": float(t.pvalue), "top1_hits": top1,
                              "top1_expected": nbt / N_BALLS}
        add_test(f"position[{k}]: log-score gain vs uniform", t.pvalue, value=float(g.mean()))
        for j in range(N_DRAWN):
            add_test(f"position[{k}] ball#{j + 1}: top-1 hit", stats.binomtest(top1[j], nbt, 1 / N_BALLS,
                                                                                alternative="greater").pvalue,
                     observed=top1[j], expected=nbt / N_BALLS)

    ps = np.array([t["p"] for t in res["tests"]])
    q = bh_qvalues(ps)
    for t, qq in zip(res["tests"], q):
        t["q_bh"] = float(qq)
    res["n_tests"] = len(ps)
    res["bonferroni_alpha"] = 0.05 / len(ps)
    res["survivors_bh"] = [t["name"] for t in res["tests"] if t["q_bh"] < 0.05]
    res["survivors_bonferroni"] = [t["name"] for t in res["tests"] if t["p"] < res["bonferroni_alpha"]]
    res["min_p"] = float(ps.min())
    per_draw = {"date": d.dates[a0:n].strftime("%Y-%m-%d")}
    for k in ll:
        per_draw[f"gain_{k}"] = ll[k] - ll["uniform"]
    for k, h in hit_sets.items():
        per_draw[f"hits_{k}"] = h
    return res, inc, wf, pwf, mix_names, pnames, pd.DataFrame(per_draw)


# --------------------------------------------------------------------------- #
# Digital twin = calibration check
# --------------------------------------------------------------------------- #
def simulate_machine(w, pos_factor, m, rng, chunk=10_000):
    """Sequential weighted draws without replacement; order preserved (Gumbel-max trick)."""
    out = np.empty((m, N_DRAWN), dtype=np.int16)
    lw = np.log(np.clip(w, 1e-12, None))
    lpf = np.log(np.clip(pos_factor, 1e-12, None))
    for s in range(0, m, chunk):
        c = min(chunk, m - s)
        taken = np.zeros((c, N_BALLS), bool)
        for j in range(N_DRAWN):
            u = rng.random((c, N_BALLS), dtype=np.float32)
            g = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-7)))
            sc = g + (lw + lpf[j])[None, :].astype(np.float32)
            sc[taken] = -np.inf
            pick = sc.argmax(1)
            out[s:s + c, j] = pick
            taken[np.arange(c), pick] = True
    return out


def fit_machine(p_target, pos_factor, rng, iters=8, m=20_000):
    w = np.clip(p_target, 1e-6, None).copy()
    err = float("nan")
    for it in range(iters):
        sims = simulate_machine(w, pos_factor, m, rng)
        pi = np.bincount(sims.ravel(), minlength=N_BALLS) / m
        err = float(np.abs(pi - p_target).max())
        log.debug("  twin fit iter %d: max |pi - target| = %.5f", it + 1, err)
        w *= (p_target / np.clip(pi, 1e-6, None)) ** 0.8
    log.info("Virtual machine weights fitted (max inclusion error %.4f; MC noise floor ~%.4f)",
             err, 2 * math.sqrt(BASE * (1 - BASE) / m))
    return w


_IU = np.triu_indices(N_DRAWN, 1)


def twin_stats(P, ex):
    n = len(P)
    cnt = np.bincount(P.ravel(), minlength=N_BALLS)
    e = n * ex["incl"]
    s_freq = float(((cnt - e) ** 2 / e).sum())
    c1 = np.bincount(P[:, 0], minlength=N_BALLS)
    e1 = n * ex["pos1"]
    s_pos1 = float(((c1 - e1) ** 2 / e1).sum())
    Xb = np.zeros((n, N_BALLS), bool)
    Xb[np.arange(n)[:, None], P] = True
    overlap = float((Xb[1:] & Xb[:-1]).sum(1).mean())
    srt = np.sort(P, axis=1)
    code = (srt[:, _IU[0]] * N_BALLS + srt[:, _IU[1]]).ravel()
    pc = np.bincount(code, minlength=N_BALLS * N_BALLS).astype(float)
    mask = ex["pair_mask"]
    s_pair = float(((pc[mask] - n * ex["pair"][mask]) ** 2 / (n * ex["pair"][mask])).sum())
    return {"ball frequency chi2": s_freq, "1st-ball chi2": s_pos1, "pair co-occurrence chi2": s_pair,
            "mean draw sum": float(P.sum(1).mean() + N_DRAWN), "overlap with previous draw": overlap}


def pair_probs(P):
    srt = np.sort(P, axis=1)
    code = (srt[:, _IU[0]] * N_BALLS + srt[:, _IU[1]]).ravel()
    return np.bincount(code, minlength=N_BALLS * N_BALLS) / len(P)


def run_twin(d: Data, p_final, pos_final, args, rng):
    n = d.n
    log.info("=== Digital twin (calibration check) ===")
    pos_factor = np.clip(pos_final * N_BALLS, 0.5, 2.0)
    w = fit_machine(p_final, pos_factor, rng)
    log.info("Simulating %d virtual draws", args.twin_draws)
    t = time.time()
    sims = simulate_machine(w, pos_factor, args.twin_draws, rng)
    log.info("  done in %.1fs", time.time() - t)
    incl = np.bincount(sims.ravel(), minlength=N_BALLS) / len(sims)
    pos1 = np.bincount(sims[:, 0], minlength=N_BALLS) / len(sims)
    pp = pair_probs(sims)
    ex_twin = {"incl": np.clip(incl, 1e-9, None), "pos1": np.clip(pos1, 1e-9, None), "pair": pp,
               "pair_mask": pp > 0}
    up = np.zeros(N_BALLS * N_BALLS)
    iu = np.triu_indices(N_BALLS, 1)
    up[iu[0] * N_BALLS + iu[1]] = N_DRAWN * (N_DRAWN - 1) / (N_BALLS * (N_BALLS - 1))
    ex_unif = {"incl": np.full(N_BALLS, BASE), "pos1": np.full(N_BALLS, 1 / N_BALLS), "pair": up,
               "pair_mask": up > 0}
    real = twin_stats(d.P[:n], ex_twin)
    real_u = twin_stats(d.P[:n], ex_unif)
    R = args.twin_reps
    log.info("Generating %d replicate histories of %d draws for reference distributions", R, n)
    pool = simulate_machine(w, pos_factor, R * n, rng)
    ref_t = {k: [] for k in real}
    ref_u = {k: [] for k in real}
    for r in range(R):
        st = twin_stats(pool[r * n:(r + 1) * n].astype(int), ex_twin)
        U = np.argsort(rng.random((n, N_BALLS)), axis=1)[:, :5]
        su = twin_stats(U, ex_unif)
        for k in real:
            ref_t[k].append(st[k])
            ref_u[k].append(su[k])
        if (r + 1) % max(R // 5, 1) == 0:
            log.info("  replicate %d/%d", r + 1, R)

    def pval(obs, ref, two_sided):
        ref = np.asarray(ref)
        hi, lo = (ref >= obs).sum() + 1, (ref <= obs).sum() + 1
        p = (hi if not two_sided else min(hi, lo) * 2) / (len(ref) + 1)
        return float(min(p, 1.0))

    rows = []
    for k in real:
        two = not k.endswith("chi2")
        rows.append({"statistic": k, "real": real[k],
                     "twin_p": pval(real[k], ref_t[k], two),
                     "uniform_machine_p": pval(real_u[k], ref_u[k], two)})
        log.info("  twin check | %-28s real=%.4f  twin p=%.3f  uniform-machine p=%.3f",
                 k, real[k], rows[-1]["twin_p"], rows[-1]["uniform_machine_p"])
    top = np.argsort(-incl)[:5]
    se = math.sqrt(BASE * (1 - BASE) / len(sims))
    return {"rows": rows, "sim_top": [{"ball": int(b) + 1, "freq": float(incl[b])} for b in top],
            "sim_se": se, "n_sims": int(len(sims)), "weights_error_note": "see log"}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    tag = Path(args.csv).stem
    logpath = setup_logging(out_dir, tag)
    rng = np.random.default_rng(args.seed)
    log.info("Stage 2 start | csv=%s | log=%s | args=%s", args.csv, logpath, vars(args))

    d = load_data(args.csv, out_dir, args.next_date, args.next_shifted, args.next_holiday_shift)
    n = d.n
    n_bt = min(args.backtest, n - args.min_train)
    if n_bt < 100:
        raise ValueError(f"only {n} clean draws; need at least min_train+100 = {args.min_train + 100}")
    a0 = n - n_bt
    log.info("Backtest: last %d draws (%s .. %s), training starts with %d draws",
             n_bt, d.dates[a0].date(), d.dates[n - 1].date(), a0)
    leakage_check(d, args.half_life)

    log.info("Building features")
    F = build_ball_features(d)
    log.info("Per-ball feature tensor %s (%.1f MB): %s", F.shape, F.nbytes / 1e6, FEATURE_NAMES)
    macro, micro, posf = freq_experts(d, args.half_life)

    inc = {"uniform": np.full((n_bt, N_BALLS), BASE), "freq_macro": macro[a0:n], "freq_micro": micro[a0:n]}
    posd = {"uniform": np.full((n_bt, N_DRAWN, N_BALLS), 1 / N_BALLS), "pos_freq": posf[a0:n]}
    nxt_inc = {"freq_macro": macro[n], "freq_micro": micro[n]}
    nxt_pos = {"uniform": np.full((N_DRAWN, N_BALLS), 1 / N_BALLS), "pos_freq": posf[n]}
    importance = None

    if not args.no_gbm:
        inc["gbm"] = walk_forward_gbm(F, d.X, a0, n, args)
        model, imp = fit_gbm(F, d.X, n, args)
        nxt_inc["gbm"] = gbm_predict(model, F, n, n + 1)[0]
        if imp is not None:
            importance = sorted(zip(FEATURE_NAMES, map(float, imp)), key=lambda x: -x[1])
            log.info("GBM feature importance (gain share): %s", [(k, round(v, 3)) for k, v in importance[:8]])
    if not args.no_lstm:
        try:
            import torch  # noqa: F401
            dev = torch_device(args)
            S = build_seq_inputs(d)
            li, lp = walk_forward_lstm(S, d, a0, n, args, dev)
            inc["lstm"], posd["lstm"] = li, lp
            bundle = fit_lstm(S, d, n, args, dev)
            ni, npos = lstm_predict(bundle, n, n + 1, dev)
            nxt_inc["lstm"], nxt_pos["lstm"] = ni[0], npos[0]
        except ImportError:
            log.warning("torch not installed: LSTM skipped (pip install torch, or use Colab)")

    log.info("=== Evaluation ===")
    res, inc, wf, pwf, mix_names, pnames, per_draw = evaluate(d, a0, inc, posd, args, rng)
    for k, v in res["experts"].items():
        log.info("  %-11s log-score gain/draw=%+.5f (p=%s) | BCE=%.4f vs uniform %.4f | mean hits=%s",
                 k, v["gain_per_draw"], f"{v.get('gain_p', float('nan')):.3f}", v["bce_mean"], v["bce_uniform"],
                 f"{res['hits'][k]['mean_hits']:.4f}" if k in res["hits"] else "-")

    # next-draw distribution
    p_next = sum(wf[i] * (np.full(N_BALLS, BASE) if k == "uniform" else nxt_inc[k]) for i, k in enumerate(mix_names))
    pos_next = sum(pwf[i] * nxt_pos[k] for i, k in enumerate(pnames))
    pos_next = pos_next / pos_next.sum(1, keepdims=True)
    twin = run_twin(d, p_next, pos_next, args, rng)

    # verdict
    sig = res["survivors_bh"]
    verdict = ("SIGNAL CANDIDATE: " + "; ".join(sig)) if sig else \
        "NO SIGNAL: nothing survives multiple-testing correction; models are indistinguishable from random."

    # ------------------------------------------------------------------ report
    L = []
    w = L.append
    w("=" * 78)
    w(f"STAGE 2 REPORT | {Path(args.csv).name} | draws {d.info['date_min']} .. {d.info['date_max']} | N={n}")
    w("=" * 78)
    w(f"Backtest: last {res['n_backtest']} draws, walk-forward, GBM refit every {args.gbm_every}, "
      f"LSTM every {args.lstm_every}. Tests run: {res['n_tests']}. Bonferroni alpha = {res['bonferroni_alpha']:.5f}")
    w(f"Shifted draws in file: {d.info['shifted_rows']} | holiday rows: {d.info['holiday_rows']}")
    w("")
    w("1) PROBABILISTIC SCORES vs UNIFORM (higher gain = better; 0 = no better than uniform)")
    w(f"{'expert':<12}{'logscore gain/draw':>20}{'p (1-sided)':>13}{'BCE':>10}{'BCE gain':>11}{'mean hits':>11}")
    for k, v in res["experts"].items():
        mh = res["hits"].get(k, {}).get("mean_hits")
        w(f"{k:<12}{v['gain_per_draw']:>+20.5f}{v.get('gain_p', float('nan')):>13.3f}{v['bce_mean']:>10.4f}"
          f"{v['bce_gain_per_draw']:>+11.5f}{(f'{mh:.4f}' if mh is not None else '-'):>11}")
    w(f"  (uniform expected mean hits for a 5-ball pick: {res['hits']['ensemble']['expected_mean']:.4f})")
    w("")
    w("2) HIT TABLE: top-5 pick, ENSEMBLE vs RANDOM control vs chance (exactly-k counts)")
    w(f"{'hits':<6}{'ensemble':>10}{'random':>10}{'expected':>10}{'P(>=k) binom p ens':>22}{'p random':>10}")
    for j in range(2, 6):
        w(f"{j:<6}{res['hits']['ensemble']['exact'][str(j)]:>10}{res['hits']['random']['exact'][str(j)]:>10}"
          f"{res['hits']['expected_exact'][str(j)]:>10.2f}{res['hits']['ensemble']['binom_p_at_least'][str(j)]:>22.4f}"
          f"{res['hits']['random']['binom_p_at_least'][str(j)]:>10.4f}")
    tp = res["top2_pair"]
    w(f"Top-2 pair both drawn: ensemble {tp['ensemble']['both_hit']} (p={tp['ensemble']['p']:.3f}) | "
      f"random {tp['random']['both_hit']} | expected {tp['ensemble']['expected']:.2f}")
    w("")
    w("3) SIGNIFICANCE (every test run, sorted by raw p)")
    w(f"{'test':<58}{'raw p':>8}{'BH q':>8}")
    for t in sorted(res["tests"], key=lambda x: x["p"])[:15]:
        w(f"{t['name'][:57]:<58}{t['p']:>8.4f}{t['q_bh']:>8.3f}")
    w(f"Survive BH (q<0.05): {len(res['survivors_bh'])} | survive Bonferroni: {len(res['survivors_bonferroni'])}")
    w(f"VERDICT: {verdict}")
    w("")
    w("4) DIGITAL TWIN CALIBRATION (twin fitted to the final ensemble; real history compared with its replicates)")
    w(f"{'statistic':<30}{'real':>12}{'twin p':>9}{'uniform p':>11}")
    for r in twin["rows"]:
        w(f"{r['statistic']:<30}{r['real']:>12.3f}{r['twin_p']:>9.3f}{r['uniform_machine_p']:>11.3f}")
    w("  Read: a twin only earns trust if real data is reproduced by it AND rejected by the uniform machine.")
    w("  Frequency checks are partly circular (twin was fitted to this history); pair/sum/overlap are the real tests.")
    w("")
    w("5) DEEP DIVE")
    w(f"Ensemble weights (mean over backtest): "
      f"{ {k: round(v, 3) for k, v in res['ensemble_weights_mean'].items()} }")
    w(f"Ensemble weights (final, used for next draw): {({k: round(v, 3) for k, v in res['ensemble_weights_final'].items()})}")
    nd_name = d.dates[n].day_name()
    for k in ("freq_macro", "freq_micro", "gbm", "lstm"):
        if k in nxt_inc:
            w(f"  {k:<11} top5 for {nd_name}: {[int(b) + 1 for b in np.argsort(-nxt_inc[k])[:5]]}")
    if importance:
        w(f"GBM top features: {[(k, round(v, 3)) for k, v in importance[:6]]}")
        hs = dict(importance)
        w(f"  holiday/shift features' share: was_shifted={hs['was_shifted']:.3f}, holiday_shift={hs['holiday_shift']:.3f} "
          f"(compare with dow={hs['dow']:.3f}, gap={hs['gap']:.3f})")
    for k, v in res["position"].items():
        w(f"Position model {k}: logscore gain/draw={v['gain_per_draw']:+.5f} (p={v['p']:.3f}); "
          f"top-1 hits by position {v['top1_hits']} (chance {v['top1_expected']:.1f} each)")
    w("")
    order = np.argsort(-p_next)
    w(f"6) CONFIDENCE BOARD for the draw on {d.info['next_date']} ({nd_name})")
    w("   'Confidence' = ensemble probability that the ball appears among the 5 drawn.")
    w(f"   Chance for any ball = {100 * BASE:.2f}%. Differences of ~1 point are inside sampling noise.")
    for r_, b in enumerate(order[:15], 1):
        w(f"   {r_:>2}. ball {b + 1:>2}   {100 * p_next[b]:5.2f}%")
    w("")
    top2 = [int(b) + 1 for b in order[:2]]
    w("7) FINAL CALL (Top 2 by the ensemble)")
    w(f"   {top2[0]} and {top2[1]}   -> pick-both hit chance if the machine is fair: {100 * N_DRAWN * (N_DRAWN - 1) / (N_BALLS * (N_BALLS - 1)):.2f}%")
    if any(s.startswith("ensemble") for s in sig):
        w("   Status: the ENSEMBLE shows a candidate signal that survived correction. Still treat as unconfirmed")
        w("   until it holds on new draws you did not use here.")
    elif sig:
        w("   Status: signals survived correction in individual components (see section 3), but NOT in the ensemble")
        w("   that produces these two numbers. Use the component that carries the signal, and confirm it on new draws.")
    else:
        w("   Status: NOT VALIDATED. The backtest found no edge over random, so these two numbers are no better than any other.")
    sim_top = ", ".join(f"{s['ball']} ({100 * s['freq']:.2f}%)" for s in twin["sim_top"][:2])
    w(f"   Twin top-2 across {twin['n_sims']} simulated draws: {sim_top} (circular by construction; MC SE {100 * twin['sim_se']:.2f}%)")
    w("")
    w(f"Elapsed: {(time.time() - T0) / 60:.1f} min")
    report = "\n".join(L)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"stage2_report_{tag}_{stamp}.txt").write_text(report, encoding="utf-8")
    per_draw.to_csv(out_dir / f"stage2_backtest_{tag}_{stamp}.csv", index=False)
    payload = {"info": d.info, "results": res, "twin": twin, "verdict": verdict,
               "next": {"date": d.info["next_date"], "p": p_next.tolist(), "top2": top2}}
    (out_dir / f"stage2_results_{tag}_{stamp}.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n" + report)
    log.info("Saved report/CSV/JSON to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
