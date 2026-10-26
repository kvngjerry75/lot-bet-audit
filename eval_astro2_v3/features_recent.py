"""
features_recent.py: recent-momentum feature builder (leak-free), banned-column guards, shared helpers.

Row t of every feature uses draws STRICTLY BEFORE t, plus the calendar of draw t (known in advance).
Nothing here looks at the all-time (8-year) frequency: rate_all / rate250 / rate500 / frozen_frequency and
everything derived from them are banned and checked by assert_no_banned. Exception: features.allow_long_run (set by
main_beast for Afriluck only, where a physical colour bias is proven) adds longrun_ball_rate, longrun_half_rate and
low_half; they are banned whenever the switch is off.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("nla.features")

N_DRAWN = int(__import__("os").environ.get("ASTRO_K", "5"))


class BannedFeatureError(AssertionError):
    pass


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def make_deterministic() -> None:
    """Bit-for-bit repeatable GPU training: without this, cuDNN picks non-deterministic LSTM kernels and the same
    command can give a different wheel core on a T4."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def seed_everything(seed: int) -> None:
    """Seeds python / numpy / torch. Training randomness is not output randomness: this only makes runs reproducible."""
    make_deterministic()
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def norm_inc(p):
    """Rescale so each row sums to N_DRAWN (inclusion probabilities of 5 distinct balls); works on 1-D and 2-D."""
    p = np.asarray(p, dtype=float)
    for _ in range(4):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        p = p * (N_DRAWN / p.sum(axis=-1, keepdims=True))
    return np.clip(p, 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def assert_no_banned(names, cfg, where: str = "", allow_rate100: bool = False) -> None:
    """Fail the run if a banned (long-history) column reaches a model."""
    banned = list(cfg["banned_features"])
    if not allow_rate100 and not cfg["features"].get("use_rate100", False):
        banned.append(cfg["banned_rate100"])
    if allow_rate100:
        banned = [b for b in banned if b != cfg["banned_rate100"]]
    if not cfg["features"].get("allow_long_run", False):
        banned += ["longrun_", "low_half"]          # the Afriluck-only long-run columns stay banned everywhere else
    bad = sorted({n for n in names for b in banned if b.lower() in str(n).lower()})
    if bad:
        raise BannedFeatureError(f"BANNED long-history feature(s) {bad} reached {where or 'a model'}")


def assert_no_banned_experts(names, cfg, where: str = "") -> None:
    banned = [b.lower() for b in cfg["banned_experts"]]
    bad = sorted({n for n in names for b in banned if b in str(n).lower()})
    if bad:
        raise BannedFeatureError(f"BANNED expert(s) {bad} reached {where or 'the ensemble'}")


# --------------------------------------------------------------------------- #
# Feature bundle
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    F: np.ndarray            # (n+1, nb, nf) float32
    names: list

    def col(self, name: str) -> np.ndarray:
        return self.F[:, :, self.names.index(name)]


CAL_FEATURES = ["dow", "month", "day_of_month", "days_since_prev", "was_shifted", "holiday_shift"]


def build_features(d, cfg) -> FeatureBundle:
    """(n+1, nb, nf). d is the stage2 Data object (n real draws, n+1 rows, last row = draw to predict)."""
    fc = cfg["features"]
    n = d.n
    N1 = n + 1
    nb = d.X.shape[1]
    base = N_DRAWN / nb
    X = np.array(d.X, dtype=np.float64)
    X[n:] = 0.0                                      # the draw to predict is never visible
    P = np.asarray(d.P, dtype=int)
    t_idx = np.arange(N1)

    cols, names = {}, []

    def put(name, arr):
        names.append(name)
        cols[name] = np.asarray(arr, dtype=np.float64)

    C = np.vstack([np.zeros((1, nb)), np.cumsum(X, axis=0)])          # C[t] = sum of X[:t]

    wins = list(fc["windows"]) + ([100] if fc.get("use_rate100") else [])
    for w in sorted(set(wins)):
        lo = np.maximum(0, t_idx - w)
        r = (C[t_idx] - C[lo]) / np.maximum(t_idx - lo, 1)[:, None]
        r[0] = base
        put(f"rate{w}", r)

    # recent same-weekday rate
    dw = int(fc["dow_window"])
    rdow = np.full((N1, nb), base)
    dow = np.asarray(d.dow)
    for k in range(7):
        idx = np.where((dow[:n] == k))[0]
        Ck = np.vstack([np.zeros((1, nb)), np.cumsum(X[idx], axis=0)]) if len(idx) else np.zeros((1, nb))
        rows = np.where(dow == k)[0]
        m = np.searchsorted(idx, rows, side="left")                  # same-weekday draws strictly before row t
        lo = np.maximum(0, m - dw)
        val = (Ck[m] - Ck[lo]) / np.maximum(m - lo, 1)[:, None]
        val[m == 0] = base
        rdow[rows] = val
    put("rate_dow8", rdow)

    # gaps + percentile of the current gap in the ball's own trailing completed-gap history
    H = int(fc["gap_hist"])
    cap = float(fc["gap_cap_mult"]) * nb / N_DRAWN
    last = np.full(nb, -1)
    G = np.zeros((nb, H))
    gcount = np.zeros(nb, dtype=int)
    gptr = np.zeros(nb, dtype=int)
    gap_cur = np.empty((N1, nb))
    gap_pct = np.empty((N1, nb))
    ar_h = np.arange(H)[None, :]
    for t in range(N1):
        gap = np.minimum(t - last, cap).astype(float)                # last = -1 -> never seen: t + 1
        gap_cur[t] = gap
        valid = ar_h < gcount[:, None]
        less = (valid & (G < gap[:, None])).sum(1)
        eq = (valid & (G == gap[:, None])).sum(1)
        gap_pct[t] = np.where(gcount > 0, (less + 0.5 * eq) / np.maximum(gcount, 1), 0.5)
        if t < n:
            for b in P[t]:
                if last[b] >= 0:
                    G[b, gptr[b] % H] = t - last[b]
                    gptr[b] += 1
                    gcount[b] = min(gcount[b] + 1, H)
                last[b] = t
    put("gap_current", gap_cur)
    put("gap_ratio", gap_cur / (nb / N_DRAWN))
    put("gap_percentile", gap_pct)

    # momentum delta
    if "rate20" in cols and "rate50" in cols:
        put("momentum_20_50", cols["rate20"] - cols["rate50"])

    # decayed Markov transition score: mean over the 5 balls of draw t-1 of P(b in draw t | a in draw t-1)
    dec = 0.5 ** (1.0 / float(fc["markov_half_life"]))
    alpha = float(fc["markov_alpha"])
    T = np.zeros((nb, nb))
    Na = np.zeros(nb)
    mk = np.full((N1, nb), base)
    for t in range(1, N1):
        rows = P[t - 1]
        prob = (T[rows] + alpha * base) / (Na[rows, None] + alpha)
        mk[t] = prob.mean(0)
        if t < n:                                                    # pair (t-1 -> t) becomes known after draw t
            T *= dec
            Na *= dec
            T += np.outer(X[t - 1], X[t])
            Na += X[t - 1]
    put("markov", mk)

    # calendar (known in advance) incl. WasShifted
    cal = {"dow": d.dow, "month": d.month, "day_of_month": d.dom, "days_since_prev": d.dsp,
           "was_shifted": d.shifted, "holiday_shift": d.hshift}
    for k in CAL_FEATURES:
        put(k, np.repeat(np.asarray(cal[k], dtype=np.float64)[:, None], nb, axis=1))

    # Long-run frequency, ONLY when features.allow_long_run is set (main_beast sets it for Afriluck). The old ban
    # (rate_all / rate250 / rate500) stopped a frozen-frequency leak on the NLA games; on a machine with a real,
    # stable physical bias (Afriluck's pink balls) the long-run rate is exactly where the signal lives. These columns
    # use draws strictly before t (C[t] = sum of X[:t]), shrunk towards chance by a0 draws.
    if fc.get("allow_long_run"):
        a0 = float(fc.get("long_run_prior", 200.0))
        Ct = C[t_idx]
        put("longrun_ball_rate", (Ct + a0 * base) / (t_idx[:, None] + a0))
        half = np.arange(nb) < nb // 2
        lo_r = (Ct[:, half].sum(1) + a0 * base * half.sum()) / (t_idx * half.sum() + a0 * half.sum())
        hi_r = (Ct[:, ~half].sum(1) + a0 * base * (~half).sum()) / (t_idx * (~half).sum() + a0 * (~half).sum())
        put("longrun_half_rate", np.where(half[None, :], lo_r[:, None], hi_r[:, None]))
        put("low_half", np.repeat(half[None, :].astype(float), N1, axis=0))

    F = np.stack([cols[k] for k in names], axis=2).astype(np.float32)
    assert_no_banned(names, cfg, "build_features", allow_rate100=bool(fc.get("use_rate100")))
    return FeatureBundle(F, names)
