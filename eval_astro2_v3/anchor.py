"""
anchor.py: TWO-TIMESCALE ANCHOR expert = long-term signal that is NOT allowed to freeze the picks.

Why the old model froze: rate_all / rate250 fed a tree ranker next to fast features. Trees split on the slow column first, so the
ball order was decided by an almost-constant number and the fast features only broke ties.

Design here (all leak-free: row t uses draws strictly before t, and the fit window ends before the block it predicts):

    logit p[t,b] = logit(5/nb) + wL * zL[t,b]  +  sc * ( f1*zM + f2*zO + f3*zK )[t,b]
                         slow, shrunk, relative          fast: momentum, overdue-ness, markov (cross-sectional z-scores)

  1. LONG enters ONLY as a shrunk log-ratio, cross-sectionally z-scored per draw, with a bounded weight wL. It shifts a ball
     by a bounded amount in logit space; it never decides the order alone (additive, not a tree split).
  2. FAST terms are signed and scaled by what the fit window shows (mean z of the balls actually drawn).
  3. (wL, sc) are chosen by validation log-loss SUBJECT TO A TURNOVER CAP: the top-K sets of consecutive draws may overlap
     (Jaccard) at most `max_top_jaccard`. Stagnation is a hard constraint of the fit, not a hope.
  4. HONEST GATE: weights are first fitted on the older 2/3 of the window and must show log-loss skill > 0 on the newest 1/3
     before the expert is active. If the long signal is not real, the gate shuts the expert (uniform) instead of faking it.
  5. MEASUREMENT MODE (anchor.always_on): the gate and the turnover cap can shut the expert in most walk-forward blocks, so
     its out-of-sample record is empty and nothing can be measured. With always_on the expert forecasts in EVERY block
     (backtest and next draw alike): if no grid point meets the cap, the cap is relaxed to the lowest achievable Jaccard
     + cap_relax. The hold-out skill is still computed and logged, it just no longer switches the expert off.
     Every fit records `reason` (min_fit | turnover_cap | gate | ok | ok_relaxed_cap | ok_ungated).

The long columns are internal to this expert; they never enter fb.F, the ranker, the TFT or the twin, so the
long-history bans in features_recent stay fully intact.
"""
from __future__ import annotations

import logging

import numpy as np

import features_recent as fr
from nla_config import N_DRAWN, core_size, uniform_bce

log = logging.getLogger("nla.anchor")
ZNAMES = ["long", "momentum", "overdue", "markov"]


def _bce(p, X):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((-(X * np.log(p) + (1 - X) * np.log1p(-p)).sum(1)).mean())


def _zrow(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def build_z(d, fb, cfg) -> np.ndarray:
    """(n+1, nb, 4) cross-sectional z-scores [long, momentum, overdue, markov]. Row t uses draws < t only."""
    ac = cfg["anchor"]
    nb = d.X.shape[1]
    n = d.n
    N1 = n + 1
    base = N_DRAWN / nb
    X = np.array(d.X, dtype=np.float64)
    X[n:] = 0.0
    C = np.vstack([np.zeros((1, nb)), np.cumsum(X, axis=0)])          # C[t] = sum of X[:t]
    t = np.arange(N1)
    k0 = float(ac["long_shrink"])
    lr = []
    for w in ac["long_windows"]:                                      # 0 = all history up to t
        lo = np.maximum(0, t - w) if w else np.zeros(N1, dtype=int)
        cnt = C[t] - C[lo]
        exp_ = ((t - lo) * base)[:, None]
        lr.append(np.log((cnt + k0) / (exp_ + k0)))
    zl = _zrow(np.mean(lr, axis=0))

    k1 = float(ac["fast_shrink"])
    mom = []
    for w in ac["fast_windows"]:
        r = fb.col(f"rate{w}").astype(float)
        mom.append(np.log((r * w + k1) / (base * w + k1)))
    zm = _zrow(np.mean(mom, axis=0))
    zo = _zrow(fb.col("gap_percentile").astype(float))
    zk = _zrow(np.log(np.clip(fb.col("markov").astype(float), 1e-6, None) / base))
    return np.stack([zl, zm, zo, zk], axis=2)


def _probs(logits, nb):
    b = N_DRAWN / nb
    return fr.norm_inc(1.0 / (1.0 + np.exp(-(np.log(b / (1 - b)) + logits))))


def top_jaccard(scores: np.ndarray, K: int) -> float:
    """Mean Jaccard overlap of the top-K sets of consecutive rows (1.0 = frozen picks)."""
    if len(scores) < 2:
        return 0.0
    idx = np.argpartition(-scores, K - 1, axis=1)[:, :K]
    A = np.zeros(scores.shape, dtype=bool)
    A[np.arange(len(scores))[:, None], idx] = True
    inter = (A[1:] & A[:-1]).sum(1)
    return float(np.mean(inter / (2 * K - inter)))


def _select(Zf, Xf, nb, K, ac, cap=None):
    """Best (wL, sc) by log-loss under the turnover cap. Returns (w (4,), bce, jaccard) or None."""
    cap = ac["max_top_jaccard"] if cap is None else cap
    drawn = Xf.sum()
    m = np.array([(Zf[:, :, i] * Xf).sum() / max(drawn, 1.0) for i in range(4)])   # mean z of drawn balls (0 = no signal)
    mf = m[1:]
    fdir = mf / max(np.abs(mf).max(), 1e-9)
    sL = 1.0 if m[0] >= 0 else -1.0
    best = None
    for wl in ac["wL_grid"]:
        for sc in ac["fast_grid"]:
            w = np.array([sL * wl, sc * fdir[0], sc * fdir[1], sc * fdir[2]])
            lg = Zf @ w
            jac = top_jaccard(lg, K)
            if jac > cap:
                continue
            b = _bce(_probs(lg, nb), Xf)
            if best is None or b < best[1]:
                best = (w, b, jac)
    return best


def _min_jaccard(Zf, Xf, K, ac):
    """Lowest top-K Jaccard any grid point achieves on this window (scaling w does not change the order, so this is
    a property of the window's signal directions, not of the grid)."""
    drawn = Xf.sum()
    m = np.array([(Zf[:, :, i] * Xf).sum() / max(drawn, 1.0) for i in range(4)])
    fdir = m[1:] / max(np.abs(m[1:]).max(), 1e-9)
    sL = 1.0 if m[0] >= 0 else -1.0
    return min(top_jaccard(Zf @ np.array([sL * wl, sc * fdir[0], sc * fdir[1], sc * fdir[2]]), K)
               for wl in ac["wL_grid"] for sc in ac["fast_grid"])


def _select_or_relax(Zf, Xf, nb, K, ac):
    """(_select result, relaxed?) ; relaxed only in always_on mode when nothing meets the cap."""
    A = _select(Zf, Xf, nb, K, ac)
    if A is not None or not ac.get("always_on"):
        return A, False
    return _select(Zf, Xf, nb, K, ac, cap=_min_jaccard(Zf, Xf, K, ac) + ac.get("cap_relax", 0.05)), True


def fit_anchor(Z, X, a, cfg):
    """Fit on draws [a - fit_len, a). Returns dict(w, active, skill, jaccard, wL, reason).
    The SAME function is used for every walk-forward block and for the next draw, so the two can only differ through
    the data in the window, never through the code path."""
    ac = cfg["anchor"]
    nb = X.shape[1]
    K = core_size(cfg, nb)
    lo = max(ac["warmup"], a - ac["fit_len"])
    dead = dict(w=np.zeros(4), active=False, skill=0.0, jaccard=0.0, wL=0.0, reason="min_fit")
    if a - lo < ac["min_fit"]:
        return dead
    cut = lo + int(round((a - lo) * 2 / 3))
    A, relaxed = _select_or_relax(Z[lo:cut], X[lo:cut], nb, K, ac)
    if A is None:
        dead["reason"] = "turnover_cap"
        dead["jaccard"] = _min_jaccard(Z[lo:cut], X[lo:cut], K, ac)
        return dead
    skill = 1.0 - _bce(_probs(Z[cut:a] @ A[0], nb), X[cut:a]) / uniform_bce(nb)    # honest hold-out: newest third
    if skill <= ac["gate_margin"] and not ac.get("always_on"):
        dead.update(skill=skill, reason="gate")
        return dead
    F, relaxed_f = _select_or_relax(Z[lo:a], X[lo:a], nb, K, ac)
    F = F or A
    reason = "ok_relaxed_cap" if (relaxed or relaxed_f) else ("ok" if skill > ac["gate_margin"] else "ok_ungated")
    return dict(w=F[0], active=True, skill=skill, jaccard=F[2], wL=float(F[0][0]), reason=reason)


def anchor_predict(fit, Z, a, b, nb):
    if not fit["active"]:
        return np.full((b - a, nb), N_DRAWN / nb)
    return _probs(Z[a:b] @ fit["w"], nb)


def _row(a, f, final=False):
    r = dict(block_start=int(a), active=bool(f["active"]), reason=f.get("reason", ""), skill=round(f["skill"], 5),
             wL=round(f["wL"], 3),
             jaccard=round(f["jaccard"], 3), w=[round(float(x), 3) for x in f["w"]])
    if final:
        r["final"] = True
    return r


def run_anchor(d, fb, cfg, span_start, n, cad, predict_next=False, quiet=False):
    """Walk-forward by blocks. Returns (probs (n-span_start, nb), active (rows,), info list, next=(vec, active) | None)."""
    nb = d.X.shape[1]
    X = np.array(d.X, dtype=np.float64)
    X[n:] = 0.0
    Z = build_z(d, fb, cfg)
    out = np.empty((n - span_start, nb))
    act = np.zeros(n - span_start, dtype=bool)
    info = []
    for a in range(span_start, n, cad):
        b = min(a + cad, n)
        f = fit_anchor(Z, X, a, cfg)
        out[a - span_start:b - span_start] = anchor_predict(f, Z, a, b, nb)
        act[a - span_start:b - span_start] = f["active"]
        info.append(_row(a, f))
    nxt = None
    if predict_next:
        f = fit_anchor(Z, X, n, cfg)
        nxt = (anchor_predict(f, Z, n, n + 1, nb)[0], f["active"])
        info.append(_row(n, f, True))
    for r in ([] if quiet else info):
        log.info("  anchor fit @%d%s: active=%s reason=%s hold-out skill %+.5f top-K Jaccard %.3f w=%s", r["block_start"],
                 " (NEXT DRAW)" if r.get("final") else "", r["active"], r["reason"], r["skill"], r["jaccard"], r["w"])
    return out, act, info, nxt
