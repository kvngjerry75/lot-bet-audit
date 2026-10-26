#!/usr/bin/env python3
"""
Stage 3 ("the beast"): TFT-style attention model + Optuna-tuned LightGBM, built on stage2_powerhouse.py.

Colab (T4):
    !pip -q install lightgbm optuna scipy scikit-learn
    # upload stage2_powerhouse.py and stage3_beast.py to the same folder, then:
    !python stage3_beast.py --csv daywa_blind.csv
Smoke test:  add --quick

What is new compared with Stage 2
  * rate100 (rolling 100-draw frequency) plus rate250/rate500 and rank/deviation versions are explicit GBM features,
    and the attention model receives rate100 / long-run frequency as a dedicated, gated input group whose
    selection weight is reported.
  * Optuna tunes LightGBM using ONLY draws before the backtest window (3 expanding inner folds, fixed tree count,
    no early stopping on the scoring fold). Parameters are then frozen for the whole walk-forward backtest.
  * The attention model ("TFT-lite": gated residual networks, variable selection, LSTM encoder, interpretable
    multi-head attention over the last L draws) predicts a CORRECTION on top of the frozen-frequency prior, starting at
    exactly that prior, so it can only help or fall back to it.
  * A frozen-frequency benchmark (expanding counts, no decay) is a first-class expert. Section 2 of the report asks the
    only question that matters: does anything heavier beat it, on draws it has never seen?
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse

import stage2_powerhouse as s2
from stage2_powerhouse import BASE, N_BALLS, N_DRAWN, log

EXT_NAMES = ["rate250", "rate500", "rate100_rank", "rate_all_rank", "rate100_minus_all", "static_z"]
ALL_NAMES = s2.FEATURE_NAMES + EXT_NAMES
I_RATE100 = s2.FEATURE_NAMES.index("rate100")
I_RATEALL = s2.FEATURE_NAMES.index("rate_all")
I_GAP = s2.FEATURE_NAMES.index("gap")
STATIC_HL = 1e9  # half-life so large that decay is negligible = frozen (expanding) frequency


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    args = s2.parse_args(argv)  # shared options (tolerates extras)
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60, help="Optuna trials for LightGBM")
    ap.add_argument("--tune-timeout", type=float, default=1500.0, help="Optuna wall-clock cap in seconds")
    ap.add_argument("--no-optuna", action="store_true")
    ap.add_argument("--no-tft", action="store_true")
    ap.add_argument("--tft-every", type=int, default=125)
    ap.add_argument("--tft-epochs", type=int, default=30)
    ap.add_argument("--tft-seq-len", type=int, default=64)
    ap.add_argument("--tft-hidden", type=int, default=64)
    ap.add_argument("--tft-heads", type=int, default=4)
    ap.add_argument("--tft-dropout", type=float, default=0.2)
    ap.add_argument("--payout", type=float, default=0.0,
                    help="optional payout multiple for a correct 2-number pick, to print break-even maths")
    extra, _ = ap.parse_known_args(argv)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    if args.quick:
        args.trials, args.tune_timeout = 8, 120.0
        args.tft_epochs, args.tft_every, args.tft_hidden, args.tft_seq_len = 4, 100, 32, 32
    args.no_lstm = True  # the attention model replaces the plain LSTM
    return args


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def extend_features(d: s2.Data, F: np.ndarray) -> np.ndarray:
    """Append long-window rates and rank/deviation versions of rate100. Row t uses draws < t only."""
    N1 = len(d.X)
    t_idx = np.arange(N1)
    C = np.vstack([np.zeros((1, N_BALLS)), np.cumsum(d.X, axis=0)])
    out = np.zeros((N1, N_BALLS, len(EXT_NAMES)), dtype=np.float32)
    for col, w in ((0, 250), (1, 500)):
        lo = np.maximum(0, t_idx - w)
        r = (C[t_idx] - C[lo]) / np.maximum(t_idx - lo, 1)[:, None]
        r[0] = BASE
        out[:, :, col] = r
    r100, rall = F[:, :, I_RATE100], F[:, :, I_RATEALL]
    out[:, :, 2] = np.argsort(np.argsort(r100, axis=1), axis=1) / (N_BALLS - 1)
    out[:, :, 3] = np.argsort(np.argsort(rall, axis=1), axis=1) / (N_BALLS - 1)
    out[:, :, 4] = r100 - rall
    out[:, :, 5] = (rall - BASE) / np.sqrt(BASE * (1 - BASE) / np.maximum(t_idx, 1))[:, None]
    return np.concatenate([F, out], axis=2)


def leakage_check_ext(d: s2.Data):
    t0 = d.n // 2
    rng = np.random.default_rng(0)
    d2 = s2.Data(**{**d.__dict__})
    d2.P, d2.X = d.P.copy(), d.X.copy()
    for t in range(t0, d.n):
        d2.P[t] = rng.permutation(N_BALLS)[:5]
        d2.X[t] = 0
        d2.X[t, d2.P[t]] = 1
    A = extend_features(d, s2.build_ball_features(d))
    B = extend_features(d2, s2.build_ball_features(d2))
    pa, pb = s2.freq_experts(d, STATIC_HL), s2.freq_experts(d2, STATIC_HL)
    ok = np.allclose(A[:t0 + 1], B[:t0 + 1]) and all(np.allclose(x[:t0 + 1], y[:t0 + 1]) for x, y in zip(pa, pb))
    log.info("Leakage self-test, extended features + static priors (rows <= %d vs future noise): %s", t0,
             "PASSED" if ok else "FAILED")
    if not ok:
        raise RuntimeError("leakage detected in extended features")


# --------------------------------------------------------------------------- #
# Optuna-tuned LightGBM
# --------------------------------------------------------------------------- #
def _rows(F, X, a, b):
    nf = F.shape[2]
    return F[a:b].reshape(-1, nf), X[a:b].reshape(-1).astype(int)


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def tune_gbm(F, X, a0, args):
    """Tune on draws < a0 only. Returns (params, info)."""
    try:
        import lightgbm as lgb
        import optuna
    except ImportError as e:
        log.warning("Optuna/LightGBM missing (%s): using default parameters. pip install optuna lightgbm", e)
        return None, {"skipped": True}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lo = args.warmup
    V = max(150, (a0 - lo) // 8)
    folds = []
    for k in (3, 2, 1):
        vs, ve = a0 - k * V, a0 - (k - 1) * V
        ts = max(lo, vs - args.train_window)
        if vs - ts >= 300:
            folds.append((ts, vs, ve))
    if not folds:
        log.warning("Not enough pre-backtest history to tune; using defaults")
        return None, {"skipped": True}
    data = [(*_rows(F, X, ts, vs), *_rows(F, X, vs, ve)) for ts, vs, ve in folds]
    const = float(np.mean([_ll(yv, np.full(len(yv), yt.mean())) for _, yt, _, yv in data]))
    log.info("=== Optuna: %d trials (cap %.0fs) on draws < %d, %d inner folds of %d draws ===",
             args.trials, args.tune_timeout, a0, len(folds), V)
    log.info("  constant-rate inner log-loss = %.5f (to beat)", const)

    def build(trial):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 40, 700, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 63, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 1500, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
            max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
            random_state=args.seed, n_jobs=-1, verbose=-1)

    def score(params):
        ls = []
        for xt, yt, xv, yv in data:
            m = lgb.LGBMClassifier(**params)
            m.fit(xt, yt)  # fixed tree count, no early stopping on the scoring fold
            ls.append(_ll(yv, m.predict_proba(xv)[:, 1]))
        return float(np.mean(ls))

    t_start = time.time()

    def objective(trial):
        v = score(build(trial))
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            log.info("  trial %3d | inner log-loss %.5f | %.0fs", trial.number + 1, v, time.time() - t_start)
        return v

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, timeout=args.tune_timeout)
    best = {**study.best_trial.params, "subsample_freq": 1, "random_state": args.seed, "n_jobs": -1, "verbose": -1}
    default = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, random_state=args.seed, n_jobs=-1, verbose=-1)
    d_ll = score(default)
    info = {"trials": len(study.trials), "best_inner_logloss": float(study.best_value),
            "default_inner_logloss": d_ll, "constant_inner_logloss": const,
            "best_params": {k: v for k, v in best.items() if k not in ("n_jobs", "verbose", "random_state")}}
    log.info("Optuna done: %d trials | best %.5f | default params %.5f | constant %.5f", info["trials"],
             study.best_value, d_ll, const)
    log.info("  best params: %s", info["best_params"])
    if study.best_value >= const:
        log.warning("  Tuned GBM does NOT beat the constant-rate model inside the tuning window.")
    return best, info


def fit_tuned(F, X, t_end, params, args):
    import lightgbm as lgb
    t0 = max(args.warmup, t_end - args.train_window)
    xa, ya = _rows(F, X, t0, t_end)
    m = lgb.LGBMClassifier(**params)
    t = time.time()
    m.fit(xa, ya)
    imp = m.booster_.feature_importance("gain")
    log.info("  GBM[tuned] fit t_end=%d rows=%d %.1fs", t_end, len(ya), time.time() - t)
    return m, imp / max(imp.sum(), 1e-12)


def walk_forward_tuned(F, X, a0, n, params, args):
    out = np.empty((n - a0, N_BALLS))
    blocks = list(range(a0, n, args.gbm_every))
    log.info("Tuned-GBM walk-forward: %d refits, %d test draws", len(blocks), n - a0)
    model = None
    for i, a in enumerate(blocks):
        b = min(a + args.gbm_every, n)
        if model is None or not s2.over_cap(args):
            model, _ = fit_tuned(F, X, a, params, args)
        else:
            log.warning("  time cap reached: reusing previous tuned GBM")
        out[a - a0:b - a0] = s2.gbm_predict(model, F, a, b)
        log.info("  tuned-GBM block %d/%d done (draws %d..%d) elapsed %.0fs", i + 1, len(blocks), a, b - 1,
                 time.time() - s2.T0)
    return out


# --------------------------------------------------------------------------- #
# TFT-lite attention model
# --------------------------------------------------------------------------- #
class TFTData:
    """Everything the attention model needs, as tensors on the device."""

    def __init__(self, d, F, prior_inc, prior_pos, dev):
        import torch
        S = s2.build_seq_inputs(d)                               # (N1, 3*90+13): inclusion, first ball, last ball, calendar
        r100 = (F[:, :, I_RATE100] - BASE) / BASE
        rall = (F[:, :, I_RATEALL] - BASE) / BASE
        gap = F[:, :, I_GAP] / 300.0
        self.sizes = dict(r100=N_BALLS, rall=N_BALLS, gap=N_BALLS, cal=d.cal.shape[1])
        ctx = np.concatenate([r100, rall, gap, d.cal], axis=1).astype(np.float32)
        self.S = torch.from_numpy(S).to(dev)
        self.C = torch.from_numpy(ctx).to(dev)
        self.X = torch.from_numpy(d.X).to(dev)
        self.P = torch.from_numpy(d.P).long().to(dev)
        pr = np.clip(prior_inc, 1e-4, 1 - 1e-4)
        self.PR = torch.from_numpy(np.log(pr / (1 - pr)).astype(np.float32)).to(dev)
        self.PP = torch.from_numpy(np.log(np.clip(prior_pos, 1e-8, None)).astype(np.float32)).to(dev)
        self.din = S.shape[1]


def make_net(td: TFTData, args):
    import torch
    import torch.nn as nn

    H, drop = args.tft_hidden, args.tft_dropout

    class GRN(nn.Module):
        def __init__(s, din, hid, dout, ctx=0):
            super().__init__()
            s.l1 = nn.Linear(din, hid)
            s.lc = nn.Linear(ctx, hid, bias=False) if ctx else None
            s.l2 = nn.Linear(hid, hid)
            s.glu = nn.Linear(hid, 2 * dout)
            s.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
            s.norm = nn.LayerNorm(dout)
            s.drop = nn.Dropout(drop)

        def forward(s, x, c=None):
            h = s.l1(x)
            if s.lc is not None and c is not None:
                h = h + s.lc(c)
            h = s.l2(nn.functional.elu(h))
            a, b = s.glu(s.drop(h)).chunk(2, dim=-1)
            return s.norm(s.skip(x) + a * torch.sigmoid(b))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            sz = td.sizes
            s.groups = list(sz)
            s.emb = nn.ModuleDict({k: nn.Linear(v, H) for k, v in sz.items()})
            s.vsn = GRN(H * len(sz), H, len(sz))                  # variable selection over {rate100, long-run freq, gap, calendar}
            s.static_ctx = GRN(H, H, H)
            s.h0 = nn.Linear(H, H)
            s.c0 = nn.Linear(H, H)
            s.step = GRN(td.din, H, H)
            s.lstm = nn.LSTM(H, H, batch_first=True)
            s.gate1 = GRN(H, H, H)
            s.enrich = GRN(H, H, H, ctx=H)
            s.attn = nn.MultiheadAttention(H, args.tft_heads, dropout=drop, batch_first=True)
            s.gate2 = GRN(H, H, H)
            s.out = GRN(H, H, H)
            s.inc = nn.Linear(H, N_BALLS)
            s.pos = nn.Linear(H, N_DRAWN * N_BALLS)
            for m in (s.inc, s.pos):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)                            # starts exactly at the frozen-frequency prior

        def forward(s, seq, ctx, pr, pp):
            parts, o = {}, 0
            for k in s.groups:
                w = td.sizes[k]
                parts[k] = s.emb[k](ctx[:, o:o + w])
                o += w
            flat = torch.cat([parts[k] for k in s.groups], -1)
            sel = torch.softmax(s.vsn(flat), -1)                  # (B, groups): which input group matters
            stack = torch.stack([parts[k] for k in s.groups], 1)
            c = s.static_ctx((sel.unsqueeze(-1) * stack).sum(1))
            e = s.step(seq)
            h0, c0 = torch.tanh(s.h0(c)).unsqueeze(0), torch.tanh(s.c0(c)).unsqueeze(0)
            lo, _ = s.lstm(e, (h0.contiguous(), c0.contiguous()))
            enr = s.enrich(s.gate1(lo + e), c.unsqueeze(1).expand(-1, lo.size(1), -1))
            q = enr[:, -1:, :]
            att, w = s.attn(q, enr, enr, need_weights=True)      # attention over history, query = latest state
            z = s.out(s.gate2(att[:, 0] + enr[:, -1]))
            li = pr + s.inc(z)
            lp = pp + s.pos(z).view(-1, N_DRAWN, N_BALLS)
            return li, lp, w[:, 0, :], sel

    return Net().to(td.S.device)


def _batch(td, ix, L, dev):
    import torch
    ix = torch.as_tensor(ix, device=dev)
    ar = torch.arange(-L, 0, device=dev)
    return td.S[ix[:, None] + ar], td.C[ix], td.PR[ix], td.PP[ix], td.X[ix], td.P[ix]


def tft_selftest(td, d, args, dev):
    """Output for target row t must not change when rows >= t of the sequence tensor are replaced by noise."""
    import torch
    torch.manual_seed(0)
    net = make_net(td, args)
    for m in (net.inc, net.pos):
        torch.nn.init.normal_(m.weight, std=0.1)
    net.eval()
    L, t0 = args.tft_seq_len, d.n // 2
    ix = np.arange(t0 - 5, t0 + 1)
    with torch.no_grad():
        a = net(*_batch(td, ix, L, dev)[:4])[0]
        keep = td.S.clone()
        td.S[t0:] = torch.randn_like(td.S[t0:])
        b = net(*_batch(td, ix, L, dev)[:4])[0]
        td.S.copy_(keep)
    ok = bool(torch.allclose(a, b, atol=1e-5))
    log.info("Attention-model leakage self-test (target row never sees rows >= t): %s", "PASSED" if ok else "FAILED")
    if not ok:
        raise RuntimeError("attention model sees the future")


def fit_tft(td, t_end, args, dev):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.manual_seed(args.seed)
    L = args.tft_seq_len
    t_lo = max(L, t_end - args.train_window)
    idx = np.arange(t_lo, t_end)
    n_val = max(int(0.15 * len(idx)), 60)
    tr, va = idx[:-n_val], idx[-n_val:]
    net = make_net(td, args)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)

    def losses(li, lp, y, p):
        a = Fn.binary_cross_entropy_with_logits(li, y, reduction="none").sum(1).mean()
        b = Fn.cross_entropy(lp.reshape(-1, N_BALLS), p.reshape(-1)) * N_DRAWN
        return a, b

    def val_loss():
        net.eval()
        with torch.no_grad():
            seq, c, pr, pp, y, p = _batch(td, va, L, dev)
            li, lp, _, _ = net(seq, c, pr, pp)
            return tuple(float(x) for x in losses(li, lp, y, p))

    vi0, vp0 = val_loss()                                         # untrained network == frozen-frequency prior
    best = vi0 + 0.5 * vp0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    log.info("    prior only: val inc=%.4f pos=%.4f", vi0, vp0)
    bad, t_fit = 0, time.time()
    for ep in range(args.tft_epochs):
        net.train()
        perm = np.random.default_rng(args.seed + ep).permutation(tr)
        tl = 0.0
        for s in range(0, len(perm), 128):
            seq, c, pr, pp, y, p = _batch(td, perm[s:s + 128], L, dev)
            li, lp, _, _ = net(seq, c, pr, pp)
            a, b = losses(li, lp, y, p)
            loss = a + 0.5 * b
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(perm[s:s + 128])
        vi, vp = val_loss()
        log.info("    TFT epoch %2d/%d train=%.4f | val inc=%.4f (prior %.4f) pos=%.4f (prior %.4f)",
                 ep + 1, args.tft_epochs, tl / len(perm), vi, vi0, vp, vp0)
        if vi + 0.5 * vp < best - 1e-5:
            best, bad = vi + 0.5 * vp, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                log.info("    TFT early stop at epoch %d", ep + 1)
                break
    net.load_state_dict(best_state)
    net.eval()
    log.info("  TFT fit t_end=%d samples=%d best_val=%.4f vs prior %.4f (%s) %.1fs", t_end, len(tr), best,
             vi0 + 0.5 * vp0, "improves on prior" if best < vi0 + 0.5 * vp0 - 1e-5 else "fell back to prior",
             time.time() - t_fit)
    return net


def tft_predict(net, td, a, b, args, dev):
    import torch
    L = args.tft_seq_len
    pi, pp, aw, vs = [], [], [], []
    with torch.no_grad():
        for s in range(a, b, 512):
            ix = np.arange(s, min(s + 512, b))
            seq, c, pr, pprior, _, _ = _batch(td, ix, L, dev)
            li, lp, w, sel = net(seq, c, pr, pprior)
            pi.append(torch.sigmoid(li).cpu().numpy())
            pp.append(torch.softmax(lp, -1).cpu().numpy())
            aw.append(w.cpu().numpy())
            vs.append(sel.cpu().numpy())
    return (s2._norm_inc(np.vstack(pi).astype(float)), np.vstack(pp).astype(float),
            np.vstack(aw).mean(0), np.vstack(vs).mean(0))


def walk_forward_tft(td, a0, n, args, dev):
    inc = np.empty((n - a0, N_BALLS))
    pos = np.empty((n - a0, N_DRAWN, N_BALLS))
    blocks = list(range(a0, n, args.tft_every))
    log.info("TFT walk-forward: %d refits, %d test draws, device=%s", len(blocks), n - a0, dev)
    net, aws, vss = None, [], []
    for i, a in enumerate(blocks):
        b = min(a + args.tft_every, n)
        if net is None or not s2.over_cap(args):
            net = fit_tft(td, a, args, dev)
        else:
            log.warning("  time cap reached: reusing previous TFT")
        inc[a - a0:b - a0], pos[a - a0:b - a0], aw, vs = tft_predict(net, td, a, b, args, dev)
        aws.append(aw)
        vss.append(vs)
        log.info("  TFT block %d/%d done (draws %d..%d) elapsed %.0fs", i + 1, len(blocks), a, b - 1, time.time() - s2.T0)
    return inc, pos, np.mean(aws, 0), np.mean(vss, 0)


# --------------------------------------------------------------------------- #
# Head-to-head against the frozen-frequency benchmark
# --------------------------------------------------------------------------- #
def versus_static(per_draw: pd.DataFrame, experts, mu0):
    rows = []
    g0 = per_draw["gain_freq_static"].to_numpy()
    h0 = per_draw["hits_freq_static"].to_numpy() if "hits_freq_static" in per_draw else None
    for k in experts:
        if k in ("uniform", "freq_static"):
            continue
        dg = per_draw[f"gain_{k}"].to_numpy() - g0
        t = stats.ttest_1samp(dg, 0.0, alternative="greater")
        row = {"expert": k, "gain_diff": float(dg.mean()), "se": float(dg.std(ddof=1) / math.sqrt(len(dg))),
               "p_better": float(t.pvalue), "p_worse": float(1 - t.pvalue)}
        if h0 is not None and f"hits_{k}" in per_draw:
            dh = per_draw[f"hits_{k}"].to_numpy() - h0
            row["hits_diff"] = float(dh.mean())
            row["p_hits"] = float(stats.ttest_1samp(dh, 0.0, alternative="greater").pvalue) if dh.std() > 0 else 1.0
        rows.append(row)
    if rows:
        q = s2.bh_qvalues([r["p_better"] for r in rows])
        for r, qq in zip(rows, q):
            r["q_bh"] = float(qq)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    s2.T0 = time.time()
    out_dir = Path(args.out_dir)
    tag = Path(args.csv).stem
    logpath = s2.setup_logging(out_dir, "stage3_" + tag)
    rng = np.random.default_rng(args.seed)
    log.info("Stage 3 start | csv=%s | log=%s", args.csv, logpath)
    log.info("args=%s", vars(args))

    d = s2.load_data(args.csv, out_dir, args.next_date, args.next_shifted, args.next_holiday_shift)
    n = d.n
    n_bt = min(args.backtest, n - args.min_train)
    if n_bt < 100:
        raise ValueError(f"only {n} clean draws; need at least {args.min_train + 100}")
    a0 = n - n_bt
    log.info("Backtest: last %d draws (%s .. %s); %d draws available before it", n_bt, d.dates[a0].date(),
             d.dates[n - 1].date(), a0)
    s2.leakage_check(d, args.half_life)
    leakage_check_ext(d)

    F = extend_features(d, s2.build_ball_features(d))
    log.info("Feature tensor %s (%.1f MB). rate100 is feature #%d of %d: %s", F.shape, F.nbytes / 1e6,
             I_RATE100, F.shape[2], ALL_NAMES)
    macro, micro, posf = s2.freq_experts(d, args.half_life)
    static, _, pos_static = s2.freq_experts(d, STATIC_HL, k_macro=200.0)

    inc = {"uniform": np.full((n_bt, N_BALLS), BASE), "freq_static": static[a0:n], "freq_macro": macro[a0:n],
           "freq_micro": micro[a0:n]}
    posd = {"uniform": np.full((n_bt, N_DRAWN, N_BALLS), 1 / N_BALLS), "pos_static": pos_static[a0:n],
            "pos_freq": posf[a0:n]}
    nxt_inc = {"freq_static": static[n], "freq_macro": macro[n], "freq_micro": micro[n]}
    nxt_pos = {"uniform": np.full((N_DRAWN, N_BALLS), 1 / N_BALLS), "pos_static": pos_static[n], "pos_freq": posf[n]}
    extras = {}

    # default GBM (Stage 2) on the extended features
    if not args.no_gbm:
        inc["gbm"] = s2.walk_forward_gbm(F, d.X, a0, n, args)
        m, imp = s2.fit_gbm(F, d.X, n, args)
        nxt_inc["gbm"] = s2.gbm_predict(m, F, n, n + 1)[0]

    # Optuna-tuned GBM (tuned on draws < a0 only)
    tuned_params, tune_info = (None, {"skipped": True})
    if not args.no_gbm and not args.no_optuna:
        tuned_params, tune_info = tune_gbm(F, d.X, a0, args)
    extras["optuna"] = tune_info
    if tuned_params is not None:
        inc["gbm_tuned"] = walk_forward_tuned(F, d.X, a0, n, tuned_params, args)
        m, imp = fit_tuned(F, d.X, n, tuned_params, args)
        nxt_inc["gbm_tuned"] = s2.gbm_predict(m, F, n, n + 1)[0]
        extras["gbm_importance"] = sorted(zip(ALL_NAMES, map(float, imp)), key=lambda x: -x[1])
        log.info("Tuned GBM gain share (NOT a measure of skill): %s",
                 [(k, round(v, 3)) for k, v in extras["gbm_importance"][:8]])

    # TFT-lite
    if not args.no_tft:
        try:
            import torch
            dev = s2.torch_device(args)
            td = TFTData(d, F, static, pos_static, dev)
            tft_selftest(td, d, args, dev)
            ti, tp, aw, vs = walk_forward_tft(td, a0, n, args, dev)
            inc["tft"], posd["tft"] = ti, tp
            net = fit_tft(td, n, args, dev)
            ni, npos, aw_n, vs_n = tft_predict(net, td, n, n + 1, args, dev)
            nxt_inc["tft"], nxt_pos["tft"] = ni[0], npos[0]
            extras["attention_lag"] = aw_n.tolist()
            extras["group_weights"] = dict(zip(["rate100", "long_run_freq", "gap", "calendar"], map(float, vs_n)))
            extras["group_weights_backtest"] = dict(zip(["rate100", "long_run_freq", "gap", "calendar"], map(float, vs)))
            log.info("TFT input-group selection weights (final model): %s",
                     {k: round(v, 3) for k, v in extras["group_weights"].items()})
        except ImportError:
            log.warning("torch not installed: TFT skipped")

    log.info("=== Evaluation ===")
    res, inc, wf, pwf, mix_names, pnames, per_draw = s2.evaluate(d, a0, inc, posd, args, rng)
    mu0 = N_DRAWN * N_DRAWN / N_BALLS
    vs_static = versus_static(per_draw, list(inc), mu0)
    for k, v in res["experts"].items():
        log.info("  %-11s log-score gain/draw vs uniform=%+.5f | mean hits=%s", k, v["gain_per_draw"],
                 f"{res['hits'][k]['mean_hits']:.4f}" if k in res["hits"] else "-")
    for r in vs_static:
        log.info("  vs frozen frequency | %-10s gain diff=%+.5f (p better=%.3f, p worse=%.3f)", r["expert"],
                 r["gain_diff"], r["p_better"], r["p_worse"])

    # top-2 pair hit rates per expert (the 'Perm 2' style pick)
    Xbt = d.X[a0:n]
    ar = np.arange(n_bt)[:, None]
    pair_rows = {}
    p0 = N_DRAWN * (N_DRAWN - 1) / (N_BALLS * (N_BALLS - 1))
    for k in inc:
        if k == "uniform":
            continue
        tp = np.argsort(-inc[k], axis=1)[:, :2]
        both = int((Xbt[ar, tp].sum(1) == 2).sum())
        lo_, hi_ = wilson(both, n_bt)
        pair_rows[k] = {"both": both, "rate": both / n_bt, "ci": [lo_, hi_],
                        "p_vs_chance": float(stats.binomtest(both, n_bt, p0, alternative="greater").pvalue)}
    rnd = np.argsort(rng.random((n_bt, N_BALLS)), axis=1)[:, :2]
    rb = int((Xbt[ar, rnd].sum(1) == 2).sum())
    pair_rows["random"] = {"both": rb, "rate": rb / n_bt, "ci": list(wilson(rb, n_bt)), "p_vs_chance": float("nan")}

    # next-draw distribution
    p_next = sum(wf[i] * (np.full(N_BALLS, BASE) if k == "uniform" else nxt_inc[k]) for i, k in enumerate(mix_names))
    pos_next = sum(pwf[i] * nxt_pos[k] for i, k in enumerate(pnames))
    pos_next = pos_next / pos_next.sum(1, keepdims=True)
    twin = s2.run_twin(d, p_next, pos_next, args, rng)

    # ------------------------------------------------------------------ report
    sig = res["survivors_bh"]
    better = [r for r in vs_static if r.get("q_bh", 1) < 0.05]
    worse = [r for r in vs_static if r["p_worse"] < 0.01]
    L = []
    w = L.append
    w("=" * 78)
    w(f"STAGE 3 REPORT | {Path(args.csv).name} | draws {d.info['date_min']} .. {d.info['date_max']} | N={n}")
    w("=" * 78)
    w(f"Walk-forward backtest on the last {n_bt} draws. Optuna tuned on draws < {a0} only. Tests: {res['n_tests']}.")
    w("")
    w("1) EXPERTS vs UNIFORM (log-score gain per draw; mean hits of the top-5 pick; chance = %.4f)" % mu0)
    w(f"{'expert':<13}{'gain/draw':>12}{'p':>9}{'mean hits':>11}{'z':>7}")
    for k, v in res["experts"].items():
        h = res["hits"].get(k, {})
        w(f"{k:<13}{v['gain_per_draw']:>+12.5f}{v.get('gain_p', float('nan')):>9.4f}"
          f"{h.get('mean_hits', float('nan')):>11.4f}{h.get('z', float('nan')):>7.1f}")
    w("")
    w("2) DOES ANYTHING HEAVIER BEAT THE FROZEN-FREQUENCY BENCHMARK? (paired, same draws)")
    w(f"{'expert':<13}{'gain diff':>11}{'se':>9}{'p better':>10}{'q (BH)':>9}{'p worse':>9}{'hit diff':>10}")
    for r in vs_static:
        w(f"{r['expert']:<13}{r['gain_diff']:>+11.5f}{r['se']:>9.5f}{r['p_better']:>10.3f}{r.get('q_bh', float('nan')):>9.3f}"
          f"{r['p_worse']:>9.3f}{r.get('hits_diff', float('nan')):>+10.4f}")
    if better:
        w("  -> Beating the benchmark (BH q<0.05): " + ", ".join(r["expert"] for r in better))
    elif worse:
        w("  -> Nothing beats the benchmark. Significantly WORSE than it: " + ", ".join(r["expert"] for r in worse))
    else:
        w("  -> Nothing beats the benchmark; the heavy models add no measurable skill beyond long-run frequency.")
    w("")
    w("3) TOP-2 PICK (both numbers drawn), %d test draws; chance %.2f%% = %.1f draws" % (n_bt, 100 * p0, n_bt * p0))
    w(f"{'expert':<13}{'both':>6}{'rate':>8}{'95% CI':>18}{'p vs chance':>13}")
    for k, v in pair_rows.items():
        ci_txt = "[%.2f%%, %.2f%%]" % (100 * v["ci"][0], 100 * v["ci"][1])
        w(f"{k:<13}{v['both']:>6}{100 * v['rate']:>7.2f}%{ci_txt:>18}{v['p_vs_chance']:>13.4f}")
    if args.payout > 0:
        be = 100 / args.payout
        w(f"  Break-even at a {args.payout:g}x payout needs a pair hit rate above {be:.2f}% "
          f"(stake returns p*{args.payout:g}). Compare the CI above, not the point estimate.")
    w("")
    w("4) HIT TABLE (top-5 pick of the ensemble vs random control vs chance)")
    w(f"{'hits':<6}{'ensemble':>10}{'random':>10}{'expected':>10}{'P(>=k) p':>11}")
    for j in range(2, 6):
        w(f"{j:<6}{res['hits']['ensemble']['exact'][str(j)]:>10}{res['hits']['random']['exact'][str(j)]:>10}"
          f"{res['hits']['expected_exact'][str(j)]:>10.2f}{res['hits']['ensemble']['binom_p_at_least'][str(j)]:>11.4f}")
    w(f"Tests surviving BH: {len(sig)} of {res['n_tests']} | Bonferroni: {len(res['survivors_bonferroni'])}")
    w("")
    w("5) OPTUNA")
    o = extras["optuna"]
    if o.get("skipped"):
        w("  skipped")
    else:
        w(f"  {o['trials']} trials on pre-backtest data. Inner log-loss: tuned {o['best_inner_logloss']:.5f} | "
          f"default {o['default_inner_logloss']:.5f} | constant {o['constant_inner_logloss']:.5f}")
        w(f"  best params: {o['best_params']}")
        w("  (best-of-N inner score is optimistic by construction; the backtest above is the honest number)")
    w("")
    w("6) WHAT THE MODELS LOOK AT")
    if "gbm_importance" in extras:
        gi = dict(extras["gbm_importance"])
        w(f"  Tuned GBM gain share: " + ", ".join(f"{k}={v:.3f}" for k, v in extras["gbm_importance"][:6]))
        w(f"  rate100={gi['rate100']:.3f}, rate_all={gi['rate_all']:.3f}, ball_id={gi['ball_id']:.3f}. Gain share says which "
          "feature the trees split on, not whether it predicts better than the frozen table.")
    if "group_weights" in extras:
        w("  TFT input-group weights: " + ", ".join(f"{k}={v:.3f}" for k, v in extras["group_weights"].items()))
        lag = np.array(extras["attention_lag"])
        Lq = len(lag)
        edges = [(1, 5), (6, 15), (16, 30), (31, Lq)]
        parts = []
        for a_, b_ in edges:
            if a_ > Lq:
                continue
            b_ = min(b_, Lq)
            mass = lag[Lq - b_:Lq - a_ + 1].sum()
            parts.append(f"lag {a_}-{b_}: {100 * mass:.1f}% (uniform {100 * (b_ - a_ + 1) / Lq:.1f}%)")
        w("  TFT attention mass by lag: " + " | ".join(parts))
        w("  Attention near-uniform over lags = the model is using long-run structure, not 'recent draws matter'.")
    w("")
    nd = d.dates[n].day_name()
    order = np.argsort(-p_next)
    w(f"7) CONFIDENCE BOARD for the draw on {d.info['next_date']} ({nd}); probability a ball is among the 5 drawn")
    w(f"   chance for a ball under a fair machine = {100 * BASE:.2f}%")
    ws = {k: round(float(v), 3) for k, v in res["ensemble_weights_final"].items()}
    w(f"   ensemble weights: {ws}")
    w(f"   {'#':>3}{'ball':>6}{'ensemble':>10}{'frozen freq':>13}")
    for r_, b in enumerate(order[:15], 1):
        w(f"   {r_:>3}{b + 1:>6}{100 * p_next[b]:>9.2f}%{100 * nxt_inc['freq_static'][b]:>12.2f}%")
    w("")
    top2 = [int(b) + 1 for b in order[:2]]
    top2s = [int(b) + 1 for b in np.argsort(-nxt_inc['freq_static'])[:2]]
    w("8) FINAL CALL")
    w(f"   Ensemble top 2: {top2[0]} and {top2[1]}   | frozen-frequency top 2: {top2s[0]} and {top2s[1]}")
    pe = pair_rows["freq_static"]
    w(f"   Backtested top-2 pair hit rate of the frozen table: {100 * pe['rate']:.2f}% "
      f"(CI {100 * pe['ci'][0]:.2f}-{100 * pe['ci'][1]:.2f}%) vs {100 * p0:.2f}% under a fair machine.")
    edge = any(x.startswith(("freq_static", "ensemble", "freq_macro", "gbm", "tft")) and "mean hits" in x or
               x.startswith(("freq_static", "ensemble")) for x in sig)
    if better:
        w("   A heavier model beat the benchmark. Confirm on draws not in this file before relying on it.")
    elif edge:
        w("   The edge is the long-run frequency table. Use the simplest model that carries it, and confirm on draws")
        w("   that arrive AFTER this file ends.")
    else:
        w("   NO EDGE: nothing survived multiple-testing correction against a fair machine. These picks are arbitrary.")
    w("   Draws are still random events: a rate well above chance can still be below break-even for the payout.")
    w("")
    w("9) DIGITAL TWIN CALIBRATION")
    w(f"{'statistic':<30}{'real':>12}{'twin p':>9}{'uniform p':>11}")
    for r in twin["rows"]:
        w(f"{r['statistic']:<30}{r['real']:>12.3f}{r['twin_p']:>9.3f}{r['uniform_machine_p']:>11.3f}")
    w("  The twin is a calibration check, not a predictor. The uniform column should reject a biased file. The twin")
    w("  column is partly circular (its weights were fitted to this very history), so a p near 1.000 on the frequency")
    w("  and pair rows is expected and proves nothing. Only a LOW twin p on sum/overlap/pair would point to structure")
    w("  beyond single-ball frequencies. Its top-2 simply echoes the table above.")
    w("")
    w(f"Elapsed: {(time.time() - s2.T0) / 60:.1f} min")
    report = "\n".join(L)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"stage3_report_{tag}_{stamp}.txt").write_text(report, encoding="utf-8")
    per_draw.to_csv(out_dir / f"stage3_backtest_{tag}_{stamp}.csv", index=False)
    payload = {"info": d.info, "results": res, "versus_static": vs_static, "pairs": pair_rows, "extras": extras,
               "twin": twin, "next": {"date": d.info["next_date"], "p": p_next.tolist(), "top2": top2}}
    (out_dir / f"stage3_results_{tag}_{stamp}.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n" + report)
    log.info("Saved report/CSV/JSON to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
