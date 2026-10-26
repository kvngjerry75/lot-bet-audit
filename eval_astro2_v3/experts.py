"""
experts.py: hazard, markov, LambdaRank ranker and the TFT army, plus the walk-forward runner.

Every expert returns per-draw INCLUSION probabilities (each row sums to 5) for a block of draws, trained only on draws
before the block. An expert is flagged `active` only when its validation log-loss beats the uniform forecast;
inactive experts get vote weight 0 in the ensemble (nothing is faked).

No randomness is added to any output: seeds only make training reproducible. Ties are broken by ball order (stable sort).
"""
from __future__ import annotations

import copy
import logging
import math
import time

import warnings

import numpy as np

import features_recent as fr
from nla_config import N_DRAWN, uniform_bce

log = logging.getLogger("nla.experts")
warnings.filterwarnings("ignore", message=".*eval_set.*")   # newer lightgbm renames the argument; both work
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")   # numpy arrays in, numpy arrays out


# --------------------------------------------------------------------------- #
# shared scoring
# --------------------------------------------------------------------------- #
def bce_rows(p: np.ndarray, Xv: np.ndarray) -> float:
    """Mean over draws of the per-draw summed binary cross-entropy."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((-(Xv * np.log(p) + (1 - Xv) * np.log1p(-p)).sum(1)).mean())


def skill_of(bce: float, nb: int) -> float:
    return 1.0 - bce / uniform_bce(nb)


# --------------------------------------------------------------------------- #
# Hazard expert: P(ball drawn | how overdue it is), pooled across balls, estimated on a trailing window
# --------------------------------------------------------------------------- #
def _hz_cells(ratio, pct, hc):
    re, pe = np.asarray(hc["ratio_edges"]), np.asarray(hc["pct_edges"])
    return np.digitize(ratio, re) * (len(pe) + 1) + np.digitize(pct, pe), (len(re) + 1) * (len(pe) + 1)


def hazard_block(ratio, pct, X, a, b, cfg, nb):
    hc = cfg["hazard"]
    base = N_DRAWN / nb
    h0 = max(hc["warmup"], a - hc["window"])
    cell, ncell = _hz_cells(ratio[h0:a], pct[h0:a], hc)
    cnt = np.bincount(cell.ravel(), minlength=ncell).astype(float)
    hit = np.bincount(cell.ravel(), weights=X[h0:a].ravel(), minlength=ncell)
    s = hc["prior"]
    table = (hit + s * base) / (cnt + s)
    cb, _ = _hz_cells(ratio[a:b], pct[a:b], hc)
    return fr.norm_inc(table[cb])


# --------------------------------------------------------------------------- #
# Ranker (LightGBM LambdaRank; sklearn HistGB fallback)
# --------------------------------------------------------------------------- #
def _scores_to_probs(s: np.ndarray, tau: float) -> np.ndarray:
    z = s / tau
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return fr.norm_inc(e / e.sum(1, keepdims=True) * N_DRAWN)


def fit_ranker(F, X, t_end, cfg, seed):
    rc = cfg["ranker"]
    nb, nf = X.shape[1], F.shape[2]
    t0 = max(rc["warmup"], t_end - rc["train_window"])
    n_val = max(int(rc["val_frac"] * (t_end - t0)), rc["min_val"])
    split = t_end - n_val
    Xa, ya = F[t0:split].reshape(-1, nf), X[t0:split].reshape(-1).astype(int)
    Xv, yv = F[split:t_end].reshape(-1, nf), X[split:t_end].reshape(-1).astype(int)
    try:
        import lightgbm as lgb
        model = lgb.LGBMRanker(objective="lambdarank", random_state=seed, verbose=-1, **rc["params"])
        model.fit(Xa, ya, group=[nb] * (split - t0), eval_set=[(Xv, yv)], eval_group=[[nb] * (t_end - split)],
                  eval_at=[N_DRAWN], callbacks=[lgb.early_stopping(rc["early_stopping"], verbose=False)])
        score = lambda A: model.predict(A)
        kind = "lightgbm-lambdarank"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=10.0, min_samples_leaf=200, early_stopping=True,
                                               validation_fraction=0.15, random_state=seed)
        model.fit(np.vstack([Xa, Xv]), np.concatenate([ya, yv]))
        score = lambda A: model.decision_function(A)
        kind = "sklearn-hgb"
    sv = score(Xv).reshape(t_end - split, nb)
    Xval = X[split:t_end].astype(float)
    best_tau, best_bce = None, float("inf")
    for tau in rc["tau_grid"]:
        bc = bce_rows(_scores_to_probs(sv, tau), Xval)
        if bc < best_bce:
            best_tau, best_bce = tau, bc
    sk = skill_of(best_bce, nb)
    log.debug("  ranker[%s] t_end=%d rows=%d tau=%.2f val_skill=%+.5f", kind, t_end, len(ya) // nb, best_tau, sk)
    return dict(score=score, tau=best_tau, skill=sk, active=bool(sk > 0.0))


def ranker_predict(bundle, F, a, b, nb):
    nf = F.shape[2]
    s = bundle["score"](F[a:b].reshape(-1, nf)).reshape(b - a, nb)
    return _scores_to_probs(s, bundle["tau"])


def walk_forward_ranker(F, X, P, a0, n, cfg, seed, cad, predict_next=False):
    """Returns (probs (n-a0, nb), info dict, next) with next = (vector, active) when predict_next else None."""
    nb = X.shape[1]
    out = np.empty((n - a0, nb))
    act = np.zeros(n - a0, dtype=bool)
    skills = []
    for a in range(a0, n, cad):
        b = min(a + cad, n)
        bd = fit_ranker(F, X, a, cfg, seed)
        out[a - a0:b - a0] = ranker_predict(bd, F, a, b, nb)
        act[a - a0:b - a0] = bd["active"]
        skills.append(bd["skill"])
    nxt = None
    if predict_next:
        bd = fit_ranker(F, X, n, cfg, seed)
        nxt = (ranker_predict(bd, F, n, n + 1, nb)[0], bd["active"])
        skills.append(bd["skill"])
    return out, dict(active=act, skills=skills), nxt


# --------------------------------------------------------------------------- #
# Pointwise learners: MLP and gradient booster (per-ball rows, binary target "ball drawn")
# --------------------------------------------------------------------------- #
def _make_pointwise(kind, cfg, seed):
    pc = cfg[kind]
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(pc["hidden"]), alpha=pc["alpha"], learning_rate_init=pc["lr"],
            batch_size=pc["batch"], max_iter=pc["max_iter"], early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=pc["patience"], random_state=seed))
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", random_state=seed, verbose=-1, **pc["params"])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=10.0,
                                              min_samples_leaf=300, random_state=seed)


def fit_pointwise(kind, F, X, t_end, cfg, seed):
    """Train on draws [t0, split), choose a temperature on [split, t_end); active only if validation skill > 0."""
    pc = cfg[kind]
    nb, nf = X.shape[1], F.shape[2]
    t0 = max(pc["warmup"], t_end - pc["train_window"])
    n_val = max(int(pc["val_frac"] * (t_end - t0)), pc["min_val"])
    split = t_end - n_val
    model = _make_pointwise(kind, cfg, seed)
    model.fit(F[t0:split].reshape(-1, nf), X[t0:split].reshape(-1).astype(int))
    score = lambda A: np.log(np.clip(model.predict_proba(A)[:, 1], 1e-9, 1.0))
    sv = score(F[split:t_end].reshape(-1, nf)).reshape(t_end - split, nb)
    Xval = X[split:t_end].astype(float)
    best_tau, best_bce = None, float("inf")
    for tau in cfg["ranker"]["tau_grid"]:
        bc = bce_rows(_scores_to_probs(sv, tau), Xval)
        if bc < best_bce:
            best_tau, best_bce = tau, bc
    sk = skill_of(best_bce, nb)
    log.debug("  %s t_end=%d tau=%.2f val_skill=%+.5f", kind, t_end, best_tau, sk)
    return dict(score=score, tau=best_tau, skill=sk, active=bool(sk > 0.0))


def walk_forward_pointwise(kind, F, X, a0, n, cfg, seed, cad, predict_next=False, quiet=False):
    nb = X.shape[1]
    out = np.empty((n - a0, nb))
    act = np.zeros(n - a0, dtype=bool)
    skills = []
    for a in range(a0, n, cad):
        b = min(a + cad, n)
        bd = fit_pointwise(kind, F, X, a, cfg, seed)
        out[a - a0:b - a0] = ranker_predict(bd, F, a, b, nb)
        act[a - a0:b - a0] = bd["active"]
        skills.append(bd["skill"])
        if not quiet:
            log.info("  %s fit @%d: val skill %+.5f -> %s", kind, a, bd["skill"], "active" if bd["active"] else "silent")
    nxt = None
    if predict_next:
        bd = fit_pointwise(kind, F, X, n, cfg, seed)
        nxt = (ranker_predict(bd, F, n, n + 1, nb)[0], bd["active"])
        skills.append(bd["skill"])
        if not quiet:
            log.info("  %s fit @%d (NEXT DRAW): val skill %+.5f -> %s", kind, n, bd["skill"],
                     "active" if bd["active"] else "silent")
    return out, dict(active=act, skills=skills), nxt


# --------------------------------------------------------------------------- #
# TFT army
# --------------------------------------------------------------------------- #
def torch_device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_torch_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    class GRN(nn.Module):
        """Gated residual network (TFT building block)."""
        def __init__(self, d_in, d_h, d_out, drop):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h)
            self.gate = nn.Linear(d_h, 2 * d_out)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.norm, self.drop = nn.LayerNorm(d_out), nn.Dropout(drop)

        def forward(self, x):
            h = self.drop(self.fc2(Fnn.elu(self.fc1(x))))
            a, g = self.gate(h).chunk(2, dim=-1)
            return self.norm(self.skip(x) + a * torch.sigmoid(g))

    class TFTLite(nn.Module):
        """Sequence context (GRN -> LSTM -> self-attention -> GRN, enriched by the target draw's calendar) combined with
        a per-ball head over the recent-momentum features.  logit = logit(k/N) + residual; the residual's last layer is
        ZERO-initialised, so a network that learns nothing outputs UNIFORM scores (never the historical frequency)."""
        def __init__(self, nb, n_seq, n_cal, nf, hidden, heads, drop):
            super().__init__()
            self.inp = nn.Linear(n_seq, hidden)
            self.grn_in = GRN(hidden, hidden, hidden, drop)
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=drop, batch_first=True)
            self.grn_attn = GRN(hidden, hidden, hidden, drop)
            self.cal = nn.Linear(n_cal, hidden)
            self.enrich = GRN(2 * hidden, hidden, hidden, drop)
            self.ball_in = nn.Linear(nf + hidden, hidden)
            self.ball_drop = nn.Dropout(drop)
            self.out = nn.Linear(hidden, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            k = N_DRAWN / nb
            self.register_buffer("prior_logit", torch.tensor(math.log(k / (1 - k)), dtype=torch.float32))

        def forward(self, seq, cal_t, fb):
            h = self.grn_in(self.inp(seq))
            o, _ = self.lstm(h)
            a, _ = self.attn(o, o, o, need_weights=False)
            z = self.grn_attn(o + a)[:, -1]
            z = self.enrich(torch.cat([z, torch.relu(self.cal(cal_t))], dim=-1))
            zb = z[:, None, :].expand(-1, fb.shape[1], -1)
            u = self.ball_drop(Fnn.elu(self.ball_in(torch.cat([fb, zb], dim=-1))))
            return self.prior_logit + self.out(u).squeeze(-1)

    return TFTLite


class TFTArmy:
    """Per-run holder of the (read-only) tensors; trains one block of the army at a time."""

    def __init__(self, d, fb, cfg, device="auto", quiet=False):
        import torch
        self.torch = torch
        self.cfg, self.tc = cfg, cfg["tft"]
        self.quiet = quiet
        self.dev = torch_device(device)
        self.TFTLite = _build_torch_classes()
        nb = d.X.shape[1]
        self.nb, self.n = nb, d.n
        X = np.array(d.X, dtype=np.float32)
        X[d.n:] = 0.0
        cal = np.asarray(d.cal, dtype=np.float32)
        self.Xnp = X
        self.F = fb.F
        self.n_cal = cal.shape[1]
        self.S = torch.from_numpy(np.concatenate([X, cal], axis=1)).to(self.dev)
        self.Cal = torch.from_numpy(cal).to(self.dev)
        self.Y = torch.from_numpy(X).to(self.dev)
        self.nf = fb.F.shape[2]

    # ----- data for rows -----
    def _seq(self, rows):
        L = self.tc["seq_len"]
        idx = self.torch.as_tensor(rows, device=self.dev)[:, None] + self.torch.arange(-L, 0, device=self.dev)[None, :]
        return self.S[idx]

    def _batch(self, rows, Fz, off):
        r = self.torch.as_tensor(rows, device=self.dev)
        return self._seq(rows), self.Cal[r], Fz[r - off], self.Y[r]

    # ----- one member -----
    def _train_member(self, rows_tr, rows_va, Fz, off, seed):
        torch, tc, nb = self.torch, self.tc, self.nb
        torch.manual_seed(seed)
        model = self.TFTLite(nb, self.S.shape[1], self.n_cal, self.nf, tc["hidden"], tc["heads"], tc["dropout"]).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        gen = torch.Generator().manual_seed(seed)
        uni = uniform_bce(nb)
        best_bce, best_state, best_ep, bad = uni, copy.deepcopy(model.state_dict()), 0, 0   # epoch 0 = uniform forecast
        ntr, bs = len(rows_tr), tc["batch"]

        def val_bce():
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for i in range(0, len(rows_va), 512):
                    seq, cal, fz, y = self._batch(rows_va[i:i + 512], Fz, off)
                    p = torch.sigmoid(model(seq, cal, fz))
                    p = (p * (N_DRAWN / p.sum(1, keepdim=True))).clamp(1e-6, 1 - 1e-6)
                    tot += float(-(y * p.log() + (1 - y) * (1 - p).log()).sum())
            return tot / len(rows_va)

        for ep in range(1, tc["max_epochs"] + 1):
            model.train()
            perm = torch.randperm(ntr, generator=gen).numpy()
            for i in range(0, ntr, bs):
                seq, cal, fz, y = self._batch(rows_tr[perm[i:i + bs]], Fz, off)
                logits = model(seq, cal, fz)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1).mean()
                lst = -(y / N_DRAWN * torch.log_softmax(logits, dim=1)).sum(1).mean()      # listwise softmax CE over the pool
                l2 = tc["l2_head"] * model.out.weight.pow(2).sum()                         # L2 on the residual head
                loss = bce + tc["list_weight"] * lst + l2
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()
            vb = val_bce()
            if vb < best_bce - 1e-6:
                best_bce, best_state, best_ep, bad = vb, copy.deepcopy(model.state_dict()), ep, 0
            else:
                bad += 1
            if ep >= tc["min_epochs"] and bad >= tc["patience"]:
                break
        model.load_state_dict(best_state)
        model.eval()
        return model, skill_of(best_bce, nb), best_ep, ep

    def _predict(self, model, rows, Fz, off):
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(rows), 512):
                seq, cal, fz, _ = self._batch(rows[i:i + 512], Fz, off)
                outs.append(torch.sigmoid(model(seq, cal, fz)).cpu().numpy())
        return fr.norm_inc(np.concatenate(outs, 0).astype(float))

    # ----- one block of the army -----
    def block(self, t_end, a, b, seed):
        """Train on draws < t_end, predict rows a..b-1. Returns (probs, active, skills)."""
        tc, nb = self.tc, self.nb
        t0 = max(tc["warmup"], t_end - tc["train_window"])
        n_val = max(int(tc["val_frac"] * (t_end - t0)), tc["min_val"])
        split = t_end - n_val
        rows_tr, rows_va = np.arange(t0, split), np.arange(split, t_end)
        flat = self.F[t0:split].reshape(-1, self.nf)                   # standardise on TRAIN rows only
        mu, sd = flat.mean(0), flat.std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        hi = max(b, t_end)
        Fz = self.torch.from_numpy(((self.F[t0:hi] - mu) / sd).astype(np.float32)).to(self.dev)
        rows_pred = np.arange(a, b)
        probs, skills = [], []
        for m in range(tc["army"]):
            model, sk, best_ep, last_ep = self._train_member(rows_tr, rows_va, Fz, t0, seed * 1_000_003 + t_end * 101 + m)
            skills.append(float(sk))
            ok = sk > tc["skill_margin"]
            if not self.quiet:
                (log.info if ok else log.warning)(
                    "  TFT member %d/%d t_end=%d: val skill %+.5f (best epoch %d of %d)%s", m + 1, tc["army"], t_end, sk,
                    best_ep, last_ep, "" if ok else "  -> NO validated skill: vote weight 0")
            if ok:
                probs.append(self._predict(model, rows_pred, Fz, t0))
        if probs:
            return fr.norm_inc(np.mean(probs, axis=0)), True, skills
        return np.full((b - a, nb), N_DRAWN / nb), False, skills


# --------------------------------------------------------------------------- #
# Walk-forward runner
# --------------------------------------------------------------------------- #
def run_experts(d, fb, cfg, span_start, n, seed, enabled, cad, predict_next=False, device="auto", quiet=False):
    """Blocks of `cad` draws from span_start to n; every block's experts are fitted on draws before the block."""
    nb = d.X.shape[1]
    X = np.array(d.X, dtype=np.float32)
    X[n:] = 0.0
    nrow = n - span_start
    names = [k for k in ("tft_army", "ranker", "hazard", "markov", "anchor", "mlp", "booster", "rulemem", "rulemine")
             if k in enabled]
    fr.assert_no_banned_experts(names, cfg, "run_experts")
    fr.assert_no_banned(fb.names, cfg, "run_experts", allow_rate100=bool(cfg["features"].get("use_rate100")))
    res = dict(probs={k: np.empty((nrow, nb)) for k in names}, active={k: np.ones(nrow, dtype=bool) for k in names},
               next={}, next_active={}, tft_skills=[])
    blocks = [(a, min(a + cad, n)) for a in range(span_start, n, cad)]
    t0 = time.time()

    if "hazard" in names:
        ratio, pct = fb.col("gap_ratio"), fb.col("gap_percentile")
        for a, b in blocks:
            res["probs"]["hazard"][a - span_start:b - span_start] = hazard_block(ratio, pct, X, a, b, cfg, nb)
        if predict_next:
            res["next"]["hazard"] = hazard_block(ratio, pct, X, n, n + 1, cfg, nb)[0]
            res["next_active"]["hazard"] = True
    if "markov" in names:
        mk = fb.col("markov")
        res["probs"]["markov"][:] = fr.norm_inc(mk[span_start:n].astype(float))
        if predict_next:
            res["next"]["markov"] = fr.norm_inc(mk[n].astype(float))
            res["next_active"]["markov"] = True
    if "ranker" in names:
        pr, info, nxt = walk_forward_ranker(fb.F, X, d.P, span_start, n, cfg, seed, cad, predict_next)
        res["probs"]["ranker"][:] = pr
        res["active"]["ranker"][:] = info["active"]
        if nxt is not None:
            res["next"]["ranker"], res["next_active"]["ranker"] = nxt[0], bool(nxt[1])
    for kind in ("mlp", "booster"):
        if kind in names:
            pr, info, nxt = walk_forward_pointwise(kind, fb.F, X, span_start, n, cfg, seed, cad, predict_next, quiet)
            res["probs"][kind][:] = pr
            res["active"][kind][:] = info["active"]
            if nxt is not None:
                res["next"][kind], res["next_active"][kind] = nxt[0], bool(nxt[1])
    if "rulemem" in names:
        import rule_memory as rm
        rc = cfg.get("rulemem", {})
        pr = rm.hedge_probs(np.asarray(d.P[:n], dtype=int) + 1, d.dates[:n + 1], nb, span_start,
                            rc.get("hedge_window", 300), rc.get("eta", 0.5), rc.get("alpha", 0.5))
        res["probs"]["rulemem"][:] = pr[:nrow]
        if predict_next:
            res["next"]["rulemem"], res["next_active"]["rulemem"] = pr[nrow], True
    if "rulemine" in names:
        import rule_miner as rmn
        mc = cfg.get("rulemine", {})
        others = {}
        for g, path in (mc.get("others") or {}).items():          # optional cross-game sources (audited CSVs)
            others[g] = rmn.load(path)
        pr = rmn.miner_probs(np.asarray(d.P[:n], dtype=np.int64), d.dates[:n + 1], nb, span_start, others,
                             mc.get("alpha", 0.05))
        res["probs"]["rulemine"][:] = pr[:nrow]
        if predict_next:
            res["next"]["rulemine"], res["next_active"]["rulemine"] = pr[nrow], True
    if "anchor" in names:
        import anchor as an
        pr, act, info, nxt = an.run_anchor(d, fb, cfg, span_start, n, cad, predict_next, quiet=quiet)
        res["probs"]["anchor"][:] = pr
        res["active"]["anchor"][:] = act
        res["anchor_info"] = info
        if nxt is not None:
            res["next"]["anchor"], res["next_active"]["anchor"] = nxt[0], bool(nxt[1])
    if "tft_army" in names:
        army = TFTArmy(d, fb, cfg, device, quiet=quiet)
        for i, (a, b) in enumerate(blocks):
            p, ok, sk = army.block(a, a, b, seed)
            res["probs"]["tft_army"][a - span_start:b - span_start] = p
            res["active"]["tft_army"][a - span_start:b - span_start] = ok
            res["tft_skills"].append(dict(block_start=int(a), skills=sk, final=False))
            if not quiet:
                log.info("TFT block %d/%d (draws %d..%d) done | active=%s | %.0fs", i + 1, len(blocks), a, b - 1, ok, time.time() - t0)
        if predict_next:
            p, ok, sk = army.block(n, n, n + 1, seed)
            res["next"]["tft_army"], res["next_active"]["tft_army"] = p[0], bool(ok)
            res["tft_skills"].append(dict(block_start=int(n), skills=sk, final=True))
    return res
