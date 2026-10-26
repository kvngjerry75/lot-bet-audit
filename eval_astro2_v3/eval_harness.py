"""
eval_harness.py: the mandatory evaluation harness.

  * walk-forward evaluation over the last >= 300 draws, experts retrained at a fixed cadence
  * ensemble weights tuned ONLY on a rolling validation window that precedes every test block
  * metrics vs chance derived from N_BALLS: top-5 hits, top-K hits, Perm-2 core hit rate, log-loss skill vs uniform
  * null test: the entire evaluation repeated R times on data whose ball labels are permuted within each draw
    (resumable: replicates are checkpointed to disk, so a Colab disconnect does not lose the work)
  * max-T p-value over the metrics inside a game, Bonferroni across the games, 95% CIs
  * ablation table, chi-square uniformity test, Jaccard overlap (diagnostics only)
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats

import features_recent as fr
from experts import run_experts, walk_forward_ranker
from nla_config import (N_DRAWN, chance_core, chance_hits, chance_jaccard, core_size, topk_big, uniform_bce)

log = logging.getLogger("nla.eval")

ENSEMBLE_EXPERTS = ("tft_army", "ranker", "hazard", "markov", "anchor", "mlp", "booster", "rulemem", "rulemine")


# --------------------------------------------------------------------------- #
# Layout of the walk-forward span
# --------------------------------------------------------------------------- #
def layout(n: int, cfg, quick: bool = False):
    """span_start .. eval_start: validation pre-roll (weights tuned here); eval_start .. n: the test draws."""
    ec = cfg["eval"]
    cad = int(ec["retrain_every"])
    want = int(ec["eval_draws"])
    if want <= 0:
        # v3 "full" mode: score EVERY draw after the warm-up and the first validation window (expanding history)
        min_hist0 = max(cfg["tft"]["warmup"], cfg["tft"]["seq_len"], cfg["ranker"]["warmup"], cfg["hazard"]["warmup"]) \
            + int(ec.get("full_warmup", 200))
        val0 = int(math.ceil(cfg["ensemble"]["val_len"] / cad) * cad)
        want = max(cad, ((n - min_hist0 - val0) // cad) * cad)
        ec = dict(ec, min_eval_draws=min(ec["min_eval_draws"], want))
        cfg = dict(cfg, eval=ec)
    if not quick and want < ec["min_eval_draws"]:
        raise ValueError(f"eval_draws={want} < {ec['min_eval_draws']}: the harness needs at least "
                         f"{ec['min_eval_draws']} walk-forward draws per game")
    eval_n = int(math.ceil(want / cad) * cad)
    val_len = int(math.ceil(cfg["ensemble"]["val_len"] / cad) * cad)
    span_start = n - eval_n - val_len
    min_hist = max(cfg["tft"]["warmup"], cfg["tft"]["seq_len"], cfg["ranker"]["warmup"], cfg["hazard"]["warmup"]) \
        + int(ec.get("full_warmup", 300) if int(cfg["eval"]["eval_draws"]) <= 0 else 300)
    if span_start < min_hist:
        raise ValueError(f"only {n} draws: need >= {min_hist + eval_n + val_len} for eval={eval_n}, val={val_len}")
    return dict(cad=cad, eval_n=eval_n, val_len=val_len, span_start=span_start, eval_start=n - eval_n)


# --------------------------------------------------------------------------- #
# Ensemble: multipliers x validation-window skill weights
# --------------------------------------------------------------------------- #
def _rank_points(p: np.ndarray) -> np.ndarray:
    nb = p.shape[1]
    order = np.argsort(-p, axis=1, kind="stable")
    rank = np.empty_like(order)
    ar = np.arange(len(p))[:, None]
    rank[ar, order] = np.arange(nb)[None, :]
    return (nb - rank).astype(float)


def _val_gain(p: np.ndarray, P_rows: np.ndarray) -> float:
    """Total log-score gain vs uniform of the drawn balls under p (rows normalised)."""
    nb = p.shape[1]
    q = p / p.sum(1, keepdims=True)
    ar = np.arange(len(p))[:, None]
    return float((np.log(q[ar, P_rows]).sum(1) + N_DRAWN * math.log(nb)).sum())


def block_weights(res, P, cfg, a, span_start, val_len, active_override=None):
    """Weights for the block starting at draw a. Tuned on draws [a - val_len, a) ONLY (out-of-sample forecasts that
    were made before those draws were seen), never on the block itself or anything after it."""
    ec = cfg["ensemble"]
    names = [k for k in ENSEMBLE_EXPERTS if k in res["probs"]]
    fr.assert_no_banned_experts(names, cfg, "ensemble weights")
    lo = a - val_len
    assert lo >= span_start, "validation window must lie inside the walk-forward span"
    gains = np.array([_val_gain(res["probs"][k][lo - span_start:a - span_start], P[lo:a]) for k in names])
    h = np.exp(ec["hedge_eta"] * (gains - gains.max()))
    h /= h.sum()
    mult = np.array([ec["multipliers"][k] for k in names])
    if active_override is None:
        act = np.array([bool(res["active"][k][a - span_start]) for k in names])
    else:
        act = np.array([bool(active_override[k]) for k in names])
    # Track-record rule (same for every walk-forward block and for the next draw): an expert that was silent (uniform)
    # on most of the validation window has gain ~0 there, which beats any expert with a slightly negative gain. Without
    # this rule the hedge hands its largest weight to whichever expert said NOTHING during validation.
    track = np.array([float(np.mean(res["active"][k][lo - span_start:a - span_start])) for k in names])
    act = act & (track >= ec.get("min_track_record", 0.0))
    if ec.get("require_positive_gain"):
        act = act & (gains > 0)
    w = mult * h * act
    return dict(zip(names, w)), dict(zip(names, gains)), dict(zip(names, h))


def consensus_scores(rows, w, vote, share):
    """Consensus score per ball: the WORST rank it gets from any heavy expert (weight share >= `share`), so a ball only
    scores high when every heavy expert ranks it high. Ties are broken by the ordinary vote (always < 1 point).
    rows: {expert: (T, nb) probabilities}; w: {expert: weight}; vote: (T, nb)."""
    sw = sum(w.values())
    if sw <= 0:
        return vote
    heavy = [k for k, v in w.items() if v / sw >= share] or [max(w, key=w.get)]
    pts = np.min(np.stack([_rank_points(rows[k]) for k in heavy]), axis=0)
    v = vote - vote.min(1, keepdims=True)
    return pts + 0.999 * v / (v.max(1, keepdims=True) + 1e-12)          # rank points are integers: this only breaks ties


def top2_metrics(score, X):
    """Top-2 of each row of `score` against the drawn balls X: mean hits among the 2 and rate of both drawn."""
    order = np.argsort(-score, axis=1, kind="stable")[:, :2]
    h = X[np.arange(len(X))[:, None], order].sum(1)
    return dict(hits=float(h.mean()), both=float((h >= 2).mean()), any=float((h >= 1).mean()), n=int(len(X)),
                h=h, order=order)


def combine(res, P, cfg, lay):
    """Ensemble probabilities (for log-loss) and vote scores (for picks) over the evaluation draws."""
    span, ev, n_eval, cad, vl = lay["span_start"], lay["eval_start"], lay["eval_n"], lay["cad"], lay["val_len"]
    names = [k for k in ENSEMBLE_EXPERTS if k in res["probs"]]
    nb = next(iter(res["probs"].values())).shape[1]
    p_ens = np.empty((n_eval, nb))
    vote = np.empty((n_eval, nb))
    cons = np.empty((n_eval, nb))
    share = cfg["ensemble"].get("consensus_share", 0.25)
    wlog = []
    for a in range(ev, ev + n_eval, cad):
        b = a + cad
        w, gains, h = block_weights(res, P, cfg, a, span, vl)
        sw = sum(w.values())
        sl = slice(a - ev, b - ev)
        if sw <= 0:
            log.debug("  ensemble block at draw %d: every expert has weight 0 -> uniform forecast, picks fall back to "
                      "the fixed ball order (no randomness is added)", a)
            p_ens[sl] = N_DRAWN / nb
            vote[sl] = 0.0
        else:
            pe = sum(w[k] * res["probs"][k][a - span:b - span] for k in names) / sw
            p_ens[sl] = fr.norm_inc(pe)
            if cfg["ensemble"]["vote_mode"] == "rank":
                pts = sum(w[k] * _rank_points(res["probs"][k][a - span:b - span]) for k in names) / sw
                vote[sl] = pts + 1e-6 * p_ens[sl]            # tie-break by the weighted probability (deterministic)
            else:
                vote[sl] = p_ens[sl]
        rows = {k: res["probs"][k][a - span:b - span] for k in names}
        cons[sl] = consensus_scores(rows, w, vote[sl], share) if sw > 0 else vote[sl]
        wlog.append({"block_start": int(a), "weights": {k: float(v) for k, v in w.items()},
                     "val_gain": {k: float(v) for k, v in gains.items()}})
    return p_ens, vote, wlog, cons


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(p_ens, vote, X, cfg):
    nb = X.shape[1]
    T = len(X)
    ar = np.arange(T)[:, None]
    K, V = topk_big(cfg, nb), core_size(cfg, nb)
    order = np.argsort(-vote, axis=1, kind="stable")
    h5 = X[ar, order[:, :N_DRAWN]].sum(1)
    hK = X[ar, order[:, :K]].sum(1)
    core = (X[ar, order[:, :V]].sum(1) >= 2).astype(float)
    p = np.clip(p_ens, 1e-9, 1 - 1e-9)
    bce = -(X * np.log(p) + (1 - X) * np.log1p(-p)).sum(1)
    sc = dict(hits_top5=float(h5.mean()), hits_topK=float(hK.mean()), core_hit=float(core.mean()),
              logloss_skill=float(1.0 - bce.mean() / uniform_bce(nb)))
    return sc, dict(h5=h5, hK=hK, core=core, bce=bce, order5=order[:, :N_DRAWN], K=K, V=V)


def chance_levels(nb, cfg):
    K, V = topk_big(cfg, nb), core_size(cfg, nb)
    return dict(hits_top5=chance_hits(N_DRAWN, nb), hits_topK=chance_hits(K, nb), core_hit=chance_core(V, nb),
                logloss_skill=0.0, K=K, V=V)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def confidence_intervals(pdraw, nb, cfg, seed):
    """95% CIs: bootstrap over test draws for the means / skill, Wilson for the binomial core rate."""
    rng = np.random.default_rng(seed)
    T, B = len(pdraw["h5"]), int(cfg["eval"]["boot_reps"])
    idx = rng.integers(0, T, size=(B, T))
    uni = uniform_bce(nb)
    ci = {}
    for key, arr in (("hits_top5", pdraw["h5"]), ("hits_topK", pdraw["hK"])):
        m = arr[idx].mean(1)
        ci[key] = (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
    s = 1.0 - pdraw["bce"][idx].mean(1) / uni
    ci["logloss_skill"] = (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    ci["core_hit"] = wilson(int(pdraw["core"].sum()), T)
    return ci


# --------------------------------------------------------------------------- #
# One full pass of the pipeline (used for the real data AND for every null replicate)
# --------------------------------------------------------------------------- #
def evaluate_pipeline(d, cfg, lay, seed, enabled, device="auto", predict_next=False, fb=None, quiet=False):
    n = d.n
    if fb is None:
        fb = fr.build_features(d, cfg)
    res = run_experts(d, fb, cfg, lay["span_start"], n, seed, enabled, lay["cad"], predict_next=predict_next,
                      device=device, quiet=quiet)
    p_ens, vote, wlog, cons = combine(res, d.P, cfg, lay)
    Xe = d.X[lay["eval_start"]:n].astype(float)
    sc, pdraw = compute_metrics(p_ens, vote, Xe, cfg)
    top2 = {"vote": top2_metrics(vote, Xe), "consensus": top2_metrics(cons, Xe)}
    return dict(fb=fb, res=res, p_ens=p_ens, vote=vote, cons=cons, wlog=wlog, scalars=sc, pdraw=pdraw, Xe=Xe, top2=top2)


# --------------------------------------------------------------------------- #
# Null test: ball labels permuted within each draw, whole evaluation repeated
# --------------------------------------------------------------------------- #
def permute_within_draws(d, rng):
    nb, n = d.X.shape[1], d.n
    P2 = d.P.copy()
    for t in range(n):
        P2[t] = rng.permutation(nb)[d.P[t]]          # a random relabelling of the balls, different for every draw
    X2 = np.zeros_like(d.X)
    X2[np.repeat(np.arange(n), N_DRAWN), P2[:n].ravel()] = 1
    return dataclasses.replace(d, P=P2, X=X2)


def _fingerprint(d, cfg, lay, enabled, seed, metrics):
    c = copy.deepcopy(cfg)
    for k in ("null_reps", "early_stop_h", "early_stop_min"):              # these change how many, not which replicates
        c["eval"].pop(k, None)
    blob = json.dumps(c, sort_keys=True, default=str)
    return dict(cfg=hashlib.md5(blob.encode()).hexdigest(), n=int(d.n), seed=int(seed), metrics=list(metrics),
                data=int(np.asarray(d.P[:d.n]).sum()), lay={k: int(v) for k, v in lay.items()}, enabled=list(enabled))


def _atomic_json(path: Path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _null_replicate(d, cfg, lay, enabled, device, seed, r, metrics):
    """One null replicate. Depends only on (seed, r), so serial and parallel runs compute the same replicate."""
    rng = np.random.default_rng(seed + 10_000 + r)
    d2 = permute_within_draws(d, rng)
    out = evaluate_pipeline(d2, cfg, lay, seed + 10_000 + r, enabled, device, quiet=True)
    return [float(out["scalars"][m]) for m in metrics]


_W = {}


def _worker_init(d, cfg, lay, enabled, device, seed, metrics, threads):
    """Pool worker: limit every library to `threads` CPU threads so N workers do not fight over the same cores."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    try:                                                                 # numpy/BLAS is already loaded: env vars are too late
        from threadpoolctl import threadpool_limits
        _W["tp"] = threadpool_limits(threads)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    fr.make_deterministic()
    cfg = copy.deepcopy(cfg)
    cfg["ranker"]["params"]["n_jobs"] = threads
    cfg["booster"]["params"]["n_jobs"] = threads
    logging.getLogger().setLevel(logging.WARNING)                        # replicates are quiet; the parent logs progress
    _W.update(d=d, cfg=cfg, lay=lay, enabled=enabled, device=device, seed=seed, metrics=metrics)


def _worker_run(r):
    w = _W
    return r, _null_replicate(w["d"], w["cfg"], w["lay"], w["enabled"], w["device"], w["seed"], r, w["metrics"])


def null_test(d, cfg, lay, enabled, device, reps, seed, observed, progress_every=5, ckpt=None, workers=1):
    """Replicate r permutes the labels with seed + 10000 + r. With `ckpt`, finished replicates are saved as they finish
    and a restarted run resumes (the checkpoint is ignored if data, config or layout changed). workers > 1 runs the
    replicates in parallel processes (they share the GPU when there is one; the TFT is small, so 2-3 processes fit on a
    T4 and overlap its CPU-bound ranker work with GPU work); the replicates and their seeds are identical to a serial run."""
    metrics = list(cfg["eval"]["metrics"])
    fp = _fingerprint(d, cfg, lay, enabled, seed, metrics)
    done_rows = {}
    if ckpt is not None and Path(ckpt).exists():
        try:
            j = json.loads(Path(ckpt).read_text())
            if j.get("fp") == fp:
                if "by_rep" in j:
                    done_rows = {int(k): v for k, v in j["by_rep"].items() if int(k) < reps}
                else:                                                    # old checkpoint format: replicates 0..len-1
                    done_rows = dict(enumerate(j["rows"][:reps]))
                log.info("  null test: RESUMING from checkpoint, %d of %d replicates already done", len(done_rows), reps)
            else:
                log.warning("  null test: checkpoint %s belongs to a different data/config/layout; starting fresh", ckpt)
        except (ValueError, KeyError):
            log.warning("  null test: unreadable checkpoint %s; starting fresh", ckpt)
    todo = [r for r in range(reps) if r not in done_rows]
    t0 = time.time()
    n_new = 0
    ec = cfg["eval"]
    h_stop, min_stop = int(ec.get("early_stop_h", 0)), int(ec.get("early_stop_min", 20))
    obs_vec = np.array([observed[m] for m in metrics])

    def exceedances(rows_):
        """Null replicates whose max-T statistic is at least the observed one (scales from the replicates so far)."""
        N_ = np.array(rows_)
        mu_, sd_ = N_.mean(0), N_.std(0, ddof=1)
        sd_ = np.where(sd_ > 0, sd_, 1.0)
        return int((((N_ - mu_) / sd_).max(1) >= ((obs_vec - mu_) / sd_).max()).sum())

    def should_stop():
        """Besag-Clifford sequential Monte Carlo test: once h replicates beat the observed result the p-value is
        already >= h / r, far above any threshold that could matter, so further replicates cannot change the verdict."""
        return h_stop > 0 and len(done_rows) >= min_stop and exceedances(list(done_rows.values())) >= h_stop

    def record(r, row):
        nonlocal n_new
        done_rows[r] = row
        n_new += 1
        if ckpt is not None:
            _atomic_json(Path(ckpt), dict(fp=fp, by_rep={str(k): v for k, v in sorted(done_rows.items())}))
        if n_new % progress_every == 0 or len(done_rows) == reps:
            el = time.time() - t0
            log.info("  null replicate %d/%d | %.0fs this session | ~%.0fs left", len(done_rows), reps, el,
                     el / n_new * (reps - len(done_rows)))

    workers = max(1, min(int(workers), len(todo) or 1))
    stopped = should_stop()
    if workers == 1:
        for r in todo:
            if stopped:
                break
            record(r, _null_replicate(d, cfg, lay, enabled, device, seed, r, metrics))
            stopped = should_stop()
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        threads = max(1, (os.cpu_count() or workers) // workers)
        log.info("  null test: %d replicates to run on %d parallel workers x %d CPU threads", len(todo), workers, threads)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                 initargs=(d, cfg, lay, enabled, device, seed, metrics, threads)) as ex:
            futs = [] if stopped else [ex.submit(_worker_run, r) for r in todo]
            for f in as_completed(futs):
                if f.cancelled():
                    continue
                record(*f.result())
                if not stopped and should_stop():
                    stopped = True
                    for g in futs:
                        g.cancel()
    if stopped:
        log.info("  null test: STOPPED EARLY after %d replicates: %d of them already match or beat the observed result, "
                 "so p >= %.3f and more replicates cannot turn this into a significant result", len(done_rows),
                 h_stop, h_stop / len(done_rows))
    rows = [done_rows[r] for r in sorted(done_rows)]
    reps = len(rows)
    N = np.array(rows)                                                   # (reps, n_metrics)
    obs = np.array([observed[m] for m in metrics])
    mu = N.mean(0)
    sd = N.std(0, ddof=1) if reps > 1 else np.ones(len(metrics))
    sd = np.where(sd > 0, sd, 1.0)
    p_raw = (1.0 + (N >= obs[None, :]).sum(0)) / (reps + 1.0)
    z_obs = (obs - mu) / sd
    t_obs = float(z_obs.max())
    t_null = ((N - mu) / sd).max(1)
    p_max = float((1.0 + (t_null >= t_obs).sum()) / (reps + 1.0))
    if stopped:
        p_max = max(p_max, float((t_null >= t_obs).sum()) / reps)          # Besag-Clifford p = h / r
    return dict(metrics=metrics, reps=reps, early_stopped=bool(stopped), null_mean=dict(zip(metrics, map(float, mu))),
                null_sd=dict(zip(metrics, map(float, sd))), p_raw=dict(zip(metrics, map(float, p_raw))),
                z_obs=dict(zip(metrics, map(float, z_obs))), t_obs=t_obs, p_maxT=p_max,
                resolution=1.0 / (reps + 1.0))


def verdict(null, cfg, game_name):
    """SKILL DETECTED only if the max-T p-value, Bonferroni-corrected across the games, is below alpha."""
    ec = cfg["eval"]
    p_corr = min(1.0, null["p_maxT"] * ec["n_games"])
    ok = p_corr < ec["alpha"]
    reachable = null["resolution"] * ec["n_games"] < ec["alpha"] or bool(null.get("early_stopped"))
    return ok, p_corr, reachable


# --------------------------------------------------------------------------- #
# Ablation (feature sets) with the LambdaRank learner, same cadence / window / seed
# --------------------------------------------------------------------------- #
def _ranker_eval(F, d, cfg, lay, seed):
    n = d.n
    pr, _, _ = walk_forward_ranker(F, d.X, d.P, lay["eval_start"], n, cfg, seed, lay["cad"])
    Xe = d.X[lay["eval_start"]:n].astype(float)
    sc, pdraw = compute_metrics(pr, pr, Xe, cfg)
    return sc, pdraw


def ablation(d, cfg, lay, seed, fb_new, legacy_builder=None):
    nb = d.X.shape[1]
    rows = {}
    ch = chance_levels(nb, cfg)

    def add(name, sc, pdraw, note=""):
        ci = confidence_intervals(pdraw, nb, cfg, seed) if pdraw is not None else None
        rows[name] = dict(scalars=sc, ci=ci, note=note)

    if legacy_builder is not None:
        F_old, names_old = legacy_builder(d)
        log.warning("ABLATION (a): the OLD feature set (contains rate_all / rate250 / rate500 and derived columns) is "
                    "used here ONLY as a reference row. It never reaches the vote, the twin or the final picks.")
        sc, pdr = _ranker_eval(F_old, d, cfg, lay, seed)
        add("(a) old feature set", sc, pdr, "reference only; banned columns included on purpose")
    fr.assert_no_banned(fb_new.names, cfg, "ablation (b)", allow_rate100=False)
    sc, pdr = _ranker_eval(fb_new.F, d, cfg, lay, seed)
    add("(b) new feature set", sc, pdr)
    c100 = copy.deepcopy(cfg)
    c100["features"]["use_rate100"] = True
    fb100 = fr.build_features(d, c100)
    fr.assert_no_banned(fb100.names, c100, "ablation (c)", allow_rate100=True)
    sc, pdr = _ranker_eval(fb100.F, d, c100, lay, seed)
    add("(c) new set + rate100", sc, pdr)
    add("(d) uniform baseline", dict(hits_top5=ch["hits_top5"], hits_topK=ch["hits_topK"], core_hit=ch["core_hit"],
                                     logloss_skill=0.0), None, "analytic: equals chance by construction")
    # (e) pure frequency model on the trailing 20 draws only (Laplace-shrunk), deterministic tie-break by ball order
    r20 = fb_new.col("rate20").astype(float)[lay["eval_start"]:d.n]
    s = cfg["eval"]["freq20_prior"]
    base = N_DRAWN / nb
    p20 = fr.norm_inc((20 * r20 + s * base) / (20 + s))
    Xe = d.X[lay["eval_start"]:d.n].astype(float)
    sc, pdr = compute_metrics(p20, p20, Xe, cfg)
    add("(e) trailing-20 frequency", sc, pdr)
    return rows


# --------------------------------------------------------------------------- #
# Diagnostics (never objectives)
# --------------------------------------------------------------------------- #
def chi_square_uniformity(d):
    """Full-history ball frequencies vs uniform. Counts come from 5-of-N draws WITHOUT replacement, so the naive
    statistic has mean (N-5), not (N-1): it is rescaled by (1-p) * N/(N-1) (derived from the exact covariance of the
    inclusion indicators) before the chi-square(N-1) reference is used."""
    nb, n = d.X.shape[1], d.n
    cnt = d.X[:n].sum(0)
    p = N_DRAWN / nb
    e = n * p
    raw = float(((cnt - e) ** 2 / e).sum())
    scale = (1 - p) * nb / (nb - 1)
    stat = raw / scale
    pv = float(stats.chi2.sf(stat, nb - 1))
    return dict(chi2_raw=raw, chi2_scaled=float(stat), df=nb - 1, p=pv, min=int(cnt.min()), max=int(cnt.max()),
                min_ball=int(cnt.argmin()) + 1, max_ball=int(cnt.argmax()) + 1, expected=float(e), n_draws=int(n))


def jaccard_consecutive(order5):
    sets = [set(map(int, r)) for r in order5]
    j = [len(a & b) / len(a | b) for a, b in zip(sets[:-1], sets[1:])]
    return float(np.mean(j)) if j else float("nan")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def markdown_report(game, nb, n, lay, obs, ci, null, abl, expert_tbl, chi, jac, jac_chance, verdict_txt, cfg, tft_skills):
    ch = chance_levels(nb, cfg)
    K, V = ch["K"], ch["V"]
    ec = cfg["eval"]
    L = []
    w = L.append
    w(f"## {game}: {N_DRAWN}-of-{nb} ({n} clean draws)")
    w("")
    w(f"Walk-forward over the last **{lay['eval_n']}** draws, retrain every **{lay['cad']}**, ensemble weights tuned on the "
      f"preceding **{lay['val_len']}**-draw validation window only. Null test: **{null['reps']}** label-permuted replicates "
      f"(p-value resolution {null['resolution']:.4f})"
      + (", stopped early: enough replicates already beat the observed result to rule out significance." if null.get(
          "early_stopped") else "."))
    w("")
    w("### Main result (ensemble vote)")
    w("")
    w(f"| metric | observed | chance | 95% CI | null mean | raw p | Bonferroni p (x{ec['n_games']} games) |")
    w("|---|---|---|---|---|---|---|")
    labels = {"hits_top5": f"mean hits top-{N_DRAWN}", "hits_topK": f"mean hits top-{K}",
              "core_hit": f"Perm-2 core hit rate (core {V})", "logloss_skill": "log-loss skill vs uniform"}
    for m in null["metrics"]:
        w(f"| {labels[m]} | {obs[m]:.4f} | {ch[m]:.4f} | {_fmt_ci(ci[m])} | {null['null_mean'][m]:.4f} | "
          f"{null['p_raw'][m]:.4f} | {min(1.0, ec['n_games'] * null['p_raw'][m]):.4f} |")
    w("")
    w(f"Max-T over the {len(null['metrics'])} metrics: p = {null['p_maxT']:.4f}; Bonferroni across {ec['n_games']} games: "
      f"p = {min(1.0, ec['n_games'] * null['p_maxT']):.4f} (alpha = {ec['alpha']}).")
    w("")
    w("### Ablation (LambdaRank learner, same cadence and seed; hazard/markov/TFT are not part of these rows)")
    w("")
    w(f"| feature set | hits top-{N_DRAWN} (chance {ch['hits_top5']:.3f}) | hits top-{K} (chance {ch['hits_topK']:.3f}) | "
      f"core hit (chance {ch['core_hit']:.4f}) | log-loss skill |")
    w("|---|---|---|---|---|")
    for name, r in abl.items():
        s = r["scalars"]
        c = r["ci"]
        t5 = f"{s['hits_top5']:.3f}" + (f" {_fmt_ci(c['hits_top5'])}" if c else "")
        tk = f"{s['hits_topK']:.3f}" + (f" {_fmt_ci(c['hits_topK'])}" if c else "")
        co = f"{s['core_hit']:.4f}" + (f" {_fmt_ci(c['core_hit'])}" if c else "")
        sk = f"{s['logloss_skill']:+.5f}" + (f" {_fmt_ci(c['logloss_skill'])}" if c else "")
        w(f"| {name} | {t5} | {tk} | {co} | {sk} |")
    w("")
    w("### Experts inside the ensemble (diagnostic)")
    w("")
    w(f"| expert | mean weight | log-loss skill | hits top-{N_DRAWN} | active share |")
    w("|---|---|---|---|---|")
    for name, r in expert_tbl.items():
        h5 = "n/a (silent)" if r["h5"] != r["h5"] else f"{r['h5']:.3f}"
        w(f"| {name} | {r['mean_weight']:.3f} | {r['skill']:+.5f} | {h5} | {r['active_share']:.2f} |")
    if tft_skills:
        last = [b for b in tft_skills if not b.get("final")]
        sk = [s for b in last for s in b["skills"]]
        if sk:
            w("")
            w(f"TFT member validation skill across blocks: mean {np.mean(sk):+.5f}, max {np.max(sk):+.5f}, "
              f"{int(np.sum(np.array(sk) > 0))}/{len(sk)} members above 0 (members at <= 0 are given vote weight 0).")
    w("")
    w("### Diagnostics (not objectives)")
    w("")
    w(f"* Chi-square uniformity of full-history ball frequencies: stat {chi['chi2_scaled']:.1f} on {chi['df']} df, "
      f"p = {chi['p']:.4f} (counts {chi['min']} [ball {chi['min_ball']}] .. {chi['max']} [ball {chi['max_ball']}], "
      f"expected {chi['expected']:.1f}).")
    w(f"* Mean Jaccard overlap of the Top-{N_DRAWN} on consecutive test dates: {jac:.3f} (two independent random picks: "
      f"{jac_chance:.3f}). Printed for information only; it is never optimised.")
    w("")
    w(f"**Verdict {game}: {verdict_txt}**")
    w("")
    return "\n".join(L)
