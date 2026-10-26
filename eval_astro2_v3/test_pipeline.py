#!/usr/bin/env python3
"""
test_pipeline.py: unit tests for the refactor.   python test_pipeline.py     (also works with pytest)

  1. LEAKAGE: features for 20 random draws t recomputed from truncated history must equal the pipeline's values exactly;
     a deliberately leaky builder must FAIL the same check (so the test is real)
  2. banned long-history columns / linear experts are rejected
  3. expert forecasts for early blocks do not change when future draws are replaced by noise
  4. zero-initialised TFT outputs the UNIFORM forecast; the army runs and silences members without skill
  5. determinism, pool-size generality (5/39 and 5/90), chance maths, wheel guarantee, permutation null
"""
from __future__ import annotations

import dataclasses
import itertools
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as eh
import experts as ex
import features_recent as fr
import stage2_powerhouse as s2
from nla_config import N_DRAWN, chance_core, chance_jaccard, chance_pair, get_config, uniform_bce


# --------------------------------------------------------------------------- #
# synthetic data in the stage2.Data layout (n real draws, n+1 rows)
# --------------------------------------------------------------------------- #
def make_synthetic(n=900, nb=90, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n + 1, freq="D")
    P = np.zeros((n + 1, N_DRAWN), dtype=int)
    for t in range(n):
        P[t] = rng.choice(nb, N_DRAWN, replace=False)
    X = np.zeros((n + 1, nb), dtype=np.float32)
    X[np.repeat(np.arange(n), N_DRAWN), P[:n].ravel()] = 1
    dow, month, dom = dates.dayofweek.to_numpy(), dates.month.to_numpy(), dates.day.to_numpy()
    dsp = np.ones(n + 1, dtype=int)
    shifted = (rng.random(n + 1) < 0.05).astype(int)
    hshift = (rng.random(n + 1) < 0.03).astype(int)
    cal = np.zeros((n + 1, len(s2.CAL_NAMES)), dtype=np.float32)
    cal[np.arange(n + 1), dow] = 1
    cal[:, 7] = np.sin(2 * np.pi * (month - 1) / 12)
    cal[:, 8] = np.cos(2 * np.pi * (month - 1) / 12)
    cal[:, 9] = dom / 31
    cal[:, 10] = np.clip(dsp, 0, 7) / 7
    cal[:, 11], cal[:, 12] = shifted, hshift
    return s2.Data(n, dates, P, X, cal, dow, month, dom, dsp, shifted, hshift, {"next_date": str(dates[n].date())})


def truncate(d, t):
    """What the pipeline would see when predicting draw t: draws 0..t-1 known, row t is the unknown draw."""
    P, X = d.P[:t + 1].copy(), d.X[:t + 1].copy()
    P[t], X[t] = 0, 0
    return dataclasses.replace(d, n=t, dates=d.dates[:t + 1], P=P, X=X, cal=d.cal[:t + 1], dow=d.dow[:t + 1],
                               month=d.month[:t + 1], dom=d.dom[:t + 1], dsp=d.dsp[:t + 1], shifted=d.shifted[:t + 1],
                               hshift=d.hshift[:t + 1])


def check_leakage(builder, d, cfg, ts):
    """True iff the features at row t are identical when computed from the truncated history."""
    full = builder(d, cfg)
    for t in ts:
        part = builder(truncate(d, int(t)), cfg)
        if not np.array_equal(full.F[t], part.F[t]):
            return False
    return True


def leaky_builder(d, cfg):
    """Deliberately peeks at the draw being predicted (X[t]); must be caught."""
    fb = fr.build_features(d, cfg)
    F = np.concatenate([fb.F, np.asarray(d.X, dtype=np.float32)[:, :, None]], axis=2)
    return fr.FeatureBundle(F, fb.names + ["peek"])


def _cfg():
    return get_config()


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_leakage_20_random_t():
    for nb in (90, 39):
        d, cfg = make_synthetic(700, nb, 1), _cfg()
        ts = np.random.default_rng(7).choice(np.arange(130, 700), 20, replace=False)
        assert check_leakage(fr.build_features, d, cfg, ts), f"leakage in build_features (nb={nb})"
        cfg100 = _cfg()
        cfg100["features"]["use_rate100"] = True
        assert check_leakage(fr.build_features, d, cfg100, ts[:5]), "leakage with rate100 on"


def test_leakage_test_is_sensitive():
    d, cfg = make_synthetic(500, 90, 2), _cfg()
    ts = np.random.default_rng(3).choice(np.arange(130, 500), 20, replace=False)
    assert not check_leakage(leaky_builder, d, cfg, ts), "the leakage test failed to catch a leaky feature"


def test_banned_columns_rejected():
    cfg = _cfg()
    for bad in ("rate_all", "rate250", "rate500", "frozen_frequency", "my_rate_all_zscore", "rate_same_dow", "ball_id",
                "pos1_rate", "last_digit"):
        try:
            fr.assert_no_banned(["rate20", bad], cfg, "test")
        except fr.BannedFeatureError:
            continue
        raise AssertionError(f"banned column {bad!r} was not rejected")
    try:
        fr.assert_no_banned(["rate20", "rate100"], cfg, "test")
        raise AssertionError("rate100 must be banned by default")
    except fr.BannedFeatureError:
        pass
    fr.assert_no_banned(["rate20", "rate100"], cfg, "test", allow_rate100=True)
    d = make_synthetic(400, 90, 4)
    fb = fr.build_features(d, cfg)
    fr.assert_no_banned(fb.names, cfg, "pipeline features")
    for must in ("rate5", "rate20", "gap_ratio", "gap_percentile", "markov", "was_shifted", "momentum_20_50"):
        assert must in fb.names, f"missing feature {must}"
    for bad in ("logreg", "logistic_regression", "ridge"):
        try:
            fr.assert_no_banned_experts(["ranker", bad], cfg, "test")
        except fr.BannedFeatureError:
            continue
        raise AssertionError(f"banned expert {bad!r} not rejected")


def _noise_future(d, t0, seed=0):
    rng = np.random.default_rng(seed)
    d2 = dataclasses.replace(d, P=d.P.copy(), X=d.X.copy())
    nb = d.X.shape[1]
    for t in range(t0, d.n):
        d2.P[t] = rng.choice(nb, N_DRAWN, replace=False)
        d2.X[t] = 0
        d2.X[t, d2.P[t]] = 1
    return d2


def _small_cfg():
    cfg = _cfg()
    cfg["ranker"]["params"].update(n_estimators=40)
    cfg["tft"].update(army=2, hidden=16, heads=2, seq_len=12, max_epochs=4, min_epochs=2, patience=2, warmup=60)
    cfg["ranker"]["warmup"] = cfg["hazard"]["warmup"] = 60
    return cfg


def test_experts_do_not_see_the_future():
    d, cfg = make_synthetic(700, 90, 5), _small_cfg()
    d2 = _noise_future(d, 600)
    a = ex.run_experts(d, fr.build_features(d, cfg), cfg, 400, 700, 11, ["ranker", "hazard", "markov"], 50)
    b = ex.run_experts(d2, fr.build_features(d2, cfg), cfg, 400, 700, 11, ["ranker", "hazard", "markov"], 50)
    keep = 600 - 400
    for k in a["probs"]:
        assert np.allclose(a["probs"][k][:keep], b["probs"][k][:keep], atol=1e-9), f"{k}: forecasts depend on future draws"


def test_determinism_no_hidden_randomness():
    d, cfg = make_synthetic(600, 90, 6), _small_cfg()
    r1 = ex.run_experts(d, fr.build_features(d, cfg), cfg, 400, 600, 5, ["ranker", "hazard", "markov"], 50, predict_next=True)
    r2 = ex.run_experts(d, fr.build_features(d, cfg), cfg, 400, 600, 5, ["ranker", "hazard", "markov"], 50, predict_next=True)
    for k in r1["probs"]:
        assert np.array_equal(r1["probs"][k], r2["probs"][k]), f"{k} is not deterministic"
        assert np.array_equal(r1["next"][k], r2["next"][k])
    for k, p in r1["probs"].items():
        assert np.allclose(p.sum(1), N_DRAWN, atol=1e-3), f"{k}: rows must sum to 5"


def test_pool_size_generality():
    for nb in (39, 90):
        d, cfg = make_synthetic(650, nb, 8), _small_cfg()
        r = ex.run_experts(d, fr.build_features(d, cfg), cfg, 400, 650, 3, ["ranker", "hazard", "markov"], 50)
        for k, p in r["probs"].items():
            assert p.shape == (250, nb) and np.allclose(p.sum(1), N_DRAWN, atol=1e-3)
        assert abs(uniform_bce(nb) - (-(5 * np.log(5 / nb) + (nb - 5) * np.log(1 - 5 / nb)))) < 1e-9


def test_chance_maths():
    assert abs(chance_core(12, 90) - 0.1306) < 0.002, chance_core(12, 90)       # "13%" in the dossier
    assert abs(chance_core(10, 39) - 0.381) < 0.002, chance_core(10, 39)        # "38.1%"
    assert abs(chance_pair(90) - 20 / (90 * 89)) < 1e-12
    rng = np.random.default_rng(0)                                              # Monte-Carlo cross-check
    hits = sum(1 for _ in range(40000) if (rng.choice(90, 5, replace=False) < 12).sum() >= 2) / 40000
    assert abs(hits - chance_core(12, 90)) < 0.01
    assert 0.0 < chance_jaccard(90) < 0.1


def test_wheel_guarantee():
    import wheel_generator as wg
    for v in (12, 10):
        tickets, _ = wg.min_cover(v, 2)
        assert wg.verify(tickets, v) and len(tickets) == v * (v - 1) // 2


def test_permutation_null_keeps_draw_structure():
    d = make_synthetic(300, 39, 9)
    d2 = eh.permute_within_draws(d, np.random.default_rng(1))
    assert (d2.X[:d.n].sum(1) == N_DRAWN).all()
    assert all(len(set(r)) == N_DRAWN for r in d2.P[:d.n])
    assert not np.array_equal(d.X, d2.X)


def test_tft_zero_init_is_uniform():
    import torch
    cls = ex._build_torch_classes()
    for nb in (39, 90):
        m = cls(nb, nb + 13, 13, 17, 16, 2, 0.0).eval()
        out = m(torch.randn(4, 12, nb + 13), torch.randn(4, 13), torch.randn(4, nb, 17))
        p = torch.sigmoid(out)
        assert torch.allclose(p, torch.full_like(p, N_DRAWN / nb), atol=1e-6), "untrained TFT must output uniform"


def test_tft_army_runs_and_silences_unskilled_members():
    d, cfg = make_synthetic(520, 90, 10), _small_cfg()
    fb = fr.build_features(d, cfg)
    army = ex.TFTArmy(d, fb, cfg, "cpu", quiet=True)
    p, ok, skills = army.block(450, 450, 480, 3)
    assert p.shape == (30, 90) and np.allclose(p.sum(1), N_DRAWN, atol=1e-3)
    assert len(skills) == cfg["tft"]["army"]
    if not ok:                                                                   # no validated skill -> exactly uniform
        assert np.allclose(p, N_DRAWN / 90)
    assert all(s <= 0.05 for s in skills), "random data should show (almost) no skill"


def test_anchor_no_future_and_turnover_cap():
    import anchor as an
    d, cfg = make_synthetic(700, 90, 12), _cfg()
    d2 = _noise_future(d, 600)
    fa, fb2 = fr.build_features(d, cfg), fr.build_features(d2, cfg)
    a = an.run_anchor(d, fa, cfg, 400, 700, 50)
    b = an.run_anchor(d2, fb2, cfg, 400, 700, 50)
    assert np.allclose(a[0][:200], b[0][:200], atol=1e-9), "anchor forecasts depend on future draws"
    assert np.allclose(a[0].sum(1), N_DRAWN, atol=1e-3)
    for row in a[2]:
        assert (not row["active"]) or row["jaccard"] <= cfg["anchor"]["max_top_jaccard"] + 1e-9
    Z = an.build_z(d, fa, cfg)                                   # pure-long ranking IS frozen: the cap must forbid it alone
    assert an.top_jaccard(Z[300:600, :, 0], 12) > 0.4


def test_track_record_rule_blocks_silent_expert():
    """An expert that was silent (uniform) on the validation window must get weight 0 even if it is active now; without the
    rule its gain of 0 beats experts with negative gain and it takes the biggest weight (the live-vs-backtest anchor bug)."""
    cfg = _cfg()
    nb, rows, span = 39, 300, 1000
    rng = np.random.default_rng(5)
    P = np.array([rng.choice(nb, N_DRAWN, replace=False) for _ in range(span + rows + 1)])
    noisy = fr.norm_inc(rng.uniform(0.05, 0.25, size=(rows, nb)))          # random forecasts: negative gain on average
    res = dict(probs={"markov": noisy, "anchor": np.full((rows, nb), N_DRAWN / nb)},
               active={"markov": np.ones(rows, bool), "anchor": np.zeros(rows, bool)})
    over = {"markov": True, "anchor": True}                               # anchor wakes up only for the final fit
    cfg["ensemble"]["min_track_record"] = 0.0
    w_old, g, _ = eh.block_weights(res, P, cfg, span + rows, span, 200, active_override=over)
    assert abs(g["anchor"]) < 1e-6 and g["markov"] < 0 and w_old["anchor"] > w_old["markov"], "bug reproduction failed"
    cfg["ensemble"]["min_track_record"] = 0.5
    w_new, _, _ = eh.block_weights(res, P, cfg, span + rows, span, 200, active_override=over)
    assert w_new["anchor"] == 0.0 and w_new["markov"] > 0


def test_consensus_top2_needs_every_heavy_expert():
    nb = 10
    a = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], float) / 55 * N_DRAWN     # A ranks ball 0 first
    b = np.array([[1, 9, 2, 8, 10, 5, 4, 3, 6, 7]], float) / 55 * N_DRAWN      # B ranks ball 4 first, ball 0 last
    w = {"A": 1.0, "B": 1.0, "C": 0.01}
    c = np.full((1, nb), 0.5)
    vote = sum(w[k] * eh._rank_points(p) for k, p in (("A", a), ("B", b), ("C", c)))
    s = eh.consensus_scores({"A": a, "B": b, "C": c}, w, vote, 0.25)
    top2 = list(np.argsort(-s[0], kind="stable")[:2])
    assert top2 == [1, 3], top2                          # ball 1 is #2 for both, ball 3 is #4/#3: no veto from either
    assert 0 not in top2                                 # ball 0 is #1 for A but last for B


def test_anchor_always_on_forecasts_every_block():
    import anchor as an
    d, cfg = make_synthetic(700, 39, 13), _cfg()
    cfg["anchor"]["always_on"] = True
    fa = fr.build_features(d, cfg)
    out, act, info, nxt = an.run_anchor(d, fa, cfg, 400, 700, 50, predict_next=True, quiet=True)
    assert act.all() and nxt[1], [r["reason"] for r in info]
    assert all(r["reason"].startswith("ok") for r in info)


def test_pointwise_experts_run_and_gate():
    d, cfg = make_synthetic(700, 39, 21), _cfg()
    fb = fr.build_features(d, cfg)
    for kind in ("mlp", "booster"):
        cfg[kind].update(train_window=400, min_val=60)
        if kind == "mlp":
            cfg["mlp"]["max_iter"] = 5
        else:
            cfg["booster"]["params"]["n_estimators"] = 30
        pr, info, nxt = ex.walk_forward_pointwise(kind, fb.F, d.X.astype(np.float32), 600, 700, cfg, 1, 50, True, quiet=True)
        assert pr.shape == (100, 39) and np.allclose(pr.sum(1), N_DRAWN, atol=1e-3)
        assert nxt[0].shape == (39,)


def test_tracker_logs_once_and_flags_late(tmp_path=None):
    import tempfile
    import tracker as tk
    p = Path(tempfile.mkdtemp()) / "t.csv"
    assert tk.log_prediction(p, "g", "2026-10-21", 39, [1, 2], [1, 2, 3, 4, 5], list(range(1, 11)))
    assert not tk.log_prediction(p, "g", "2026-10-21", 39, [9, 8], [9, 8, 7, 6, 5], list(range(30, 40)))
    rows = tk.read_ledger(p)
    assert len(rows) == 1 and rows[0]["top2"] == "1 2"


def test_rulemem_expert_uses_only_past_draws():
    import pandas as pd
    import rule_memory as rm
    rng = np.random.default_rng(3)
    v = np.array([rng.choice(39, 5, replace=False) + 1 for _ in range(260)])
    dates = pd.date_range("2024-01-01", periods=261, freq="D")
    p = rm.hedge_probs(v, dates, 39, 200)
    assert p.shape == (61, 39) and np.allclose(p.sum(1), 5.0)
    v2 = v.copy()
    v2[230:] = np.array([rng.choice(39, 5, replace=False) + 1 for _ in range(30)])     # change the future
    p2 = rm.hedge_probs(v2, dates, 39, 200)
    assert np.allclose(p[:31], p2[:31])                                                 # rows up to draw 230 unchanged


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    fails = 0
    for t in TESTS:
        if "tft" in t.__name__ and not have_torch:
            print(f"SKIP  {t.__name__} (torch not installed)")
            continue
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print("\nALL TESTS PASSED" if not fails else f"\n{fails} TEST(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
