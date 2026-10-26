#!/usr/bin/env python3
"""
test_audit.py: the checks behind the October 2026 audit. Run from the astro-latest folder:

    python test_audit.py            # ~2 minutes
    python test_audit.py --full     # + the original stage 4 in the leakage test (slower)

  leakage            scramble the last 200 draws: no forecast for an earlier draw may change (every component)
  loader             Afriluck timeline: one draw per date+session, Anopa < Midday < evening, no repeated set in 3 days
  payout maths       fair-machine EV and break-even lifts of every bet
  bet lab +control   a machine with a big planted pink edge (w = 2.5) MUST show a profitable Perm-2, significantly
  bet lab -control   a fair machine must NOT
  detectors +control planted x1.77 pairs are found; a planted ball-set change is found
  forward test       a pick logged after its draw started is marked late and never counted
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ASTRO_K", "6")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "eval_astro2_v3"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

AFL = str(HERE / "afriluck_output")
FULL = "--full" in sys.argv


def test_no_future_leakage():
    import main_beast as mb
    d = mb.load("Afriluck", "", AFL)
    n = len(d["W"])
    cut = n - 200
    rng = np.random.default_rng(1)
    W2 = d["W"].copy()
    W2[cut:] = rng.random((n - cut, 57)).argsort(1)[:, :6]
    M2 = d["M"].copy()
    M2[cut:] = -1
    d2 = dict(d, W=W2, M=M2)
    c1, c2 = mb.components(d, full=FULL), mb.components(d2, full=FULL)
    for members in (mb.PRUNED, None):
        c1["BEAST"] = mb.beast(c1, d["W"], 57, 6, members=members)[0]
        c2["BEAST"] = mb.beast(c2, W2, 57, 6, members=members)[0]
        for nm in c1:
            diff = np.abs(c1[nm][:cut + 1] - c2[nm][:cut + 1]).max()
            assert diff < 1e-9, f"{nm}: forecasts for draws before {cut} changed by {diff} when later draws changed"
            assert np.abs(c1[nm][cut + 1:] - c2[nm][cut + 1:]).max() > 0, f"{nm}: test is blind (nothing changed)"


def test_afriluck_loader():
    import forecaster as fc
    import argparse
    d = fc.load(argparse.Namespace(game="Afriluck", afriluck_dir=AFL, nla_dir=""))
    df = pd.DataFrame({"d": d["dates"], "s": d["session"]})
    assert df.groupby(["d", "s"]).size().max() == 1, "two draws on the same date and session"
    rank = df["s"].map({"Anopa": 0, "Midday": 1, "evening": 2}).to_numpy()
    key = d["dates"].values.astype("datetime64[D]").astype(np.int64) * 3 + rank
    assert (np.diff(key) > 0).all(), "timeline not sorted by date then Anopa < Midday < evening"
    seen = {}
    for dt, w in zip(d["dates"], d["W"]):
        kk = tuple(sorted(w))
        assert kk not in seen or (dt - seen[kk]).days > 3, f"set {kk} repeated within 3 days"
        seen[kk] = dt
    assert ((d["W"] >= 0) & (d["W"] < 57)).all() and all(len(set(r)) == 6 for r in d["W"])


def test_payout_maths():
    import bet_lab as bl
    fe = bl.fair_ev()
    assert abs(fe["perm2"] - 60 * 30 / (57 * 56)) < 1e-12 and abs(fe["perm2"] - 0.5639) < 1e-3
    assert abs(fe["direct1"] - 30 / 57) < 1e-12
    assert abs(fe["banker"] - fe["perm2"]) < 1e-12            # 5 winning pairs x 60 x P(ball drawn) / 56 lines
    assert abs(1 / fe["perm2"] - 1.773) < 1e-3                # break-even pair lift x1.77
    assert abs(1 / fe["direct1"] - 1.90) < 1e-3
    assert abs(1 / fe["direct3"] - 2.44) < 1e-2
    # banker break-even: one ball drawn with probability >= 56/300 = 18.7% (fair 10.5%)
    assert abs(56 / 300 - 0.1867) < 1e-3 and abs((56 / 300) / (6 / 57) - 1.773) < 1e-3
    c = bl.ceiling(1.0, N=200_000)
    for b in fe:
        assert abs(c[b] - fe[b]) / fe[b] < 0.05, (b, c[b], fe[b])


def _sim(w, n, seed):
    rng = np.random.default_rng(seed)
    wt = np.where(np.arange(57) < 28, w, 1.0)
    return np.argsort(-np.log(rng.random((n, 57))) / wt, 1)[:, :6]


def test_bet_lab_positive_control():
    import bet_lab as bl
    W = _sim(2.5, 2061, 3)
    p = bl.base_forecasts(W)["sharp"]
    rows = np.arange(300, len(W))
    ret = bl.returns("perm2_10", bl.rank(p[rows]), W[rows])
    r = bl.money(ret, 45)
    expect = bl.ceiling(2.5, N=200_000)["perm2"]
    assert r["back"] > 1.0 and r["p_be"] < 0.01, f"planted edge missed: {r['back']:.3f}, p={r['p_be']:.3f}"
    assert abs(r["back"] - expect) < 0.12, (r["back"], expect)


def test_bet_lab_negative_control():
    import bet_lab as bl
    W = _sim(1.0, 2061, 4)
    p = bl.base_forecasts(W)["sharp"]
    rows = np.arange(300, len(W))
    r = bl.money(bl.returns("perm2_10", bl.rank(p[rows]), W[rows]), 45)
    assert r["back"] < 0.8 and r["p_be"] > 0.5, r


def test_detectors_find_planted_effects():
    import dependence as dp
    rng = np.random.default_rng(9)
    n = 2061
    w = np.where(np.arange(57) < 28, 1.11, 1.0)
    acc = np.zeros(len(dp.IU[0]))
    for _ in range(60):
        Xs = dp.onehot(dp.sim(w, n, rng))
        acc += (Xs.T @ Xs)[dp.IU]
    E2 = acc / 60
    null = np.array([dp.dep_stats(dp.sim(w, n, rng), E2)["most over-represented pair (z)"] for _ in range(60)])
    found = 0
    for _ in range(5):
        pairs = [tuple(rng.choice(57, 2, replace=False)) for _ in range(15)]
        st = dp.dep_stats(dp.plant_pairs(dp.sim(w, n, rng), pairs, 1.77, rng), E2)
        found += dp.pval(st["most over-represented pair (z)"], null) < 0.05
    assert found >= 4, f"planted x1.77 pairs found only {found}/5 times"
    W0 = dp.sim(w, n, rng)
    null = np.array([dp.split_lr(dp.onehot(W0[rng.permutation(n)])) for _ in range(60)])
    w2 = w.copy()
    ch = rng.choice(57, 20, replace=False)
    w2[ch] *= np.where(np.arange(20) % 2 == 0, 1.25, 0.8)
    Wc = np.vstack([dp.sim(w, n // 2, rng), dp.sim(w2, n - n // 2, rng)])
    assert dp.pval(dp.split_lr(dp.onehot(Wc)), null) < 0.05, "planted ball-set change missed"


def test_forward_test_late_flag():
    import subprocess
    tmp = Path(tempfile.mkdtemp())
    log = tmp / "log.csv"
    # pretend the clock is far in the future: the next draw has already happened -> LATE
    subprocess.run([sys.executable, str(HERE / "forward_test.py"), "log", "--afriluck-dir", AFL, "--log", str(log),
                    "--now", "2099-01-01 00:00"], capture_output=True, text=True, check=True)
    assert bool(pd.read_csv(log)["late"].iloc[0]) is True
    # pretend it is long before: on time
    subprocess.run([sys.executable, str(HERE / "forward_test.py"), "log", "--afriluck-dir", AFL, "--log", str(log),
                    "--bet", "perm2_3", "--now", "2000-01-01 00:00"], capture_output=True, text=True, check=True)
    assert bool(pd.read_csv(log)["late"].iloc[1]) is False


if __name__ == "__main__":
    tests = [test_afriluck_loader, test_payout_maths, test_bet_lab_positive_control, test_bet_lab_negative_control,
             test_detectors_find_planted_effects, test_forward_test_late_flag, test_no_future_leakage]
    bad = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}", flush=True)
        except Exception as e:                                  # noqa: BLE001
            bad += 1
            print(f"FAIL  {t.__name__}: {e}", flush=True)
    print("\nALL AUDIT TESTS PASSED" if not bad else f"\n{bad} TEST(S) FAILED")
    sys.exit(1 if bad else 0)
