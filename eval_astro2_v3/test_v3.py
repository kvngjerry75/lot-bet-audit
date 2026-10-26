#!/usr/bin/env python3
"""test_v3.py: unit tests for the v3 additions. Run after test_pipeline.py:  !python test_v3.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_miner as rmn


def _dates(n):
    return pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n + 1, freq="D"))


def test_miner_uses_only_past_draws():
    rng = np.random.default_rng(1)
    nb, n = 39, 400
    P = rmn.fair(n, nb, rng)
    p1 = rmn.run(P, _dates(n), nb, {}, warmup=50, return_p=True)["p"]
    P2 = P.copy()
    P2[300:] = rmn.fair(n - 300, nb, rng)               # change the future
    p2 = rmn.run(P2, _dates(n), nb, {}, warmup=50, return_p=True)["p"]
    assert np.allclose(p1[:301], p2[:301]), "forecast for draw t changed when draws >= t changed"


def test_miner_quiet_on_fair_machine():
    rng = np.random.default_rng(2)
    nb, n = 90, 800
    r = rmn.run(rmn.fair(n, nb, rng), _dates(n), nb, {}, warmup=100)
    assert r["proven_now"] == 0 and r["max_shift"] < 0.01, r


def test_miner_finds_planted_rule():
    rng = np.random.default_rng(3)
    nb, n = 90, 1500
    P = rmn.plant(n, nb, rng, 0.12)
    r = rmn.run(P, _dates(n), nb, {}, warmup=100, keep_memory=True)
    assert r["proven_now"] >= 1, "planted rule not found"
    assert "yesterday's 1st ball + 17" in r["memory"][0]["rule"], r["memory"][0]
    assert r["top5"] > 25 / nb


def test_clean_data_removes_copies():
    import clean_data as cd
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rows = [("2024-01-03", 1, 2, 3, 4, 5), ("2024-01-02", 1, 2, 3, 4, 5), ("2024-01-01", 6, 7, 8, 9, 10),
                ("2024-01-01", 11, 12, 13, 14, 50)]
        pd.DataFrame(rows, columns=["Date"] + cd.NUM).to_csv(d / "g_clean.csv", index=False)
        pd.DataFrame([("2024-01-02", 31, 32, 33, 34, 35)], columns=["Date"] + cd.NUM).to_csv(d / "h_clean.csv", index=False)
        summ, log = cd.clean({"G": (str(d / "g_clean.csv"), 39), "H": (str(d / "h_clean.csv"), 39)}, d)
        reasons = sorted(r["reason"].split(" (")[0] for r in log)
        assert any("outside the pool" in x for x in reasons), reasons
        assert any("copy of the draw" in x for x in reasons), reasons
        out = pd.read_csv(d / "g_audited.csv")
        assert len(out) == 2, out


def test_full_layout_scores_everything():
    import eval_harness as eh
    from nla_config import get_config
    cfg = get_config()
    lay = eh.layout(2000, cfg)
    assert lay["eval_n"] >= 2000 - 120 - 200 - 200 - 100, lay


def test_tracker_same_day_not_counted():
    import tracker
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        csv = d / "g.csv"
        pd.DataFrame([("2026-01-02", 1, 2, 3, 4, 5)], columns=["Date"] + [f"Number{i}" for i in range(1, 6)]).to_csv(csv, index=False)
        led = d / "t.csv"
        tracker.log_prediction(led, "g", "2026-01-02", 39, [1, 2], [1, 2, 3, 4, 5], list(range(1, 11)))
        rows = tracker.read_ledger(led)
        rows[0]["logged_at"] = "2026-01-02 15:00:00"
        import csv as _c
        with led.open("w", newline="") as f:
            w = _c.DictWriter(f, fieldnames=tracker.FIELDS)
            w.writeheader()
            w.writerows(rows)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tracker.score(led, csv, "g")
        assert "Counted picks: 0" in buf.getvalue(), buf.getvalue()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                       # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {e!r}")
    print("\nALL V3 TESTS PASSED" if not fails else f"\n{fails} TEST(S) FAILED")
    sys.exit(1 if fails else 0)
