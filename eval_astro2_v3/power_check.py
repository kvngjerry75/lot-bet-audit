#!/usr/bin/env python3
"""
power_check.py: does the harness behave? (optional sanity tool, runs without the TFT)

  python power_check.py --mode time                       time one full-size pass (default config, ~2900 draws)
  python power_check.py --mode null  --reps 40 --trials 6  random data: p-values should look uniform (no false alarms)
  python power_check.py --mode power --reps 40 --delta 0.6 data with a real "recent-draw momentum" effect: should be detected
"""
import argparse, logging, sys, time
import numpy as np
sys.path.insert(0, ".")
import eval_harness as eh, features_recent as fr, stage2_powerhouse as s2
from nla_config import N_DRAWN, get_config
from test_pipeline import make_synthetic
import dataclasses


def make_momentum(n, nb, delta, seed):
    """Fair-looking lottery EXCEPT a ball drawn in the last 3 draws has weight (1 + delta) next draw."""
    rng = np.random.default_rng(seed)
    d = make_synthetic(n, nb, seed)
    P, X = d.P.copy(), d.X.copy()
    recent = []
    for t in range(n):
        w = np.ones(nb)
        for r in recent[-3:]:
            w[r] *= (1 + delta)
        w /= w.sum()
        P[t] = rng.choice(nb, N_DRAWN, replace=False, p=w)
        X[t] = 0
        X[t, P[t]] = 1
        recent.append(P[t].copy())
    return dataclasses.replace(d, P=P, X=X)


def run(d, cfg, reps, seed, enabled):
    nb = d.X.shape[1]
    s2.N_BALLS, s2.BASE = nb, N_DRAWN / nb
    lay = eh.layout(d.n, cfg)
    out = eh.evaluate_pipeline(d, cfg, lay, seed, enabled, "cpu")
    null = eh.null_test(d, cfg, lay, enabled, "cpu", reps, seed, out["scalars"], progress_every=10 ** 9)
    return out["scalars"], null


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="time", choices=["time", "null", "power"])
    ap.add_argument("--nb", type=int, default=90)
    ap.add_argument("--n", type=int, default=2900)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--delta", type=float, default=0.6)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = get_config()
    enabled = ["ranker", "hazard", "markov"]
    if a.mode == "time":
        d = make_synthetic(a.n, a.nb, 0)
        s2.N_BALLS, s2.BASE = a.nb, N_DRAWN / a.nb
        lay = eh.layout(d.n, cfg)
        t = time.time(); fb = fr.build_features(d, cfg); print(f"features: {time.time() - t:.1f}s")
        t = time.time(); eh.evaluate_pipeline(d, cfg, lay, 1, enabled, "cpu", fb=fb); print(f"pass without TFT: {time.time() - t:.1f}s")
    elif a.mode == "null":
        ps = []
        for k in range(a.trials):
            sc, nl = run(make_synthetic(a.n, a.nb, 100 + k), cfg, a.reps, 5 + k, enabled)
            ps.append(nl["p_maxT"]); print(f"trial {k}: max-T p = {nl['p_maxT']:.3f}  obs={ {m: round(v, 4) for m, v in sc.items()} }")
        print("p-values:", np.round(sorted(ps), 3), "| fraction below 0.05:", np.mean(np.array(ps) < 0.05))
    else:
        for k in range(a.trials):
            sc, nl = run(make_momentum(a.n, a.nb, a.delta, 200 + k), cfg, a.reps, 5 + k, enabled)
            print(f"trial {k}: max-T p = {nl['p_maxT']:.3f}  z={ {m: round(v, 1) for m, v in nl['z_obs'].items()} }")
