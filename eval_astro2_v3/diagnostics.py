"""Randomness diagnostics for a 5/90 draw history.

Null hypothesis for every test: each draw is 5 balls chosen uniformly without
replacement from 90, independently of all other draws. Where an asymptotic
approximation is shaky (frequency chi-square, pair dispersion), the p-value is
calibrated by Monte Carlo against truly random draws instead.

Many tests are run, so some raw p < 0.05 are EXPECTED by chance. Read the
Benjamini-Hochberg q and Bonferroni columns, not the raw flags.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import DEFAULT_SEED, N_BALLS, N_DRAWN

log = logging.getLogger("nla.diagnostics")

P_BALL = N_DRAWN / N_BALLS  # 5/90
N_PAIRS = N_BALLS * (N_BALLS - 1) // 2  # 4005
P_PAIR = (N_DRAWN * (N_DRAWN - 1)) / (N_BALLS * (N_BALLS - 1))  # 10/4005
ALPHA = 0.05


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float  # the single p-value this test contributes to the summary
    detail: dict = field(default_factory=dict)
    note: str = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def random_draws(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random 5-of-90 draws, as ball indices (0..89), shape (n, 5)."""
    return np.argpartition(rng.random((n, N_BALLS)), N_DRAWN, axis=1)[:, :N_DRAWN]


def random_tensor(rng: np.random.Generator, n: int) -> np.ndarray:
    idx = random_draws(rng, n)
    X = np.zeros((n, N_BALLS), dtype=np.uint8)
    X[np.arange(n)[:, None], idx] = 1
    return X


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def _two_sided_binom_p(k: np.ndarray, n: int, p: float) -> np.ndarray:
    lo = stats.binom.cdf(k, n, p)
    hi = stats.binom.sf(k - 1, n, p)
    return np.minimum(1.0, 2 * np.minimum(lo, hi))


def _chisq_gof(obs, probs, min_exp: float = 5.0):
    """Chi-square goodness of fit, merging bins until every expected count >= min_exp."""
    n = float(np.sum(obs))
    o = [float(x) for x in obs]
    e = list(n * np.asarray(probs, dtype=float))
    while len(e) > 2 and min(e) < min_exp:
        i = int(np.argmin(e))
        if i == 0:
            j = 1
        elif i == len(e) - 1:
            j = i - 1
        else:
            j = i - 1 if e[i - 1] < e[i + 1] else i + 1
        lo, hi = sorted((i, j))
        o[lo] += o[hi]
        e[lo] += e[hi]
        del o[hi], e[hi]
    o, e = np.array(o), np.array(e)
    stat = float(((o - e) ** 2 / e).sum())
    dof = len(e) - 1
    return stat, dof, float(stats.chi2.sf(stat, dof))


def _timed(label):
    class _T:
        def __enter__(self):
            self.t = time.time()
            log.info(">> %s", label)
            return self

        def __exit__(self, *a):
            log.info("<< %s (%.1fs)", label, time.time() - self.t)

    return _T()


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_frequency(X, df, rng, n_sims) -> TestResult:
    """Do all 90 balls come up equally often? (chi-square, Monte Carlo calibrated)"""
    N = len(X)
    counts = X.sum(axis=0).astype(float)
    exp = N * P_BALL
    stat = float(((counts - exp) ** 2 / exp).sum())
    p_asym = float(stats.chi2.sf(stat, N_BALLS - 1))

    sims = np.empty(n_sims)
    for i in range(n_sims):
        c = np.bincount(random_draws(rng, N).ravel(), minlength=N_BALLS)
        sims[i] = ((c - exp) ** 2 / exp).sum()
        if (i + 1) % 500 == 0:
            log.debug("frequency MC %d/%d", i + 1, n_sims)
    p_mc = float((1 + (sims >= stat).sum()) / (n_sims + 1))
    log.info("   chi2=%.2f (df=89) | p_asymptotic=%.4f | p_MC=%.4f (sim mean %.1f)", stat, p_asym, p_mc, sims.mean())
    return TestResult("Ball frequency (chi-square, MC)", stat, p_mc,
                      {"p_asymptotic": p_asym, "expected_per_ball": exp, "n_draws": N, "n_sims": n_sims})


def test_ball_binomial(X, df, rng, n_sims) -> TestResult:
    """Is any single ball over/under-represented? (exact binomial, BH-corrected)"""
    N = len(X)
    k = X.sum(axis=0).astype(int)
    p = _two_sided_binom_p(k, N, P_BALL)
    q = bh_qvalues(p)
    exp = N * P_BALL
    sd = np.sqrt(N * P_BALL * (1 - P_BALL))
    order = np.argsort(p)[:5]
    top = [{"ball": int(b + 1), "count": int(k[b]), "expected": round(exp, 1),
            "z": round(float((k[b] - exp) / sd), 2), "p": float(p[b]), "q_BH": float(q[b])} for b in order]
    n_sig = int((q < ALPHA).sum())
    p_bonf = float(min(1.0, p.min() * N_BALLS))
    for t in top[:3]:
        log.info("   ball %2d: count=%d (exp %.1f) z=%+.2f p=%.4f q=%.3f", t["ball"], t["count"], exp, t["z"], t["p"], t["q_BH"])
    log.info("   balls with BH q<%.2f: %d | min p Bonferroni-adjusted: %.4f", ALPHA, n_sig, p_bonf)
    return TestResult("Single-ball bias (binomial, min-p Bonferroni)", float(p.min()), p_bonf,
                      {"top5": top, "n_balls_q_lt_0.05": n_sig})


def test_runs(X, df, rng, n_sims) -> TestResult:
    """Wald-Wolfowitz runs test per ball: do appearances cluster or alternate too regularly?"""
    N = len(X)
    Xb = X.astype(np.int64)
    n1 = Xb.sum(axis=0)
    n0 = N - n1
    R = 1 + (Xb[1:] != Xb[:-1]).sum(axis=0)
    mean = 2.0 * n1 * n0 / N + 1
    var = (mean - 1) * (mean - 2) / (N - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, (R - mean) / np.sqrt(var), 0.0)
    p = 2 * stats.norm.sf(np.abs(z))
    ks = stats.kstest(p, "uniform")
    n_sig = int((p < ALPHA).sum())
    p_binom = float(stats.binomtest(n_sig, N_BALLS, ALPHA).pvalue)
    log.info("   balls with runs p<0.05: %d (expected ~%.1f) | KS-uniformity p=%.4f | mean z=%+.3f",
             n_sig, N_BALLS * ALPHA, ks.pvalue, z.mean())
    worst = int(np.argmin(p))
    return TestResult("Runs test per ball (KS of 90 p-values)", float(ks.statistic), float(ks.pvalue),
                      {"n_p_lt_0.05": n_sig, "expected": N_BALLS * ALPHA, "binomial_p_on_count": p_binom,
                       "mean_z": float(z.mean()), "most_extreme_ball": worst + 1, "its_p": float(p[worst])},
                      note="z>0 = appearances alternate more than chance; z<0 = they clump.")


def test_autocorrelation(X, df, rng, n_sims, max_lag: int = 10) -> TestResult:
    """Per-ball indicator autocorrelation at lags 1..max_lag, pooled across balls."""
    N = len(X)
    xc = X.astype(np.float64) - X.mean(axis=0)
    den = (xc ** 2).sum(axis=0)
    rows, pvals = [], []
    for lag in range(1, max_lag + 1):
        acf = (xc[:-lag] * xc[lag:]).sum(axis=0) / den
        z = acf * np.sqrt(N)
        chi = float((z ** 2).sum())
        p = float(stats.chi2.sf(chi, N_BALLS))
        frac = float((np.abs(z) > 1.96).mean())
        rows.append({"lag": lag, "mean_acf": float(acf.mean()), "chi2_p": p, "frac_|z|>1.96": frac})
        pvals.append(p)
        log.debug("   lag %2d: mean acf=%+.4f chi2 p=%.4f frac|z|>1.96=%.3f (null 0.05)", lag, acf.mean(), p, frac)
    p_adj = float(min(1.0, min(pvals) * max_lag))
    log.info("   lag-wise chi2 p-values: %s | Bonferroni-adjusted min p=%.4f",
             [round(x, 3) for x in pvals], p_adj)
    return TestResult(f"Per-ball autocorrelation (lags 1-{max_lag})", float(min(pvals)), p_adj,
                      {"lags": rows}, note="Approximate: balls within a draw are weakly dependent.")


def test_overlap(X, df, rng, n_sims, max_lag: int = 10) -> TestResult:
    """How many balls repeat from k draws ago? Null: Hypergeometric(90, 5, 5), mean 25/90."""
    h = stats.hypergeom(N_BALLS, N_DRAWN, N_DRAWN)
    mu, sd = h.mean(), h.std()
    rows, pvals = [], []
    for k in range(1, max_lag + 1):
        ov = (X[:-k].astype(np.int16) * X[k:]).sum(axis=1)
        n = len(ov)
        z = (ov.mean() - mu) / (sd / np.sqrt(n))
        p = float(2 * stats.norm.sf(abs(z)))
        rows.append({"lag": k, "mean_overlap": float(ov.mean()), "expected": float(mu), "z": float(z), "p": p})
        pvals.append(p)
        log.debug("   lag %2d: mean overlap=%.4f (exp %.4f) z=%+.2f p=%.4f", k, ov.mean(), mu, z, p)
    p_adj = float(min(1.0, min(pvals) * max_lag))
    log.info("   overlap lag p-values: %s | Bonferroni-adjusted min p=%.4f", [round(x, 3) for x in pvals], p_adj)
    return TestResult(f"Ball repeats vs {max_lag} previous draws (hypergeometric)", float(min(pvals)), p_adj,
                      {"lags": rows}, note="Directly tests whether the machine 'remembers' recent draws.")


def test_gaps(X, df, rng, n_sims) -> TestResult:
    """Gap between appearances of a ball must be Geometric(5/90) (mean 18)."""
    gaps = [np.diff(np.flatnonzero(X[:, b])) for b in range(N_BALLS)]
    g = np.concatenate([x for x in gaps if len(x) > 0])
    G = len(g)
    p = P_BALL
    kmax = 1
    while G * (1 - p) ** kmax * p >= 5 and kmax < 300:  # keep expected count >= 5 in each bin
        kmax += 1
    kmax = max(kmax - 1, 1)
    obs = np.bincount(np.minimum(g, kmax + 1), minlength=kmax + 2)[1:]
    probs = np.array([(1 - p) ** (k - 1) * p for k in range(1, kmax + 1)] + [(1 - p) ** kmax])
    stat, dof, pval = _chisq_gof(obs, probs)
    exp_var = (1 - p) / p ** 2
    log.info("   gaps=%d | mean=%.2f (exp %.2f) var=%.1f (exp %.1f) | chi2=%.2f df=%d p=%.4f",
             G, g.mean(), 1 / p, g.var(), exp_var, stat, dof, pval)
    return TestResult("Gap distribution (chi-square vs Geometric)", stat, pval,
                      {"n_gaps": G, "mean_gap": float(g.mean()), "expected_mean_gap": 1 / p,
                       "gap_variance": float(g.var()), "expected_variance": exp_var, "dof": dof},
                      note="Approximate: pooled across balls.")


def test_draw_structure(X, df, rng, n_sims) -> list[TestResult]:
    """Odd/even split, low/high split, and sum of the five numbers."""
    N = len(X)
    out = []
    h = stats.hypergeom(N_BALLS, 45, N_DRAWN)
    probs = h.pmf(np.arange(N_DRAWN + 1))

    odd = X[:, 0::2].sum(axis=1)  # balls 1,3,5,... (indices 0,2,4,...)
    stat, dof, p = _chisq_gof(np.bincount(odd, minlength=6), probs)
    log.info("   odd-count distribution: chi2=%.2f df=%d p=%.4f | observed %s", stat, dof, p, np.bincount(odd, minlength=6).tolist())
    out.append(TestResult("Odd/even split (chi-square vs hypergeometric)", stat, p,
                          {"observed": np.bincount(odd, minlength=6).tolist(), "expected": (N * probs).round(1).tolist()}))

    low = X[:, :45].sum(axis=1)  # balls 1-45
    stat, dof, p = _chisq_gof(np.bincount(low, minlength=6), probs)
    log.info("   low-count (1-45) distribution: chi2=%.2f df=%d p=%.4f | observed %s", stat, dof, p, np.bincount(low, minlength=6).tolist())
    out.append(TestResult("Low/high split (chi-square vs hypergeometric)", stat, p,
                          {"observed": np.bincount(low, minlength=6).tolist(), "expected": (N * probs).round(1).tolist()}))

    sums = X.astype(np.float64) @ np.arange(1, N_BALLS + 1)
    sigma2 = (N_BALLS ** 2 - 1) / 12
    mu = N_DRAWN * (N_BALLS + 1) / 2
    var = N_DRAWN * sigma2 * (N_BALLS - N_DRAWN) / (N_BALLS - 1)
    z = (sums.mean() - mu) / np.sqrt(var / N)
    p_z = float(2 * stats.norm.sf(abs(z)))
    ref = (random_tensor(rng, 100_000).astype(np.float64) @ np.arange(1, N_BALLS + 1))
    ks = stats.ks_2samp(sums, ref)
    log.info("   draw sum: mean=%.2f (exp %.2f) z=%+.2f p=%.4f | KS vs 100k random draws p=%.4f",
             sums.mean(), mu, z, p_z, ks.pvalue)
    out.append(TestResult("Sum of 5 numbers (KS vs simulated random)", float(ks.statistic), float(ks.pvalue),
                          {"mean": float(sums.mean()), "expected_mean": mu, "mean_z_p": p_z}))
    return out


def test_pairs(X, df, rng, n_sims) -> TestResult:
    """Do specific pairs of balls co-occur more than chance? (4005 pairs)"""
    N = len(X)
    Xf = X.astype(np.float32)
    iu = np.triu_indices(N_BALLS, 1)
    cnt = (Xf.T @ Xf)[iu].astype(int)
    e = N * P_PAIR
    stat = float(((cnt - e) ** 2 / e).sum())

    n_pair_sims = min(n_sims, 300)
    sims = np.empty(n_pair_sims)
    for i in range(n_pair_sims):
        Xs = random_tensor(rng, N).astype(np.float32)
        c = (Xs.T @ Xs)[iu]
        sims[i] = ((c - e) ** 2 / e).sum()
    p_mc = float((1 + (sims >= stat).sum()) / (n_pair_sims + 1))

    p = _two_sided_binom_p(cnt, N, P_PAIR)
    q = bh_qvalues(p)
    order = np.argsort(p)[:5]
    top = [{"pair": (int(iu[0][i] + 1), int(iu[1][i] + 1)), "count": int(cnt[i]), "expected": round(e, 2),
            "p": float(p[i]), "q_BH": float(q[i])} for i in order]
    log.info("   pair dispersion=%.1f (MC mean %.1f) p_MC=%.4f | pairs with BH q<0.05: %d | top pair %s",
             stat, sims.mean(), p_mc, int((q < ALPHA).sum()), top[0]["pair"])
    return TestResult("Pair co-occurrence (dispersion, MC)", stat, p_mc,
                      {"top5": top, "n_pairs_q_lt_0.05": int((q < ALPHA).sum()), "n_sims": n_pair_sims})


def test_weekday(X, df, rng, n_sims) -> TestResult | None:
    """Does the ball distribution differ by weekday? (the premise of the Micro-engine)"""
    dow = df["DayOfWeek"].to_numpy()
    days = [d for d in range(7) if (dow == d).sum() > 0]
    if len(days) < 2:
        log.warning("   weekday test skipped: fewer than 2 weekdays present")
        return None
    table = np.array([X[dow == d].sum(axis=0) for d in days])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    per_day = []
    for d, row in zip(days, table):
        n_d = int((dow == d).sum())
        e = n_d * P_BALL
        c = float(((row - e) ** 2 / e).sum())
        per_day.append({"day": names[d], "n_draws": n_d, "chi2": c, "p": float(stats.chi2.sf(c, N_BALLS - 1))})
        log.info("   %s: n=%d chi2=%.1f p=%.4f", names[d], n_d, c, per_day[-1]["p"])
    log.info("   weekday x ball contingency: chi2=%.1f dof=%d p=%.4f", chi2, dof, p)
    note = ""
    if expected.min() < 5:
        note = f"Some expected cell counts < 5 (min {expected.min():.1f}); p-value is approximate."
        log.warning("   %s", note)
    return TestResult("Weekday x ball heterogeneity (chi-square)", float(chi2), float(p),
                      {"per_weekday": per_day, "dof": int(dof)}, note=note)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_all(X: np.ndarray, df: pd.DataFrame, seed: int = DEFAULT_SEED, n_sims: int = 2000) -> list[TestResult]:
    rng = np.random.default_rng(seed)
    log.info("=== RANDOMNESS DIAGNOSTICS | draws=%d | seed=%d | MC sims=%d ===", len(X), seed, n_sims)
    suite = [test_frequency, test_ball_binomial, test_runs, test_autocorrelation, test_overlap,
             test_gaps, test_draw_structure, test_pairs, test_weekday]
    results: list[TestResult] = []
    for fn in suite:
        with _timed(fn.__name__):
            try:
                r = fn(X, df, rng, n_sims)
            except Exception:
                log.exception("Test %s failed; continuing", fn.__name__)
                continue
        if r is None:
            continue
        results.extend(r if isinstance(r, list) else [r])
    log.info("=== %d diagnostics completed ===", len(results))
    return results


def summarise(results: list[TestResult]) -> tuple[pd.DataFrame, str]:
    results = [r for r in results if np.isfinite(r.p_value)]
    K = len(results)
    p = np.array([r.p_value for r in results])
    q = bh_qvalues(p)
    bonf = ALPHA / K
    tbl = pd.DataFrame({
        "test": [r.name for r in results],
        "p_value": p,
        "BH_q": q,
        "raw_p<0.05": p < ALPHA,
        "survives_Bonferroni": p < bonf,
    })
    n_raw = int((p < ALPHA).sum())
    n_bonf = int((p < bonf).sum())
    lines = [
        f"{K} tests run. At alpha=0.05 about {K * ALPHA:.1f} raw flags are expected by chance; observed {n_raw}.",
        f"Bonferroni threshold across tests: p < {bonf:.4f}; tests passing it: {n_bonf}.",
    ]
    if n_bonf == 0:
        lines.append("VERDICT: no test survives multiple-testing correction. The history is statistically "
                     "consistent with a fair 5/90 draw. There is nothing here a model could legitimately exploit.")
    else:
        names = ", ".join(tbl.loc[tbl["survives_Bonferroni"], "test"])
        lines.append(f"VERDICT: flagged after correction -> {names}. This is a lead, not proof: it must be "
                     "confirmed out-of-sample (walk-forward + permutation tests) before it counts as real.")
    return tbl, "\n".join(lines)


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_report(results, table, verdict, out_dir, game_name, n_draws) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"diagnostics_{game_name.lower()}.json"
    tp = out_dir / f"diagnostics_{game_name.lower()}.txt"
    jp.write_text(json.dumps(_py({"game": game_name, "n_draws": n_draws, "verdict": verdict,
                                  "results": [r.__dict__ for r in results]}), indent=2))
    with tp.open("w", encoding="utf-8") as f:
        f.write(f"RANDOMNESS DIAGNOSTICS - {game_name} ({n_draws} draws)\n{'=' * 64}\n\n")
        f.write(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write(f"\n\n{verdict}\n\n{'-' * 64}\nDETAIL\n{'-' * 64}\n")
        for r in results:
            f.write(f"\n[{r.name}]\n  statistic={r.statistic:.4f}  p={r.p_value:.4f}\n")
            if r.note:
                f.write(f"  note: {r.note}\n")
            for k, v in r.detail.items():
                f.write(f"  {k}: {json.dumps(_py(v))}\n")
    log.info("Diagnostics saved: %s , %s", tp, jp)
    return tp, jp
