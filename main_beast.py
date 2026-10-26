#!/usr/bin/env python3
"""
main_beast.py: EVERYTHING we built, on one game, working together.

    python main_beast.py --game Afriluck --afriluck-dir afriluck_output
    python main_beast.py --game DayWA --nla-dir nla_audited
    python main_beast.py --game Afriluck --holdout 1        # hide the latest draw, forecast it, show the result
    python main_beast.py --game Afriluck --save afl.npz     # keep every forecast (profit_history.py, bet_lab.py)

Components (each makes a forecast for every draw from the draws before it only):
  stage 4 ORIGINAL  the real stage4_astronomical.py pipeline: TFT army (5 neural nets), LightGBM ranker, hazard,
                    markov, anchor, MLP, booster, rule engine and its own ensemble (skip with --fast). On Afriluck it
                    now also sees long-run frequency (the pink bias lives there); --no-long-run restores the old ban
  quick experts     ball frequency, recent-50, overdue, follow-ups (markov), gradient boosting
  rule engine       the data-mined rule engine (up to 108,000 rules, proven only on evidence)
  stage 5           the forecaster: laws, precedents, partners, analogues, base rates, groups + ML brain
  sharp             the self-tuning hierarchical colour/low-high model
  colour only       one rate for the low half (Afriluck: pink 1-28) and one for the high half
THE BEAST (v2): before every draw it weights its members by their forecasting record on the draws before it
(Bayesian model averaging, slowly forgetting), with "pure chance" always in the mix. Since the 2026-10 audit the
default members are chance + sharp + colour only (--combiner pruned): averaging all 15 components diluted sharp and
lost money on the untouched half of the history, while the pruned mix matched sharp exactly. --combiner all gives the
old 15-model average. Every component is still run and shown in the backtest table, so you can see the comparison.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# The original stage 4 modules read the balls-per-draw (ASTRO_K) when they are imported: 6 for Afriluck, 5 for
# DayWA. From the command line --game decides; when main_beast is imported as a module, an ASTRO_K that is already
# set is respected (the old code silently forced 6, which broke DayWA when imported).
if "--game" in sys.argv[:-1]:
    os.environ["ASTRO_K"] = "6" if sys.argv[sys.argv.index("--game") + 1] == "Afriluck" else "5"
else:
    os.environ.setdefault("ASTRO_K", "6")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "eval_astro2_v3"))

import afriluck_lab as lab          # noqa: E402
import forecaster as fc             # noqa: E402
import sharp                        # noqa: E402

PRUNED = ("sharp", "colour only")    # the BEAST's members by default (plus pure chance)


def load(game, nla_dir, afl_dir):
    a = argparse.Namespace(game=game, nla_dir=nla_dir, afriluck_dir=afl_dir)
    d = fc.load(a)
    return d


def norm(p, k):
    p = np.clip(p, 1e-6, None)
    for _ in range(5):
        p = np.clip(p * (k / p.sum(-1, keepdims=True)), 1e-6, 0.95)
    return p


def next_slot(d, game):
    """(date, session) of the draw after the latest one in the data, from the real weekly schedule."""
    dates = d["dates"]
    rec = dates >= dates[-1] - pd.Timedelta(days=120)
    if game == "Afriluck":
        seq = ["Anopa", "Midday", "evening"]
        runs = {(dw, s) for dw, s in zip(dates[rec].dayofweek, d["session"][rec])}
        slot_d, slot_i = dates[-1], seq.index(d["session"][-1])
        for _ in range(30):
            slot_i += 1
            if slot_i == 3:
                slot_i, slot_d = 0, slot_d + pd.Timedelta(days=1)
            if (slot_d.dayofweek, seq[slot_i]) in runs:
                break
        return slot_d, seq[slot_i]
    cnt = pd.Series(dates[rec].dayofweek).value_counts()
    days = set(cnt[cnt >= 0.5 * cnt.max()].index)                # a weekday with only a delayed draw or two is not a draw day
    nd = dates[-1] + pd.Timedelta(days=1)
    for _ in range(7):
        if nd.dayofweek in days:
            break
        nd += pd.Timedelta(days=1)
    return nd, None


STAGE4_START = 0


def original_stage4(d, device="auto", long_run=False):
    """The ORIGINAL stage4_astronomical.py pipeline (TFT army, LightGBM ranker, hazard, markov, anchor, MLP, booster,
    rule engine + its own leak-free ensemble), one walk-forward pass on exactly the draws main_beast uses.
    Its 500-replicate null test is NOT run: that is a significance test, not a forecast. Here the BEAST's own
    record-vs-chance weighting decides how much stage 4 is trusted.
    long_run=True (Afriluck) lets the feature-based experts see long-run ball and half rates.
    Returns (name -> (n+1, nb)); draws before stage 4's first scored draw are filled with chance."""
    import importlib.util
    import logging
    import tempfile
    import time
    nb, k, W, dates = d["nb"], d["k"], d["W"], d["dates"]
    n = len(W)
    import eval_harness as eh
    import features_recent as fr
    import stage2_powerhouse as s2
    import stage4_astronomical as s4
    from nla_config import get_config
    if s2.N_DRAWN != k:
        raise RuntimeError(f"stage 4 was loaded for {s2.N_DRAWN}-ball draws but this game has {k}. Run main_beast.py "
                           f"with --game, or set the environment variable ASTRO_K={k} before importing it.")
    logging.getLogger().setLevel(logging.WARNING)
    s2.log.setLevel(logging.WARNING)
    cfg = get_config()
    cfg["eval"]["null_reps"] = 0
    cfg["features"]["allow_long_run"] = bool(long_run)
    s4.set_pool(nb)
    tmp = Path(tempfile.mkdtemp())
    csv = tmp / "draws.csv"
    pd.DataFrame({"Date": [x.strftime("%Y-%m-%d") for x in dates],
                  **{f"Number{i + 1}": W[:, i] + 1 for i in range(k)}}).to_csv(csv, index=False)   # oldest first
    sd = s2.load_data(str(csv), tmp)
    assert sd.n == n, f"stage 4 read {sd.n} draws, expected {n}"
    lay = eh.layout(n, cfg)
    enabled = [e for e in eh.ENSEMBLE_EXPERTS if e != "rulemem"]
    if importlib.util.find_spec("torch") is None:
        enabled.remove("tft_army")
        print("  !! torch is not installed: the TFT army (5 neural nets) is SKIPPED. In Colab it is pre-installed; "
              "on a PC: pip install torch", flush=True)
    t0 = time.time()
    print(f"  original stage 4: experts {enabled}, scoring draws {lay['span_start']}..{n - 1} "
          f"(retrained every {lay['cad']} draws){', long-run frequency ALLOWED' if long_run else ''}. "
          f"This is the slow part.", flush=True)
    fb = fr.build_features(sd, cfg)
    run = eh.evaluate_pipeline(sd, cfg, lay, cfg["seed"], enabled, device, predict_next=True, fb=fb, quiet=True)
    nxt = s4.next_play(run["res"], sd, cfg, lay)
    print(f"  original stage 4 finished in {time.time() - t0:.0f}s", flush=True)
    q = k / nb
    out = {}
    ens = np.full((n + 1, nb), q)
    ens[lay["eval_start"]:n] = run["p_ens"]
    ens[n] = nxt["p_play"]
    out["stage4 ORIGINAL ensemble"] = norm(ens, k)
    global STAGE4_START
    STAGE4_START = lay["eval_start"]
    for e, pr in run["res"]["probs"].items():
        if e not in enabled or e == "rulemine":        # rulemine = the same rule engine main_beast already runs
            continue
        x = np.full((n + 1, nb), q)
        x[lay["span_start"]:n] = pr
        if e in run["res"]["next"]:
            x[n] = run["res"]["next"][e]
        out[f"stage4 {e}"] = norm(x, k)
    return out


def components(d, full=True, device="auto", long_run=None, next_date=None):
    """(name -> (n+1, nb)) walk-forward forecasts; row n = the next, unseen draw."""
    nb, k, W, M, dates = d["nb"], d["k"], d["W"], d["M"], d["dates"]
    n = len(W)
    if long_run is None:
        long_run = nb == 57
    if next_date is None:
        next_date = dates[-1] + pd.Timedelta(days=1)
    out = {}
    if full:
        out.update(original_stage4(d, device, long_run))
    # stage 4 experts (afriluck_lab is written for any pool once its constants are set)
    lab.NB, lab.K, lab.PINK = nb, k, nb // 2
    Wx = np.vstack([W, W[-1:]])                                   # dummy last row: row n only uses draws < n
    dx = pd.DatetimeIndex(list(dates) + [pd.Timestamp(next_date)])
    ex = lab.experts(Wx, dx, use_rule=False, use_boost=True)
    for nm in ("frequency", "recent50", "overdue", "markov", "boosting"):
        out[f"quick {nm}"] = ex[nm]
    # rule engine
    import rule_miner as rmn
    rmn.K = k
    out["rule engine"] = norm(rmn.run(np.asarray(W, dtype=np.int64), dx, nb, {}, warmup=min(100, n - 1),
                                      return_p=True)["p"], k)
    # stage 5 forecaster with brain
    eng = fc.Forecaster(d, warm=100, use_brain=True)
    p5, _, _, _ = eng.run(W, M, dates, next_date=next_date)
    out["stage5 forecaster"] = norm(p5, k)
    # sharp + colour only
    X = np.zeros((n + 1, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    P, colour = sharp.predict_all(X, nb, k)
    sp, _ = sharp.bma(P, X)
    out["sharp"] = norm(sp, k)
    out["colour only"] = norm(colour, k)
    return out


def beast(comp, W, nb, k, forget=0.999, members=None):
    """Bayesian model averaging of `members` (default: every component) + pure chance, walk-forward."""
    n = len(W)
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    use = [c for c in comp if members is None or c in members]
    names = ["chance"] + use
    q = k / nb
    P = np.stack([np.full((n + 1, nb), q)] + [comp[c] for c in use])
    ls = np.zeros(len(names))
    res = np.empty((n + 1, nb))
    wh = []
    for t in range(n + 1):
        w = np.exp(ls - ls.max())
        w /= w.sum()
        res[t] = (w[:, None] * P[:, t]).sum(0)
        wh.append(w)
        if t < n:
            y = X[t]
            ls = forget * ls + (y * np.log(P[:, t]) + (1 - y) * np.log1p(-P[:, t])).sum(1)
    return norm(res, k), names, np.array(wh)


def payout_lines(game, picks, drawn, k):
    """What the forecast would have won on the held-out draw."""
    core = picks[:10]
    h = len(set(core) & set(drawn))
    lines = [f"  10-ball core {core}: {h} of the {k} winning numbers inside"]
    if game == "Afriluck":
        pairs = math.comb(h, 2)
        lines.append(f"  Perm-2 on the core (45 pairs x GH₵1 = GH₵45): {pairs} winning pair(s) x 60 = GH₵{60 * pairs} "
                     f"-> {'PROFIT' if 60 * pairs > 45 else 'LOSS'} of GH₵{abs(60 * pairs - 45)}")
        h3 = len(set(picks[:3]) & set(drawn))
        lines.append(f"  Perm-2 on the top 3 {picks[:3]} (3 pairs = GH₵3): {math.comb(h3, 2)} winning pair(s) -> "
                     f"GH₵{60 * math.comb(h3, 2) - 3:+}")
        top2 = picks[:2]
        lines.append(f"  Direct-2 on {top2} (GH₵1): {'WIN GH₵60' if set(top2) <= set(drawn) else 'lost GH₵1'}")
        lines.append(f"  Direct-1 first ball on {picks[0]} (GH₵1): "
                     f"{'WIN GH₵30' if picks[0] == drawn[0] else 'lost GH₵1'} (first ball drawn: {drawn[0]})")
    else:
        t5 = picks[:5]
        m5 = len(set(t5) & set(drawn))
        lines.append(f"  DayWA ticket {t5}: {m5} correct -> "
                     + ("consolation prize (2+ correct; amount not published)" if m5 >= 2 else "no prize"))
    return lines


def money_verdict(W):
    """The honest money line for Afriluck, from the Plackett-Luce ceiling (bet_lab.py has the full table)."""
    import bet_lab as bl
    lw, se = bl.fit_colour_pl(W)
    c = bl.ceiling(math.exp(lw), N=300_000)
    hi = bl.ceiling(math.exp(lw + 1.96 * se), N=300_000)
    best = max(c, key=c.get)
    return (f"MONEY: even if the pink bias were known exactly, the best bet ({best}) returns about "
            f"{c[best]:.0%} per GH₵1 (at most {hi[best]:.0%} at the top of its 95% interval). Perm-2 {c['perm2']:.0%}, "
            f"Banker {c['banker']:.0%}, Direct-1 {c['direct1']:.0%}, Direct-3 {c['direct3']:.0%}, Direct-4 "
            f"{c['direct4']:.0%}. NO Afriluck bet has a positive expected value. If you play anyway, the smallest "
            f"expected loss is Perm-2 on pink balls only; the top 3 (3 pairs, GH₵3 a draw) keeps the loss per draw "
            f"smallest: about GH₵{3 * (1 - c['perm2']):.2f} a draw on average. Run bet_lab.py for the full GH₵ table.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Afriluck", choices=["Afriluck", "DayWA"])
    ap.add_argument("--nla-dir", default="nla_audited")
    ap.add_argument("--afriluck-dir", default="afriluck_output")
    ap.add_argument("--holdout", type=int, default=0, help="hide the latest N draws and show the forecast for each")
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--fast", action="store_true", help="skip the ORIGINAL stage 4 (TFT army etc.): about 30s instead of "
                                                        "several minutes")
    ap.add_argument("--combiner", default="pruned", choices=["pruned", "all"],
                    help="pruned (default) = chance + sharp + colour; all = the old average of every component")
    ap.add_argument("--no-long-run", action="store_true", help="Afriluck: keep the old long-run frequency ban in stage 4")
    ap.add_argument("--save", default="", help="save every walk-forward forecast to this .npz (profit_history.py, "
                                               "bet_lab.py)")
    ap.add_argument("--device", default="auto", help="for the TFT army: auto / cpu / cuda")
    a = ap.parse_args()
    d = load(a.game, a.nla_dir, a.afriluck_dir)
    full_W, full_M, full_dates = d["W"], d["M"], d["dates"]
    if a.holdout:
        cut = full_dates[-a.holdout]
        if "session" in d:
            d["session"] = d["session"][:-a.holdout]
        others = {g: (o[0][o[0] < cut], o[1][(o[0] < cut)], o[2][(o[0] < cut)]) + tuple(o[3:])
                  for g, o in d["others"].items()}                 # other games: only draws before the hidden one
        d = dict(d, W=full_W[:-a.holdout], M=full_M[:-a.holdout], dates=full_dates[:-a.holdout], others=others)
    nb, k, W = d["nb"], d["k"], d["W"]
    n = len(W)
    nxt_date, nxt_sess = next_slot(d, a.game)
    print(f"MAIN BEAST: {d['name']} | {n} draws used ({d['dates'][0].date()} to {d['dates'][-1].date()})"
          + (f" | latest {a.holdout} draw(s) HIDDEN" if a.holdout else ""), flush=True)
    import time
    t0 = time.time()
    comp = components(d, full=not a.fast, device=a.device,
                      long_run=(a.game == "Afriluck" and not a.no_long_run), next_date=nxt_date)
    members = PRUNED if a.combiner == "pruned" else [c for c in comp if c != "colour only"]
    p, names, wh = beast(comp, W, nb, k, members=members)
    if a.save:
        np.savez_compressed(a.save, p=p, W=W, dates=np.array([str(x.date()) for x in d["dates"]]),
                            session=np.array(d.get("session", np.array(["DayWA"] * n))), game=a.game, nb=nb, k=k,
                            combiner=a.combiner, **{"c__" + c: v for c, v in comp.items()})
        print(f"Forecasts saved to {a.save}", flush=True)
    # backtest on the visible history
    X = np.zeros((n, nb))
    X[np.repeat(np.arange(n), k), W.ravel()] = 1
    if a.fast:                                       # same scored draws as a full run, so the tables compare
        import eval_harness as eh
        from nla_config import get_config
        start = max(a.warm, eh.layout(n, get_config())["eval_start"])
    else:
        start = max(a.warm, STAGE4_START)
    print(f"\nBacktest on draws {start}..{n - 1} ({d['dates'][start].date()} to {d['dates'][n - 1].date()}; every draw "
          f"forecast from earlier draws only; same draws for every row so the rows compare fairly):")
    afl = a.game == "Afriluck"
    print("| component | 2+ in 10-core | hits in top k |" + (" Perm-2 back per GH₵1 (top 10 / top 3) |" if afl else "")
          + " trust now |")
    print("|---|---|---|" + ("---|" if afl else "") + "---|")
    from math import comb
    ch = sum(comb(10, j) * comb(nb - 10, k - j) for j in range(2, k + 1)) / comb(nb, k)
    rows = {f"MAIN BEAST ({'chance + sharp + colour' if a.combiner == 'pruned' else 'all combined'})": p, **comp}
    for nm, pr in rows.items():
        s = slice(start, n)
        o = np.argsort(-(pr[s] + np.random.default_rng(1).random((n - start, nb)) * 1e-12), axis=1)
        ar = np.arange(n - start)[:, None]
        hc = X[s][ar, o[:, :10]].sum(1)
        h3 = X[s][ar, o[:, :3]].sum(1)
        hk = X[s][ar, o[:, :k]].sum(1).mean()
        trust = "-" if nm.startswith("MAIN") else (f"{wh[-1][names.index(nm)]:.2f}" if nm in names else "not a member")
        money = (f" {(60 * hc * (hc - 1) / 2).mean() / 45:.3f} / {(60 * h3 * (h3 - 1) / 2).mean() / 3:.3f} |"
                 if afl else "")
        print(f"| {nm} | {(hc >= 2).mean():.1%} | {hk:.3f} |{money} {trust} |")
    chp = 60 * k * (k - 1) / (nb * (nb - 1))
    print(f"| pure chance | {ch:.1%} | {k * k / nb:.3f} |" + (f" {chp:.3f} / {chp:.3f} |" if afl else "")
          + f" {wh[-1][0]:.2f} |")
    print(f"\n{len(comp)} models run in {time.time() - t0:.0f}s"
          + (" (FAST mode: the original stage 4 was skipped)" if a.fast else "")
          + f"; the BEAST combines {', '.join(names)}")
    nxt = p[n]
    order = [int(b) + 1 for b in np.argsort(-nxt)]
    last_d = d["dates"][-1].date()
    if afl:
        what = (f"the next Afriluck draw after {last_d} {d['session'][n - 1]}, i.e. {nxt_sess} on "
                f"{nxt_date.day_name()} {nxt_date.date()}. All three Afriluck games use the same machine and balls, "
                f"so this core also holds for any later Anopa / Midday / evening draw until you add new results")
    else:
        what = f"the next DayWA draw after {last_d}, i.e. {nxt_date.day_name()} {nxt_date.date()}"
    print(f"\nLATEST DRAW IN YOUR DATA: {last_d}" + (f" ({d['session'][n - 1]})" if afl else ""))
    print(f"FORECASTING: {what}.")
    print(f"Forecast: top {k} {order[:k]} | 10-ball core {order[:10]} | strongest ball x{nxt.max() / (k / nb):.3f} chance")
    if afl:
        print("\n" + money_verdict(W))
    else:
        print("\nMONEY: DayWA shows no edge that survives fair-machine controls and correction for the number of models "
              "tried, and its prize table is not published, so no DayWA bet can be shown to pay.")
    stale = (pd.Timestamp.now().normalize() - d["dates"][-1]).days
    if stale >= 2 and not a.holdout:
        print(f"NOTE: your data is {stale} days old. Re-scrape first for a forecast of today's draw.")
    if a.holdout:
        print("\nThe 10 draws before it, each forecast blind by the same combined system:")
        for t in range(n - 10, n):
            o10 = [int(b) + 1 for b in np.argsort(-p[t])]
            dr = [int(x) + 1 for x in W[t]]
            h = len(set(o10[:10]) & set(dr))
            extra = (f" | Perm-2 GH₵45 -> GH₵{60 * math.comb(h, 2)}" if afl else
                     f" | ticket {o10[:5]}: {len(set(o10[:5]) & set(dr))} correct")
            print(f"  {d['dates'][t].date()}: drawn {dr} | {h} in the 10-core{extra}")
        drawn = [int(x) + 1 for x in full_W[n]]
        print(f"\nHIDDEN DRAW {full_dates[n].date()}: actual numbers {drawn}")
        for line in payout_lines(a.game, order, drawn, k):
            print(line)


if __name__ == "__main__":
    main()
