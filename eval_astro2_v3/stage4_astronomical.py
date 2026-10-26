#!/usr/bin/env python3
"""
Stage 4 (refactored): recent-window features, zero-init TFT army, hazard / markov / ranker experts, leak-free ensemble,
mandatory evaluation harness with a label-permutation null test, per-game verdict.

Colab (T4):
    !pip -q install lightgbm scipy scikit-learn torch
    !python test_pipeline.py                                              # unit tests first
    !python stage4_astronomical.py --csv daywa_blind2.csv --quick         # smoke test (NOT a verdict)
    !python stage4_astronomical.py --csv daywa_blind2.csv --null-reps 100 # one game; repeat for the 4 games
    !python stage4_astronomical.py --aggregate outputs/                   # combined table across the games
The null test is checkpointed: if Colab disconnects, run the same command again and it resumes.

What changed vs the old stage 4
  * rate_all, rate250, rate500, frozen_frequency and everything derived from them are gone from every model, the vote
    and the digital-twin calibration inputs (assertion in features_recent.assert_no_banned; unit-tested).
  * per-ball recent-window features (rate5/10/20/50, gap_current, gap_ratio, gap_percentile, momentum, decayed Markov).
  * TFT output = logit(k/N) + zero-initialised residual: "no signal" means UNIFORM scores.
  * every pool-size dependent number derives from N_BALLS (detected from the CSV or --n-balls).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as eh
import features_recent as fr
import stage2_powerhouse as s2
from nla_config import N_DRAWN, chance_jaccard, chance_pair, core_size, get_config

log = s2.log


# --------------------------------------------------------------------------- #
# Arguments / config
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    args = s2.parse_args(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-balls", type=int, default=0, help="pool size; 0 = detect from the CSV")
    ap.add_argument("--eval-draws", type=int, default=0, help="walk-forward test draws (>= 300); 0 = config default")
    ap.add_argument("--retrain-every", type=int, default=0)
    ap.add_argument("--null-reps", type=int, default=-1, help="null-test replicates; -1 = config default (500)")
    ap.add_argument("--null-workers", type=int, default=1,
                    help="parallel processes for the null test; 0 = auto (2 on a GPU runtime, else one per 2 CPU cores). "
                         "Same replicates as serial")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="run every null replicate even when the result is already clearly not significant")
    ap.add_argument("--no-tft", action="store_true")
    ap.add_argument("--no-ranker", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--no-booster", action="store_true")
    ap.add_argument("--no-rulemem", action="store_true", help="leave the rule-memory expert out")
    ap.add_argument("--rulemem", action="store_true", help="v3: the old fixed-library rule memory is OFF by default; this "
                                                           "switches it back on")
    ap.add_argument("--no-rulemine", action="store_true", help="leave the v3 data-mined rule engine out")
    ap.add_argument("--feature-windows", default="", help="v3: comma list of count windows, e.g. 5,10,20,50,100,200,400")
    ap.add_argument("--tft-seq-len", type=int, default=0, help="v3: TFT sequence length (v2: 20)")
    ap.add_argument("--markov-half-life", type=float, default=0.0, help="v3: Markov follow-up memory (v2: 40 draws)")
    ap.add_argument("--last300", action="store_true", help="v2 layout: exam = last 300 draws only (v3 default: every "
                                                           "draw after the warm-up)")
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-wheel", action="store_true")
    ap.add_argument("--vote-mode", default="", choices=["", "rank", "prob"])
    ap.add_argument("--top2-mode", default="", choices=["", "vote", "consensus"],
                    help="vote: Top-2 = first 2 of the weighted vote; consensus: balls every heavy expert ranks high")
    ap.add_argument("--anchor-always-on", action="store_true",
                    help="measurement mode: the anchor forecasts in every walk-forward block AND the next draw")
    ap.add_argument("--min-track-record", type=float, default=-1.0,
                    help="share of the validation window an expert must have been active to get weight; -1 = config")
    ap.add_argument("--require-positive-gain", action="store_true",
                    help="experts with validation log-score gain <= 0 vs uniform get weight 0")
    ap.add_argument("--payout", type=float, default=0.0)
    ap.add_argument("--wheel-size", type=int, default=0)
    ap.add_argument("--wheel-ticket", type=int, default=2)
    ap.add_argument("--wheel-price", type=float, default=1.0)
    ap.add_argument("--wheel-pair-price", action="store_true")
    ap.add_argument("--game", default="", help="label for the report (default: CSV name)")
    ap.add_argument("--tracker", default="", help="live-test ledger the pick is appended to (default: <out-dir>/tracker.csv)")
    ap.add_argument("--no-tracker", action="store_true")
    extra, _ = ap.parse_known_args(argv)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    return args


def apply_overrides(cfg, args):
    if args.seed != cfg["seed"]:
        cfg["seed"] = args.seed
    if args.eval_draws:
        cfg["eval"]["eval_draws"] = args.eval_draws
    if getattr(args, "feature_windows", ""):
        w = sorted({int(x) for x in args.feature_windows.split(",") if x.strip()})
        cfg["features"]["use_rate100"] = 100 in w
        cfg["features"]["windows"] = [x for x in w if x != 100]
    if getattr(args, "tft_seq_len", 0):
        cfg["tft"]["seq_len"] = int(args.tft_seq_len)
    if getattr(args, "markov_half_life", 0):
        cfg["features"]["markov_half_life"] = float(args.markov_half_life)
    if getattr(args, "last300", False):
        cfg["eval"]["eval_draws"] = 300
        for k in ("ranker", "mlp", "booster", "tft"):
            cfg[k]["train_window"] = 1500
        cfg["hazard"]["window"] = 1200
    if args.retrain_every:
        cfg["eval"]["retrain_every"] = args.retrain_every
    if args.null_reps >= 0:
        cfg["eval"]["null_reps"] = args.null_reps
    if args.no_early_stop:
        cfg["eval"]["early_stop_h"] = 0
    if args.vote_mode:
        cfg["ensemble"]["vote_mode"] = args.vote_mode
    if args.top2_mode:
        cfg["ensemble"]["top2_mode"] = args.top2_mode
    if args.anchor_always_on:
        cfg["anchor"]["always_on"] = True
    if args.min_track_record >= 0:
        cfg["ensemble"]["min_track_record"] = args.min_track_record
    if args.require_positive_gain:
        cfg["ensemble"]["require_positive_gain"] = True
    if args.quick:
        log.warning("--quick: relaxed floors (eval draws, TFT epochs, null reps). Results are a smoke test, NOT a verdict.")
        cfg["eval"].update(eval_draws=100, retrain_every=50,
                           null_reps=min(cfg["eval"]["null_reps"], 20), boot_reps=300)
        cfg["ensemble"]["val_len"] = 50
        cfg["tft"].update(army=2, hidden=16, heads=2, seq_len=16, max_epochs=6, min_epochs=3, patience=2)
        cfg["ranker"]["params"].update(n_estimators=60)
        args.twin_draws, args.twin_reps = 20_000, 20
    return cfg


# --------------------------------------------------------------------------- #
# Pool size
# --------------------------------------------------------------------------- #
def detect_n(path: str, override: int = 0) -> int:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in s2.NUM_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    nums = raw[s2.NUM_COLS].apply(lambda c: pd.to_numeric(c.astype(str).str.strip(), errors="coerce"))
    ok = nums.notna().all(axis=1) & ((nums % 1) == 0).all(axis=1) & (nums >= 1).all(axis=1)
    v = nums[ok].to_numpy(dtype=int)
    if v.size == 0:
        raise ValueError("no numeric draws found")
    mx = int(v.max())
    counts = np.bincount(v.ravel(), minlength=mx + 1)[1:]
    if override:
        log.info("Pool size forced by --n-balls: %d (largest number seen: %d)", override, mx)
        return int(override)
    med = float(np.median(counts[counts > 0]))
    floor = max(5.0, 0.1 * med)
    n = int(np.where(counts >= floor)[0].max() + 1)
    if mx > n:
        log.warning("Numbers above %d appear only as stray values: treated as bad rows, not as part of the pool", n)
    log.info("DETECTED POOL SIZE N = %d (draws are %d-of-%d); count per ball min %d, median %d, max %d", n, N_DRAWN, n,
             counts[:n].min(), np.median(counts[:n]), counts[:n].max())
    return n


def set_pool(nb: int):
    s2.N_BALLS, s2.BASE = int(nb), N_DRAWN / int(nb)       # stage2's run_twin / simulate_machine read these globals


# --------------------------------------------------------------------------- #
# Next-draw play (final fit on all draws, weights tuned on the last validation window)
# --------------------------------------------------------------------------- #
def next_play(res, d, cfg, lay):
    n, nb = d.n, d.X.shape[1]
    names = [k for k in eh.ENSEMBLE_EXPERTS if k in res["next"]]
    fr.assert_no_banned_experts(names, cfg, "next-draw vote")
    w, gains, hedge = eh.block_weights(res, d.P, cfg, n, lay["span_start"], lay["val_len"], active_override=res["next_active"])
    sw = sum(w.values())
    ec = cfg["ensemble"]
    for k in names:
        log.info("Next-draw weight | %-8s val gain %+8.3f | hedge %.3f | active now %s | active on validation window %.0f%% "
                 "| weight %.3f", k, gains[k], hedge[k], bool(res["next_active"][k]),
                 100 * float(np.mean(res["active"][k][n - lay["val_len"] - lay["span_start"]:])), w[k])
    if sw <= 0:
        log.warning("Next draw: every expert has weight 0 (none shows validated signal). Uniform forecast; the picks below "
                    "fall back to fixed ball order and carry NO information.")
        p = np.full(nb, N_DRAWN / nb)
        score = np.zeros(nb)
        cons = score
    else:
        p = fr.norm_inc(sum(w[k] * res["next"][k] for k in names) / sw)
        if ec["vote_mode"] == "rank":
            score = sum(w[k] * eh._rank_points(res["next"][k][None, :])[0] for k in names) / sw + 1e-6 * p
        else:
            score = p.copy()
        cons = eh.consensus_scores({k: res["next"][k][None, :] for k in names}, w, score[None, :],
                                   ec.get("consensus_share", 0.25))[0]
    order = np.argsort(-score, kind="stable")
    corder = np.argsort(-cons, kind="stable")
    top2 = corder[:2] if ec.get("top2_mode", "vote") == "consensus" else order[:2]
    return dict(p_play=p, score=score, order=[int(b) + 1 for b in order], cons_order=[int(b) + 1 for b in corder],
                top2=[int(b) + 1 for b in top2], weights=w, names=names)


# --------------------------------------------------------------------------- #
# Aggregation across the games
# --------------------------------------------------------------------------- #
def aggregate(folder: str):
    files = sorted(Path(folder).glob("eval_*.json"), key=lambda f: f.stat().st_mtime)
    latest = {}
    for f in files:
        j = json.loads(f.read_text())
        latest[j["game"]] = j
    if not latest:
        raise SystemExit(f"no eval_*.json in {folder}")
    print("| game | pool | draws tested | null reps | max-T p | Bonferroni p | verdict |")
    print("|---|---|---|---|---|---|---|")
    for g, j in latest.items():
        print(f"| {g} | {j['N']} | {j['eval_draws']} | {j['null_reps']} | {j['p_maxT']:.4f} | {j['p_corrected']:.4f} | "
              f"{j['verdict']} |")
    if len(latest) != 4:
        print(f"\n(note: {len(latest)} of the 4 games found; Bonferroni always uses the configured family size of 4)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--aggregate" in argv:
        i = argv.index("--aggregate")
        return aggregate(argv[i + 1] if i + 1 < len(argv) else "outputs")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    tag = Path(args.csv).stem
    logpath = s2.setup_logging(out_dir, "stage4_" + tag)
    cfg = apply_overrides(get_config(), args)
    s2.T0 = time.time()
    fr.seed_everything(cfg["seed"])
    log.info("Stage 4 (refactor) start | csv=%s | log=%s", args.csv, logpath)
    log.info("config=%s", json.dumps(cfg, default=str))

    nb = detect_n(args.csv, args.n_balls)
    set_pool(nb)
    K, V = eh.topk_big(cfg, nb), core_size(cfg, nb)
    log.info("Chance levels for %d-of-%d (all from N_BALLS): top-5 hits %.4f | top-%d hits %.4f | any ball %.2f%% | "
             "P(>=2 of 5 in a %d-ball core) %.4f | P(both of a pair) %.4f%%", N_DRAWN, nb, eh.chance_hits(5, nb), K,
             eh.chance_hits(K, nb), 100 * N_DRAWN / nb, V, eh.chance_core(V, nb), 100 * chance_pair(nb))

    d = s2.load_data(args.csv, out_dir, args.next_date, args.next_shifted, args.next_holiday_shift)
    n = d.n
    if pd.Timestamp(d.info["next_date"]).date() < pd.Timestamp.now().date():
        log.warning("!!! The draw being predicted (%s) is in the PAST. Add the latest results to %s first (python "
                    "tracker.py add-draw ...). This pick is a re-run of an old draw and the tracker will mark it LATE.",
                    d.info["next_date"], args.csv)
    lay = eh.layout(n, cfg, quick=args.quick)
    log.info("Layout: %d draws | validation pre-roll draws %d..%d | TEST draws %d..%d (%s .. %s) | cadence %d",
             n, lay["span_start"], lay["eval_start"] - 1, lay["eval_start"], n - 1, d.dates[lay["eval_start"]].date(),
             d.dates[n - 1].date(), lay["cad"])

    off = {"tft_army": args.no_tft, "ranker": args.no_ranker, "anchor": args.no_anchor, "mlp": args.no_mlp,
           "booster": args.no_booster, "rulemem": args.no_rulemem or not args.rulemem, "rulemine": args.no_rulemine}
    enabled = [k for k in eh.ENSEMBLE_EXPERTS if not off.get(k, False)]
    fr.assert_no_banned_experts(enabled, cfg, "enabled experts")
    seed = cfg["seed"]

    # ---------------- observed run (also yields the next-draw forecasts)
    log.info("=== Walk-forward evaluation (observed data) ===")
    fb = fr.build_features(d, cfg)
    log.info("Feature columns reaching the models (%d): %s", len(fb.names), fb.names)
    t_pass = time.time()
    obs_run = eh.evaluate_pipeline(d, cfg, lay, seed, enabled, args.device, predict_next=True, fb=fb)
    t_pass = time.time() - t_pass
    obs = obs_run["scalars"]
    ci = eh.confidence_intervals(obs_run["pdraw"], nb, cfg, seed)
    log.info("Observed: %s", {k: round(v, 5) for k, v in obs.items()})

    # ---------------- null test
    reps = int(cfg["eval"]["null_reps"])
    log.info("=== Null test: %d replicates, ball labels permuted within each draw, whole evaluation repeated ===", reps)
    log.info("One full pass took %.0fs -> the null test needs roughly %.1f hours (checkpointed; rerun the same command "
             "after a disconnect). Use --null-reps to trade time for p-value resolution.", t_pass, t_pass * reps / 3600)
    if reps < 79:
        log.warning("With %d replicates the smallest attainable p is %.4f; Bonferroni x%d can never reach alpha. "
                    "Use >= 100 (ideally the default 500) for a real verdict.", reps, 1 / (reps + 1), cfg["eval"]["n_games"])
    workers = args.null_workers
    if workers <= 0:
        try:
            import torch
            gpu = torch.cuda.is_available() and args.device != "cpu"
        except ImportError:
            gpu = False
        workers = 2 if gpu else max(1, (os.cpu_count() or 2) // 2)
    null = eh.null_test(d, cfg, lay, enabled, args.device, reps, seed, obs, ckpt=out_dir / f"null_ckpt_{tag}.json",
                        workers=workers)
    ok, p_corr, reachable = eh.verdict(null, cfg, tag)

    # ---------------- ablation + diagnostics
    abl = {}
    if not args.no_ablation:
        log.info("=== Ablation ===")
        from legacy_features import build_old_features
        abl = eh.ablation(d, cfg, lay, seed, fb, legacy_builder=build_old_features)
    chi = eh.chi_square_uniformity(d)
    jac = eh.jaccard_consecutive(obs_run["pdraw"]["order5"])
    jac_ch = chance_jaccard(nb)
    log.info("Diagnostic | chi-square uniformity p=%.4f | Top-5 Jaccard on consecutive test dates %.3f (random %.3f)",
             chi["p"], jac, jac_ch)

    # expert table (diagnostic)
    res = obs_run["res"]
    expert_tbl = {}
    Xe = obs_run["Xe"]
    for k, p in res["probs"].items():
        pe = p[lay["eval_start"] - lay["span_start"]:]
        sc, pdr = eh.compute_metrics(pe, pe, Xe, cfg)
        wm = np.mean([b["weights"].get(k, 0.0) for b in obs_run["wlog"]])
        ash = float(res["active"][k][lay["eval_start"] - lay["span_start"]:].mean())
        # a silent expert's forecast is uniform, so its "top-5" is just balls 1..5 (stable tie-break): not a measurement
        expert_tbl[k] = dict(mean_weight=float(wm), skill=sc["logloss_skill"], h5=sc["hits_top5"] if ash > 0 else float("nan"),
                             active_share=ash)

    # ---------------- verdict text
    ch = eh.chance_levels(nb, cfg)
    nums = (f"top-5 hits {obs['hits_top5']:.3f} (chance {ch['hits_top5']:.3f}); top-{K} hits {obs['hits_topK']:.3f} "
            f"(chance {ch['hits_topK']:.3f}); core-{V} hit rate {100 * obs['core_hit']:.2f}% (chance {100 * ch['core_hit']:.2f}%); "
            f"log-loss skill {obs['logloss_skill']:+.5f}; max-T p={null['p_maxT']:.4f}, Bonferroni p={p_corr:.4f}")
    verdict_txt = ("SKILL DETECTED" if ok else "NO SIGNIFICANT SKILL vs chance") + " | " + nums
    if not ok and not reachable:
        verdict_txt += (f" | NOTE: only {reps} null replicates, so alpha={cfg['eval']['alpha']} after Bonferroni is unreachable; "
                        f"this is not a real verdict")
    md = eh.markdown_report(args.game or tag, nb, n, lay, obs, ci, null, abl, expert_tbl, chi, jac, jac_ch, verdict_txt, cfg,
                            res["tft_skills"])

    # ---------------- next-draw play + twin + wheel
    nxt = next_play(res, d, cfg, lay)
    final2, final5 = nxt["top2"], nxt["order"][:N_DRAWN]
    wheel_k = args.wheel_size or core_size(cfg, nb)
    wheel_core = nxt["order"][:wheel_k]
    twin = None
    if not args.no_twin:
        fr.assert_no_banned_experts(nxt["names"], cfg, "digital twin inputs")
        fr.assert_no_banned(fb.names, cfg, "digital twin inputs", allow_rate100=False)
        pos_uniform = np.full((N_DRAWN, nb), 1.0 / nb)       # no all-history positional table feeds the twin
        twin = s2.run_twin(d, nxt["p_play"], pos_uniform, args, np.random.default_rng(seed))

    t2, ch2 = obs_run["top2"], 2 * N_DRAWN / nb
    L = [md, "### Top-2 over the walk-forward (diagnostic, not part of the max-T null test)", "",
         f"| Top-2 mode | mean hits in Top-2 (chance {ch2:.3f}) | at least 1 drawn (chance {1 - math.comb(nb - 2, N_DRAWN) / math.comb(nb, N_DRAWN):.3f}) "
         f"| both drawn (chance {chance_pair(nb):.4f}) | draws |", "|---|---|---|---|---|"]
    for mname, r in t2.items():
        lo_, hi_ = eh.wilson(int((r["h"] >= 2).sum()), r["n"])
        L.append(f"| {mname}{' (used for the play)' if mname == cfg['ensemble'].get('top2_mode', 'vote') else ''} | "
                 f"{r['hits']:.3f} | {r['any']:.3f} | {r['both']:.4f} [{lo_:.4f}, {hi_:.4f}] | {r['n']} |")
    L += ["", "With ~300 test draws a 'both drawn' rate is a handful of events: the CI above is what it can and cannot rule out.",
          "", "## Next draw", ""]
    nd = d.dates[n].day_name()
    status = ("validated by the harness" if ok else "NOT validated: no skill vs chance was detected, treat as arbitrary picks")
    L.append(f"Draw {d.info['next_date']} ({nd}). Vote of {', '.join(nxt['names'])} with weights "
             f"{ {k: round(float(v), 3) for k, v in nxt['weights'].items()} }.")
    L.append(f"Top-2 mode: {cfg['ensemble'].get('top2_mode', 'vote')} | vote Top-2 = {nxt['order'][0]}, {nxt['order'][1]} | "
             f"consensus Top-2 = {nxt['cons_order'][0]}, {nxt['cons_order'][1]}")
    L.append(f"TOP 2 = {final2[0]}, {final2[1]} | TOP 5 = {', '.join(map(str, final5))} | WHEEL CORE ({wheel_k}) = "
             f"{', '.join(map(str, wheel_core))} | status: {status}")
    L.append(f"Chance for both top-2 numbers: {100 * chance_pair(nb):.3f}%; at least 2 of the {wheel_k} core numbers: "
             f"{100 * eh.chance_core(wheel_k, nb):.2f}%.")
    if twin:
        L += ["", "### Digital twin calibration (inputs: new-feature ensemble only)", "",
              "| statistic | real | twin p | uniform p |", "|---|---|---|---|"]
        for r in twin["rows"]:
            L.append(f"| {r['statistic']} | {r['real']:.3f} | {r['twin_p']:.3f} | {r['uniform_machine_p']:.3f} |")
    wheel_text = ""
    if not args.no_wheel:
        try:
            import wheel_generator as wg
            ns = argparse.Namespace(ticket=args.wheel_ticket, price=args.wheel_price, pair_price=args.wheel_pair_price,
                                    payout=args.payout, draw=N_DRAWN, size=wheel_k)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                wg.print_wheel(wheel_core, nb, "stage4 refactor vote", d.info["next_date"], ns)
            wheel_text = buf.getvalue()
            L += ["", "```", wheel_text.rstrip(), "```"]
        except (ImportError, SystemExit) as e:
            log.warning("Wheel skipped (%s)", e)
    report = "\n".join(L)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report_{tag}_{stamp}.md").write_text(report, encoding="utf-8")
    if wheel_text:
        (out_dir / f"wheel_{tag}.txt").write_text(wheel_text, encoding="utf-8")
    payload = {"N": nb, "info": d.info, "game": args.game or tag, "observed": obs, "ci": {k: list(v) for k, v in ci.items()},
               "null": null, "p_maxT": null["p_maxT"], "p_corrected": p_corr, "verdict": "SKILL DETECTED" if ok else
               "NO SIGNIFICANT SKILL vs chance", "eval_draws": lay["eval_n"], "null_reps": reps,
               "ablation": {k: {"scalars": v["scalars"], "ci": v["ci"]} for k, v in abl.items()},
               "chi_square": chi, "jaccard_consecutive": jac, "jaccard_chance": jac_ch,
               "weights_by_block": obs_run["wlog"], "tft_skills": res["tft_skills"], "anchor_info": res.get("anchor_info", []),
               "top2_walk_forward": {k: {m: v[m] for m in ("hits", "both", "any", "n")} for k, v in obs_run["top2"].items()},
               "next": {"date": d.info["next_date"], "p_play": nxt["p_play"].tolist(), "final_top2": final2,
                        "final_top5": final5, "vote_top2": nxt["order"][:2], "consensus_top2": nxt["cons_order"][:2],
                        "wheel_core": wheel_core, "wheel_size": wheel_k,
                        "play_models": nxt["names"], "play_weights": [float(nxt["weights"][k]) for k in nxt["names"]]},
               "config": cfg}
    (out_dir / f"stage4_results_{tag}_{stamp}.json").write_text(json.dumps(payload, indent=2, default=float))
    (out_dir / f"eval_{tag}_{stamp}.json").write_text(json.dumps(
        {k: payload[k] for k in ("N", "game", "p_maxT", "p_corrected", "verdict", "eval_draws", "null_reps")}, indent=2))
    if not args.no_tracker:
        import tracker
        tpath = Path(args.tracker) if args.tracker else out_dir / "tracker.csv"
        added = tracker.log_prediction(tpath, args.game or tag, d.info["next_date"], nb, final2, final5, wheel_core,
                                       cfg["ensemble"].get("top2_mode", "vote"), "SKILL" if ok else "no skill")
        log.info("Live tracker: %s for %s %s in %s (score later with: python tracker.py score --csv %s)",
                 "pick LOGGED" if added else "a pick was already logged (first one kept)", args.game or tag,
                 d.info["next_date"], tpath, args.csv)
    print("\n" + report)
    print("\n" + "#" * 78)
    print(f"VERDICT {args.game or tag}: {verdict_txt}")
    print("#" * 78)
    log.info("Saved report/JSON to %s | elapsed %.1f min", out_dir.resolve(), (time.time() - s2.T0) / 60)


if __name__ == "__main__":
    main()