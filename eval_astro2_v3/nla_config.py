"""
nla_config.py: the ONE place for every weight / hyper-parameter, plus chance-level maths.
Every pool-size dependent number is derived from N_BALLS (never hardcoded for 5/90 or 5/39).
"""
from __future__ import annotations

import copy
import math

N_DRAWN = int(__import__("os").environ.get("ASTRO_K", "5"))   # balls per draw (main_beast sets 6 for Afriluck)

CONFIG = {
    "seed": 42,

    # ------------------------------------------------------------------ features
    "features": {
        "windows": [5, 10, 20, 50],          # rate5, rate10, rate20, rate50
        "use_rate100": False,                # only switched on by ablation row (c)
        "dow_window": 8,                     # recent same-weekday draws for rate_dow8
        "gap_cap_mult": 40.0,                # gap_current is capped at gap_cap_mult * (N/5) draws
        "gap_hist": 100,                     # trailing completed gaps used for gap_percentile
        "markov_half_life": 40.0,            # draws
        "markov_alpha": 50.0,                # Laplace / shrinkage strength (in draw-equivalents)
    },
    # Columns / experts that must never reach a model, the vote or the twin.
    "banned_features": ["rate_all", "rate250", "rate500", "frozen_frequency", "frozen_freq",
                        "pos1_rate", "pos5_rate", "rate_same_dow", "ball_id", "last_digit"],
    "banned_rate100": "rate100",             # banned unless features.use_rate100
    "banned_experts": ["logreg", "logistic", "linear", "ridge", "lasso", "frozen", "freq_macro", "freq_micro"],

    # ------------------------------------------------------------------ hazard expert
    "hazard": {
        "window": 10**9,                    # v3: ALL prior draws (expanding). v2 used 1200
        "warmup": 120,
        "ratio_edges": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],   # gap / (N/5)
        "pct_edges": [0.2, 0.4, 0.6, 0.8],   # gap_percentile bins
        "prior": 400.0,                      # shrinkage towards the uniform rate (observations)
    },

    # ------------------------------------------------------------------ ranker (LambdaRank)
    "ranker": {
        "warmup": 120,
        "train_window": 10**9,               # v3: expanding window from the first usable draw (v2: 1500)
        "val_frac": 0.15,
        "min_val": 60,
        "early_stopping": 20,
        "params": dict(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=-1,
                       deterministic=True, force_row_wise=True),
        "tau_grid": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],   # score temperature for p; "inf" = uniform is implicit
    },

    # ------------------------------------------------------------------ MLP (pointwise, per-ball rows)
    "mlp": {
        "warmup": 120,
        "train_window": 10**9,               # v3: expanding window from the first usable draw (v2: 1500)
        "val_frac": 0.15,
        "min_val": 60,
        "hidden": [64, 32],
        "alpha": 1e-3,                       # L2
        "lr": 1e-3,
        "batch": 512,
        "max_iter": 60,
        "patience": 6,
    },

    # ------------------------------------------------------------------ gradient booster (pointwise, binary)
    "booster": {
        "warmup": 120,
        "train_window": 10**9,               # v3: expanding window from the first usable draw (v2: 1500)
        "val_frac": 0.15,
        "min_val": 60,
        "params": dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=-1,
                       deterministic=True, force_row_wise=True),
    },

    # ------------------------------------------------------------------ TFT army
    "tft": {
        "army": 5,
        "hidden": 32,
        "heads": 4,
        "seq_len": 20,
        "warmup": 120,
        "train_window": 10**9,               # v3: expanding window from the first usable draw (v2: 1500)
        "val_frac": 0.15,
        "min_val": 60,
        "dropout": 0.1,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "batch": 128,
        "max_epochs": 40,
        "min_epochs": 15,                    # epoch floor: early stopping cannot trigger before this
        "patience": 6,                       # on validation log-loss
        "list_weight": 5.0,                  # weight of the listwise softmax CE (5 balances its gradient against BCE)
        "l2_head": 1e-3,                     # L2 on the residual head
        "grad_clip": 1.0,
        "skill_margin": 0.0,                 # member is silenced when skill <= this
    },

    # ------------------------------------------------------------------ anchor (two-timescale: long signal under a turnover cap)
    "anchor": {
        "long_windows": [250, 500, 0],       # 0 = all history up to t (as-of t, never future data)
        "long_shrink": 30.0,                 # hits-equivalents shrinking each ball's long ratio to 1
        "fast_windows": [10, 20],
        "fast_shrink": 3.0,
        "fit_len": 600,                      # draws before the block used to choose weights
        "warmup": 150,
        "min_fit": 240,
        "wL_grid": [0.0, 0.02, 0.04, 0.08, 0.15, 0.3],       # logit shift per cross-sectional SD
        "fast_grid": [0.02, 0.05, 0.1, 0.2, 0.4],
        "max_top_jaccard": 0.40,             # HARD anti-stagnation cap: mean Jaccard of top-K of consecutive draws (1.0 = frozen)
        "gate_margin": 0.0,                  # hold-out log-loss skill must exceed this, else the expert is silent
        "always_on": False,                  # MEASUREMENT mode: forecast in every block (gate logged, not enforced; cap relaxed
                                             # to the lowest achievable Jaccard + cap_relax when nothing meets it)
        "cap_relax": 0.05,
    },

    # ------------------------------------------------------------------ rule memory (hedge vote over ~2,300 rules)
    "rulemem": {
        "hedge_window": 300,                 # draws of history behind each rule weight
        "eta": 0.5,                          # hedge learning rate
        "alpha": 0.5,                        # 0 = chance forecast, 1 = raw rule vote (shrinkage for log-loss)
    },

    # ------------------------------------------------------------------ v3 data-mined rule engine (rule_miner.py)
    "rulemine": {
        "alpha": 0.05,                       # family-wise error for calling a rule PROVEN (evidence > rules / alpha)
        "others": {},                        # optional {game: audited csv} for cross-game rules (never the same game)
    },

    # ------------------------------------------------------------------ ensemble
    "ensemble": {
        "multipliers": {"tft_army": 2.0, "ranker": 2.0, "hazard": 1.5, "markov": 1.5, "anchor": 2.0, "mlp": 2.0,
                        "booster": 2.0, "rulemem": 2.0, "rulemine": 2.0},
        "val_len": 200,                      # validation window (draws) that precedes every test block
        "hedge_eta": 0.5,                    # exponent on the validation log-score gain (v2: 2.0 = winner-take-all on noise)
        "vote_mode": "rank",                 # rank | prob
        "min_track_record": 0.5,             # an expert must have been ACTIVE on >= this share of the validation window to
                                             # get any weight (a silent expert scores gain 0 there, which beats every expert
                                             # with a slightly negative gain, so without this rule silence is rewarded)
        "require_positive_gain": False,      # True: experts whose validation log-score gain is <= 0 get weight 0
        "top2_mode": "vote",                 # vote | consensus (Top-2 = balls ranked high by EVERY heavy expert)
        "consensus_share": 0.25,             # an expert is "heavy" when its share of the total vote weight is >= this
    },

    # ------------------------------------------------------------------ evaluation
    "eval": {
        "eval_draws": 0,                     # v3: 0 = score EVERY draw after the warm-up (v2: last 300)
        "full_warmup": 200,                  # draws after the expert warm-up before the first validation window
        "min_eval_draws": 300,
        "retrain_every": 100,
        "null_reps": 500,
        "early_stop_h": 10,                  # stop the null test once this many replicates match/beat the observed result
        "early_stop_min": 20,                # ... but never before this many replicates (0 for h = no early stop)
        "boot_reps": 1000,
        "n_games": 4,                        # Bonferroni family (Main, Noonrush, Vag, Daywa)
        "alpha": 0.05,
        "metrics": ["hits_top5", "hits_topK", "core_hit", "logloss_skill"],
        "freq20_prior": 20.0,
    },

    # core / top-K sizes by pool size (same rule as wheel_generator.default_core_size)
    "core": {"90": 12, "39": 10, "default": 8},
}


def get_config() -> dict:
    return copy.deepcopy(CONFIG)


# --------------------------------------------------------------------------- #
# Chance levels (all from the pool size nb)
# --------------------------------------------------------------------------- #
def core_size(cfg: dict, nb: int) -> int:
    c = cfg["core"]
    return int(c.get(str(int(nb)), c["default"]))


def topk_big(cfg: dict, nb: int) -> int:
    return core_size(cfg, nb)


def chance_hits(k: int, nb: int) -> float:
    """Expected number of drawn balls among k distinct picks."""
    return k * N_DRAWN / nb


def p_at_least(nb: int, core: int, m: int) -> float:
    return sum(math.comb(core, j) * math.comb(nb - core, N_DRAWN - j) for j in range(m, min(core, N_DRAWN) + 1)) \
        / math.comb(nb, N_DRAWN)


def chance_core(core: int, nb: int) -> float:
    """P(>= 2 of the 5 drawn balls fall inside a fixed core of `core` balls)  (the Perm-2 wheel)."""
    return p_at_least(nb, core, 2)


def chance_pair(nb: int) -> float:
    return N_DRAWN * (N_DRAWN - 1) / (nb * (nb - 1))


def uniform_bce(nb: int) -> float:
    """Per-draw summed binary cross-entropy of the uniform forecast p = 5/nb (constant: every draw has exactly 5 ones)."""
    p = N_DRAWN / nb
    return -(N_DRAWN * math.log(p) + (nb - N_DRAWN) * math.log(1 - p))


def chance_jaccard(nb: int) -> float:
    """Expected Jaccard overlap of two independent uniform random 5-subsets of nb balls."""
    tot = 0.0
    for i in range(0, N_DRAWN + 1):
        pr = math.comb(N_DRAWN, i) * math.comb(nb - N_DRAWN, N_DRAWN - i) / math.comb(nb, N_DRAWN)
        tot += pr * i / (2 * N_DRAWN - i)
    return tot
