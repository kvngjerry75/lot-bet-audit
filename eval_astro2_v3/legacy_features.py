"""
legacy_features.py: the OLD feature set (stage 2), used ONLY as the reference row (a) of the ablation.
It contains rate_all, rate_same_dow, ball_id, ... on purpose. It never reaches the vote, the twin or the final picks.
"""
from __future__ import annotations

import stage2_powerhouse as s2


def build_old_features(d):
    """Returns (F (n+1, nb, 17), names). stage2 reads its module-level N_BALLS / BASE, set by stage4.set_pool()."""
    F = s2.build_ball_features(d)
    return F, list(s2.FEATURE_NAMES)
