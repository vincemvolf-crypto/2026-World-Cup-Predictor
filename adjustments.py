"""
adjustments.py

Shared post-processing helpers used by predict.py and
export_predictions.py to apply the accuracy features on top of the raw
model output:

  - possession_share_est: feature, computed from Elo gap (same formula
    as feature_engineering, kept in sync here)
  - expected-minutes scaling: scales counting stats by expected_minutes/90
  - venue (altitude/heat) factor: scales work-rate stats for the host city
  - set-piece taker bonus: bumps crosses/shots_assisted for designated takers
"""

import os
import numpy as np
import pandas as pd

import config


def possession_share(team_elo, opponent_elo):
    elo_diff = (team_elo or 0) - (opponent_elo or 0)
    poss = 50 + 15 * np.tanh(elo_diff / 200)
    return poss, 100 - poss


def load_expected_minutes():
    path = f"{config.PROCESSED_DIR}/expected_minutes.parquet"
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["player"], df["expected_minutes"]))


def load_set_piece_takers():
    try:
        sp = pd.read_csv(config.SET_PIECE_TAKERS_FILE)
        return set(sp["player"].dropna())
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return set()


def load_venue_factors():
    try:
        return pd.read_csv(config.VENUE_FACTORS_FILE).set_index("venue")
    except FileNotFoundError:
        return None


def get_venue_factor(venue_factors, venue_name):
    """Returns (altitude_factor, heat_factor), defaulting to (1.0, 1.0)
    if the venue isn't found / no venue factors loaded."""
    if venue_factors is None or venue_name is None:
        return 1.0, 1.0
    # exact match, then loose contains-match (FBref/fixture venue names
    # don't always match exactly)
    if venue_name in venue_factors.index:
        row = venue_factors.loc[venue_name]
        return float(row["altitude_factor"]), float(row["heat_factor"])
    for idx in venue_factors.index:
        if idx.lower() in venue_name.lower() or venue_name.lower() in idx.lower():
            row = venue_factors.loc[idx]
            return float(row["altitude_factor"]), float(row["heat_factor"])
    return 1.0, 1.0


def apply_adjustments(preds: dict, position_group: str, player_name: str,
                       expected_minutes, venue_factor_tuple,
                       set_piece_players: set) -> dict:
    """Apply expected-minutes scaling, venue factor, and set-piece bonus
    to a raw prediction dict (in place + returned)."""
    altitude_factor, heat_factor = venue_factor_tuple
    venue_factor = altitude_factor * heat_factor

    minutes_scale = 1.0
    if expected_minutes is not None:
        minutes_scale = max(0.0, min(1.15, expected_minutes / 90.0))

    for stat in config.MINUTES_SCALED_STATS:
        if stat not in preds:
            continue
        value = preds[stat] * minutes_scale
        if stat in config.VENUE_SENSITIVE_STATS:
            value *= venue_factor
        preds[stat] = round(value, 2)

    # goals/assists scaled more gently by minutes (model already accounts
    # for some of this via form features, so apply a softer factor)
    soft_scale = 0.5 + 0.5 * minutes_scale
    for stat in ["goals", "assists"]:
        if stat in preds:
            preds[stat] = round(preds[stat] * soft_scale, 3)

    if player_name in set_piece_players:
        for stat in config.SET_PIECE_STATS:
            if stat in preds:
                preds[stat] = round(preds[stat] * config.SET_PIECE_BONUS, 2)

    return preds
