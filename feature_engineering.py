"""
feature_engineering.py

Turns raw per-match stat rows into a model-ready feature table.

For each player, for each match, builds:
  - Rolling averages of each of the 12 target stats over the last N matches
    (this is the "recent form" signal)
  - Per-90-minute normalized versions of those averages
  - Opponent strength (Elo) for that match
  - Context flags: home/neutral venue, days since last match (fatigue/rest)

Output: data/processed/training_table.parquet
  One row per (player, match), with feature columns (X) and the actual
  stat values for that match (y / targets).
"""

import pandas as pd
import numpy as np

import config


def _pivot_long_to_wide(raw: pd.DataFrame) -> pd.DataFrame:
    """
    The raw FBref dump has one row per (player, match, stat_type) with many
    stat-specific columns. Collapse to one row per (player, match) with our
    12 named target columns, plus identifying info (team, opponent, date,
    position, minutes).
    """
    id_cols = ["player", "team", "opponent", "date", "pos", "min", "game"]
    id_cols = [c for c in id_cols if c in raw.columns]

    wide = raw[id_cols].drop_duplicates(subset=[c for c in id_cols if c != "min"])
    wide = wide.drop_duplicates(subset=["player", "date", "team"])

    for stat_name, (stat_type, candidates) in config.TARGET_STATS.items():
        sub = raw[raw["stat_type"] == stat_type]
        if sub.empty:
            print(f"  WARNING: no rows for stat_type '{stat_type}' "
                  f"(needed for '{stat_name}') - setting to NaN.")
            wide[stat_name] = np.nan
            continue

        found_col = next((c for c in candidates if c in sub.columns), None)
        if found_col is None:
            available = sorted(c for c in sub.columns
                                if c not in ("player", "date", "team", "stat_type"))
            print(f"  WARNING: none of {candidates} found for '{stat_name}' "
                  f"(stat_type='{stat_type}'). '{stat_name}' will be 0/NaN. "
                  f"Available columns in '{stat_type}': {available}")
            wide[stat_name] = np.nan
            continue

        merge_cols = [c for c in ["player", "date", "team"] if c in sub.columns]
        wide = wide.merge(
            sub[merge_cols + [found_col]].rename(columns={found_col: stat_name}),
            on=merge_cols, how="left",
        )

    return wide


def _add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, compute rolling mean of each target stat over the
    previous N matches (NOT including the current match - that would be
    target leakage).
    """
    df = df.sort_values(["player", "date"]).reset_index(drop=True)

    stat_cols = list(config.TARGET_STATS.keys())
    for stat in stat_cols:
        df[f"form_{stat}"] = (
            df.groupby("player")[stat]
              .transform(lambda s: s.shift(1).rolling(
                  config.FORM_WINDOW_MATCHES, min_periods=2).mean())
        )

    # Minutes-based per-90 normalization of the form features
    df["form_minutes"] = (
        df.groupby("player")["min"]
          .transform(lambda s: s.shift(1).rolling(
              config.FORM_WINDOW_MATCHES, min_periods=2).mean())
    )
    for stat in stat_cols:
        df[f"form_{stat}_per90"] = (
            df[f"form_{stat}"] / df["form_minutes"].clip(lower=1) * 90
        )

    return df


def _add_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    try:
        elo = pd.read_csv(f"{config.RAW_DIR}/national_elo.csv")
        elo_map = dict(zip(elo["team"], elo["elo"]))
    except FileNotFoundError:
        print("  WARNING: national_elo.csv not found - opponent_elo will be NaN. "
              "Run data_collection.fetch_national_team_elo() or create it manually.")
        elo_map = {}

    df["opponent_elo"] = df["opponent"].map(elo_map)
    df["team_elo"] = df["team"].map(elo_map)
    df["elo_diff"] = df["team_elo"] - df["opponent_elo"]
    return df


def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player", "date"])
    df["rest_days"] = df.groupby("player")["date"].diff().dt.days
    df["rest_days"] = df["rest_days"].fillna(7)  # default assumption
    return df


def _add_position_group(df: pd.DataFrame) -> pd.DataFrame:
    def map_group(pos):
        if not isinstance(pos, str):
            return "MF"
        for group, codes in config.POSITION_GROUPS.items():
            if pos in codes:
                return group
        # fallback: first letter
        if pos.startswith("GK"):
            return "GK"
        if pos.startswith("DF"):
            return "DF"
        if pos.startswith("FW"):
            return "FW"
        return "MF"

    df["position_group"] = df["pos"].apply(map_group)
    return df


def _add_possession_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate each team's likely possession share for the match from the
    Elo gap, using a logistic curve scaled to a realistic ~35-65% range.
    This feeds the model a sense of "will this team dominate the ball" -
    which correlates strongly with passes_attempted, dribbles, crosses
    (higher for the dominant team) vs tackles/clearances (higher for the
    team without the ball).
    """
    df["possession_share_est"] = 50 + 15 * np.tanh(df["elo_diff"].fillna(0) / 200)
    df["opp_possession_share_est"] = 100 - df["possession_share_est"]
    return df


def _add_set_piece_flag(df: pd.DataFrame) -> pd.DataFrame:
    try:
        sp = pd.read_csv(config.SET_PIECE_TAKERS_FILE)
        sp_players = set(sp["player"].dropna())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        sp_players = set()
    df["is_set_piece_taker"] = df["player"].isin(sp_players).astype(int)
    return df


def _blend_international_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    If international match data has been collected (see
    data_collection.fetch_international_matches), blend each player's
    international-form rolling averages into their club-form rolling
    averages. International samples are small, so this is a weighted blend
    rather than a replacement - see config.INTERNATIONAL_FORM_WEIGHT.

    Players with no international data are left on club form only.
    """
    import os
    if not os.path.exists(config.INTERNATIONAL_TABLE):
        print("  no international_form_table.parquet found - skipping "
              "international blend (club form only). See README for how "
              "to add this.")
        return df

    intl = pd.read_parquet(config.INTERNATIONAL_TABLE)
    intl_latest = (intl.sort_values("date")
                        .groupby("player").tail(1))

    form_cols = [c for c in df.columns if c.startswith("form_")]
    intl_cols = {c: f"intl_{c}" for c in form_cols if c in intl_latest.columns}
    intl_latest = intl_latest.rename(columns=intl_cols)
    intl_latest = intl_latest[["player"] + list(intl_cols.values())]

    df = df.merge(intl_latest, on="player", how="left")

    w = config.INTERNATIONAL_FORM_WEIGHT
    for col in form_cols:
        intl_col = f"intl_{col}"
        if intl_col in df.columns:
            has_intl = df[intl_col].notna()
            df.loc[has_intl, col] = (
                (1 - w) * df.loc[has_intl, col].fillna(0) + w * df.loc[has_intl, intl_col]
            )
            df = df.drop(columns=[intl_col])

    return df


def build_training_table(raw_path: str, output_name: str) -> pd.DataFrame:
    print(f"Loading {raw_path} ...")
    raw = pd.read_parquet(raw_path)

    print("  pivoting long -> wide ...")
    wide = _pivot_long_to_wide(raw)

    print("  filtering low-minute cameos ...")
    wide = wide[wide["min"].fillna(0) >= 0]  # keep all for form calc; filter later for training

    print("  adding rolling form features ...")
    wide = _add_rolling_form(wide)

    print("  adding opponent strength (Elo) ...")
    wide = _add_opponent_strength(wide)

    print("  adding rest days ...")
    wide = _add_rest_days(wide)

    print("  adding position groups ...")
    wide = _add_position_group(wide)

    print("  adding possession share estimate ...")
    wide = _add_possession_estimate(wide)

    print("  flagging set-piece takers ...")
    wide = _add_set_piece_flag(wide)

    print("  blending international form (if available) ...")
    wide = _blend_international_form(wide)

    out_path = f"{config.PROCESSED_DIR}/{out
