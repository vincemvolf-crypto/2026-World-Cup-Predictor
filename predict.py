"""
predict.py

Generates the 12-stat prediction table for every starting player in a
match, plus a fantasy score, and rolls it up into a match summary.

Usage:
    python predict.py --home "Brazil" --away "France" --neutral

You'll need a lineups file at data/processed/lineups.csv with columns:
    team, player, position, is_starter

(Until ~75 min before kickoff, official lineups aren't out - use your best
guess at the XI based on recent matches; rerun once confirmed lineups drop.)
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

import config
import adjustments


def _latest_form_row(player_table: pd.DataFrame, player_name: str):
    """Most recent row of feature data we have for this player (their
    current 'form' snapshot going into the next match)."""
    rows = player_table[player_table["player"] == player_name]
    if rows.empty:
        return None
    return rows.sort_values("date").iloc[-1]


def _fantasy_score(row: dict, position_group: str) -> float:
    weights = config.GK_FANTASY_WEIGHTS if position_group == "GK" else config.FANTASY_WEIGHTS
    score = 0.0
    for stat, w in weights.items():
        score += w * row.get(stat, 0.0)
    return round(score, 2)


def predict_player(models, player_form_row, position_group, opponent_elo, team_elo,
                    rest_days, player_name=None, expected_minutes=None,
                    venue_factor_tuple=(1.0, 1.0), set_piece_players=None):
    if player_form_row is None:
        return None

    feature_row = {}
    for col in [f"form_{s}" for s in config.TARGET_STATS]:
        feature_row[col] = player_form_row.get(col, 0.0)
    for col in [f"form_{s}_per90" for s in config.TARGET_STATS]:
        feature_row[col] = player_form_row.get(col, 0.0)
    feature_row["form_minutes"] = player_form_row.get("form_minutes", 70.0)
    feature_row["opponent_elo"] = opponent_elo
    feature_row["team_elo"] = team_elo
    feature_row["elo_diff"] = (team_elo or 0) - (opponent_elo or 0)
    feature_row["rest_days"] = rest_days

    poss, opp_poss = adjustments.possession_share(team_elo, opponent_elo)
    feature_row["possession_share_est"] = poss
    feature_row["opp_possession_share_est"] = opp_poss
    feature_row["is_set_piece_taker"] = int(
        player_name in (set_piece_players or set())
    )

    X = pd.DataFrame([feature_row]).fillna(0)

    preds = {stat: 0.0 for stat in config.TARGET_STATS}
    group_models = models.get(position_group, {})
    for stat, bundle in group_models.items():
        feats = bundle["features"]
        x_row = X[feats] if all(f in X.columns for f in feats) else X.reindex(columns=feats, fill_value=0)
        pred = bundle["model"].predict(x_row)[0]
        preds[stat] = max(0.0, round(float(pred), 2))

    # zero out stats not relevant to this position group
    for stat in preds:
        if stat not in config.RELEVANT_STATS.get(position_group, []):
            preds[stat] = 0.0

    preds["fantasy_score"] = _fantasy_score(preds, position_group)

    preds = adjustments.apply_adjustments(
        preds, position_group, player_name,
        expected_minutes, venue_factor_tuple, set_piece_players,
    )
    # fantasy score recomputed after adjustments so it reflects scaled stats
    preds["fantasy_score"] = _fantasy_score(preds, position_group)

    return preds


def predict_match(home_team, away_team, neutral, lineups_path, elo_path,
                   form_table_path, models_path, venue=None):
    models = joblib.load(models_path)
    lineups = pd.read_csv(lineups_path)
    elo = pd.read_csv(elo_path)
    elo_map = dict(zip(elo["team"], elo["elo"]))
    form_table = pd.read_parquet(form_table_path)

    expected_minutes_map = adjustments.load_expected_minutes()
    set_piece_players = adjustments.load_set_piece_takers()
    venue_factors = adjustments.load_venue_factors()
    venue_factor_tuple = adjustments.get_venue_factor(venue_factors, venue)

    home_elo = elo_map.get(home_team, np.nan)
    away_elo = elo_map.get(away_team, np.nan)

    results = []
    for team, opponent, team_elo, opp_elo in [
        (home_team, away_team, home_elo, away_elo),
        (away_team, home_team, away_elo, home_elo),
    ]:
        team_lineup = lineups[(lineups["team"] == team) & (lineups["is_starter"] == 1)]
        for _, p in team_lineup.iterrows():
            position_group = p["position"]
            form_row = _latest_form_row(form_table, p["player"])
            rest_days = form_row["rest_days"] if form_row is not None else 7
            expected_minutes = expected_minutes_map.get(p["player"])

            preds = predict_player(
                models, form_row, position_group, opp_elo, team_elo, rest_days,
                player_name=p["player"], expected_minutes=expected_minutes,
                venue_factor_tuple=venue_factor_tuple,
                set_piece_players=set_piece_players,
            )
            if preds is None:
                print(f"  WARNING: no form data for {p['player']} - skipping")
                continue

            row = {"team": team, "opponent": opponent, "player": p["player"],
                   "position": position_group}
            row.update(preds)
            results.append(row)

    out_df = pd.DataFrame(results)

    cols = (["team", "opponent", "player", "position"] +
            list(config.TARGET_STATS.keys()) + ["fantasy_score"])
    out_df = out_df[[c for c in cols if c in out_df.columns]]

    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    out_name = f"{home_team}_vs_{away_team}".replace(" ", "_")
    out_path = f"{config.PREDICTIONS_DIR}/{out_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved player predictions -> {out_path}")

    # Match-level summary: sum of team totals for each stat
    summary = out_df.groupby("team")[list(config.TARGET_STATS.keys())].sum().reset_index()
    summary_path = f"{config.PREDICTIONS_DIR}/{out_name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved match summary -> {summary_path}")

    return out_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--lineups", default=f"{config.PROCESSED_DIR}/lineups.csv")
    parser.add_argument("--elo", default=f"{config.RAW_DIR}/national_elo.csv")
    parser.add_argument("--form-table", default=f"{config.PROCESSED_DIR}/club_form_table.parquet")
    parser.add_argument("--models", default=f"{config.MODEL_DIR}/stat_models.joblib")
    parser.add_argument("--venue", default=None,
                         help="Host city/venue name, for altitude/heat adjustment")
    args = parser.parse_args()

    predict_match(args.home, args.away, args.neutral, args.lineups,
                   args.elo, args.form_table, args.models, venue=args.venue)


if __name__ == "__main__":
    main()
