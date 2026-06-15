"""
export_predictions.py

Runs predict.py logic for every upcoming fixture (from wc2026_fixtures.csv)
and bundles the results into a single predictions.json that the static
dashboard (site/index.html) fetches.

If lineups.csv doesn't have confirmed starters for a team yet, falls back
to that team's 11 most-recently-most-minutes players from the form table
as a "projected XI" (clearly marked as such in the output).

Usage:
    python export_predictions.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import joblib

import config
import adjustments
from predict import predict_player, _latest_form_row


def _projected_xi(form_table: pd.DataFrame, team: str, n=11):
    """Fallback XI: each player's most recent appearance for `team`,
    ranked by recent average minutes, top N."""
    team_rows = form_table[form_table["team"] == team]
    latest = (team_rows.sort_values("date")
                        .groupby("player").tail(1))
    latest = latest.sort_values("form_minutes", ascending=False).head(n)
    return latest[["player", "position_group"]].rename(
        columns={"position_group": "position"})


def build_match_predictions(home, away, elo_map, form_table, models, lineups,
                             expected_minutes_map, set_piece_players,
                             venue_factor_tuple):
    home_elo = elo_map.get(home)
    away_elo = elo_map.get(away)

    players_out = []
    lineup_is_projected = {}

    for team, opponent, team_elo, opp_elo in [
        (home, away, home_elo, away_elo),
        (away, home, away_elo, home_elo),
    ]:
        team_lineup = lineups[(lineups["team"] == team) & (lineups["is_starter"] == 1)]
        projected = False
        if team_lineup.empty:
            team_lineup = _projected_xi(form_table, team)
            projected = True
        lineup_is_projected[team] = projected

        for _, p in team_lineup.iterrows():
            position_group = p["position"]
            form_row = _latest_form_row(form_table, p["player"])
            if form_row is None:
                continue
            rest_days = form_row.get("rest_days", 7)
            expected_minutes = expected_minutes_map.get(p["player"])

            preds = predict_player(
                models, form_row, position_group, opp_elo, team_elo, rest_days,
                player_name=p["player"], expected_minutes=expected_minutes,
                venue_factor_tuple=venue_factor_tuple,
                set_piece_players=set_piece_players,
            )
            if preds is None:
                continue

            row = {"team": team, "opponent": opponent,
                   "player": p["player"], "position": position_group}
            row.update(preds)
            players_out.append(row)

    return players_out, lineup_is_projected, home_elo, away_elo


def main():
    models = joblib.load(f"{config.MODEL_DIR}/stat_models.joblib")
    form_table = pd.read_parquet(f"{config.PROCESSED_DIR}/club_form_table.parquet")

    elo = pd.read_csv(f"{config.RAW_DIR}/national_elo.csv")
    elo_map = dict(zip(elo["team"], elo["elo"]))

    lineups_path = f"{config.PROCESSED_DIR}/lineups.csv"
    if os.path.exists(lineups_path):
        lineups = pd.read_csv(lineups_path)
    else:
        lineups = pd.DataFrame(columns=["team", "player", "position", "is_starter"])

    fixtures = pd.read_csv(f"{config.RAW_DIR}/wc2026_fixtures.csv")

    expected_minutes_map = adjustments.load_expected_minutes()
    set_piece_players = adjustments.load_set_piece_takers()
    venue_factors = adjustments.load_venue_factors()

    # Only upcoming matches (no final score yet)
    score_col = "score.ft" if "score.ft" in fixtures.columns else None
    if score_col:
        upcoming = fixtures[fixtures[score_col].isna()]
    else:
        upcoming = fixtures

    matches_out = []
    for _, fx in upcoming.iterrows():
        home, away = fx.get("team1"), fx.get("team2")
        if not home or not away:
            continue
        venue_name = fx.get("ground")
        venue_factor_tuple = adjustments.get_venue_factor(venue_factors, venue_name)
        print(f"Predicting {home} vs {away} ...")
        try:
            players, projected_flags, home_elo, away_elo = build_match_predictions(
                home, away, elo_map, form_table, models, lineups,
                expected_minutes_map, set_piece_players, venue_factor_tuple)
        except Exception as e:
            print(f"  skipped ({e})")
            continue

        if not players:
            continue

        matches_out.append({
            "id": f"{home}_vs_{away}".replace(" ", "_"),
            "home": home,
            "away": away,
            "date": fx.get("date"),
            "venue": fx.get("ground"),
            "group": fx.get("group"),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "lineup_projected": projected_flags,
            "players": players,
        })

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stat_columns": list(config.TARGET_STATS.keys()) + ["fantasy_score"],
        "matches": matches_out,
    }

    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    out_path = f"{config.PREDICTIONS_DIR}/predictions.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)
    print(f"Saved {len(matches_out)} matches -> {out_path}")

    # Also copy to site/ so it's served alongside the dashboard
    os.makedirs("site", exist_ok=True)
    with open("site/predictions.json", "w") as f:
        json.dump(bundle, f, indent=2, default=str)


if __name__ == "__main__":
    main()
