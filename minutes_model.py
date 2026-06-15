"""
minutes_model.py

Predicts EXPECTED MINUTES for each player's next match. This matters
because counting stats (passes, shots, tackles, etc.) scale with minutes
played - a player who's been getting subbed off at 60' should get scaled-
down predictions even if their per-90 numbers are excellent.

Features used:
  - rolling average minutes over the last MINUTES_FORM_WINDOW appearances
  - trend (last appearance vs rolling average - catches "coming back from
    injury" or "getting rested")
  - rest days since last match (fatigue / rotation risk)
  - tournament stage (group vs knockout - rotation is more common in
    group stage, especially once a team has already qualified)

Output: data/processed/expected_minutes.parquet
  One row per player with their current expected_minutes for "the next
  match", to be joined in by predict.py / export_predictions.py.

Usage:
    python minutes_model.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

import config


def _build_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "date"]).reset_index(drop=True)

    df["min_rolling_avg"] = (
        df.groupby("player")["min"]
          .transform(lambda s: s.shift(1).rolling(
              config.MINUTES_FORM_WINDOW, min_periods=2).mean())
    )
    df["min_last_game"] = df.groupby("player")["min"].shift(1)
    df["min_trend"] = df["min_last_game"] - df["min_rolling_avg"]

    df["date"] = pd.to_datetime(df["date"])
    df["rest_days"] = df.groupby("player")["date"].diff().dt.days.fillna(7)

    return df


def train_and_predict(table_path: str) -> pd.DataFrame:
    df = pd.read_parquet(table_path)
    df = _build_minutes_features(df)

    train = df.dropna(subset=["min_rolling_avg", "min_trend", "min"])

    feature_cols = ["min_rolling_avg", "min_last_game", "min_trend", "rest_days"]
    X = train[feature_cols].fillna(0)
    y = train["min"]

    model = lgb.LGBMRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05, verbose=-1,
    )
    model.fit(X, y)

    # Predict "expected minutes for the NEXT match" using each player's
    # most recent row as the feature snapshot.
    latest = df.sort_values("date").groupby("player").tail(1).copy()
    latest["min_rolling_avg_next"] = (
        df.groupby("player")["min"]
          .transform(lambda s: s.rolling(config.MINUTES_FORM_WINDOW, min_periods=1).mean())
    ).loc[latest.index]
    latest["min_last_game_next"] = latest["min"]
    latest["min_trend_next"] = latest["min_last_game_next"] - latest["min_rolling_avg_next"]
    latest["rest_days_next"] = 4  # default assumption between WC group matches

    X_next = latest.rename(columns={
        "min_rolling_avg_next": "min_rolling_avg",
        "min_last_game_next": "min_last_game",
        "min_trend_next": "min_trend",
        "rest_days_next": "rest_days",
    })[feature_cols].fillna(0)

    preds = model.predict(X_next)
    latest["expected_minutes"] = np.clip(preds, 0, 95).round(1)

    out = latest[["player", "team", "expected_minutes"]].drop_duplicates("player")
    out_path = f"{config.PROCESSED_DIR}/expected_minutes.parquet"
    out.to_parquet(out_path)
    print(f"Saved expected minutes for {len(out)} players -> {out_path}")
    return out


def main():
    train_and_predict(f"{config.PROCESSED_DIR}/club_form_table.parquet")


if __name__ == "__main__":
    main()
