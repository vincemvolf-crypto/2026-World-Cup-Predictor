"""
model.py

Trains one model per (position group x target stat) using LightGBM.
Position-grouping matters because the relationship between "form" and
"shots" looks very different for a striker vs a center-back.

A simple LightGBM regressor per stat (rather than one big multi-output net)
is used deliberately: it's robust on small/medium tabular data, handles
missing values natively, gives feature importances for sanity-checking,
and is easy to retrain incrementally as World Cup data accumulates.

Usage:
    python model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

import config


FEATURE_COLS_TEMPLATE = (
    [f"form_{s}" for s in config.TARGET_STATS] +
    [f"form_{s}_per90" for s in config.TARGET_STATS] +
    ["form_minutes", "opponent_elo", "team_elo", "elo_diff", "rest_days",
     "possession_share_est", "opp_possession_share_est", "is_set_piece_taker"]
)


def _prep_training_data(table_path: str) -> pd.DataFrame:
    df = pd.read_parquet(table_path)
    # Drop cameo appearances and rows with no form history yet
    df = df[df["min"].fillna(0) >= config.MIN_MINUTES_FOR_SAMPLE]
    df = df.dropna(subset=["form_minutes"])
    return df


def train_position_group(df: pd.DataFrame, group: str):
    """Train one LightGBM model per relevant stat for this position group."""
    sub = df[df["position_group"] == group].copy()
    if sub.empty:
        print(f"  no rows for group {group}, skipping")
        return {}

    models = {}
    for stat in config.RELEVANT_STATS[group]:
        target_col = stat
        feature_cols = [c for c in FEATURE_COLS_TEMPLATE if c in sub.columns]

        data = sub.dropna(subset=[target_col])
        if len(data) < 50:
            print(f"  [{group}] {stat}: only {len(data)} rows, skipping (need >=50)")
            continue

        X = data[feature_cols].fillna(0)
        y = data[target_col]

        # Simple time-based CV to sanity-check, then refit on all data
        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for train_idx, test_idx in tscv.split(X):
            m = lgb.LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
            )
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = m.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))

        final_model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
        )
        final_model.fit(X, y)

        models[stat] = {
            "model": final_model,
            "features": feature_cols,
            "cv_mae": float(np.mean(maes)),
        }
        print(f"  [{group}] {stat}: cv_mae={np.mean(maes):.3f} "
              f"(mean actual={y.mean():.2f}, n={len(data)})")

    return models


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Train primarily on club-season data (much larger sample of "recent
    # form -> next match output" pairs). World Cup data, once enough
    # matches have been played, can be concatenated in for a fine-tune pass.
    df = _prep_training_data(f"{config.PROCESSED_DIR}/club_form_table.parquet")

    wc_path = f"{config.PROCESSED_DIR}/world_cup_table.parquet"
    if os.path.exists(wc_path):
        wc_df = _prep_training_data(wc_path)
        if len(wc_df) > 0:
            df = pd.concat([df, wc_df], ignore_index=True)
            print(f"Added {len(wc_df)} World Cup rows to training data.")

    all_models = {}
    for group in config.POSITION_GROUPS:
        print(f"Training group: {group}")
        all_models[group] = train_position_group(df, group)

    out_path = f"{config.MODEL_DIR}/stat_models.joblib"
    joblib.dump(all_models, out_path)
    print(f"\nSaved all models -> {out_path}")


if __name__ == "__main__":
    main()
