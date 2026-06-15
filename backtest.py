"""
backtest.py

Walk-forward validation: for a cutoff date, train only on matches before
that date, then predict every match after it using ONLY the form features
that would have been available the day before (no leakage), and compare
predictions against what actually happened.

Reports, per (position group, stat):
  - model MAE / RMSE
  - a "naive baseline" MAE (predicting the player's rolling-average form
    directly, with no model at all) - if the model doesn't beat this,
    it isn't adding value
  - bias (mean predicted - mean actual), to catch systematic over/under-
    prediction
  - correlation between predicted and actual (rank-order usefulness,
    important for prop betting where you mostly care about "who's
    likely to go over")

Usage:
    python backtest.py --cutoff 2026-03-01
    python backtest.py --cutoff 2026-03-01 --table data/processed/club_form_table.parquet
"""

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config


def run_backtest(table_path: str, cutoff: str):
    df = pd.read_parquet(table_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["min"].fillna(0) >= config.MIN_MINUTES_FOR_SAMPLE]
    df = df.dropna(subset=["form_minutes"])

    cutoff = pd.Timestamp(cutoff)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    print(f"Train: {len(train):,} rows before {cutoff.date()}")
    print(f"Test:  {len(test):,} rows on/after {cutoff.date()}\n")

    if len(test) == 0:
        print("No test rows after cutoff - pick an earlier date.")
        return None

    feature_cols = (
        [f"form_{s}" for s in config.TARGET_STATS] +
        [f"form_{s}_per90" for s in config.TARGET_STATS] +
        ["form_minutes", "opponent_elo", "team_elo", "elo_diff", "rest_days"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    results = []

    for group in config.POSITION_GROUPS:
        train_g = train[train["position_group"] == group]
        test_g = test[test["position_group"] == group]
        if train_g.empty or test_g.empty:
            continue

        for stat in config.RELEVANT_STATS[group]:
            train_d = train_g.dropna(subset=[stat])
            test_d = test_g.dropna(subset=[stat])
            if len(train_d) < 50 or len(test_d) < 5:
                continue

            X_train = train_d[feature_cols].fillna(0)
            y_train = train_d[stat]
            X_test = test_d[feature_cols].fillna(0)
            y_test = test_d[stat]

            model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)

            # Naive baseline: predict = the player's own rolling average
            # (i.e. "form_<stat>") with no model at all
            baseline = test_d[f"form_{stat}"].fillna(y_train.mean())

            corr = np.corrcoef(preds, y_test)[0, 1] if y_test.std() > 0 else np.nan

            results.append({
                "group": group,
                "stat": stat,
                "n_test": len(test_d),
                "actual_mean": round(y_test.mean(), 2),
                "model_mae": round(mean_absolute_error(y_test, preds), 3),
                "baseline_mae": round(mean_absolute_error(y_test, baseline), 3),
                "model_rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 3),
                "bias": round((preds.mean() - y_test.mean()), 3),
                "corr_pred_actual": round(corr, 3) if not np.isnan(corr) else None,
            })

    report = pd.DataFrame(results)
    report["model_better_than_baseline"] = report["model_mae"] < report["baseline_mae"]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=f"{config.PROCESSED_DIR}/club_form_table.parquet")
    parser.add_argument("--cutoff", required=True,
                         help="YYYY-MM-DD - test on matches on/after this date")
    parser.add_argument("--out", default="backtest_report.csv")
    args = parser.parse_args()

    report = run_backtest(args.table, args.cutoff)
    if report is None:
        return

    pd.set_option("display.width", 160)
    print(report.to_string(index=False))

    n_better = report["model_better_than_baseline"].sum()
    print(f"\nModel beats the naive 'use rolling average' baseline on "
          f"{n_better}/{len(report)} stat/position combos.")
    print("If a stat is NOT beating baseline, the model isn't learning "
          "anything useful from opponent strength / rest days for that "
          "stat - either drop it back to the rolling average, or it needs "
          "more/better features.")

    report.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
