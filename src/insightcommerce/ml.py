"""Calendar-based daily demand prediction for the capstone's predictive feature."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = (
    "month",
    "quarter",
    "day_of_week",
    "day_of_month",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "is_q4",
)


@dataclass
class DemandModelArtifact:
    """Trained model, held-out metrics, and daily evaluation frame."""

    model: RandomForestRegressor
    metrics: dict[str, float]
    daily_frame: pd.DataFrame
    feature_importance: pd.DataFrame
    median_order_value: float


def calendar_features(values: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Convert dates to non-sensitive, known-in-advance forecasting features."""

    dates = pd.DatetimeIndex(pd.to_datetime(values))
    day_of_year = dates.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "month": dates.month,
            "quarter": dates.quarter,
            "day_of_week": dates.dayofweek,
            "day_of_month": dates.day,
            "day_of_year_sin": np.sin(2 * math.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * math.pi * day_of_year / 365.25),
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "is_q4": (dates.quarter == 4).astype(int),
        }
    )


def build_daily_demand_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction rows to daily order volume and revenue."""

    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            orders=("order_id", "count"),
            revenue=("order_value", "sum"),
            units=("quantity", "sum"),
            average_order_value=("order_value", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    features = calendar_features(daily["date"])
    return pd.concat([daily, features], axis=1)


def train_demand_model(frame: pd.DataFrame) -> DemandModelArtifact:
    """Train and evaluate a calendar model on a deterministic weekly holdout."""

    daily = build_daily_demand_frame(frame)
    holdout = np.arange(len(daily)) % 7 == 0
    train = daily.loc[~holdout]
    test = daily.loc[holdout]
    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[list(FEATURE_COLUMNS)], train["orders"])
    predictions = model.predict(test[list(FEATURE_COLUMNS)])
    daily["predicted_orders"] = model.predict(daily[list(FEATURE_COLUMNS)])
    metrics = {
        "mae_orders": float(mean_absolute_error(test["orders"], predictions)),
        "rmse_orders": float(mean_squared_error(test["orders"], predictions) ** 0.5),
        "r2": float(r2_score(test["orders"], predictions)),
        "test_days": float(len(test)),
    }
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return DemandModelArtifact(
        model=model,
        metrics=metrics,
        daily_frame=daily,
        feature_importance=importance,
        median_order_value=float(frame["order_value"].median()),
    )


def predict_daily_demand(artifact: DemandModelArtifact, target_date: date) -> dict[str, float | str]:
    """Predict daily order count and translate it to indicative revenue."""

    features = calendar_features(pd.DatetimeIndex([pd.Timestamp(target_date)]))
    predicted_orders = max(0.0, float(artifact.model.predict(features[list(FEATURE_COLUMNS)])[0]))
    return {
        "date": target_date.isoformat(),
        "predicted_orders": round(predicted_orders),
        "indicative_revenue": round(predicted_orders * artifact.median_order_value, 2),
        "median_order_value_assumption": round(artifact.median_order_value, 2),
    }

