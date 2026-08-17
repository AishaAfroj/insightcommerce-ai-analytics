"""Unsupervised transaction anomaly detection with human-readable reason codes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyArtifact:
    """Scored public transaction fields and anomaly summary statistics."""

    scored_frame: pd.DataFrame
    anomalies: pd.DataFrame
    metrics: dict[str, float]


def detect_anomalies(frame: pd.DataFrame, contamination: float = 0.01) -> AnomalyArtifact:
    """Fit Isolation Forest and flag the most unusual price/quantity/value combinations."""

    working = frame.copy()
    product_median = working.groupby("product")["order_value"].transform("median").clip(lower=0.01)
    working["relative_to_product_median"] = working["order_value"] / product_median
    features = pd.DataFrame(
        {
            "log_price": np.log1p(working["price"]),
            "quantity": working["quantity"].astype(float),
            "log_order_value": np.log1p(working["order_value"]),
            "relative_to_product_median": working["relative_to_product_median"],
        }
    )
    model = IsolationForest(
        n_estimators=160,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = model.fit_predict(features)
    working["anomaly_score"] = -model.decision_function(features)
    working["is_anomaly"] = labels.eq(-1) if isinstance(labels, pd.Series) else labels == -1

    value_q99 = float(working["order_value"].quantile(0.99))
    price_q99 = float(working["price"].quantile(0.99))
    relative_q99 = float(working["relative_to_product_median"].quantile(0.99))

    def reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["order_value"] >= value_q99:
            reasons.append("exceptionally high order value")
        if row["price"] >= price_q99:
            reasons.append("premium unit price")
        if row["relative_to_product_median"] >= relative_q99:
            reasons.append("well above the product's typical order")
        if row["quantity"] >= 9:
            reasons.append("maximum observed quantity")
        return "; ".join(reasons) or "unusual multivariate price-and-quantity combination"

    working["anomaly_reason"] = working.apply(reason, axis=1)
    public_columns = [
        "order_id",
        "date",
        "country",
        "product_category",
        "product",
        "price",
        "quantity",
        "order_value",
        "relative_to_product_median",
        "anomaly_score",
        "is_anomaly",
        "anomaly_reason",
    ]
    scored = working[public_columns].sort_values("anomaly_score", ascending=False)
    anomalies = scored.loc[scored["is_anomaly"]].copy()
    metrics = {
        "rows_scored": float(len(scored)),
        "anomalies": float(len(anomalies)),
        "anomaly_rate": float(len(anomalies) / len(scored)),
        "contamination": float(contamination),
        "highest_anomaly_score": float(anomalies["anomaly_score"].max()),
    }
    return AnomalyArtifact(scored_frame=scored, anomalies=anomalies, metrics=metrics)

