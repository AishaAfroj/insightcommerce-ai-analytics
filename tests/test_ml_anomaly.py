"""Tests for predictive analytics and anomaly detection."""

import pandas as pd

from insightcommerce.anomaly import detect_anomalies
from insightcommerce.data import load_processed_dataset
from insightcommerce.ml import predict_daily_demand, train_demand_model


def test_demand_model_produces_finite_metrics_and_prediction() -> None:
    frame = load_processed_dataset()
    artifact = train_demand_model(frame)
    assert artifact.metrics["test_days"] >= 50
    assert pd.notna(artifact.metrics["mae_orders"])
    prediction = predict_daily_demand(artifact, pd.Timestamp("2026-11-15").date())
    assert prediction["predicted_orders"] > 0
    assert prediction["indicative_revenue"] > 0


def test_anomaly_detector_flags_requested_fraction_without_pii() -> None:
    frame = load_processed_dataset().head(10_000)
    artifact = detect_anomalies(frame, contamination=0.02)
    assert 150 <= len(artifact.anomalies) <= 250
    assert "anomaly_reason" in artifact.anomalies.columns
    assert "customer_email" not in artifact.scored_frame.columns

