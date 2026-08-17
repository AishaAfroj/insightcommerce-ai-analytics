# Predictive and anomaly model card

## Daily demand model

- Task: estimate daily order count from known-in-advance calendar features.
- Model: Random Forest regressor, fixed random seed 42.
- Features: month, quarter, weekday, day of month, annual sine/cosine, weekend,
  and Q4 indicator.
- Evaluation: every seventh calendar day is held out, producing 53 test days.
- Current result: MAE 12.4 orders, RMSE 18.2 orders, R² 0.995.
- Revenue display: predicted orders × historical median order value; it is an
  indicative conversion, not a second revenue model.

Limitations: the source has one synthetic year and an intentionally strong Q4
pattern. This is an academic demonstration and not a causal or production forecast.

## Transaction anomaly detector

- Task: rank unusual price, quantity, order value, and product-relative order value.
- Model: Isolation Forest, fixed random seed 42, nominal contamination 1%.
- Output: anomaly score, flag, and transparent reason code.

Limitations: anomalies are unusual observations, not confirmed fraud, error, or
misconduct. Tied scores can make the observed flagged rate differ slightly from
the nominal contamination setting.

