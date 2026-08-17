"""Tests for source validation and deterministic feature engineering."""

import numpy as np

from insightcommerce.data import clean_and_enrich, load_raw_dataset
from insightcommerce.schema import RAW_COLUMNS


def test_verified_dataset_schema_and_shape() -> None:
    frame = load_raw_dataset()
    assert tuple(frame.columns) == RAW_COLUMNS
    assert frame.shape == (108_300, 10)
    assert frame.isna().sum().sum() == 0
    assert frame["order_id"].is_unique


def test_enrichment_preserves_financial_identity() -> None:
    clean = clean_and_enrich(load_raw_dataset().head(2_000))
    assert np.isclose(clean["order_value"], clean["price"] * clean["quantity"]).all()
    assert clean["product_category"].nunique() >= 8
    assert set(clean["quarter"].unique()).issubset({1, 2, 3, 4})
