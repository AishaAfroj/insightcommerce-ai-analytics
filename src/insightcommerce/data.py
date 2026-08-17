"""Raw CSV validation, cleaning, feature derivation, and Parquet preparation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config import PARQUET_PATH, PROCESSED_DIR, QUALITY_PATH, RAW_CSV_PATH, SCHEMA_PATH
from .schema import RAW_COLUMNS, schema_catalog


class DatasetValidationError(ValueError):
    """Raised when the downloaded Kaggle file does not match its data contract."""


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Computers & Components": (
        "laptop", "macbook", "chromebook", "desktop", "ram", "ssd", "hard drive",
        "graphics card", "motherboard", "cpu", "power supply", "pc case", "thermal",
        "cooling pad", "monitor", "usb-c hub", "cable management",
    ),
    "Mobile & Wearables": (
        "smartphone", "phone", "tablet", "smartwatch", "fitness tracker", "smart ring",
        "power bank", "wireless charger", "stylus",
    ),
    "Audio": (
        "speaker", "headphone", "earbud", "earphone", "soundbar", "microphone",
        "audio interface", "boombox", "mp3", "turntable", "sound card", "headset",
        "home theater", "dj controller",
    ),
    "Cameras & Imaging": (
        "camera", "camcorder", "drone", "tripod", "ring light", "external flash",
        "photo frame", "camera lens", "camera bag", "webcam",
    ),
    "Gaming & VR": (
        "gaming", "console", "joystick", "racing wheel", "vr ", "capture card",
        "rgb mousepad", "controller",
    ),
    "TV & Entertainment": (
        "tv", "projector", "blu-ray", "streaming stick", "remote", "antenna",
    ),
    "Smart Home": (
        "smart light", "smart plug", "thermostat", "security camera", "video doorbell",
        "robot vacuum", "air purifier", "mesh wifi", "wifi router",
    ),
    "Office & Accessories": (
        "printer", "scanner", "shredder", "keyboard", "mouse", "desk lamp", "chair",
        "graphics tablet", "hdmi", "monitor arm", "headphone stand", "scale",
        "toothbrush", "gps", "binoculars",
    ),
}


def categorize_product(product: str) -> str:
    """Map an electronics product name to a transparent, reproducible category."""

    value = str(product).strip().lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in value for keyword in keywords):
            return category
    return "Other Electronics"


def validate_raw_columns(columns: Iterable[str]) -> None:
    """Require the exact 10-column Kaggle schema before processing."""

    actual = tuple(columns)
    if actual != RAW_COLUMNS:
        raise DatasetValidationError(
            f"Unexpected columns. Expected {list(RAW_COLUMNS)}, received {list(actual)}"
        )


def load_raw_dataset(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load and validate the immutable Kaggle CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the verified Kaggle CSV first."
        )
    frame = pd.read_csv(
        path,
        dtype={
            "order_id": "string",
            "customer_id": "string",
            "customer_name": "string",
            "customer_email": "string",
            "country": "string",
            "product": "string",
        },
    )
    validate_raw_columns(frame.columns)
    return frame


def clean_and_enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning and add analytics-friendly features."""

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
    for column in ("order_id", "customer_id", "country", "product"):
        result[column] = result[column].str.strip()
    result["price"] = pd.to_numeric(result["price"], errors="raise").round(2)
    result["quantity"] = pd.to_numeric(result["quantity"], errors="raise").astype("int16")
    result["order_value"] = pd.to_numeric(result["order_value"], errors="raise").round(2)
    mismatch = (result["price"] * result["quantity"] - result["order_value"]).abs() > 0.005
    if mismatch.any():
        raise DatasetValidationError(
            f"Found {int(mismatch.sum())} rows where order_value != price * quantity."
        )
    if result["order_id"].duplicated().any():
        raise DatasetValidationError("order_id contains duplicates.")
    if result.isna().any().any():
        raise DatasetValidationError("The verified dataset is expected to contain no missing values.")

    result["product_category"] = result["product"].map(categorize_product).astype("category")
    result["year"] = result["date"].dt.year.astype("int16")
    result["quarter"] = result["date"].dt.quarter.astype("int8")
    result["month"] = result["date"].dt.month.astype("int8")
    result["month_name"] = result["date"].dt.strftime("%b").astype("category")
    result["week"] = result["date"].dt.isocalendar().week.astype("int16")
    result["day_of_week"] = result["date"].dt.day_name().astype("category")
    result["is_q4"] = result["quarter"].eq(4)
    return result.sort_values(["date", "order_id"], kind="stable").reset_index(drop=True)


def build_quality_report(frame: pd.DataFrame, source_path: Path = RAW_CSV_PATH) -> dict:
    """Create a machine-readable dataset quality profile."""

    return {
        "source_file": source_path.name,
        "source_bytes": source_path.stat().st_size,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_order_ids": int(frame["order_id"].duplicated().sum()),
        "unique_customers": int(frame["customer_id"].nunique()),
        "unique_products": int(frame["product"].nunique()),
        "unique_countries": int(frame["country"].nunique()),
        "country_values": sorted(frame["country"].astype(str).unique().tolist()),
        "price_min": float(frame["price"].min()),
        "price_max": float(frame["price"].max()),
        "quantity_min": int(frame["quantity"].min()),
        "quantity_max": int(frame["quantity"].max()),
        "order_value_min": float(frame["order_value"].min()),
        "order_value_max": float(frame["order_value"].max()),
        "total_revenue": round(float(frame["order_value"].sum()), 2),
        "order_value_formula_mismatches": int(
            ((frame["price"] * frame["quantity"] - frame["order_value"]).abs() > 0.005).sum()
        ),
        "quality_status": "PASS",
    }


def prepare_dataset(source_path: Path = RAW_CSV_PATH, parquet_path: Path = PARQUET_PATH) -> dict:
    """Validate the CSV, produce Parquet, schema metadata, and a quality report."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_dataset(source_path)
    clean = clean_and_enrich(raw)
    clean.to_parquet(parquet_path, index=False, compression="zstd")
    quality = build_quality_report(clean, source_path)
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    SCHEMA_PATH.write_text(json.dumps(schema_catalog(), indent=2), encoding="utf-8")
    return quality


def load_processed_dataset(parquet_path: Path = PARQUET_PATH) -> pd.DataFrame:
    """Load the prepared Parquet dataset, preparing it on first use if necessary."""

    if not parquet_path.exists():
        prepare_dataset(parquet_path=parquet_path)
    return pd.read_parquet(parquet_path)

