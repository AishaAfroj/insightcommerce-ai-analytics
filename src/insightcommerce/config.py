"""Central paths and application configuration for InsightCommerce."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "e_commerce_electronic_sales_2025_dataset.csv"
RAW_ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "kaggle_dataset.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_PATH = PROCESSED_DIR / "orders.parquet"
SCHEMA_PATH = PROCESSED_DIR / "schema.json"
QUALITY_PATH = PROCESSED_DIR / "quality_report.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
BENCHMARK_DIR = ARTIFACTS_DIR / "benchmarks"
EXPORT_DIR = ARTIFACTS_DIR / "exports"
REPORT_DIR = PROJECT_ROOT / "output" / "pdf"


@dataclass(frozen=True)
class LLMSettings:
    """Runtime settings for local Ollama or an OpenAI-compatible endpoint."""

    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    timeout_seconds: int = 45

    @classmethod
    def from_environment(cls) -> LLMSettings:
        """Load non-secret defaults and optional API credentials from environment variables."""

        return cls(
            provider=os.getenv("INSIGHTCOMMERCE_LLM_PROVIDER", "ollama").lower(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        )


def ensure_output_directories() -> None:
    """Create generated-output directories without touching the immutable raw dataset."""

    for directory in (PROCESSED_DIR, MODEL_DIR, BENCHMARK_DIR, EXPORT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
