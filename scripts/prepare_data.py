"""Prepare the verified Kaggle CSV for the analytics application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insightcommerce.data import prepare_dataset  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(prepare_dataset(), indent=2))
