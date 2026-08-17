#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating the InsightCommerce environment..."
  python3.12 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-dev.txt
fi

echo "Checking the verified dataset..."
.venv/bin/python scripts/prepare_data.py

echo "Starting InsightCommerce at http://localhost:8501"
exec .venv/bin/streamlit run app.py
