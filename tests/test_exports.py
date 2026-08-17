"""Smoke tests for PDF, Word, CSV, and interactive chart exports."""

import pandas as pd
import plotly.express as px

from insightcommerce.exports import (
    export_chart_html,
    export_frame_csv,
    export_summary_docx,
    export_summary_pdf,
)


def test_document_and_data_exports_have_expected_signatures() -> None:
    frame = pd.DataFrame({"country": ["USA", "India"], "revenue": [10, 20]})
    metrics = {"Revenue": "$30"}
    insights = ["India leads this two-row sample."]
    assert export_summary_pdf("Test", metrics, insights, frame).startswith(b"%PDF")
    assert export_summary_docx("Test", metrics, insights, frame).startswith(b"PK")
    assert export_frame_csv(frame).startswith(b"country,revenue")
    assert b"plotly" in export_chart_html(px.bar(frame, x="country", y="revenue")).lower()
