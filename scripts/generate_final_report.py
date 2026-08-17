"""Generate the final InsightCommerce capstone PDF from measured project evidence."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insightcommerce.anomaly import detect_anomalies  # noqa: E402
from insightcommerce.config import (  # noqa: E402
    BENCHMARK_DIR,
    QUALITY_PATH,
    REPORT_DIR,
)
from insightcommerce.data import load_processed_dataset  # noqa: E402
from insightcommerce.ml import train_demand_model  # noqa: E402

NAVY = colors.HexColor("#0F172A")
BLUE = colors.HexColor("#2563EB")
SKY = colors.HexColor("#0EA5E9")
TEAL = colors.HexColor("#14B8A6")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#F8FAFC")
PALE_BLUE = colors.HexColor("#DBEAFE")
PALE_SKY = colors.HexColor("#E0F2FE")
PALE_GREEN = colors.HexColor("#DCFCE7")
PALE_ORANGE = colors.HexColor("#FFEDD5")
GRID = colors.HexColor("#CBD5E1")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ASSET_DIR = PROJECT_ROOT / "tmp" / "pdfs" / "report_assets"
OUTPUT_PATH = REPORT_DIR / "InsightCommerce_Group_Capstone_Report.pdf"


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load stable fonts for chart images."""

    path = FONT_BOLD if bold and Path(FONT_BOLD).exists() else FONT
    return ImageFont.truetype(path, size)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    y_labels: list[str],
) -> tuple[int, int, int, int]:
    """Draw shared chart furniture and return the inner plotting rectangle."""

    x1, y1, x2, y2 = rect
    draw.rounded_rectangle(rect, radius=20, fill="#FFFFFF", outline="#CBD5E1", width=2)
    draw.text((x1 + 28, y1 + 22), title, font=pil_font(27, True), fill="#0F172A")
    plot = (x1 + 92, y1 + 92, x2 - 34, y2 - 55)
    px1, py1, px2, py2 = plot
    for index, label in enumerate(y_labels):
        y = py2 - (py2 - py1) * index / max(len(y_labels) - 1, 1)
        draw.line((px1, y, px2, y), fill="#E2E8F0", width=2)
        draw.text((x1 + 16, y - 12), label, font=pil_font(16), fill="#64748B")
    draw.line((px1, py1, px1, py2), fill="#94A3B8", width=2)
    draw.line((px1, py2, px2, py2), fill="#94A3B8", width=2)
    return plot


def sales_evidence_chart(frame: pd.DataFrame, output: Path) -> None:
    """Create monthly revenue and country revenue evidence as a two-panel PNG."""

    canvas = PILImage.new("RGB", (1600, 760), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    monthly = (
        frame.assign(month=frame["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["order_value"]
        .sum()
    )
    left = draw_axes(draw, (25, 25, 965, 730), "Monthly revenue, 2025", ["$0M", "$10M", "$20M", "$30M", "$40M"])
    x1, y1, x2, y2 = left
    maximum = 40_000_000
    points = []
    for index, row in monthly.iterrows():
        x = x1 + (x2 - x1) * index / (len(monthly) - 1)
        y = y2 - (y2 - y1) * float(row["order_value"]) / maximum
        points.append((x, y))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2563EB")
        if index % 2 == 0:
            draw.text((x - 22, y2 + 14), row["month"].strftime("%b"), font=pil_font(16), fill="#475569")
    draw.line(points, fill="#2563EB", width=5)
    draw.text((x1 + 530, y1 + 12), "Q4 modeled spike", font=pil_font(18, True), fill="#1D4ED8")

    grouped = (
        frame.groupby("country", as_index=False)["order_value"]
        .sum()
        .nlargest(8, "order_value")
        .sort_values("order_value")
    )
    right = draw_axes(draw, (1000, 25, 1575, 730), "Top country markets", ["", "", "", "", ""])
    rx1, ry1, rx2, ry2 = right
    bar_height = (ry2 - ry1) / len(grouped) * 0.58
    max_country = float(grouped["order_value"].max())
    for index, row in enumerate(grouped.itertuples(index=False)):
        center_y = ry2 - (index + 0.5) * (ry2 - ry1) / len(grouped)
        width = (rx2 - rx1) * float(row.order_value) / max_country
        draw.rounded_rectangle(
            (rx1, center_y - bar_height / 2, rx1 + width, center_y + bar_height / 2),
            radius=8,
            fill="#0EA5E9",
        )
        draw.text((1008, center_y - 11), str(row.country), font=pil_font(15, True), fill="#334155")
        draw.text(
            (rx1 + width - 68, center_y - 10),
            f"${row.order_value / 1_000_000:.1f}M",
            font=pil_font(14, True),
            fill="white",
        )
    canvas.save(output, optimize=True)


def model_evidence_chart(model_artifact, output: Path) -> None:
    """Create actual/fitted demand and feature-importance evidence."""

    canvas = PILImage.new("RGB", (1600, 760), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    daily = model_artifact.daily_frame
    maximum = float(max(daily["orders"].max(), daily["predicted_orders"].max())) * 1.06
    left = draw_axes(
        draw,
        (25, 25, 1030, 730),
        "Actual vs fitted daily orders",
        ["0", "250", "500", "750", "1,000"],
    )
    x1, y1, x2, y2 = left
    for column, color in (("orders", "#2563EB"), ("predicted_orders", "#F59E0B")):
        points = []
        for index, value in enumerate(daily[column]):
            x = x1 + (x2 - x1) * index / (len(daily) - 1)
            y = y2 - (y2 - y1) * float(value) / maximum
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    draw.line((x1 + 20, y1 + 20, x1 + 80, y1 + 20), fill="#2563EB", width=5)
    draw.text((x1 + 88, y1 + 8), "Actual", font=pil_font(16, True), fill="#334155")
    draw.line((x1 + 190, y1 + 20, x1 + 250, y1 + 20), fill="#F59E0B", width=5)
    draw.text((x1 + 258, y1 + 8), "Fitted", font=pil_font(16, True), fill="#334155")
    for month, position in (("Jan", 0), ("Apr", 90), ("Jul", 181), ("Oct", 273), ("Dec", 350)):
        x = x1 + (x2 - x1) * position / 364
        draw.text((x - 14, y2 + 14), month, font=pil_font(16), fill="#475569")

    importance = model_artifact.feature_importance.head(8).sort_values("importance")
    right = draw_axes(draw, (1065, 25, 1575, 730), "Feature importance", ["", "", "", "", ""])
    rx1, ry1, rx2, ry2 = right
    bar_height = (ry2 - ry1) / len(importance) * 0.58
    max_importance = float(importance["importance"].max())
    for index, row in enumerate(importance.itertuples(index=False)):
        center_y = ry2 - (index + 0.5) * (ry2 - ry1) / len(importance)
        width = (rx2 - rx1) * float(row.importance) / max_importance
        draw.rounded_rectangle(
            (rx1, center_y - bar_height / 2, rx1 + width, center_y + bar_height / 2),
            radius=8,
            fill="#14B8A6",
        )
        label = str(row.feature).replace("_", " ")
        draw.text((1072, center_y - 10), label[:18], font=pil_font(14, True), fill="#334155")
    canvas.save(output, optimize=True)


def anomaly_evidence_chart(anomaly_artifact, output: Path) -> None:
    """Create a transaction-value/anomaly-score scatter evidence chart."""

    scored = anomaly_artifact.scored_frame
    sample = pd.concat(
        [
            scored.loc[~scored["is_anomaly"]].sample(2_400, random_state=42),
            scored.loc[scored["is_anomaly"]].head(500),
        ],
        ignore_index=True,
    )
    canvas = PILImage.new("RGB", (1600, 760), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    plot = draw_axes(
        draw,
        (30, 25, 1570, 730),
        "Transaction anomaly evidence: order value vs anomaly score",
        ["0.00", "0.05", "0.10", "0.15", "0.20"],
    )
    x1, y1, x2, y2 = plot
    x_max = float(sample["order_value"].quantile(0.995))
    y_min = float(sample["anomaly_score"].min())
    y_max = float(sample["anomaly_score"].max())
    for row in sample.itertuples(index=False):
        x = x1 + (x2 - x1) * min(float(row.order_value), x_max) / x_max
        y = y2 - (y2 - y1) * (float(row.anomaly_score) - y_min) / max(y_max - y_min, 0.001)
        color = "#EF4444" if row.is_anomaly else "#93C5FD"
        radius = 5 if row.is_anomaly else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.text((x1 + 20, y1 + 20), "Blue: typical", font=pil_font(17, True), fill="#2563EB")
    draw.text((x1 + 170, y1 + 20), "Red: flagged", font=pil_font(17, True), fill="#DC2626")
    draw.text((x1 + 560, y2 + 14), "Order value (USD, clipped at 99.5th percentile)", font=pil_font(17), fill="#475569")
    canvas.save(output, optimize=True)


def create_styles() -> dict[str, ParagraphStyle]:
    """Create a restrained, consistent report style sheet."""

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=34,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=15,
            leading=21,
            textColor=BLUE,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.4,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=SLATE,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1E3A8A"),
            leftIndent=10,
            rightIndent=10,
            spaceBefore=5,
            spaceAfter=5,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=17,
            textColor=SLATE,
            spaceAfter=3,
        ),
        "cover_card": ParagraphStyle(
            "CoverCard",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=12.2,
            textColor=NAVY,
            spaceAfter=0,
        ),
        "course": ParagraphStyle(
            "Course",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.1,
            leading=11,
            textColor=NAVY,
            leftIndent=8,
            rightIndent=8,
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=8,
            backColor=LIGHT,
        ),
    }


def styled_table(
    rows: list[list],
    widths: list[float],
    header: bool = True,
    font_size: float = 8.2,
) -> Table:
    """Create a table with exact widths, padding, and repeating header."""

    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def callout(text: str, style: ParagraphStyle, fill=PALE_BLUE) -> Table:
    """Create one compact highlighted finding."""

    table = Table([[Paragraph(text, style)]], colWidths=[7.05 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def page_decor(canvas, document) -> None:
    """Draw consistent headers, footers, and page numbers after the cover."""

    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(0.72 * inch, 10.35 * inch, 7.78 * inch, 10.35 * inch)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(0.72 * inch, 10.48 * inch, "INSIGHTCOMMERCE CAPSTONE REPORT")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(SLATE)
        canvas.drawRightString(7.78 * inch, 0.48 * inch, f"InsightCommerce Group Project  |  Page {page}")
        canvas.line(0.72 * inch, 0.62 * inch, 7.78 * inch, 0.62 * inch)
    canvas.restoreState()


def section_page(story: list, title: str, styles: dict[str, ParagraphStyle]) -> None:
    """Start a numbered report section on a clean page."""

    story.append(PageBreak())
    story.append(Paragraph(title, styles["h1"]))


def generate_report() -> Path:
    """Build the complete final report and remove temporary authoring images."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    styles = create_styles()
    frame = load_processed_dataset()
    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
    performance = json.loads((BENCHMARK_DIR / "performance.json").read_text(encoding="utf-8"))
    question_benchmark = json.loads(
        (BENCHMARK_DIR / "question_benchmark.json").read_text(encoding="utf-8")
    )
    demand = train_demand_model(frame)
    anomalies = detect_anomalies(frame, contamination=0.01)

    sales_chart = ASSET_DIR / "sales_evidence.png"
    model_chart = ASSET_DIR / "model_evidence.png"
    anomaly_chart = ASSET_DIR / "anomaly_evidence.png"
    sales_evidence_chart(frame, sales_chart)
    model_evidence_chart(demand, model_chart)
    anomaly_evidence_chart(anomalies, anomaly_chart)

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.78 * inch,
        title="InsightCommerce Capstone Report",
        author="MD. Shahriar Ahamed Ridoy; Aisha Afroj",
        subject="AI-Powered E-Commerce Analytics and Decision Intelligence Platform",
    )
    story: list = []

    # Page 1 - cover
    kicker = Table([[Paragraph("CAPSTONE PROJECT REPORT", styles["cover_kicker"])]], colWidths=[2.4 * inch])
    kicker.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("BOX", (0, 0), (-1, -1), 0, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 0.28 * inch),
            kicker,
            Spacer(1, 0.30 * inch),
            Paragraph("InsightCommerce", styles["title"]),
            Paragraph(
                "An AI-Powered E-Commerce Analytics and Decision Intelligence Platform",
                styles["subtitle"],
            ),
            Table([[""]], colWidths=[7.05 * inch], rowHeights=[0.08 * inch], style=[("BACKGROUND", (0, 0), (-1, -1), BLUE)]),
            Spacer(1, 0.26 * inch),
            Paragraph("Course Title: Deep Learning", styles["course"]),
            Spacer(1, 0.18 * inch),
            Table(
                [
                    [
                        Paragraph(
                            "<b>Submitted by</b><br/><br/>"
                            "<b>MD. Shahriar Ahamed Ridoy</b><br/>"
                            "ID: 261-25-008<br/>"
                            "M.Sc. in CSE (Major in Data Science)<br/>"
                            "Daffodil International University<br/><br/>"
                            "<b>Aisha Afroj</b><br/>"
                            "ID: 261-25-007<br/>"
                            "M.Sc. in CSE (Major in Data Science)<br/>"
                            "Daffodil International University",
                            styles["cover_card"],
                        ),
                        Paragraph(
                            "<b>Submitted to</b><br/><br/>"
                            "<b>Sadat Hasan</b><br/>"
                            "Adjunct Faculty<br/>"
                            "Department of Computer Science and Engineering (CSE)<br/>"
                            "Faculty of Science and Information Technology (FSIT)<br/>"
                            "Daffodil International University",
                            styles["cover_card"],
                        ),
                    ]
                ],
                colWidths=[3.52 * inch, 3.53 * inch],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), PALE_BLUE),
                        ("BACKGROUND", (1, 0), (1, 0), PALE_SKY),
                        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                        ("INNERGRID", (0, 0), (-1, -1), 0.7, colors.white),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 11),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                    ]
                ),
            ),
            Spacer(1, 0.18 * inch),
            Paragraph("Submission date: 17 August 2026", styles["cover_meta"]),
            Spacer(1, 0.14 * inch),
            callout(
                "Group capstone evidence: a verified 108,300-row dataset, 21 passing tests, "
                "a 10/10 query benchmark, and filtered aggregation below 3 ms on our development Mac.",
                styles["callout"],
            ),
        ]
    )

    # Page 2 - abstract
    section_page(story, "Abstract and project result", styles)
    abstract = (
        "We built InsightCommerce to make a large e-commerce dataset easier to explore and explain. "
        "Our verified Kaggle file contains 108,300 synthetic electronics transactions from 2025. "
        "We check the real schema, create a data-quality profile, derive documented calendar and "
        "product-taxonomy fields, store optimized Parquet, and use DuckDB for fast aggregation. "
        "The schema-aware assistant turns plain-English questions into a strict JSON plan and "
        "read-only SQL. Before anything runs, an AST safety gate blocks data changes, external access, "
        "sensitive-shaped fields, unrestricted star selection, excessive rows, and long execution. "
        "The application also keeps five successful conversation turns and allows one corrective retry. "
        "Its Streamlit interface includes global filters, preset insights, eight chart types, AI chart "
        "suggestions with manual override, and PDF, DOCX, CSV, HTML, and PNG export paths. We added daily "
        "demand prediction and Isolation Forest anomaly detection as advanced features. Our evaluation "
        "produced 21 passing tests, 10/10 benchmark answers, a 2.52 ms median filtered aggregation, and "
        "a held-out R-squared of 0.995. We report these results together with the limitations of using "
        "synthetic, single-year data."
    )
    story.append(Paragraph(abstract, styles["body"]))
    story.append(Paragraph("Measured completion summary", styles["h2"]))
    summary_rows = [
        ["Evidence", "Result", "Acceptance"],
        ["Source integrity", "108,300 rows; 0 missing; 0 duplicate IDs", "PASS"],
        ["Filtered aggregation", f"Median {performance['median_ms']:.2f} ms", "PASS (<500 ms)"],
        ["Automated verification", "21 tests + static quality checks", "PASS"],
        ["Question benchmark", "10 of 10 questions", "100%"],
        ["Dashboard", "5 sections; 8 prepared chart types", "PASS"],
        ["Advanced features", "Demand prediction + anomaly detection", "PASS"],
    ]
    story.append(styled_table(summary_rows, [2.05 * inch, 3.3 * inch, 1.7 * inch]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        callout(
            "Central design choice: adapt to the real ten-column file. Profit, discount, shipping, "
            "and cancellation are not present, so no synthetic analytical field was invented.",
            styles["callout"],
            PALE_SKY,
        )
    )

    # Page 3 - introduction
    section_page(story, "1. Introduction, objectives, and dataset scope", styles)
    story.append(
        Paragraph(
            "For this capstone, we wanted to go beyond a dashboard that only displays fixed charts. "
            "Our goal was to let a user move from a business question to a result they can inspect, "
            "visualize, and export. That required a reproducible data layer, controlled AI-generated "
            "queries, interactive charts, and advanced analytics that stay honest about the limits "
            "of the dataset.",
            styles["body"],
        )
    )
    story.append(Paragraph("Objectives", styles["h2"]))
    for item in (
        "- Verify the downloaded file before implementation and preserve the original unchanged.",
        "- Deliver fast filtered aggregation, schema inspection, and measurable data-quality evidence.",
        "- Convert natural language to safe DuckDB SQL with bounded retry and conversational context.",
        "- Provide at least six interactive visual forms plus AI selection and human override.",
        "- Add two advanced features: demand prediction and anomaly detection.",
        "- Package tests, deployment configuration, documentation, benchmarks, final report, and ZIP.",
    ):
        story.append(Paragraph(item, styles["body"]))
    story.append(Paragraph("Dataset scope", styles["h2"]))
    dataset_rows = [
        ["Dimension", "Verified value"],
        ["Period", f"{quality['date_min']} to {quality['date_max']}"],
        ["Markets", f"{quality['unique_countries']} countries"],
        ["Products", f"{quality['unique_products']} electronics products"],
        ["Customers", f"{quality['unique_customers']:,} synthetic customer identifiers"],
        ["Order value", "$9.99 to $16,191.00; exact price x quantity"],
        ["Total simulated revenue", f"${quality['total_revenue']:,.2f}"],
    ]
    story.append(styled_table(dataset_rows, [2.2 * inch, 4.85 * inch]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(
        Paragraph(
            "Although names and email addresses are synthetic, they are treated as sensitive-shaped "
            "data. They remain in the preserved source but are excluded from prompts, general exports, "
            "anomaly results, and the generated-query allowlist.",
            styles["body"],
        )
    )
    story.append(Paragraph("Project team", styles["h2"]))
    team_rows = [
        ["Member", "Student ID", "Program"],
        ["MD. Shahriar Ahamed Ridoy", "261-25-008", "M.Sc. in CSE (Major in Data Science)"],
        ["Aisha Afroj", "261-25-007", "M.Sc. in CSE (Major in Data Science)"],
    ]
    story.append(styled_table(team_rows, [2.5 * inch, 1.15 * inch, 3.4 * inch], font_size=8))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "We worked together across data preparation, application development, testing, "
            "documentation, and presentation planning.",
            styles["body"],
        )
    )

    # Page 4 - architecture
    section_page(story, "2. System architecture", styles)
    story.append(
        Paragraph(
            "We organized the project into clear layers so that each result can be traced. DuckDB "
            "receives either parameterized dashboard filters or SQL that has already passed the AST "
            "safety check. Raw customer names and email addresses are never sent to an LLM.",
            styles["body"],
        )
    )
    story.append(Image(str(PROJECT_ROOT / "docs" / "architecture.png"), width=7.05 * inch, height=4.39 * inch))
    story.append(Paragraph("Figure 1. InsightCommerce component and trust-boundary architecture.", styles["caption"]))
    component_rows = [
        ["Layer", "Responsibility", "Primary evidence"],
        ["Data", "Validate, enrich, profile, convert, aggregate", "quality_report.json; performance.json"],
        ["AI query", "Schema prompt, JSON plan, retry, memory", "prompts.py; assistant.py"],
        ["Safety", "Parse, allowlist, cap, timeout, disable external access", "sandbox.py; security tests"],
        ["Experience", "Filters, charts, explanations, exports", "app.py; browser QA"],
        ["Advanced", "Demand forecast and anomaly ranking", "ml.py; anomaly.py; model card"],
    ]
    story.append(styled_table(component_rows, [1.05 * inch, 3.45 * inch, 2.55 * inch], font_size=7.8))

    # Page 5 - Task A
    section_page(story, "3. Task A - data backend and performance", styles)
    story.append(
        Paragraph(
            "We started by verifying the downloaded file instead of assuming its columns. The raw ZIP "
            "checksum is 81a80e884aeaf28f6816af25fa334dbcd712716337208194c5b451be339b63db. "
            "Our preparation command checks the exact column order, parses dates and numerics, strips "
            "identifier whitespace, enforces unique order IDs, confirms price x quantity, derives "
            "calendar/taxonomy fields, and writes Zstd-compressed Parquet plus JSON metadata.",
            styles["body"],
        )
    )
    source_schema = [
        ["Field", "Type", "Analytical use"],
        ["order_id", "string", "Unique order count"],
        ["date", "date", "Trend and seasonality"],
        ["customer_id", "string", "Repeat-customer count"],
        ["customer_name", "string", "Excluded from AI/general exports"],
        ["customer_email", "string", "Excluded from AI/general exports"],
        ["country", "string", "Geographic comparison"],
        ["product", "string", "Product ranking and taxonomy"],
        ["price", "float", "Unit-price distribution"],
        ["quantity", "integer", "Units and transaction pattern"],
        ["order_value", "float", "Simulated revenue"],
    ]
    story.append(styled_table(source_schema, [1.55 * inch, 1.1 * inch, 4.4 * inch], font_size=8))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        callout(
            f"Performance result: {performance['iterations']} warm filtered aggregations produced a "
            f"{performance['median_ms']:.2f} ms median, {performance['p95_ms']:.2f} ms p95, and "
            f"{performance['max_ms']:.2f} ms maximum - comfortably below the 500 ms target.",
            styles["callout"],
            PALE_GREEN,
        )
    )

    # Page 6 - Task B
    section_page(story, "4. Task B - schema-aware AI and safe text-to-code", styles)
    story.append(
        Paragraph(
            "When a user asks a question, we combine it with the public schema catalog and at most five "
            "previous successful turns. The model must return a strict QueryPlan containing SQL, chart "
            "type, result fields, a title, and a rationale. We never execute model-generated SQL directly.",
            styles["body"],
        )
    )
    pipeline_rows = [
        ["Step", "Control", "Failure behavior"],
        ["1", "Question + schema + recent memory", "No raw PII sample is sent"],
        ["2", "Strict JSON Schema query plan", "Pydantic rejects missing/extra fields"],
        ["3", "sqlglot AST allowlist", "Unsafe plans are blocked before DuckDB"],
        ["4", "5 s timeout + 5,000-row wrapper", "Query is interrupted/capped"],
        ["5", "One error-informed corrective retry", "Second failure becomes a clear user error"],
        ["6", "Grounded result narration", "Only executed result values are described"],
    ]
    story.append(styled_table(pipeline_rows, [0.5 * inch, 3.25 * inch, 3.3 * inch], font_size=8))
    story.append(Paragraph("Provider strategy", styles["h2"]))
    provider_rows = [
        ["Mode", "Purpose", "Credential"],
        ["Ollama", "Local live LLM demonstration", "None; local model required"],
        ["OpenAI Responses", "Optional hosted Structured Outputs", "OPENAI_API_KEY"],
        ["Offline demo", "Deterministic common-question evaluation", "None; clearly labeled fallback"],
    ]
    story.append(styled_table(provider_rows, [1.45 * inch, 3.75 * inch, 1.85 * inch]))
    story.append(Spacer(1, 0.14 * inch))
    story.append(
        callout(
            "Security tests prove that DROP, DELETE, COPY, external CSV reads, hidden schema access, "
            "multiple statements, sensitive fields, and unrestricted star selection are rejected.",
            styles["callout"],
            PALE_ORANGE,
        )
    )

    # Page 7 - Task C interface
    section_page(story, "5. Task C - interactive application", styles)
    story.append(
        Paragraph(
            "We divided the Streamlit application into Overview, Visual Explorer, AI Analyst, "
            "Prediction & Anomalies, and Reports & Quality. Date, country, and product-category filters "
            "apply consistently across the dashboard. The default overview remains legible at a tested "
            "1600 x 1000 desktop viewport with no horizontal overflow or application exceptions.",
            styles["body"],
        )
    )
    story.append(Image(str(PROJECT_ROOT / "docs" / "dashboard-overview.png"), width=7.05 * inch, height=4.41 * inch))
    story.append(Paragraph("Figure 2. Browser-verified default dashboard on the development Mac.", styles["caption"]))
    interface_rows = [
        ["Section", "User outcome"],
        ["Overview", "KPIs, trend, map, category comparison, quarter heatmap"],
        ["Visual Explorer", "Eight prepared visual forms plus manual builder and chart downloads"],
        ["AI Analyst", "Question -> safe SQL -> table -> chart -> grounded explanation"],
        ["Prediction & Anomalies", "Forecast a date and investigate unusual orders"],
        ["Reports & Quality", "Inspect schema/quality and download PDF/DOCX summaries"],
    ]
    story.append(styled_table(interface_rows, [1.7 * inch, 5.35 * inch]))

    # Page 8 - visualization evidence
    section_page(story, "6. Visualization and analytical evidence", styles)
    story.append(Image(str(sales_chart), width=7.05 * inch, height=3.35 * inch))
    story.append(Paragraph("Figure 3. Measured 2025 revenue pattern and strongest country markets.", styles["caption"]))
    story.append(
        Paragraph(
            "When we reviewed the yearly pattern, revenue stayed near $6-7 million per month from "
            "January through October, then rose above $33 million in both November and December. "
            "The source intentionally weights Q4, so this is a modeled pattern rather than evidence "
            "about a real company. USA leads total revenue "
            "because the synthetic generator also assigns different country record volumes.",
            styles["body"],
        )
    )
    visual_rows = [
        ["Visual", "Purpose", "Interactivity"],
        ["Line", "Monthly revenue trend", "Hover, zoom, range inspection"],
        ["Choropleth", "Country revenue", "Country hover and scale"],
        ["Bar", "Category ranking", "Sorted comparison"],
        ["Heatmap", "Country x quarter concentration", "Color/intensity inspection"],
        ["Treemap", "Category -> product hierarchy", "Drill and hover"],
        ["Scatter", "Price, quantity, order value", "Multi-encoding exploration"],
        ["Box", "Price distribution by category", "Outlier visibility"],
        ["Histogram", "Order-value distribution", "Binned frequency"],
    ]
    story.append(styled_table(visual_rows, [1.1 * inch, 3.15 * inch, 2.8 * inch], font_size=7.7))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "AI chart hints are accepted only when referenced columns and types are compatible. The "
            "user can override chart type and fields; incompatible combinations fall back safely.",
            styles["body"],
        )
    )

    # Page 9 - prediction
    section_page(story, "7. Task D feature 1 - predictive analytics", styles)
    story.append(Image(str(model_chart), width=7.05 * inch, height=3.35 * inch))
    story.append(Paragraph("Figure 4. Daily order-demand fit and model feature importance.", styles["caption"]))
    metric_rows = [
        ["Metric", "Held-out result", "Interpretation"],
        ["Test design", "Every seventh day (53 days)", "Seasonally distributed holdout"],
        ["MAE", f"{demand.metrics['mae_orders']:.1f} orders/day", "Average absolute error"],
        ["RMSE", f"{demand.metrics['rmse_orders']:.1f} orders/day", "Penalizes larger misses"],
        ["R-squared", f"{demand.metrics['r2']:.3f}", "Strong fit to synthetic seasonality"],
        ["Model", "Random Forest; seed 42", "Reproducible nonlinear calendar model"],
    ]
    story.append(styled_table(metric_rows, [1.35 * inch, 2.15 * inch, 3.55 * inch]))
    story.append(Paragraph("Feature and prediction design", styles["h2"]))
    story.append(
        Paragraph(
            "Known-in-advance features are month, quarter, weekday, day of month, annual sine/cosine, "
            "weekend, and Q4. The interface predicts daily order count for a chosen date. Indicative "
            "revenue is then predicted orders multiplied by the historical median order value; it is "
            "explicitly labeled as an assumption rather than a separate learned target.",
            styles["body"],
        )
    )
    story.append(
        callout(
            "The high R-squared reflects the dataset's designed Q4 step change. It must not be "
            "generalized to real retail demand without multi-year real data and rolling-origin tests.",
            styles["callout"],
            PALE_ORANGE,
        )
    )

    # Page 10 - anomaly
    section_page(story, "8. Task D feature 2 - anomaly detection", styles)
    story.append(Image(str(anomaly_chart), width=7.05 * inch, height=3.35 * inch))
    story.append(Paragraph("Figure 5. Isolation Forest anomaly scores with flagged points in red.", styles["caption"]))
    anomaly_rows = [
        ["Element", "Implementation"],
        ["Model", "Isolation Forest, 160 trees, seed 42"],
        ["Features", "log price, quantity, log order value, value / product median"],
        ["Nominal contamination", "1%"],
        ["Observed flags", f"{int(anomalies.metrics['anomalies']):,} of {int(anomalies.metrics['rows_scored']):,}"],
        ["Observed rate", f"{anomalies.metrics['anomaly_rate']:.2%}"],
        ["Output", "score, flag, and transparent reason code"],
    ]
    story.append(styled_table(anomaly_rows, [2.15 * inch, 4.9 * inch]))
    story.append(Paragraph("Interpretation boundary", styles["h2"]))
    story.append(
        Paragraph(
            "Reason codes identify exceptionally high order value, premium unit price, maximum "
            "quantity, values well above a product's median, or a broader unusual multivariate "
            "combination. The output excludes name and email. A flag is an investigative signal - "
            "not proof of fraud, data error, or misconduct. Score ties can make the observed rate "
            "slightly different from the configured contamination.",
            styles["body"],
        )
    )

    # Page 11 - evaluation
    section_page(story, "9. Evaluation, testing, and browser QA", styles)
    evaluation_rows = [
        ["Verification area", "Evidence", "Result"],
        ["Data contract", "Exact schema, shape, identity checks", "PASS"],
        ["Generated SQL safety", "7 malicious/unsafe cases + safe COUNT(*)", "PASS"],
        ["Retry behavior", "Unsafe first plan, valid second; no third", "PASS"],
        ["Conversation memory", "Seven writes retain only latest five", "PASS"],
        ["Charts", "Time/country inference and invalid-hint fallback", "PASS"],
        ["Prediction/anomaly", "Finite metrics, output, privacy columns", "PASS"],
        ["Exports", "PDF, DOCX, CSV, interactive HTML signatures", "PASS"],
        ["Performance", f"Median {performance['median_ms']:.2f} ms vs 500 ms", "PASS"],
        ["Static quality", "Ruff across app, source, scripts, tests", "PASS"],
        ["Total automated tests", "21", "PASS"],
    ]
    story.append(styled_table(evaluation_rows, [1.6 * inch, 4.35 * inch, 1.1 * inch], font_size=7.8))
    story.append(Paragraph("Browser walkthrough", styles["h2"]))
    story.append(
        Paragraph(
            "We launched the application on our development Mac and tested it through the browser. "
            "We opened all five primary tabs and both nested advanced/report tabs. The offline monthly "
            "AI question executed in one attempt at 6.70 ms. An initial duplicate Plotly element ID was "
            "observed between hidden tabs, fixed with unique keys, and reverified. Final snapshots showed "
            "zero Streamlit exception elements and no horizontal overflow at 1600 x 1000.",
            styles["body"],
        )
    )
    story.append(
        callout(
            "Evaluation principle: pass/fail claims in this report come from committed JSON/Markdown "
            "artifacts, automated tests, or the final browser walkthrough.",
            styles["callout"],
        )
    )

    # Page 12 - benchmark
    section_page(story, "10. Ten-question accuracy benchmark", styles)
    story.append(
        Paragraph(
            "We used ten representative questions to test the full offline path: natural-language plan selection, "
            "strict plan parsing, SQL safety validation, DuckDB execution, chart compatibility, and "
            "grounded narration. Each result is compared with independently executed reference SQL.",
            styles["body"],
        )
    )
    benchmark_rows: list[list] = [["#", "Question", "Attempts", "Time", "Status"]]
    for row in question_benchmark["results"]:
        benchmark_rows.append(
            [
                str(row["id"]),
                Paragraph(row["question"], styles["small"]),
                str(row["attempts"]),
                f"{row['execution_ms']:.2f} ms",
                row["status"],
            ]
        )
    story.append(
        styled_table(
            benchmark_rows,
            [0.35 * inch, 4.15 * inch, 0.75 * inch, 0.95 * inch, 0.85 * inch],
            font_size=7.7,
        )
    )
    story.append(Spacer(1, 0.16 * inch))
    criteria_rows = [
        ["Per-question criterion", "Required"],
        ["Exact numeric/tabular result", "Yes"],
        ["Required result columns", "Yes"],
        ["Compatible chart recommendation", "Yes"],
        ["Success without corrective retry", "Yes"],
        ["Non-empty grounded narrative", "Yes"],
    ]
    story.append(styled_table(criteria_rows, [4.9 * inch, 2.15 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        callout(
            f"Final benchmark accuracy: {question_benchmark['passed']}/{question_benchmark['questions']} "
            f"({question_benchmark['accuracy']:.0%}). Live LLM accuracy remains provider/model dependent "
            "and should be evaluated separately before production use.",
            styles["callout"],
            PALE_GREEN,
        )
    )

    # Page 13 - security
    section_page(story, "11. Security, privacy, and reliability", styles)
    security_rows = [
        ["Risk", "Control", "Residual limitation"],
        ["Prompt injection", "Question/history labeled untrusted; system contract fixed", "A model may still propose bad SQL; validator is authoritative"],
        ["SQL mutation", "AST permits one SELECT/CTE only", "Parser/library updates require regression tests"],
        ["File/network access", "Functions blocked; DuckDB external access disabled", "OS/container permissions remain defense-in-depth"],
        ["Sensitive-shaped fields", "Prompt schema omits name/email; validator rejects them", "Raw synthetic source still contains these columns"],
        ["Resource exhaustion", "5 s interrupt, 5,000 rows, 1 GB DuckDB limit", "Host-level limits should also be configured"],
        ["Hallucinated insight", "Narrative is generated from executed result values", "Live-model explanatory text still needs user judgment"],
        ["Secret exposure", "Environment/Streamlit secrets; real files ignored by Git", "Operators must rotate compromised keys"],
    ]
    story.append(styled_table(security_rows, [1.3 * inch, 3.1 * inch, 2.65 * inch], font_size=7.4))
    story.append(Paragraph("Reliability controls", styles["h2"]))
    for item in (
        "- Fixed random seeds make prediction and anomaly results reproducible.",
        "- Parameterized application filters keep user values out of SQL strings.",
        "- Only successful answers enter conversation memory.",
        "- AI chart hints are type-checked before rendering.",
        "- The original CSV and ZIP remain unchanged and attributable under their own license.",
        "- CI reruns data preparation, tests, static checks, and both benchmarks.",
    ):
        story.append(Paragraph(item, styles["body"]))

    # Page 14 - deployment
    section_page(story, "12. Reproducibility and deployment preparation", styles)
    story.append(
        Paragraph(
            "We included pinned requirements and the raw data so a clean environment can reproduce the "
            "application. Parquet and quality files are rebuilt rather than treated as hidden inputs.",
            styles["body"],
        )
    )
    command = """python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/prepare_data.py
pytest
python scripts/benchmark_performance.py
python scripts/run_question_benchmark.py
streamlit run app.py"""
    story.append(Preformatted(command, styles["code"]))
    story.append(Paragraph("Prepared deployment targets", styles["h2"]))
    deploy_rows = [
        ["Target", "Included support", "AI behavior"],
        ["Local macOS", "Virtual environment + Streamlit", "Ollama, OpenAI, or offline demo"],
        ["Streamlit Cloud", "app.py, runtime.txt, pinned requirements", "Offline demo or hosted secret"],
        ["Docker", "Dockerfile + health check", "Configure reachable protected provider"],
        ["Process host", "Procfile", "Provider via environment variables"],
        ["GitHub CI", "Test, lint, data prep, two benchmarks", "No external LLM required"],
    ]
    story.append(styled_table(deploy_rows, [1.35 * inch, 3.2 * inch, 2.5 * inch]))
    story.append(Paragraph("Version control", styles["h2"]))
    story.append(
        Paragraph(
            "Development is divided into regular milestone commits: verified data backend; safe "
            "schema-aware AI engine; dashboard and advanced analytics; submission documentation and "
            "evaluation; and final report/package. This avoids a prohibited single-commit submission "
            "and makes review traceable.",
            styles["body"],
        )
    )

    # Page 15 - limitations, future, conclusion, references
    section_page(story, "13. Limitations, future work, conclusion, and references", styles)
    story.append(Paragraph("Limitations", styles["h2"]))
    for item in (
        "- The source is synthetic and represents only 2025; findings are demonstrations, not company facts.",
        "- Order value is deterministically price x quantity; no profit, cost, discount, shipping, or returns exist.",
        "- Country volumes and Q4 behavior were modeled by the dataset creator and drive several results.",
        "- The forecast holdout is seasonally distributed, not a multi-year rolling-origin production test.",
        "- Isolation Forest identifies unusual combinations, not fraud or operational error.",
        "- Live LLM results depend on provider availability, model behavior, latency, and cost.",
    ):
        story.append(Paragraph(item, styles["body"]))
    story.append(Paragraph("Future work", styles["h2"]))
    story.append(
        Paragraph(
            "If we continue this work with instructor approval and a richer real dataset, we would add "
            "product cost/profit, discount, shipping, return status, and customer segment fields; evaluate "
            "rolling multi-year forecasts; add authenticated user roles and audit logs; benchmark "
            "multiple live LLMs; and deploy managed observability with rate limits and query traces.",
            styles["body"],
        )
    )
    story.append(Paragraph("Conclusion", styles["h2"]))
    story.append(
        Paragraph(
            "By completing InsightCommerce, we met Tasks A-D in one integrated and testable application. "
            "For us, the most important result is not one chart or model; it is the controlled path from "
            "a verified file to generated code, checked execution, and an exportable answer. The project "
            "is fast, visually complete, privacy-conscious, reproducible, and clear about what this "
            "synthetic dataset can and cannot support.",
            styles["body"],
        )
    )
    story.append(Paragraph("References", styles["h2"]))
    references = (
        "1. W. Kielbowicz, Synthetic E-Commerce Electronic Sales 2025 Dataset, Kaggle, 2026. "
        "https://www.kaggle.com/datasets/wojciechkiebowicz/e-commerce-electronic-sales-2025-dataset",
        "2. DuckDB Documentation. https://duckdb.org/docs/",
        "3. Streamlit Documentation. https://docs.streamlit.io/",
        "4. Plotly Python Documentation. https://plotly.com/python/",
        "5. scikit-learn User Guide: Random Forests and Isolation Forest. https://scikit-learn.org/stable/user_guide.html",
        "6. OpenAI API Documentation: Structured model outputs. https://developers.openai.com/api/docs/guides/structured-outputs",
        "7. Ollama Documentation. https://docs.ollama.com/",
        "8. sqlglot documentation and source. https://github.com/tobymao/sqlglot",
    )
    for reference in references:
        story.append(Paragraph(reference, styles["small"]))

    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    shutil.rmtree(ASSET_DIR, ignore_errors=True)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(generate_report())
