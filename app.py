"""Streamlit entry point for the InsightCommerce capstone application."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insightcommerce.anomaly import AnomalyArtifact, detect_anomalies  # noqa: E402
from insightcommerce.assistant import AnalyticsAssistant, AnalyticsAssistantError  # noqa: E402
from insightcommerce.charts import (  # noqa: E402
    CHART_OPTIONS,
    ChartSpec,
    build_chart,
    category_price_box,
    category_revenue_chart,
    chart_is_compatible,
    country_revenue_chart,
    monthly_revenue_chart,
    order_value_histogram,
    price_order_scatter,
    product_treemap,
    quarter_country_heatmap,
    recommend_chart,
)
from insightcommerce.config import QUALITY_PATH, LLMSettings  # noqa: E402
from insightcommerce.data import load_processed_dataset  # noqa: E402
from insightcommerce.exports import (  # noqa: E402
    ExportDependencyError,
    export_chart_html,
    export_chart_png,
    export_frame_csv,
    export_summary_docx,
    export_summary_pdf,
)
from insightcommerce.memory import ConversationMemory  # noqa: E402
from insightcommerce.ml import (  # noqa: E402
    DemandModelArtifact,
    predict_daily_demand,
    train_demand_model,
)
from insightcommerce.presets import PRESET_INSIGHTS  # noqa: E402
from insightcommerce.providers import LLMProviderError, build_provider  # noqa: E402
from insightcommerce.schema import schema_catalog  # noqa: E402

st.set_page_config(
    page_title="InsightCommerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px;}
      [data-testid="stMetric"] {background: white; border: 1px solid #dbeafe;
        border-radius: 14px; padding: 0.85rem 1rem; box-shadow: 0 4px 18px rgba(15,23,42,.05);}
      .hero {background: linear-gradient(120deg,#0f172a,#1d4ed8 60%,#0ea5e9);
        padding: 1.4rem 1.7rem; border-radius: 18px; color: white; margin-bottom: 1rem;}
      .hero h1 {font-size: 2.15rem; margin: 0 0 .25rem 0; color: white;}
      .hero p {font-size: 1rem; margin: 0; opacity: .9;}
      .note {background:#eff6ff; border-left:4px solid #2563eb; padding:.8rem 1rem;
        border-radius:8px; color:#1e3a8a;}
      .small-label {font-size:.78rem; color:#64748b; text-transform:uppercase;
        letter-spacing:.08em; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    """Load prepared Parquet once per application process."""

    return load_processed_dataset()


@st.cache_resource(show_spinner="Training the calendar demand model…")
def get_demand_model() -> DemandModelArtifact:
    """Train and cache the predictive feature."""

    return train_demand_model(get_data())


@st.cache_resource(show_spinner="Scoring transaction anomalies…")
def get_anomalies() -> AnomalyArtifact:
    """Fit and cache the unsupervised anomaly detector."""

    return detect_anomalies(get_data(), contamination=0.01)


def money(value: float) -> str:
    """Format a simulated USD value for dashboard cards."""

    return f"${value:,.0f}"


def compact_money(value: float) -> str:
    """Format large values with compact M/K suffixes."""

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return money(value)


def report_content(frame: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Build grounded metrics and highlights for PDF/DOCX exports."""

    country = frame.groupby("country")["order_value"].sum().idxmax()
    category = frame.groupby("product_category", observed=True)["order_value"].sum().idxmax()
    monthly = frame.assign(month=frame["date"].dt.strftime("%b")).groupby("month")[
        "order_value"
    ].sum()
    q4_share = frame.loc[frame["quarter"].eq(4), "order_value"].sum() / frame[
        "order_value"
    ].sum()
    metrics = {
        "Filtered revenue": money(float(frame["order_value"].sum())),
        "Orders": f"{len(frame):,}",
        "Customers": f"{frame['customer_id'].nunique():,}",
        "Units": f"{int(frame['quantity'].sum()):,}",
        "Average order value": f"${frame['order_value'].mean():,.2f}",
        "Date range": f"{frame['date'].min().date()} to {frame['date'].max().date()}",
    }
    insights = [
        f"{country} leads revenue inside the current filter selection.",
        f"{category} is the strongest derived product category by revenue.",
        f"{monthly.idxmax()} is the strongest displayed month by revenue.",
        f"Q4 contributes {q4_share:.1%} of revenue in the current selection.",
        "All financial figures are synthetic and order value equals price multiplied by quantity.",
    ]
    return metrics, insights


def chart_downloads(figure, frame: pd.DataFrame, key: str) -> None:
    """Render interactive-chart and data downloads, with optional PNG generation."""

    one, two, three = st.columns(3)
    one.download_button(
        "Download chart (HTML)",
        export_chart_html(figure),
        file_name=f"{key}.html",
        mime="text/html",
        key=f"html-{key}",
    )
    two.download_button(
        "Download data (CSV)",
        export_frame_csv(frame),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"csv-{key}",
    )
    if three.button("Prepare PNG", key=f"prepare-png-{key}"):
        try:
            st.session_state[f"png-{key}"] = export_chart_png(figure)
        except ExportDependencyError as exc:
            st.warning(str(exc))
    if st.session_state.get(f"png-{key}"):
        three.download_button(
            "Download PNG",
            st.session_state[f"png-{key}"],
            file_name=f"{key}.png",
            mime="image/png",
            key=f"png-download-{key}",
        )


def llm_settings() -> LLMSettings:
    """Resolve provider settings from environment variables and optional Streamlit secrets."""

    settings = LLMSettings.from_environment()
    overrides: dict[str, str] = {}
    try:
        for field, key in (
            ("provider", "INSIGHTCOMMERCE_LLM_PROVIDER"),
            ("ollama_base_url", "OLLAMA_BASE_URL"),
            ("ollama_model", "OLLAMA_MODEL"),
            ("openai_api_key", "OPENAI_API_KEY"),
            ("openai_model", "OPENAI_MODEL"),
        ):
            if key in st.secrets:
                overrides[field] = str(st.secrets[key])
    except FileNotFoundError:
        pass
    return replace(settings, **overrides) if overrides else settings


data = get_data()

st.markdown(
    """
    <div class="hero">
      <div class="small-label" style="color:#bfdbfe">AI-powered decision intelligence</div>
      <h1>InsightCommerce</h1>
      <p>Explore 108,300 synthetic 2025 electronics transactions through fast filters,
      interactive visuals, safe natural-language queries, prediction, anomalies, and exports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters")
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()
    chosen_dates = st.date_input(
        "Order date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if not isinstance(chosen_dates, tuple) or len(chosen_dates) != 2:
        start_date, end_date = min_date, max_date
    else:
        start_date, end_date = chosen_dates
    country_values = sorted(data["country"].astype(str).unique())
    category_values = sorted(data["product_category"].astype(str).unique())
    countries = st.multiselect("Countries", country_values, placeholder="All countries")
    categories = st.multiselect("Product categories", category_values, placeholder="All categories")
    st.divider()
    st.caption("Synthetic dataset · Kaggle · CC BY-NC-SA 4.0")
    st.caption("Source names/emails never enter AI prompts or general exports.")

mask = data["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
if countries:
    mask &= data["country"].isin(countries)
if categories:
    mask &= data["product_category"].astype(str).isin(categories)
filtered = data.loc[mask].copy()

if filtered.empty:
    st.error("No transactions match these filters. Broaden the date, country, or category selection.")
    st.stop()

overview_tab, explorer_tab, ai_tab, advanced_tab, reports_tab = st.tabs(
    ["Overview", "Visual Explorer", "AI Analyst", "Prediction & Anomalies", "Reports & Quality"]
)

with overview_tab:
    revenue = float(filtered["order_value"].sum())
    orders = len(filtered)
    customers = int(filtered["customer_id"].nunique())
    units = int(filtered["quantity"].sum())
    average_order_value = float(filtered["order_value"].mean())
    top_country = filtered.groupby("country")["order_value"].sum().idxmax()
    cards = st.columns(6)
    cards[0].metric("Revenue", compact_money(revenue))
    cards[1].metric("Orders", f"{orders:,}")
    cards[2].metric("Customers", f"{customers:,}")
    cards[3].metric("Units", f"{units:,}")
    cards[4].metric("Avg. order value", f"${average_order_value:,.2f}")
    cards[5].metric("Top country", top_country)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(monthly_revenue_chart(filtered), use_container_width=True, key="overview-monthly")
    with right:
        st.plotly_chart(country_revenue_chart(filtered), use_container_width=True, key="overview-country")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(category_revenue_chart(filtered), use_container_width=True, key="overview-category")
    with right:
        st.plotly_chart(quarter_country_heatmap(filtered), use_container_width=True, key="overview-heatmap")

with explorer_tab:
    st.subheader("Eight interactive views")
    st.caption("Choose a prepared analysis or build a manual chart from the filtered records.")
    views = {
        "Monthly revenue — line": lambda: monthly_revenue_chart(filtered),
        "Country revenue — geographic map": lambda: country_revenue_chart(filtered),
        "Category revenue — bar": lambda: category_revenue_chart(filtered),
        "Country × quarter — heatmap": lambda: quarter_country_heatmap(filtered),
        "Category → product — treemap": lambda: product_treemap(filtered),
        "Price × order value — scatter": lambda: price_order_scatter(filtered),
        "Category price — box plot": lambda: category_price_box(filtered),
        "Order value — histogram": lambda: order_value_histogram(filtered),
    }
    view_name = st.selectbox("Prepared visualization", list(views))
    explorer_figure = views[view_name]()
    st.plotly_chart(explorer_figure, use_container_width=True, key="explorer-prepared")
    chart_downloads(explorer_figure, filtered.head(10_000), "insightcommerce-chart")

    with st.expander("Manual chart builder"):
        safe_columns = [
            column for column in filtered.columns if column not in {"customer_name", "customer_email"}
        ]
        numeric_columns = filtered[safe_columns].select_dtypes(include="number").columns.tolist()
        first, second, third, fourth = st.columns(4)
        manual_type = first.selectbox(
            "Chart type", [option for option in CHART_OPTIONS if option not in {"auto", "metric"}]
        )
        manual_x = second.selectbox("X", safe_columns, index=safe_columns.index("country"))
        manual_y = third.selectbox("Y / value", numeric_columns, index=numeric_columns.index("order_value"))
        color_options = [""] + safe_columns
        manual_color = fourth.selectbox("Color / heat value", color_options)
        manual_frame = filtered[safe_columns].sample(min(5_000, len(filtered)), random_state=42)
        manual_spec = ChartSpec(manual_type, manual_x, manual_y, manual_color)
        if not chart_is_compatible(manual_frame, manual_spec):
            st.info("That combination is not compatible; the safest compatible chart is shown.")
        manual_figure = build_chart(manual_frame, manual_spec, "Manual filtered exploration")
        st.plotly_chart(manual_figure, use_container_width=True, key="explorer-manual")

with ai_tab:
    st.subheader("Ask the data in plain English")
    st.markdown(
        '<div class="note">The AI receives a documented schema and up to five previous successful '
        "turns. Generated SQL must pass a read-only safety gate. One failed plan receives exactly one "
        "corrective retry.</div>",
        unsafe_allow_html=True,
    )
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory(max_turns=5)
    if "ai_question" not in st.session_state:
        st.session_state.ai_question = PRESET_INSIGHTS[0].question

    provider_label = st.radio(
        "Query planner",
        ["Offline demo", "Ollama (local LLM)", "OpenAI (hosted, optional)"],
        horizontal=True,
        help="Offline demo covers common questions without claiming to be a live model.",
    )
    preset_columns = st.columns(3)
    for index, preset in enumerate(PRESET_INSIGHTS):
        if preset_columns[index % 3].button(preset.label, key=f"preset-{index}"):
            st.session_state.ai_question = preset.question
    question = st.text_area("Question", key="ai_question", height=90)
    run_col, clear_col = st.columns([1, 1])
    run_query = run_col.button("Run safe analysis", type="primary", use_container_width=True)
    if clear_col.button("Clear five-turn memory", use_container_width=True):
        st.session_state.conversation_memory.clear()
        st.session_state.pop("last_ai_answer", None)
        st.rerun()

    if run_query:
        provider_name = {
            "Offline demo": "offline",
            "Ollama (local LLM)": "ollama",
            "OpenAI (hosted, optional)": "openai",
        }[provider_label]
        try:
            provider = build_provider(provider_name, llm_settings())
            assistant = AnalyticsAssistant(provider, memory=st.session_state.conversation_memory)
            with st.spinner("Planning, validating, and executing the query…"):
                st.session_state.last_ai_answer = assistant.ask(question)
        except (AnalyticsAssistantError, LLMProviderError, ValueError) as exc:
            st.error(str(exc))
            if provider_name != "offline":
                st.info("Use Offline demo for a no-credential walkthrough, or verify the selected LLM service.")

    answer = st.session_state.get("last_ai_answer")
    if answer is not None:
        st.success(answer.narrative)
        info_columns = st.columns(3)
        info_columns[0].metric("Execution", f"{answer.elapsed_ms:.2f} ms")
        info_columns[1].metric("Attempts", str(answer.attempts))
        info_columns[2].metric("Provider", answer.provider)
        st.dataframe(answer.frame, use_container_width=True, hide_index=True)
        suggested = recommend_chart(
            answer.frame,
            answer.plan.chart_type,
            answer.plan.x,
            answer.plan.y,
            answer.plan.color,
        )
        st.caption(
            f"AI recommendation: {suggested.chart_type}. You can override the chart and fields below."
        )
        columns = list(answer.frame.columns)
        numeric = answer.frame.select_dtypes(include="number").columns.tolist()
        control_a, control_b, control_c, control_d = st.columns(4)
        selected_type = control_a.selectbox(
            "Chart override",
            CHART_OPTIONS,
            index=CHART_OPTIONS.index(suggested.chart_type),
            key="ai-chart-type",
        )
        x_default = columns.index(suggested.x) if suggested.x in columns else 0
        selected_x = control_b.selectbox("X field", columns, index=x_default, key="ai-x")
        y_options = [""] + numeric
        y_default = y_options.index(suggested.y) if suggested.y in y_options else 0
        selected_y = control_c.selectbox("Y field", y_options, index=y_default, key="ai-y")
        color_options = [""] + columns
        color_default = color_options.index(suggested.color) if suggested.color in color_options else 0
        selected_color = control_d.selectbox(
            "Color / heat value", color_options, index=color_default, key="ai-color"
        )
        resolved_type = suggested.chart_type if selected_type == "auto" else selected_type
        manual_spec = ChartSpec(resolved_type, selected_x, selected_y, selected_color)
        if not chart_is_compatible(answer.frame, manual_spec):
            st.info("The override is incompatible with this result, so a safe automatic chart is used.")
        ai_figure = build_chart(answer.frame, manual_spec, answer.plan.title)
        st.plotly_chart(ai_figure, use_container_width=True, key="ai-result-chart")
        chart_downloads(ai_figure, answer.frame, "ai-analysis")
        with st.expander("Generated SQL and rationale"):
            st.code(answer.plan.sql, language="sql")
            st.write(answer.plan.rationale)
        st.caption(f"Memory contains {len(st.session_state.conversation_memory)} of 5 turns.")

with advanced_tab:
    prediction_tab, anomaly_tab = st.tabs(["Demand prediction", "Anomaly detection"])
    with prediction_tab:
        artifact = get_demand_model()
        st.subheader("Daily order-demand prediction")
        st.caption(
            "Random Forest uses known-in-advance calendar features. Evaluation holds out every seventh "
            "day across the year, so this is a capstone demonstration—not a production causal forecast."
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Held-out MAE", f"{artifact.metrics['mae_orders']:.1f} orders")
        metric_columns[1].metric("Held-out RMSE", f"{artifact.metrics['rmse_orders']:.1f} orders")
        metric_columns[2].metric("Held-out R²", f"{artifact.metrics['r2']:.3f}")
        metric_columns[3].metric("Test days", f"{int(artifact.metrics['test_days'])}")
        selected_prediction_date = st.date_input(
            "Date to forecast", value=date(2026, 11, 15), min_value=date(2025, 1, 1)
        )
        forecast = predict_daily_demand(artifact, selected_prediction_date)
        left, right = st.columns(2)
        left.metric("Predicted daily orders", f"{int(forecast['predicted_orders']):,}")
        right.metric("Indicative revenue", money(float(forecast["indicative_revenue"])))
        st.caption(
            f"Indicative revenue applies the historical median order value of "
            f"${forecast['median_order_value_assumption']:,.2f}; it is not a separate revenue model."
        )
        daily_figure = px.line(
            artifact.daily_frame,
            x="date",
            y=["orders", "predicted_orders"],
            labels={"value": "Daily orders", "variable": "Series"},
            title="Actual vs fitted daily orders",
        )
        daily_figure.update_layout(template="plotly_white")
        importance_figure = px.bar(
            artifact.feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Demand-model feature importance",
        )
        importance_figure.update_layout(template="plotly_white")
        left, right = st.columns(2)
        left.plotly_chart(daily_figure, use_container_width=True, key="prediction-daily")
        right.plotly_chart(importance_figure, use_container_width=True, key="prediction-importance")

    with anomaly_tab:
        anomaly_artifact = get_anomalies()
        anomaly_frame = anomaly_artifact.anomalies
        anomaly_mask = anomaly_frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        if countries:
            anomaly_mask &= anomaly_frame["country"].isin(countries)
        if categories:
            anomaly_mask &= anomaly_frame["product_category"].astype(str).isin(categories)
        visible_anomalies = anomaly_frame.loc[anomaly_mask]
        st.subheader("Unusual transaction combinations")
        st.caption(
            "Isolation Forest flags the top 1% of unusual price, quantity, order-value, and "
            "product-relative combinations. Flags are investigative signals, not fraud labels."
        )
        anomaly_cards = st.columns(4)
        anomaly_cards[0].metric("Rows scored", f"{int(anomaly_artifact.metrics['rows_scored']):,}")
        anomaly_cards[1].metric("Global anomalies", f"{int(anomaly_artifact.metrics['anomalies']):,}")
        anomaly_cards[2].metric("Global rate", f"{anomaly_artifact.metrics['anomaly_rate']:.1%}")
        anomaly_cards[3].metric("Visible after filters", f"{len(visible_anomalies):,}")
        anomaly_figure = px.scatter(
            visible_anomalies.head(1_000),
            x="order_value",
            y="anomaly_score",
            color="product_category",
            size="quantity",
            hover_data=["order_id", "country", "product", "anomaly_reason"],
            title="Anomaly score vs order value",
        )
        anomaly_figure.update_layout(template="plotly_white")
        st.plotly_chart(anomaly_figure, use_container_width=True, key="anomaly-scatter")
        st.dataframe(visible_anomalies.head(250), use_container_width=True, hide_index=True)
        st.download_button(
            "Download anomalies (CSV)",
            export_frame_csv(visible_anomalies),
            file_name="insightcommerce-anomalies.csv",
            mime="text/csv",
        )

with reports_tab:
    quality_tab, export_tab = st.tabs(["Schema & data quality", "Report exports"])
    with quality_tab:
        st.subheader("Verified source and generated schema")
        quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
        quality_cards = st.columns(5)
        quality_cards[0].metric("Rows", f"{quality['rows']:,}")
        quality_cards[1].metric("Source columns", "10")
        quality_cards[2].metric("Missing cells", f"{quality['missing_cells']:,}")
        quality_cards[3].metric("Duplicate IDs", f"{quality['duplicate_order_ids']:,}")
        quality_cards[4].metric("Quality status", quality["quality_status"])
        st.json(quality, expanded=False)
        schema_frame = pd.DataFrame(schema_catalog())
        st.dataframe(schema_frame, use_container_width=True, hide_index=True)
        st.info(
            "The real source schema has no profit, discount, category, shipping, or cancellation "
            "field. The app does not invent them; only calendar fields and a documented product "
            "taxonomy are derived."
        )

    with export_tab:
        st.subheader("Filtered management summary")
        metrics, insights = report_content(filtered)
        st.write(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
        for insight in insights:
            st.write(f"• {insight}")
        preview = (
            filtered.groupby("country", as_index=False)
            .agg(revenue=("order_value", "sum"), orders=("order_id", "count"))
            .sort_values("revenue", ascending=False)
        )
        pdf_bytes = export_summary_pdf("InsightCommerce Analytics Summary", metrics, insights, preview)
        docx_bytes = export_summary_docx("InsightCommerce Analytics Summary", metrics, insights, preview)
        left, right = st.columns(2)
        left.download_button(
            "Download PDF summary",
            pdf_bytes,
            file_name="insightcommerce-summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        right.download_button(
            "Download editable DOCX",
            docx_bytes,
            file_name="insightcommerce-summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

st.caption(
    "InsightCommerce v1.0 · Data values are synthetic · Built for transparent academic evaluation"
)
