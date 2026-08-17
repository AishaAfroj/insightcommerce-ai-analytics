# InsightCommerce presentation and testing guide

This guide provides a reliable 8–10 minute demonstration of every capstone feature.
The application works without an API key by using the clearly labelled **Offline demo**
query planner.

## 1. Start the application on the Mac

### Fastest method

Open the project folder in Finder and double-click `start_app.command`. Keep the Terminal
window open while presenting. The browser should open automatically at
`http://localhost:8501`.

If macOS asks for confirmation, choose **Open**. If double-clicking is blocked, open
Terminal and run:

```bash
cd /Users/aishaafroj/.codex/.chatgpt-projects/g-p-6a73a12238308191a280cb6e3d061809/insightcommerce-ai-analytics
./start_app.command
```

Stop the application after the presentation by selecting its Terminal window and pressing
`Control + C`.

## 2. Five-minute pre-presentation check

Run these commands once before leaving for the presentation:

```bash
cd /Users/aishaafroj/.codex/.chatgpt-projects/g-p-6a73a12238308191a280cb6e3d061809/insightcommerce-ai-analytics
source .venv/bin/activate
python scripts/prepare_data.py
pytest -q
python scripts/benchmark_performance.py
python scripts/run_question_benchmark.py
```

Expected evidence:

- dataset: 108,300 rows and 10 verified source columns;
- quality: zero missing cells and zero duplicate order IDs;
- tests: 21 passed;
- filtered aggregation: far below the 500 ms requirement (about 2–3 ms on this Mac);
- AI benchmark: 10/10 questions passed.

Then start the app with `./start_app.command` and use **Offline demo** for the most
reliable presentation. Do not depend on venue Wi-Fi or a live model.

## 3. Recommended presentation flow

### A. Dataset and backend — 60 seconds

1. Open **Reports & Quality** → **Schema & data quality**.
2. Point out 108,300 rows, 10 source columns, zero missing cells, zero duplicate IDs,
   and the PASS status.
3. Expand the quality JSON and show the schema table.
4. Explain that the downloaded CSV was inspected before coding and that the app does
   not invent profit, discount, shipping, cancellation, or category source fields.
5. Mention that calendar fields and a documented product taxonomy are the only derived
   fields, and DuckDB serves filtered aggregations from optimized Parquet.

Suggested line: “The implementation follows the real file rather than assuming a
Superstore-style schema.”

### B. Overview and global filters — 60 seconds

1. Return to **Overview**.
2. Show revenue, orders, customers, units, average order value, and top country.
3. In the left sidebar, select one country such as **USA**.
4. Confirm that the KPIs, monthly line, map, category bar, and heatmap all update.
5. Clear the country filter before continuing.

### C. Interactive visualizations and export — 90 seconds

1. Open **Visual Explorer**.
2. Use the prepared-view menu to show at least six of the eight forms: line, map, bar,
   heatmap, treemap, scatter, box plot, and histogram.
3. Hover over marks and use Plotly zoom/reset controls to demonstrate interactivity.
4. Download the interactive HTML chart and the filtered CSV.
5. For a PNG, use the camera icon in the Plotly toolbar. The separate **Prepare PNG**
   button needs a local Chrome installation; HTML and toolbar PNG export work without it.
6. Expand **Manual chart builder**, change the chart type or fields, and show that an
   incompatible selection safely falls back to a compatible chart.

### D. Schema-aware AI query and chart override — 2 minutes

1. Open **AI Analyst** and keep **Offline demo** selected.
2. Click the preset for the monthly revenue trend, then **Run safe analysis**.
3. Point out the grounded explanation, execution time, one attempt, result table, and
   recommended chart.
4. Change **Chart override** from line to bar to demonstrate human control.
5. Expand **Generated SQL and rationale**. Explain that the model proposes a strict JSON
   plan, but SQL executes only after a read-only AST safety check, a 5,000-row cap, and a
   five-second timeout. Exactly one corrective retry is allowed.
6. Ask these questions in sequence to demonstrate conversational memory:

   1. `What is the total revenue?`
   2. `How many orders are there?`
   3. `How many units were sold?`
   4. `Show the monthly revenue trend for 2025.`
   5. `Which countries generated the most revenue?`
   6. `Compare revenue across product categories.`

   After the sixth successful question, the caption should still show **5 of 5 turns**,
   proving that only the latest five are retained. Use **Clear five-turn memory** when done.

Suggested line: “The LLM is a planner, not the authority—the validator and executed data
result are authoritative.”

### E. Predictive analytics — 60 seconds

1. Open **Prediction & Anomalies** → **Demand prediction**.
2. Show the held-out MAE, RMSE, R-squared, and 53 test days.
3. Select a future date such as **15 November 2026**.
4. Show predicted daily orders and clearly label indicative revenue as predicted orders
   multiplied by historical median order value—not a second learned model.
5. Point out the actual-versus-fitted chart and feature importance.

Important limitation to say: the high R-squared reflects the synthetic dataset’s designed
Q4 jump and should not be generalized to a real retailer.

### F. Anomaly detection — 60 seconds

1. Open **Anomaly detection**.
2. Show 108,300 scored rows, 1,006 flagged records, and the approximate 0.93% rate.
3. Hover over the scatter plot and show the transparent reason codes in the table.
4. Download the anomaly CSV.
5. State that a flag is an investigative signal, not proof of fraud or data error.

### G. Reports and final evidence — 45 seconds

1. Open **Reports & Quality** → **Report exports**.
2. Download the PDF summary and editable DOCX.
3. Mention that chart HTML/PNG and CSV exports were demonstrated earlier.
4. Finish with the repository README, architecture diagram, benchmark evidence, and
   15-page final capstone report.

## 4. What to show if asked about safety

- customer names and emails are omitted from AI prompts, anomaly output, and general exports;
- only one `SELECT`/CTE statement is allowed;
- mutation, multiple statements, external files, hidden schemas, unrestricted `SELECT *`,
  and sensitive fields are rejected;
- execution is capped at 5,000 rows, five seconds, and a 1 GB DuckDB memory limit;
- only successful answers enter the five-turn memory;
- the test suite contains malicious SQL cases and a one-retry recovery case.

## 5. Live-model options

Use **Offline demo** during the presentation unless a live-model demonstration is required.
It is deterministic and requires no credentials.

For Ollama, start the local service and make sure `llama3.1:8b` is installed before opening
the app. For OpenAI, set `OPENAI_API_KEY` outside Git and select the hosted provider. Never
display or commit an API key.

## 6. Troubleshooting

- **Browser does not open:** visit `http://localhost:8501` manually.
- **Port already in use:** stop the old Terminal process with `Control + C`, then restart.
- **Live AI provider fails:** switch immediately to **Offline demo**.
- **PNG preparation warns about Chrome:** use the Plotly camera icon or download HTML.
- **Filters produce no rows:** clear country/category selections and restore the full date range.
- **App looks stale:** refresh the browser once; Streamlit preserves session state during reruns.

The strongest presentation sequence is: verified data → fast dashboard → safe AI query →
manual chart control → prediction → anomaly investigation → downloadable evidence.
