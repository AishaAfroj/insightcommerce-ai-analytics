# Security and privacy design

## Generated-query controls

- SQL is parsed with `sqlglot`; text matching alone is not the safety boundary.
- Exactly one `SELECT`/CTE statement is allowed.
- Mutation, DDL, commands, external files, URLs, extensions, and environment
  functions are blocked.
- Only the in-memory `orders` table and named CTEs are permitted.
- `customer_name`, `customer_email`, and unrestricted `SELECT *` are rejected.
- Results are wrapped with a 5,000-row limit and interrupted after five seconds.
- DuckDB external access is disabled after the trusted Parquet table is loaded.
- A failed model plan receives one corrective retry and no third attempt.

## Data handling

The Kaggle data is synthetic. Names and email addresses are nevertheless treated
as sensitive-shaped fields: they do not enter LLM prompts, AI query results,
anomaly output, benchmark artifacts, or general exports.

## Secret handling

API keys belong in environment variables or Streamlit secrets. `.env` and the
real `.streamlit/secrets.toml` are ignored by Git. Example files contain no keys.

