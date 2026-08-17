# Deployment guide

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from `app.py` on the `main` branch.
3. Use Python 3.12. The included `runtime.txt` and pinned requirements are ready.
4. The app works immediately with **Offline demo** as the AI query planner.
5. For hosted live LLM calls, add `OPENAI_API_KEY` and `OPENAI_MODEL` to the app's
   encrypted secrets. Never commit a real key.

Local Ollama cannot be reached from Streamlit Community Cloud. Use Ollama only
for a local demonstration, or deploy both services inside a network you control.

## Docker

```bash
docker build -t insightcommerce .
docker run --rm -p 8501:8501 insightcommerce
```

To connect a local Ollama service from Docker, provide an environment-specific
`OLLAMA_BASE_URL` reachable from the container. Do not expose Ollama publicly
without authentication and network controls.

## Required deployment checks

- `/_stcore/health` returns HTTP 200.
- Overview charts load with all default filters.
- Offline demo returns the monthly revenue query.
- PDF and DOCX summary buttons download non-empty files.
- Anomaly and prediction tabs load without retraining on every rerun.
- Secrets do not appear in logs, screenshots, exports, or the repository.

