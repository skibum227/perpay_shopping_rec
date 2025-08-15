# LLM + Cosine Similarity Product Finder (Dockerized)

Three containers:
- **ollama** — runs a local LLM with an OpenAI-compatible API.
- **backend** — FastAPI service that does TF–IDF + cosine similarity over your product catalog.
- **frontend** — Streamlit UI that:
  1) sends user text to the LLM to extract compact keywords,
  2) calls the backend to fetch top-5 matches,
  3) asks the LLM to draft a friendly summary of those results.

## Prereqs
- Docker & Docker Compose

## Quick Start

1) **Place your catalog** at `./data/product_catalog.csv` (a sample is included). Default title column is `name`.
   - Change the column name via `NAME_COL` in `docker-compose.yml` if needed.

2) **Start the stack**:
```bash
docker compose up --build -d
```

3) **Pull a model inside Ollama** (one-time):
```bash
docker compose exec ollama ollama pull ${OLLAMA_MODEL}
```
> Configure the model in `.env` (default: `qwen3:0.6b`). Alternatives: `qwen2:0.5b-instruct`, `llama3:8b-instruct`.

4) **Open the UI**: http://localhost:8501

5) **Try it**: e.g., _"i am looking for expensive laptops"_ and click **Search**.

## How It Works

**Flow:** UI text → LLM (keywords) → Backend (TF–IDF cosine top-5) → UI → LLM (summary).

- **Backend** builds a TF–IDF index over your product title column at startup and exposes `/search`.
- **Frontend** calls Ollama's OpenAI-compatible `/v1` API to:
  - Condense the user's sentence into compact keywords.
  - Summarize the top 5 results into a readable blurb.
- **Ollama** hosts the local model on `http://ollama:11434`.

## Backend API

- `GET /healthz` → `{ status, rows, name_col }`
- `POST /search`
  ```json
  { "query": "nike red shoe", "top_k": 5 }
  ```
  returns
  ```json
  { "items": [ { "similarity": 0.93, "name": "...", "brand": "...", "product_id": "..." }, ... ] }
  ```

## Useful Commands
```bash
# Logs
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f ollama

# Manual health checks
curl http://localhost:8000/healthz

# Manual search (bypass UI)
curl -s -X POST http://localhost:8000/search   -H "Content-Type: application/json"   -d '{"query":"apple laptop", "top_k":5}' | jq

# Pull different models
docker compose exec ollama ollama pull qwen2:0.5b-instruct
docker compose exec ollama ollama pull llama3:8b-instruct
```

## Troubleshooting
- **LLM errors / 404s**: Make sure the model is pulled (`docker compose exec ollama ollama pull <model>`).
- **CSV not found**: verify `./data/product_catalog.csv` exists; update `CSV_PATH`/`NAME_COL` if needed.
- **Slow**: choose a smaller model in `.env`.

## Next Steps
- Add auth, request logging, metrics.
- Persist models across environments by pre-baking an Ollama image with `ollama pull` in CI.
- Deploy these same containers to Cloud Run or GKE.
