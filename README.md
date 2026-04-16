# rag

A simple, local-first **Retrieval-Augmented Generation** pipeline.
Ingest PDFs and Markdown, index them with local embeddings, and query them via
a FastAPI endpoint or CLI.

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **Vector store**: ChromaDB (persistent, local)
- **LLM**: Ollama (local, free) — or OpenAI / Groq (configurable)
- **API**: FastAPI
- **CLI**: `rag` (ingest, ask, serve, stats, reset)

## Quickstart

### 1. Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### 2. (Optional) Install Ollama for local LLM

[Download Ollama](https://ollama.com), then pull a small model:

```bash
ollama pull llama3.2:3b
```

If you prefer a hosted free option instead, use Groq:

```bash
cp .env.example .env
# edit .env → LLM_PROVIDER=groq, GROQ_API_KEY=gsk_..., LLM_MODEL=llama-3.1-8b-instant
```

### 3. Ingest documents

Drop PDFs / Markdown / text files into `data/` (or any directory), then:

```bash
uv run rag ingest data/
```

### 4. Ask questions

**CLI:**

```bash
uv run rag ask "What is this project about?"
```

**HTTP API:**

```bash
uv run rag serve
# then in another terminal:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project about?"}'
```

Interactive API docs at <http://localhost:8000/docs>.

## Endpoints

| Method | Path           | Description                                    |
| ------ | -------------- | ---------------------------------------------- |
| GET    | `/health`      | Liveness check                                 |
| GET    | `/stats`       | Number of indexed chunks                       |
| POST   | `/query`       | Ask a question (`{question, top_k?}`)          |
| POST   | `/ingest`      | Upload a file to index (multipart `file`)      |
| DELETE | `/collection`  | Drop all indexed data                          |

## Configuration

All settings are read from environment variables or a `.env` file.
See [`.env.example`](.env.example) for the full list.

| Variable              | Default                                         |
| --------------------- | ----------------------------------------------- |
| `LLM_PROVIDER`        | `ollama` (or `openai`, `groq`, `none`)          |
| `LLM_MODEL`           | `llama3.2:3b`                                   |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-MiniLM-L6-v2`        |
| `CHUNK_SIZE`          | `800`                                           |
| `CHUNK_OVERLAP`       | `120`                                           |
| `TOP_K`               | `4`                                             |
| `CHROMA_DIR`          | `./.chroma`                                     |

## Project layout

```
src/rag/
├── api.py         # FastAPI app
├── cli.py         # Typer CLI
├── config.py      # Settings (pydantic-settings)
├── generator.py   # LLM providers (Ollama / OpenAI / Groq)
├── ingest.py      # Load + chunk + upsert
├── pipeline.py    # retrieve → prompt → generate
├── retriever.py   # Chroma similarity search
└── store.py       # Chroma client + embedding function
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## License

MIT
