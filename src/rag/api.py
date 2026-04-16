"""FastAPI application exposing the RAG pipeline over HTTP."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag import __version__
from rag.ingest import SUPPORTED_SUFFIXES, ingest_path, reset_collection
from rag.pipeline import answer_question
from rag.store import get_collection

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    version=__version__,
    description="A simple, local-first retrieval-augmented generation API.",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    filename: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class IngestResponse(BaseModel):
    chunks_added: int


class StatsResponse(BaseModel):
    collection: str
    count: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    col = get_collection()
    return StatsResponse(collection=col.name, count=col.count())


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(req.question, top_k=req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(
        answer=result.answer,
        sources=[SourceOut(**s.__dict__) for s in result.sources],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        added = ingest_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(chunks_added=added)


@app.delete("/collection")
def delete_collection() -> dict[str, str]:
    reset_collection()
    return {"status": "reset"}
