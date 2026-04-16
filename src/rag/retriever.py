"""Vector similarity search against the Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import settings
from rag.store import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    filename: str
    chunk_index: int
    score: float  # similarity in [0, 1] (1 = identical)


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the top_k most similar chunks for *query*."""
    if not query.strip():
        return []

    k = top_k or settings.top_k
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=k)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    # Chroma returns cosine *distance*; similarity = 1 - distance.
    distances = result.get("distances", [[]])[0] or [0.0] * len(docs)

    chunks: list[RetrievedChunk] = []
    for text, meta, dist in zip(docs, metas, distances):
        chunks.append(
            RetrievedChunk(
                text=text,
                source=str(meta.get("source", "")),
                filename=str(meta.get("filename", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=max(0.0, 1.0 - float(dist)),
            )
        )
    return chunks
