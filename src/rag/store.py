"""ChromaDB persistent client + embedding function wiring."""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import settings


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


@lru_cache(maxsize=1)
def _embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


def get_collection() -> Collection:
    """Return the Chroma collection, creating it if missing."""
    return _client().get_or_create_collection(
        name=settings.collection_name,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop and recreate the collection."""
    client = _client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
