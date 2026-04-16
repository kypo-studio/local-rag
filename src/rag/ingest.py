"""Load documents from disk, chunk them, embed them, and store in ChromaDB."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.config import settings
from rag.store import get_collection

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    """Return the raw text of a supported document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _read_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path.name})")


def iter_documents(root: Path) -> Iterable[Path]:
    """Yield supported documents under *root*, recursively."""
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping chunks using a recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def _doc_id(source: str, idx: int, chunk: str) -> str:
    digest = hashlib.sha1(f"{source}:{idx}:{chunk}".encode("utf-8")).hexdigest()
    return f"{Path(source).name}:{idx}:{digest[:10]}"


def ingest_path(path: Path) -> int:
    """Ingest a single file or a directory. Returns the number of chunks added."""
    collection = get_collection()
    total = 0
    for doc_path in iter_documents(path):
        try:
            raw = load_document(doc_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping %s: %s", doc_path, exc)
            continue
        if not raw.strip():
            logger.warning("Skipping empty document: %s", doc_path)
            continue

        chunks = chunk_text(raw)
        if not chunks:
            continue

        source = str(doc_path.resolve())
        ids = [_doc_id(source, i, c) for i, c in enumerate(chunks)]
        metadatas = [
            {"source": source, "filename": doc_path.name, "chunk_index": i}
            for i, _ in enumerate(chunks)
        ]
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        total += len(chunks)
        logger.info("Ingested %s: %d chunks", doc_path.name, len(chunks))
    return total


def reset_collection() -> None:
    """Delete and recreate the collection (clears all indexed data)."""
    from rag.store import reset_collection as _reset

    _reset()
