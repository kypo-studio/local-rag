"""Unit tests for the ingestion helpers (no LLM, no network)."""

from __future__ import annotations

from pathlib import Path

from rag.ingest import chunk_text, load_document


def test_chunk_text_splits_into_nonempty_pieces() -> None:
    text = ("Hello world. " * 400).strip()
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_short_input_returns_single_chunk() -> None:
    chunks = chunk_text("Short text.")
    assert chunks == ["Short text."]


def test_load_markdown(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome content.\n", encoding="utf-8")
    assert "Some content" in load_document(p)
