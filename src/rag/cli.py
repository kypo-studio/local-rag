"""Command-line interface for local ingestion and querying."""

from __future__ import annotations

from pathlib import Path

import typer

from rag.ingest import ingest_path, reset_collection
from rag.pipeline import answer_question
from rag.store import get_collection

app = typer.Typer(help="RAG CLI — ingest documents and ask questions locally.")


@app.command()
def ingest(path: Path = typer.Argument(..., exists=True, help="File or directory to ingest.")) -> None:
    """Ingest a file or all supported files under a directory."""
    added = ingest_path(path)
    typer.echo(f"Ingested {added} chunks from {path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of chunks to retrieve."),
) -> None:
    """Ask a question against the indexed knowledge base."""
    result = answer_question(question, top_k=top_k)
    typer.echo("\n=== Answer ===")
    typer.echo(result.answer)
    if result.sources:
        typer.echo("\n=== Sources ===")
        for s in result.sources:
            typer.echo(f"  - {s.filename} (chunk {s.chunk_index}, score={s.score:.3f})")


@app.command()
def stats() -> None:
    """Show the number of indexed chunks."""
    col = get_collection()
    typer.echo(f"Collection '{col.name}': {col.count()} chunks")


@app.command()
def reset() -> None:
    """Drop all indexed data."""
    reset_collection()
    typer.echo("Collection reset.")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("rag.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
