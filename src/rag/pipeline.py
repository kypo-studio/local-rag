"""End-to-end RAG pipeline: retrieve → format → generate."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.generator import build_prompt, get_llm
from rag.retriever import RetrievedChunk, retrieve


@dataclass
class Source:
    filename: str
    chunk_index: int
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[Source] = field(default_factory=list)


def _format_context_block(chunk: RetrievedChunk) -> str:
    return f"[{chunk.filename} #{chunk.chunk_index}]\n{chunk.text}"


def answer_question(question: str, top_k: int | None = None) -> Answer:
    chunks = retrieve(question, top_k=top_k)
    context_blocks = [_format_context_block(c) for c in chunks]
    prompt = build_prompt(question, context_blocks)
    llm = get_llm()
    text = llm.complete(prompt)
    sources = [
        Source(filename=c.filename, chunk_index=c.chunk_index, score=c.score)
        for c in chunks
    ]
    return Answer(answer=text, sources=sources)
