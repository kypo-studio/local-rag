# About this RAG

This is a sample document used to test the retrieval pipeline.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a retriever
(which finds relevant passages from a knowledge base) with a language model
(which generates a fluent answer grounded in those passages).

## How this project works

1. **Ingestion**: Documents (PDF, Markdown, text) are loaded, split into
   overlapping chunks, embedded with `sentence-transformers`, and stored in
   a local ChromaDB collection.
2. **Retrieval**: At query time, the question is embedded with the same model,
   and the top-k most similar chunks are returned by cosine similarity.
3. **Generation**: The retrieved chunks are passed as context to an LLM
   (Ollama locally by default), which produces a grounded answer.

## Stack

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: ChromaDB (persistent, local)
- LLM: Ollama, OpenAI, or Groq (configurable)
- API: FastAPI
