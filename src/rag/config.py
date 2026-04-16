"""Configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the RAG pipeline.

    Values are loaded from environment variables or a `.env` file at the repo root.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage ---
    data_dir: Path = Field(default=ROOT_DIR / "data")
    chroma_dir: Path = Field(default=ROOT_DIR / ".chroma")
    collection_name: str = Field(default="documents")

    # --- Embeddings (local, free) ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- Chunking ---
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)

    # --- Retrieval ---
    top_k: int = Field(default=4)

    # --- LLM ---
    # Supported providers: "ollama" (local, free), "openai" (API key required),
    # "groq" (API key required, free tier), or "none" (retrieval only).
    llm_provider: Literal["ollama", "openai", "groq", "none"] = Field(default="ollama")
    llm_model: str = Field(default="llama3.2:3b")
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=512)

    # Provider-specific endpoints / keys
    ollama_base_url: str = Field(default="http://localhost:11434")
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    groq_api_key: str | None = Field(default=None)
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")


settings = Settings()
