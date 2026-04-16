"""LLM generation layer. Supports Ollama (local), OpenAI, Groq, or none."""

from __future__ import annotations

from typing import Protocol

import httpx

from rag.config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. If the answer is not in the context, say that you don't know. "
    "Cite the sources you used by filename."
)


def build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in the same language as the question. Be concise."
    )


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class NoopLLM:
    """Fallback that returns the raw context (retrieval-only mode)."""

    def complete(self, prompt: str) -> str:
        return (
            "[LLM disabled — retrieval-only mode] "
            "Set LLM_PROVIDER in .env to 'ollama', 'openai', or 'groq' to enable generation."
        )


class OllamaLLM:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.llm_model

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{self.base_url}/api/generate", json=payload)
                r.raise_for_status()
                return r.json().get("response", "").strip()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed ({exc}). "
                f"Is Ollama running at {self.base_url}? "
                f"Install: https://ollama.com, then: `ollama pull {self.model}`."
            ) from exc


class OpenAICompatibleLLM:
    """Works with OpenAI and any OpenAI-compatible API (e.g. Groq)."""

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY or GROQ_API_KEY in your .env."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()


def get_llm() -> LLM:
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaLLM()
    if provider == "openai":
        return OpenAICompatibleLLM(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        )
    if provider == "groq":
        return OpenAICompatibleLLM(
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key,
            model=settings.llm_model,
        )
    return NoopLLM()
