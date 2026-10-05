"""
agents/llm_factory.py — the ONE place that knows which LLM endpoint is used.

Environment: LOCAL (Mac).

Every agent's _get_llm() calls make_chat_model(). Switching between GitHub
Models (free) and the OpenAI API (paid) is a .env change, not a code change.
Interview line: "the provider is configuration; agents only know the model name."
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

import config

# GitHub Models is OpenAI-compatible, so the same ChatOpenAI client works:
# only base_url and the credential differ.
GITHUB_MODELS_URL = "https://models.github.ai/inference"


def _github_model_name(model: str) -> str:
    # GitHub Models names models as "<publisher>/<model>", e.g. "openai/gpt-4o-mini".
    # Agents keep using the plain name from state["metadata"]["llm_model"].
    return model if "/" in model else f"openai/{model}"


def make_chat_model(model: str, temperature: float = 0.0) -> ChatOpenAI:
    # config attributes are read at CALL time (not import time) so tests can
    # monkeypatch them.
    provider = config.LLM_PROVIDER.strip().lower()

    if provider == "github":
        if not config.GITHUB_TOKEN:
            # Fail fast with a clear message instead of an opaque 401 later.
            # Agents catch this and use their deterministic fallback.
            raise ValueError("LLM_PROVIDER=github but GITHUB_TOKEN is not set in .env")
        return ChatOpenAI(
            model=_github_model_name(model),
            temperature=temperature,
            base_url=GITHUB_MODELS_URL,
            api_key=config.GITHUB_TOKEN,
            max_retries=3,  # free tier is rate-limited; retry 429s with backoff
        )

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=config.OPENAI_API_KEY or None,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' (expected 'github' or 'openai')")