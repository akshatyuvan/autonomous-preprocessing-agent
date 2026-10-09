"""
agents/llm_factory.py — the ONE place that knows which LLM endpoint is used.

Environment: LOCAL (Mac).

Every agent's _get_llm() calls make_chat_model(). Switching between a local
Ollama model (free) and the OpenAI API (paid) is a .env change, not a code change.
Interview line: "the provider is configuration; agents only ask for a chat model."
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI

import config


def make_chat_model(model: str, temperature: float = 0.0,
                    ollama_model: str | None = None) -> ChatOpenAI:
    # config attributes are read at CALL time (not import time) so tests can monkeypatch them.
    provider = config.LLM_PROVIDER.strip().lower()

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API, so the same ChatOpenAI client works.
        # The provider decides the actual model: state["metadata"]["llm_model"] names an
        # OpenAI model ("gpt-4o-mini") that doesn't exist locally, so OLLAMA_MODEL wins.
        return ChatOpenAI(
            # ollama_model lets ONE agent (the fine-tuned Critic) use a different local
            # model than the rest of the pipeline; everyone else gets OLLAMA_MODEL.
            model=ollama_model or config.OLLAMA_MODEL,
            temperature=temperature,
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",   # required by the client, ignored by Ollama
            timeout=120,        # local CPU inference is slower than a hosted API
            max_retries=1,
        )

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=config.OPENAI_API_KEY or None,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' (expected 'ollama' or 'openai')")