"""
tests/test_llm_factory.py — provider selection.

Environment: LOCAL (Mac). No network: constructing ChatOpenAI makes no API call.
"""
import pytest

import agents.llm_factory as factory
import config


def test_ollama_provider_uses_local_endpoint(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
    llm = factory.make_chat_model("gpt-4o-mini")
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_ollama_provider_uses_configured_model_not_requested_name(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "llama3.2:3b")
    assert factory.make_chat_model("gpt-4o-mini").model_name == "llama3.2:3b"


def test_openai_provider_uses_requested_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-fake")
    assert factory.make_chat_model("gpt-4o-mini").model_name == "gpt-4o-mini"


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", " Ollama ")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "llama3.2:3b")
    assert factory.make_chat_model("x").model_name == "llama3.2:3b"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        factory.make_chat_model("gpt-4o-mini")