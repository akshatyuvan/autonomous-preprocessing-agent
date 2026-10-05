"""
tests/test_llm_factory.py — provider selection.

Environment: LOCAL (Mac). No network: constructing ChatOpenAI makes no API call.
"""
import pytest

import agents.llm_factory as factory
import config


def test_github_provider_uses_github_endpoint(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "github")
    monkeypatch.setattr(config, "GITHUB_TOKEN", "fake-token")
    llm = factory.make_chat_model("gpt-4o-mini")
    assert llm.model_name == "openai/gpt-4o-mini"
    assert llm.openai_api_base == factory.GITHUB_MODELS_URL


def test_github_keeps_already_prefixed_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "github")
    monkeypatch.setattr(config, "GITHUB_TOKEN", "fake-token")
    assert factory.make_chat_model("openai/gpt-4o-mini").model_name == "openai/gpt-4o-mini"


def test_openai_provider_uses_plain_model_name(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-fake")
    llm = factory.make_chat_model("gpt-4o-mini")
    assert llm.model_name == "gpt-4o-mini"
    assert llm.openai_api_base != factory.GITHUB_MODELS_URL


def test_github_without_token_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "github")
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        factory.make_chat_model("gpt-4o-mini")


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        factory.make_chat_model("gpt-4o-mini")