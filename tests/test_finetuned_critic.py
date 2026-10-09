"""
tests/test_finetuned_critic.py -- serving the fine-tuned Critic. No model needed.
Environment: LOCAL (Mac).
"""
from types import SimpleNamespace

import pytest

from agents import critic as critic_mod
from agents.critic import (
    JSON_FORMAT_INSTRUCTION, JsonTextJudge, critic_node, make_critic_llm, parse_judgment,
)
from agents.llm_factory import make_chat_model
from tests.test_critic import _decision, _state


def test_parse_judgment_reads_the_trained_format():
    j = parse_judgment('{"reason": "rule 7: drop_rows_missing affected 12.7% > 10%", "verdict": "reject"}')
    assert j.verdict == "reject"
    assert j.reason.startswith("rule 7")


def test_parse_judgment_accepts_either_field_order_and_loose_casing():
    j = parse_judgment('{"verdict": "Accept.", "reason": "rule 2: lossless fix"}')
    assert j.verdict == "accept"


def test_parse_judgment_without_a_verdict_is_malformed():
    with pytest.raises(ValueError):
        parse_judgment("I think this transformation looks fine.")


class _FakeChatModel:
    def __init__(self, text):
        self.text, self.messages = text, None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.text)


def test_json_text_judge_appends_the_training_instruction():
    chat = _FakeChatModel('{"reason": "rule 1: no_action leaves the issue unfixed", "verdict": "reject"}')
    judgment = JsonTextJudge(chat).invoke([("system", "S"), ("human", "EVIDENCE")])
    assert judgment.verdict == "reject"
    assert chat.messages[-1] == ("human", "EVIDENCE" + JSON_FORMAT_INSTRUCTION)
    assert chat.messages[0] == ("system", "S")  # system prompt untouched


def test_critic_model_setting_selects_the_finetuned_model(monkeypatch):
    monkeypatch.setattr("config.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("config.CRITIC_MODEL", "critic-ft")
    llm = make_critic_llm("gpt-4o-mini")
    assert isinstance(llm, JsonTextJudge)
    assert llm.chat_model.model_name == "critic-ft"


def test_critic_model_needs_ollama(monkeypatch):
    monkeypatch.setattr("config.LLM_PROVIDER", "openai")
    monkeypatch.setattr("config.CRITIC_MODEL", "critic-ft")
    with pytest.raises(ValueError, match="LLM_PROVIDER=ollama"):
        make_critic_llm("gpt-4o-mini")


def test_ollama_model_override_only_affects_that_call(monkeypatch):
    monkeypatch.setattr("config.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("config.OLLAMA_MODEL", "llama3.2:3b")
    assert make_chat_model("x", ollama_model="critic-ft").model_name == "critic-ft"
    assert make_chat_model("x").model_name == "llama3.2:3b"


def test_finetuned_critic_never_uses_retrieval(monkeypatch):
    monkeypatch.setattr("config.CRITIC_MODEL", "critic-ft")
    calls = []
    monkeypatch.setattr(critic_mod, "_get_retriever", lambda: calls.append(1))
    out = critic_node(_state([_decision("clean_r1_001")]))
    assert calls == []  # retrieval was never even requested
    assert out["critic"]["current_verdict"]["verdict"] == "accept"  # conftest's fake judge