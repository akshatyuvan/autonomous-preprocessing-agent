"""
tests/conftest.py -- shared fixtures, loaded automatically by pytest before any test.

Environment: LOCAL (Mac).
"""
import json

import pytest


class _FakeProfilerLLM:
    """Stands in for ChatOpenAI(...).with_structured_output(ProfileSynthesis)."""

    def invoke(self, messages):
        from agents.profiler import ProfileSynthesis
        return ProfileSynthesis(summary="[fake] profile summary", top_concerns=[])


class _FakeCleanerLLM:
    """Always picks the FIRST allowed strategy: deterministic, offline, and it
    exercises the real LLM code path (choice -> validation -> dispatch)."""

    def invoke(self, messages):
        from agents.cleaner import StrategyChoice
        payload = json.loads(messages[-1][1])
        return StrategyChoice(strategy=payload["allowed_strategies"][0]["key"],
                              fill_value="", justification="[fake] first allowed option")

class _FakeCriticLLM:
    """Accepts everything, so graph tests flow end to end without a model."""

    def invoke(self, messages):
        from agents.critic import DecisionJudgment
        return DecisionJudgment(verdict="accept", reason="[fake] looks fine")

@pytest.fixture(autouse=True)
def _no_real_llm_in_profiler(monkeypatch):
    # autouse=True applies this to EVERY test, so no test can call a real model.
    # Tests needing specific LLM behaviour patch _get_llm again; the later patch wins.
    monkeypatch.setattr("agents.profiler._get_llm", lambda *args, **kwargs: _FakeProfilerLLM())


@pytest.fixture(autouse=True)
def _no_real_llm_in_cleaner(monkeypatch):
    monkeypatch.setattr("agents.cleaner._get_llm", lambda *args, **kwargs: _FakeCleanerLLM())


@pytest.fixture(autouse=True)
def _snapshots_go_to_tmp(monkeypatch, tmp_path):
    # Without this, every test run would write parquet files into the real data/cleaned/.
    monkeypatch.setattr("config.CLEANED_DIR", str(tmp_path / "cleaned"))


@pytest.fixture(autouse=True)
def _no_real_llm_in_critic(monkeypatch):
    monkeypatch.setattr("agents.critic._get_llm", lambda *args, **kwargs: _FakeCriticLLM())


@pytest.fixture(autouse=True)
def _no_retrieval_in_critic(monkeypatch):
    # Even if .env sets CRITIC_RETRIEVAL=true, tests never build the real store
    # (which would download an embedding model).
    monkeypatch.setattr("agents.critic._get_retriever", lambda: None)


class _FakeStageLLM:
    """Picks the FIRST allowed option for any registry-driven stage."""

    def invoke(self, messages):
        from agents.stage_runner import StageChoice
        payload = json.loads(messages[-1][1])
        return StageChoice(strategy=payload["allowed_strategies"][0]["key"],
                           justification="[fake] first allowed option")


@pytest.fixture(autouse=True)
def _no_real_llm_in_stages(monkeypatch):
    monkeypatch.setattr("agents.stage_runner._get_llm", lambda *args, **kwargs: _FakeStageLLM())


@pytest.fixture(autouse=True)
def _prompted_critic_by_default(monkeypatch):
    # Even if .env sets CRITIC_MODEL=critic-ft, tests start from the prompted Critic.
    # Tests for the fine-tuned path set it themselves (their later patch wins).
    monkeypatch.setattr("config.CRITIC_MODEL", "")