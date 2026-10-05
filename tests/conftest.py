"""
tests/conftest.py — shared fixtures, loaded automatically by pytest before any test.

Environment: LOCAL (Mac).
"""
import pytest


class _FakeProfilerLLM:
    """Stands in for ChatOpenAI(...).with_structured_output(ProfileSynthesis)."""

    def invoke(self, messages):
        from agents.profiler import ProfileSynthesis
        return ProfileSynthesis(summary="[fake] profile summary", top_concerns=[])


@pytest.fixture(autouse=True)
def _no_real_llm_in_profiler(monkeypatch):
    # autouse=True applies this to EVERY test, so no test can accidentally call
    # OpenAI (locked decision #4). Tests needing specific LLM behaviour patch
    # _get_llm again themselves, and the later patch wins.
    # raising=False: until Day 2 lands, agents.profiler has no _get_llm yet.
    monkeypatch.setattr(
        "agents.profiler._get_llm", lambda *args, **kwargs: _FakeProfilerLLM(), raising=False
    )