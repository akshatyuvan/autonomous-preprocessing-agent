"""
tests/test_retrieval.py -- BM25, RRF, the hybrid retriever and its Critic wiring.
Fake embedder only: no model download. Environment: LOCAL (Mac).
"""
import json
import math

from agents import critic as critic_mod
from agents.critic import DecisionJudgment, build_critic_messages, critic_node
from evaluation.critic_dataset import build_dataset
from tests.test_critic import _decision, _state
from vector_store.retrieval import (
    BM25, HybridRetriever, TRAIN_PATH, load_cases, reciprocal_rank_fusion, tokenize,
)


def fake_embedder(texts):
    """Deterministic bag-of-words vectors, normalised to length 1. No download."""
    vectors = []
    for text in texts:
        v = [0.0] * 64
        for token in tokenize(text):
            v[sum(map(ord, token)) % 64] += 1.0  # not hash(): that changes between runs
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors.append([x / norm for x in v])
    return vectors


def test_bm25_prefers_documents_with_the_query_term():
    docs = [tokenize("strategy drop_rows_missing rows 100"),
            tokenize("strategy impute_median rows 100"),
            tokenize("strategy cap_iqr rows 100")]
    scores = BM25(docs).scores(tokenize("drop_rows_missing"))
    assert scores[0] > 0
    assert scores[1] == 0 and scores[2] == 0


def test_rrf_rewards_agreement_between_retrievers():
    # y: 1/(60+2) + 1/(60+1) beats x: 1/(60+1) alone, which beats z: 1/(60+2) alone.
    assert reciprocal_rank_fusion([["x", "y"], ["y", "z"]]) == ["y", "x", "z"]


def test_rrf_breaks_ties_deterministically():
    # a and b get identical scores (1/61 + 1/62); the tie breaks by id.
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]]) == ["a", "b", "c", "d"]


def test_hybrid_retriever_finds_the_identical_case_first():
    cases, _ = build_dataset(n=24, seed=7)
    retriever = HybridRetriever(cases, fake_embedder)
    target = cases[5]
    ids = retriever.search(target["evidence"], top_k=4)
    assert ids[0] == target["id"]
    assert len(ids) == 4
    assert all("verdict" in text for text in retriever.retrieve(target["evidence"]))


def test_store_never_contains_test_examples():
    with open("evaluation/data/critic_test.jsonl") as f:
        test_ids = {json.loads(line)["id"] for line in f if line.strip()}
    train_ids = {c["id"] for c in load_cases(TRAIN_PATH)}
    assert train_ids and not train_ids & test_ids


def test_prompt_mentions_similar_cases_only_when_given():
    evidence = {"strategy": "no_action"}
    system, human = build_critic_messages(evidence, ["case A"])
    assert "similar_cases" in system[1]
    assert json.loads(human[1])["similar_cases"] == ["case A"]
    system, human = build_critic_messages(evidence)
    assert "similar_cases" not in system[1]
    assert "similar_cases" not in json.loads(human[1])


class _RecordingJudge:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return DecisionJudgment(verdict="accept", reason="ok")


class _FakeRetriever:
    def retrieve(self, evidence, top_k=4):
        return ["strategy impute_median | verdict accept | rule 8"]


def test_critic_node_passes_retrieved_cases_to_the_llm(monkeypatch):
    llm = _RecordingJudge()
    monkeypatch.setattr(critic_mod, "_get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(critic_mod, "_get_retriever", lambda: _FakeRetriever())
    critic_node(_state([_decision("clean_r1_001")]))
    human = json.loads(llm.messages[0][-1][1])
    assert human["similar_cases"] == ["strategy impute_median | verdict accept | rule 8"]