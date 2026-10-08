"""
vector_store/retrieval.py -- hybrid retrieval of labelled past Critic cases (Step 6).

Store: the 300 TRAIN examples only. Test rows never enter it, so no leakage.
Two retrievers over the same documents:
  - dense:  ChromaDB nearest neighbours on sentence embeddings (overall similarity)
  - sparse: BM25 on tokens (exact matches, e.g. strategy names like "drop_rows_missing")
Fused with Reciprocal Rank Fusion:  score(doc) = sum over retrievers of 1 / (k + rank).
RRF uses RANKS, not raw scores, so an embedding distance and a BM25 score -- which
live on different scales -- never have to be normalised against each other.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

Embedder = Callable[[list[str]], list[list[float]]]

TRAIN_PATH = Path("evaluation/data/critic_train.jsonl")
RRF_K = 60  # the constant from the original RRF paper; damps the gap between ranks 1 and 2
_TOKEN = re.compile(r"[a-z0-9_]+")  # keeps underscores, so "drop_rows_missing" is ONE token


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def render_evidence(evidence: dict) -> str:
    """Flat text used by BOTH retrievers. The same function renders stored cases
    and queries, so both sides are written exactly the same way."""
    issue = evidence.get("issue") or {}
    parts = [f"strategy {evidence.get('strategy')}",
             f"issue {issue.get('type')} severity {issue.get('severity')}"]
    for side in ("before", "after"):
        stats = evidence.get(side) or {}
        parts.append(side + " " + " ".join(f"{k} {v}" for k, v in stats.items()))
    change = evidence.get("change") or {}
    parts.append("change " + " ".join(f"{k} {v}" for k, v in change.items()))
    return " | ".join(parts)


def render_case(example: dict) -> str:
    """What the Critic is SHOWN for a retrieved case: the situation, the correct
    verdict, and the rule that decided it (with its numbers worked out)."""
    return f"{render_evidence(example['evidence'])} | verdict {example['label']} | {example['label_rule']}"


def load_cases(path: Path = TRAIN_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def default_embedder() -> Embedder:
    """all-MiniLM-L6-v2 via ONNX, bundled with chromadb. Downloads ~80 MB on first use."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    return lambda texts: [[float(x) for x in vector] for vector in ef(texts)]


class BM25:
    """Okapi BM25, written out so every term can be explained.

    score(doc, query) = sum over query terms t of
        idf(t) * f(t,doc) * (k1 + 1) / (f(t,doc) + k1 * (1 - b + b * len(doc) / avg_len))
    k1 caps how much repeating a term helps (term-frequency saturation);
    b controls how much long documents are penalised (length normalisation).
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in docs]
        self.avg_len = sum(self.doc_len) / len(docs)
        self.tf = [Counter(d) for d in docs]
        n = len(docs)
        df = Counter(term for d in docs for term in set(d))
        # Rare terms weigh more. The +1 inside the log keeps idf positive even for
        # terms in more than half the documents (classic BM25 idf can go negative).
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.doc_len):
            score = 0.0
            for term in query:
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out.append(score)
        return out


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked id lists. A document ranked well by BOTH retrievers beats one
    ranked first by only one of them. Ties break by id, so the order is deterministic."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


class HybridRetriever:
    def __init__(self, cases: list[dict], embedder: Embedder, candidates: int = 20):
        import chromadb
        from chromadb.config import Settings

        self.ids = [c["id"] for c in cases]
        self.case_text = {c["id"]: render_case(c) for c in cases}
        self.embedder = embedder
        self.candidates = min(candidates, len(cases))  # each retriever's shortlist size

        # Index the evidence WITHOUT the label: queries never have a label.
        texts = [render_evidence(c["evidence"]) for c in cases]
        self.bm25 = BM25([tokenize(t) for t in texts])

        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        # Unique name: in-memory clients in one process can share storage.
        self.collection = client.create_collection(name=f"critic_cases_{uuid.uuid4().hex[:8]}")
        # We pass vectors ourselves instead of a Chroma embedding function: simpler,
        # stable across Chroma versions, and tests can inject a fake embedder.
        self.collection.add(ids=self.ids, embeddings=embedder(texts), documents=texts)

    def dense_ranking(self, text: str) -> list[str]:
        result = self.collection.query(query_embeddings=self.embedder([text]),
                                       n_results=self.candidates)
        return list(result["ids"][0])

    def sparse_ranking(self, text: str) -> list[str]:
        scores = self.bm25.scores(tokenize(text))
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))
        return [self.ids[i] for i in order[: self.candidates]]

    def search(self, evidence: dict, top_k: int = 4) -> list[str]:
        """Ids of the top_k fused results."""
        text = render_evidence(evidence)
        return reciprocal_rank_fusion([self.dense_ranking(text), self.sparse_ranking(text)])[:top_k]

    def retrieve(self, evidence: dict, top_k: int = 4) -> list[str]:
        """Case texts for the Critic prompt."""
        return [self.case_text[i] for i in self.search(evidence, top_k)]