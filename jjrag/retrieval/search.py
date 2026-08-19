"""Retrieval — hybrid dense + lexical search with diversity re-ranking.

Dense-only retrieval misses exact terms (an invoice number, a policy code like
"DR-14", a surname). Lexical-only retrieval misses paraphrase. Real questions
need both, so this module runs the two in parallel and fuses them with
Reciprocal Rank Fusion, which needs no score calibration between the two very
different scales.

MMR then trims the candidate set to the final ``top_k``, trading a little
relevance for diversity so the context window is not three near-identical
paragraphs from the same page.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..config import RetrievalSettings
from ..models import Chunk, RetrievedChunk
from ..store.vectorstore import VectorIndex

logger = logging.getLogger("jjrag.retrieval.search")

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "do", "does", "can", "should", "i", "we", "you",
}


def tokenize(text: str) -> list[str]:
    return [
        token for token in (t.lower() for t in _TOKEN.findall(text))
        if token not in _STOPWORDS and len(token) > 1
    ]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
@dataclass
class BM25Index:
    """Okapi BM25 over the chunk corpus.

    Built in memory from the same chunk list the vector index holds, so the two
    halves of hybrid search can never disagree about what the corpus contains.
    """

    documents: list[list[str]]
    doc_freq: Counter
    doc_lengths: list[int]
    average_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        documents = [tokenize(t) for t in texts]
        doc_freq: Counter = Counter()
        for tokens in documents:
            doc_freq.update(set(tokens))
        lengths = [len(d) for d in documents]
        average = sum(lengths) / len(lengths) if lengths else 0.0
        return cls(documents, doc_freq, lengths, average, k1, b)

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        total_docs = len(self.documents)
        out = np.zeros(total_docs, dtype=np.float32)
        if not tokens or total_docs == 0:
            return out

        for term in Counter(tokens):
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for i, document in enumerate(self.documents):
                tf = document.count(term)
                if tf == 0:
                    continue
                length_norm = (
                    1 - self.b + self.b * (self.doc_lengths[i] / (self.average_length or 1))
                )
                out[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        return out


# ---------------------------------------------------------------------------
# Fusion + diversity
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60
) -> dict[int, float]:
    """Combine ranked id lists. Rank-based, so score scales never matter."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking):
            fused[position] = fused.get(position, 0.0) + 1.0 / (k + rank + 1)
    return fused


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_positions: Sequence[int],
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Greedy MMR: pick relevance, then penalise similarity to what's picked."""
    if len(candidate_positions) == 0:
        return []
    k = min(k, len(candidate_positions))
    relevance = candidate_vectors @ query_vector
    selected: list[int] = []
    remaining = list(range(len(candidate_positions)))

    while remaining and len(selected) < k:
        best_index, best_score = remaining[0], -math.inf
        for i in remaining:
            if selected:
                redundancy = float(
                    np.max(candidate_vectors[i] @ candidate_vectors[selected].T)
                )
            else:
                redundancy = 0.0
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        remaining.remove(best_index)
    return [candidate_positions[i] for i in selected]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """Search one index version. Rebuilt when the index is republished."""

    def __init__(self, index: VectorIndex, settings: RetrievalSettings) -> None:
        self.index = index
        self.settings = settings
        self._bm25: BM25Index | None = None
        if settings.hybrid and len(index) > 0:
            self._bm25 = BM25Index.build([c.text for c in index.chunks])

    @property
    def size(self) -> int:
        return len(self.index)

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k: int | None = None,
        filter_doc_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        if len(self.index) == 0:
            return []

        settings = self.settings
        top_k = top_k or settings.top_k
        candidate_k = max(settings.candidate_k, top_k)

        dense_hits = self.index.search(query_vector, k=min(candidate_k, len(self.index)))
        dense_scores = {hit.position: hit.score for hit in dense_hits}
        dense_ranking = [hit.position for hit in dense_hits]

        lexical_scores: dict[int, float] = {}
        rankings: list[Sequence[int]] = [dense_ranking]
        if self._bm25 is not None:
            raw = self._bm25.scores(query)
            if raw.any():
                top = np.argsort(-raw)[:candidate_k]
                lexical_ranking = [int(i) for i in top if raw[i] > 0]
                lexical_scores = {int(i): float(raw[i]) for i in lexical_ranking}
                rankings.append(lexical_ranking)

        fused = (
            reciprocal_rank_fusion(rankings, settings.rrf_k)
            if len(rankings) > 1
            else {p: 1.0 / (rank + 1) for rank, p in enumerate(dense_ranking)}
        )

        candidates = sorted(fused, key=lambda p: -fused[p])
        if filter_doc_ids is not None:
            candidates = [
                p for p in candidates
                if self.index.chunks[p].doc_id in filter_doc_ids
            ]
        candidates = candidates[:candidate_k]
        if not candidates:
            return []

        if 0.0 <= settings.mmr_lambda < 1.0:
            vectors = np.stack([self.index.vector_at(p) for p in candidates])
            ordered = maximal_marginal_relevance(
                np.asarray(query_vector, dtype=np.float32), vectors, candidates,
                top_k, settings.mmr_lambda,
            )
        else:
            ordered = candidates[:top_k]

        results: list[RetrievedChunk] = []
        for rank, position in enumerate(ordered):
            dense = dense_scores.get(position)
            if dense is not None and dense < settings.min_score:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.index.chunks[position],
                    score=float(fused.get(position, 0.0)),
                    dense_score=dense,
                    lexical_score=lexical_scores.get(position),
                    rank=rank + 1,
                )
            )
        return results

    def build_context(self, results: Sequence[RetrievedChunk]) -> str:
        """Render retrieved chunks as a numbered, citable context block."""
        budget = self.settings.max_context_chars
        parts: list[str] = []
        used = 0
        for i, result in enumerate(results, start=1):
            chunk: Chunk = result.chunk
            header = f"[{i}] {chunk.filename}"
            if chunk.segment_label:
                header += f" — {chunk.segment_label}"
            block = f"{header}\n{chunk.text}"
            if used + len(block) > budget:
                remaining = budget - used
                if remaining > 200:
                    parts.append(block[:remaining] + " …[truncated]")
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)
