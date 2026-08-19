"""Grounded answering — retrieval plus a local model, with citations.

The contract with the user is that every claim in an answer is traceable to a
retrieved excerpt. That is enforced in three places:

1. the prompt forbids using outside knowledge and requires ``[n]`` citations;
2. retrieval returns nothing → the model is never called, and the user is told
   the corpus has no answer rather than being given a plausible invention;
3. the response carries the excerpts themselves, so a reader can check.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence

from ..config import Settings
from ..llm.ollama import LocalModelError, OllamaClient
from ..models import Answer, RetrievedChunk
from ..pipeline.embed import Embedder
from .search import Retriever

logger = logging.getLogger("jjrag.retrieval.answer")

NO_CONTEXT_MESSAGE = (
    "I could not find anything relevant in the indexed documents, so I will "
    "not guess. Try rephrasing the question, or upload the document that "
    "should contain the answer."
)

PROMPT_TEMPLATE = """Answer the question using ONLY the excerpts below.

Rules:
- Use only what the excerpts say. Do not add outside knowledge.
- Cite the excerpt numbers you used, like [1] or [2, 3], next to each claim.
- If the excerpts do not answer the question, say exactly what is missing.
- Quote figures, dates and names exactly as they appear.

Excerpts:
{context}

Question: {question}

Answer:"""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_indices(text: str) -> set[int]:
    """Excerpt numbers the model actually referenced."""
    found: set[int] = set()
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return found


class AnswerEngine:
    """Ties the index, the embedder and the local model together."""

    def __init__(
        self, settings: Settings, retriever: Retriever, embedder: Embedder,
        client: OllamaClient | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.embedder = embedder
        self.client = client or OllamaClient(
            settings.llm, settings.security.extra_allowed_hosts
        )

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(question)
        return self.retriever.search(question, vector, top_k=top_k)

    def build_prompt(self, question: str, results: Sequence[RetrievedChunk]) -> str:
        return PROMPT_TEMPLATE.format(
            context=self.retriever.build_context(results), question=question
        )

    def answer(
        self, question: str, *, top_k: int | None = None, model: str | None = None,
    ) -> Answer:
        started = time.perf_counter()
        results = self.retrieve(question, top_k)
        if not results:
            return Answer(
                question=question, answer=NO_CONTEXT_MESSAGE, citations=[],
                model=model or self.settings.llm.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                index_version=self.retriever.index.version,
            )

        text = self.client.generate(
            self.build_prompt(question, results), model=model
        )
        return Answer(
            question=question,
            answer=text.strip(),
            citations=self._used_citations(text, results),
            model=model or self.settings.llm.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            index_version=self.retriever.index.version,
        )

    def stream_answer(
        self, question: str, *, top_k: int | None = None, model: str | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Yield ``(event, payload)`` pairs for the SSE endpoint.

        Sources are emitted *before* the first token so the UI can show what the
        answer is grounded in while it is still being written.
        """
        results = self.retrieve(question, top_k)
        yield "sources", [r.model_dump() for r in results]

        if not results:
            yield "token", NO_CONTEXT_MESSAGE
            yield "done", {"citations": [], "index_version": self.retriever.index.version}
            return

        started = time.perf_counter()
        collected: list[str] = []
        try:
            for token in self.client.stream(
                self.build_prompt(question, results), model=model
            ):
                collected.append(token)
                yield "token", token
        except LocalModelError as exc:
            yield "error", str(exc)
            return

        text = "".join(collected)
        yield "done", {
            "citations": [
                r.model_dump() for r in self._used_citations(text, results)
            ],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "index_version": self.retriever.index.version,
            "model": model or self.settings.llm.model,
        }

    @staticmethod
    def _used_citations(
        text: str, results: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Prefer the excerpts the model cited; fall back to all of them.

        A model that cites nothing has not proved its grounding, so in that case
        the user still sees every excerpt that was in the prompt and can judge
        for themselves.
        """
        indices = cited_indices(text)
        if not indices:
            return list(results)
        return [r for i, r in enumerate(results, start=1) if i in indices] or list(results)
