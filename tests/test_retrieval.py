"""Hybrid retrieval and grounded answering."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from jjrag.config import LLMSettings, RetrievalSettings, Settings
from jjrag.llm.ollama import LocalModelError, OllamaClient
from jjrag.pipeline.runner import Pipeline
from jjrag.retrieval.answer import NO_CONTEXT_MESSAGE, AnswerEngine, cited_indices
from jjrag.retrieval.search import (
    BM25Index,
    Retriever,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
    tokenize,
)


class TestLexicalSearch:
    def test_tokenize_drops_stopwords(self) -> None:
        assert tokenize("What is the retention policy?") == ["retention", "policy"]

    def test_bm25_ranks_the_matching_document_first(self) -> None:
        index = BM25Index.build([
            "retention of customer records is seven years",
            "access control requires multi factor authentication",
        ])
        scores = index.scores("retention records")
        assert scores[0] > scores[1]

    def test_bm25_returns_zeros_for_unknown_terms(self) -> None:
        index = BM25Index.build(["alpha beta"])
        assert not index.scores("zebra").any()


class TestFusionAndDiversity:
    def test_rrf_rewards_agreement_between_rankings(self) -> None:
        fused = reciprocal_rank_fusion([[5, 1, 2], [5, 2, 1]])
        assert max(fused, key=fused.get) == 5

    def test_mmr_avoids_returning_near_identical_passages(self) -> None:
        # Candidate 1 is nearly identical to candidate 0; candidate 2 is
        # unrelated. A diversity-weighted pick should take 0 then 2.
        vectors = np.array(
            [[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        picked = maximal_marginal_relevance(query, vectors, [0, 1, 2], k=2, lambda_=0.3)
        assert picked == [0, 2]

    def test_mmr_lambda_one_is_pure_relevance(self) -> None:
        vectors = np.array(
            [[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        picked = maximal_marginal_relevance(query, vectors, [0, 1, 2], k=2, lambda_=1.0)
        assert picked == [0, 1]


class TestRetriever:
    @pytest.fixture
    def retriever(self, pipeline: Pipeline, sample_docs: Path) -> Retriever:
        pipeline.run()
        index = pipeline.vector_store.load()
        return Retriever(index, RetrievalSettings())

    def test_finds_the_passage_that_answers_the_question(
        self, retriever: Retriever, pipeline: Pipeline
    ) -> None:
        query = "how long are customer records retained"
        vector = pipeline.embedder.embed_query(query)
        results = retriever.search(query, vector, top_k=3)

        assert results
        assert any("seven years" in r.chunk.text for r in results)
        assert results[0].rank == 1

    def test_exact_identifiers_are_found_by_the_lexical_half(
        self, retriever: Retriever, pipeline: Pipeline
    ) -> None:
        query = "DR-14"
        results = retriever.search(query, pipeline.embedder.embed_query(query), top_k=3)
        assert any("DR-14" in r.chunk.text for r in results)

    def test_context_block_is_numbered_for_citation(
        self, retriever: Retriever, pipeline: Pipeline
    ) -> None:
        query = "retention"
        results = retriever.search(query, pipeline.embedder.embed_query(query), top_k=2)
        context = retriever.build_context(results)
        assert context.startswith("[1] ")
        assert results[0].chunk.filename in context

    def test_context_respects_the_character_budget(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        pipeline.run()
        retriever = Retriever(
            pipeline.vector_store.load(), RetrievalSettings(max_context_chars=300)
        )
        query = "retention"
        results = retriever.search(query, pipeline.embedder.embed_query(query), top_k=5)
        assert len(retriever.build_context(results)) <= 400

    def test_asking_for_more_passages_than_exist_is_safe(
        self, settings: Settings
    ) -> None:
        from jjrag.models import Chunk
        from jjrag.store.vectorstore import VectorStore

        store = VectorStore(settings.paths.index_dir)
        chunk = Chunk(
            doc_id="d", source_id="s", ordinal=0, text="only chunk", filename="a.txt"
        ).finalize()
        version, _ = store.build(
            [chunk], np.ones((1, 4), dtype=np.float32), embedding_model="t"
        )
        store.publish(version)

        retriever = Retriever(store.load(), RetrievalSettings())
        results = retriever.search("anything", np.ones(4, dtype=np.float32), top_k=10)
        assert len(results) == 1


class TestCitations:
    def test_parses_single_and_grouped_citations(self) -> None:
        assert cited_indices("Yes [1]. Also [2, 3].") == {1, 2, 3}

    def test_returns_nothing_when_no_citations_present(self) -> None:
        assert cited_indices("no citations here") == set()


# ---------------------------------------------------------------------------
# A stub local model server, so the whole answer path is exercised without
# needing Ollama installed.
# ---------------------------------------------------------------------------
class _StubOllamaHandler(BaseHTTPRequestHandler):
    reply = "Records are kept for seven years [1]."

    def log_message(self, *args) -> None:  # silence test output
        pass

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._json({"models": [{"name": "test-model:8b", "size": 1,
                                    "details": {"family": "llama"}}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        if self.path == "/api/generate":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for token in self.reply.split(" "):
                self.wfile.write(
                    json.dumps({"response": token + " ", "done": False}).encode() + b"\n"
                )
            self.wfile.write(json.dumps({"response": "", "done": True}).encode() + b"\n")
        elif self.path == "/api/embeddings":
            self._json({"embedding": [0.1] * 16})
        else:
            self.send_error(404)

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_model_server():
    server = HTTPServer(("127.0.0.1", 0), _StubOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


class TestLocalModelClient:
    def test_lists_installed_models(self, stub_model_server: str) -> None:
        client = OllamaClient(LLMSettings(host=stub_model_server))
        assert client.is_available()
        assert client.model_names() == ["test-model:8b"]

    def test_matches_a_model_ignoring_the_tag(self, stub_model_server: str) -> None:
        client = OllamaClient(LLMSettings(host=stub_model_server, model="test-model"))
        assert client.ensure_model() == "test-model:8b"

    def test_streams_tokens(self, stub_model_server: str) -> None:
        client = OllamaClient(LLMSettings(host=stub_model_server))
        assert "seven years" in "".join(client.stream("question"))

    def test_refuses_a_non_local_model_host(self) -> None:
        with pytest.raises(LocalModelError, match="refusing"):
            OllamaClient(LLMSettings(host="https://api.openai.com"))

    def test_allows_an_explicitly_allowlisted_host(self) -> None:
        client = OllamaClient(
            LLMSettings(host="http://models.internal:11434"),
            allowed_hosts=["models.internal"],
        )
        assert client.host == "http://models.internal:11434"

    def test_unreachable_server_gives_an_actionable_error(self) -> None:
        client = OllamaClient(LLMSettings(host="http://127.0.0.1:1"))
        with pytest.raises(LocalModelError, match="Is Ollama running"):
            list(client.stream("question"))


class TestAnswerEngine:
    @pytest.fixture
    def engine(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings,
        stub_model_server: str,
    ) -> AnswerEngine:
        pipeline.run()
        settings.llm.host = stub_model_server
        retriever = Retriever(pipeline.vector_store.load(), settings.retrieval)
        return AnswerEngine(settings, retriever, pipeline.embedder)

    def test_answers_with_citations_to_real_passages(self, engine: AnswerEngine) -> None:
        answer = engine.answer("How long are customer records retained?")
        assert "seven years" in answer.answer
        assert answer.citations and answer.citations[0].chunk.filename

    def test_prompt_forbids_outside_knowledge(self, engine: AnswerEngine) -> None:
        results = engine.retrieve("retention")
        prompt = engine.build_prompt("retention?", results)
        assert "ONLY the excerpts" in prompt
        assert "[1]" in prompt

    def test_streaming_emits_sources_before_tokens(self, engine: AnswerEngine) -> None:
        events = list(engine.stream_answer("retention policy"))
        names = [name for name, _ in events]
        assert names[0] == "sources"
        assert "token" in names and names[-1] == "done"

    def test_refuses_to_guess_when_nothing_is_retrieved(
        self, engine: AnswerEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "retrieve", lambda *a, **k: [])
        answer = engine.answer("what is the capital of France?")
        assert answer.answer == NO_CONTEXT_MESSAGE
        assert answer.citations == []
