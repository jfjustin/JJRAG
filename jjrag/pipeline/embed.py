"""Embedding stage — chunks become vectors, locally.

Three backends, all of which run on hardware you control:

``sentence-transformers``
    The default. Downloads a model once (from Hugging Face, at install time —
    see :func:`prefetch_model`), then runs offline forever. Best quality per
    CPU-second for this workload.
``ollama``
    Uses the same local Ollama server that serves generation, so a deployment
    can run with exactly one model runtime.
``hashing``
    A deterministic, dependency-free hashed bag-of-words projection. Not
    competitive for semantic search, but it needs no model download at all,
    which makes it the right choice for CI, smoke tests, and air-gapped hosts
    that cannot fetch weights. Hybrid retrieval's lexical half still works, so
    the system stays usable.

Embeddings are cached by ``(model, text_sha256)`` in SQLite, so re-ingesting an
unchanged corpus costs no compute.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import struct
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import EmbeddingSettings

logger = logging.getLogger("jjrag.pipeline.embed")


class EmbeddingError(RuntimeError):
    pass


class EmbeddingBackend(Protocol):
    """Every backend is a pure function from texts to unit-norm vectors."""

    name: str
    model_id: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# sentence-transformers (default)
# ---------------------------------------------------------------------------
class SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self, settings: EmbeddingSettings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it "
                "(pip install sentence-transformers) or set "
                "embedding.backend to 'ollama' or 'hashing'."
            ) from exc

        self.model_id = settings.model
        self.settings = settings
        logger.info("loading local embedding model %s", settings.model)
        self._model = SentenceTransformer(settings.model, device=settings.device)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.settings.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
class OllamaEmbeddingBackend:
    name = "ollama"

    def __init__(self, settings: EmbeddingSettings, host: str) -> None:
        import requests  # local import: only this backend needs it

        self._requests = requests
        self.host = host.rstrip("/")
        self.model_id = settings.ollama_model
        self.settings = settings
        self.dimensions = self._probe_dimensions()

    def _embed_one(self, text: str) -> list[float]:
        response = self._requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model_id, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        vector = response.json().get("embedding")
        if not vector:
            raise EmbeddingError(
                f"Ollama returned no embedding for model {self.model_id}"
            )
        return vector

    def _probe_dimensions(self) -> int:
        try:
            return len(self._embed_one("dimension probe"))
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"cannot reach the local Ollama embedding model at {self.host}: "
                f"{exc}. Start Ollama and run: ollama pull {self.model_id}"
            ) from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self._embed_one(t) for t in texts]
        matrix = np.asarray(vectors, dtype=np.float32)
        return _normalize_rows(matrix) if self.settings.normalize else matrix


# ---------------------------------------------------------------------------
# Hashing (no model weights required)
# ---------------------------------------------------------------------------
class HashingBackend:
    """Hashed bag-of-words with sub-word shingles, projected to a fixed size.

    Deterministic across processes and machines, which is what makes it usable
    as a CI fixture: the same text always produces the same vector, so index
    tests are reproducible without shipping model weights.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 384) -> None:
        self.model_id = f"hashing-{dimensions}"
        self.dimensions = dimensions

    @staticmethod
    def _features(text: str) -> Iterable[str]:
        tokens = re.findall(r"\w+", text.lower())
        yield from tokens
        for i in range(len(tokens) - 1):
            yield f"{tokens[i]}_{tokens[i + 1]}"       # bigrams
        for token in tokens:
            if len(token) > 5:
                for i in range(len(token) - 3):
                    yield f"#{token[i:i + 4]}"          # char 4-grams

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(
                    feature.encode("utf-8"), digest_size=8
                ).digest()
                value = struct.unpack("<Q", digest)[0]
                index = value % self.dimensions
                sign = 1.0 if (value >> 63) & 1 else -1.0
                matrix[row, index] += sign
        return _normalize_rows(matrix)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class EmbeddingCache:
    """SQLite-backed cache keyed by (model, text hash)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model, text_sha256)
            )
            """
        )
        self._conn.commit()

    def get_many(
        self, model: str, hashes: Sequence[str]
    ) -> dict[str, np.ndarray]:
        if not hashes:
            return {}
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(hashes), 500):
                batch = hashes[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT text_sha256, vector FROM embeddings "
                    f"WHERE model = ? AND text_sha256 IN ({placeholders})",
                    (model, *batch),
                ).fetchall()
                for text_hash, blob in rows:
                    out[text_hash] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(
        self, model: str, items: Sequence[tuple[str, np.ndarray]]
    ) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(model, text_sha256, dimensions, vector) VALUES (?, ?, ?, ?)",
                [
                    (model, text_hash, len(vector),
                     np.asarray(vector, dtype=np.float32).tobytes())
                    for text_hash, vector in items
                ],
            )
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class Embedder:
    """Backend + cache + batching. The only embedding entry point."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        llm_host: str = "http://localhost:11434",
        cache_path: Path | str | None = None,
    ) -> None:
        self.settings = settings
        self.backend = build_backend(settings, llm_host)
        self.cache = (
            EmbeddingCache(cache_path)
            if settings.cache_enabled and cache_path
            else None
        )
        self.stats = {"embedded": 0, "cache_hits": 0}

    @property
    def dimensions(self) -> int:
        return self.backend.dimensions

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def embed_texts(
        self, texts: Sequence[str], hashes: Sequence[str] | None = None
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        if hashes is None:
            hashes = [
                hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts
            ]

        cached: dict[str, np.ndarray] = {}
        if self.cache is not None:
            cached = self.cache.get_many(self.model_id, list(hashes))
            self.stats["cache_hits"] += len(cached)

        missing = [i for i, h in enumerate(hashes) if h not in cached]
        fresh: dict[int, np.ndarray] = {}
        batch_size = max(1, self.settings.batch_size)
        for start in range(0, len(missing), batch_size):
            batch_indices = missing[start:start + batch_size]
            vectors = self.backend.embed([texts[i] for i in batch_indices])
            for index, vector in zip(batch_indices, vectors, strict=True):
                fresh[index] = np.asarray(vector, dtype=np.float32)
            self.stats["embedded"] += len(batch_indices)

        if self.cache is not None and fresh:
            self.cache.put_many(
                self.model_id, [(hashes[i], v) for i, v in fresh.items()]
            )

        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text_hash in enumerate(hashes):
            out[i] = fresh[i] if i in fresh else cached[text_hash]
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """Queries are never cached — they are usually unique and short."""
        return np.asarray(self.backend.embed([text])[0], dtype=np.float32)

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()


def build_backend(
    settings: EmbeddingSettings, llm_host: str = "http://localhost:11434"
) -> EmbeddingBackend:
    if settings.backend == "sentence-transformers":
        return SentenceTransformerBackend(settings)
    if settings.backend == "ollama":
        return OllamaEmbeddingBackend(settings, llm_host)
    if settings.backend == "hashing":
        return HashingBackend(settings.dimensions or 384)
    raise EmbeddingError(f"unknown embedding backend: {settings.backend}")


def prefetch_model(settings: EmbeddingSettings) -> str:
    """Download the embedding weights ahead of time.

    Run this during image build or provisioning — *before* the egress guard is
    installed — so the running service never needs network access at all.
    """
    if settings.backend != "sentence-transformers":
        return f"nothing to prefetch for backend {settings.backend}"
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(settings.model, device="cpu")
    return f"cached {settings.model}"
