"""Vector store — versioned, atomic, local.

Two backends behind one interface:

``numpy``
    Vectors in a memory-mapped ``.npy`` file, metadata in JSONL. Exact cosine
    search by matrix multiply. On a corpus of tens of thousands of chunks this
    answers in single-digit milliseconds, needs no native dependency, and is
    trivially auditable — you can read the index with three lines of Python.
``faiss``
    Used when it is installed and the corpus outgrows brute force.

Writes are **versioned and atomic**: a run builds ``index/v7/`` alongside the
live ``index/v6/`` and only flips the ``current`` pointer once validation has
passed. A failed run therefore cannot corrupt or empty a working index, and
rollback is a pointer move.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import Chunk

logger = logging.getLogger("jjrag.store.vectorstore")

CURRENT_POINTER = "current.json"


class VectorStoreError(RuntimeError):
    pass


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    position: int


class VectorIndex:
    """One immutable version of the index, loaded read-only."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "index.json"
        if not meta_path.is_file():
            raise VectorStoreError(f"no index at {self.directory}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dimensions: int = self.meta["dimensions"]
        self.metric: str = self.meta.get("metric", "cosine")
        self.model_id: str = self.meta.get("embedding_model", "unknown")
        self.version: int = self.meta.get("version", 0)

        vectors_path = self.directory / "vectors.npy"
        self._vectors = np.load(vectors_path, mmap_mode="r")
        self._chunks: list[Chunk] = [
            Chunk.model_validate_json(line)
            for line in (self.directory / "chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len(self._chunks) != self._vectors.shape[0]:
            raise VectorStoreError(
                f"index at {self.directory} is inconsistent: "
                f"{self._vectors.shape[0]} vectors vs {len(self._chunks)} chunks"
            )
        self._faiss = None
        if self.meta.get("backend") == "faiss":
            self._faiss = _load_faiss(self.directory)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        if len(self._chunks) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimensions:
            raise VectorStoreError(
                f"query vector is {query.shape[0]}-dimensional but the index is "
                f"{self.dimensions}-dimensional — the embedding model changed "
                "since this index was built; re-run ingestion"
            )
        k = min(k, len(self._chunks))

        if self._faiss is not None:
            scores, indices = self._faiss.search(query.reshape(1, -1), k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist(), strict=True))
        else:
            matrix = np.asarray(self._vectors)
            if self.metric == "cosine":
                norm = np.linalg.norm(query) or 1.0
                scores_all = matrix @ (query / norm)
            else:
                scores_all = -np.linalg.norm(matrix - query, axis=1)
            top = np.argpartition(-scores_all, k - 1)[:k]
            top = top[np.argsort(-scores_all[top])]
            pairs = [(int(i), float(scores_all[i])) for i in top]

        return [
            SearchHit(chunk=self._chunks[i], score=float(s), position=i)
            for i, s in pairs
            if 0 <= i < len(self._chunks)
        ]

    def vector_at(self, position: int) -> np.ndarray:
        return np.asarray(self._vectors[position], dtype=np.float32)

    def stats(self) -> dict:
        return {
            "version": self.version,
            "vectors": len(self._chunks),
            "dimensions": self.dimensions,
            "metric": self.metric,
            "embedding_model": self.model_id,
            "built_at": self.meta.get("built_at"),
            "backend": self.meta.get("backend", "numpy"),
            "documents": len({c.doc_id for c in self._chunks}),
        }


def _load_faiss(directory: Path):  # pragma: no cover - optional dependency
    try:
        import faiss
    except ImportError as exc:
        raise VectorStoreError(
            "this index was built with the faiss backend but faiss is not "
            "installed (pip install faiss-cpu)"
        ) from exc
    return faiss.read_index(str(directory / "index.faiss"))


class VectorStore:
    """Manages index versions inside ``index_dir``."""

    def __init__(self, index_dir: Path | str, backend: str = "numpy",
                 metric: str = "cosine", keep_versions: int = 3) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.metric = metric
        self.keep_versions = keep_versions

    # -- version bookkeeping ------------------------------------------------
    def _pointer_path(self) -> Path:
        return self.index_dir / CURRENT_POINTER

    def current_version(self) -> int | None:
        pointer = self._pointer_path()
        if not pointer.is_file():
            return None
        try:
            return int(json.loads(pointer.read_text(encoding="utf-8"))["version"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def versions(self) -> list[int]:
        out = []
        for entry in self.index_dir.glob("v*"):
            if entry.is_dir() and entry.name[1:].isdigit():
                out.append(int(entry.name[1:]))
        return sorted(out)

    def version_dir(self, version: int) -> Path:
        return self.index_dir / f"v{version}"

    def next_version(self) -> int:
        versions = self.versions()
        return (max(versions) + 1) if versions else 1

    # -- write --------------------------------------------------------------
    def build(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        *,
        embedding_model: str,
        version: int | None = None,
        run_id: str | None = None,
    ) -> tuple[int, Path]:
        """Write a new index version. Does **not** publish it."""
        from datetime import datetime, timezone

        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                f"cannot build index: {len(chunks)} chunks vs "
                f"{vectors.shape[0]} vectors"
            )
        version = version or self.next_version()
        target = self.version_dir(version)
        staging = self.index_dir / f".building_v{version}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        np.save(staging / "vectors.npy", vectors)
        with (staging / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")

        backend = self.backend
        if backend == "faiss":
            backend = self._try_build_faiss(staging, vectors)

        meta = {
            "version": version,
            "backend": backend,
            "metric": self.metric,
            "dimensions": int(vectors.shape[1]) if vectors.size else 0,
            "vectors": int(vectors.shape[0]),
            "documents": len({c.doc_id for c in chunks}),
            "embedding_model": embedding_model,
            "run_id": run_id,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "index.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        logger.info("built index v%d with %d vectors", version, len(chunks))
        return version, target

    def _try_build_faiss(self, staging: Path, vectors: np.ndarray) -> str:
        try:  # pragma: no cover - optional dependency
            import faiss
        except ImportError:
            logger.warning(
                "faiss backend requested but faiss is not installed; "
                "falling back to the numpy backend"
            )
            return "numpy"
        dim = int(vectors.shape[1])
        index = (
            faiss.IndexFlatIP(dim) if self.metric == "cosine"
            else faiss.IndexFlatL2(dim)
        )
        index.add(vectors)
        faiss.write_index(index, str(staging / "index.faiss"))
        return "faiss"

    def publish(self, version: int) -> None:
        """Atomically point ``current`` at a built version."""
        from datetime import datetime, timezone

        target = self.version_dir(version)
        if not (target / "index.json").is_file():
            raise VectorStoreError(f"index v{version} has not been built")

        pointer = self._pointer_path()
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": version,
                    "path": str(target),
                    "published_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, pointer)  # atomic on POSIX
        logger.info("published index v%d", version)
        self.prune()

    def prune(self) -> list[int]:
        """Delete old versions beyond ``keep_versions`` (never the current one)."""
        current = self.current_version()
        versions = self.versions()
        removable = [v for v in versions if v != current]
        to_delete = removable[: max(0, len(versions) - self.keep_versions)]
        for version in to_delete:
            shutil.rmtree(self.version_dir(version), ignore_errors=True)
            logger.info("pruned index v%d", version)
        return to_delete

    def rollback(self) -> int | None:
        """Publish the newest version older than the current one."""
        current = self.current_version()
        candidates = [v for v in self.versions() if current is None or v < current]
        if not candidates:
            return None
        target = max(candidates)
        self.publish(target)
        return target

    # -- read ---------------------------------------------------------------
    def load(self, version: int | None = None) -> VectorIndex | None:
        version = version or self.current_version()
        if version is None:
            return None
        directory = self.version_dir(version)
        if not (directory / "index.json").is_file():
            return None
        return VectorIndex(directory)
