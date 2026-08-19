"""Vector store versioning and catalog bookkeeping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jjrag.models import Chunk, RunManifest, RunStatus, SourceFile, TransformedDocument
from jjrag.store.catalog import Catalog
from jjrag.store.vectorstore import VectorStore, VectorStoreError


def _chunks(count: int, doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(
            doc_id=doc_id, source_id="src1", ordinal=i, text=f"chunk number {i}",
            filename="a.txt", segment_label=f"p. {i + 1}",
        ).finalize()
        for i in range(count)
    ]


class TestVectorStore:
    def test_build_does_not_publish(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        version, _ = store.build(
            _chunks(3), np.eye(3, 8, dtype=np.float32), embedding_model="test"
        )
        assert version == 1
        assert store.current_version() is None
        assert store.load() is None

    def test_publish_makes_the_version_current(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        version, _ = store.build(
            _chunks(3), np.eye(3, 8, dtype=np.float32), embedding_model="test"
        )
        store.publish(version)
        index = store.load()
        assert index is not None and len(index) == 3
        assert index.stats()["embedding_model"] == "test"

    def test_failed_build_leaves_the_live_index_serving(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        first, _ = store.build(
            _chunks(3), np.eye(3, 8, dtype=np.float32), embedding_model="test"
        )
        store.publish(first)
        # A second build that is never published must not disturb the pointer.
        store.build(_chunks(9), np.eye(9, 8, dtype=np.float32), embedding_model="test")
        assert store.current_version() == first
        assert len(store.load()) == 3

    def test_rollback_returns_to_the_previous_version(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        for size in (2, 5):
            version, _ = store.build(
                _chunks(size), np.eye(size, 8, dtype=np.float32),
                embedding_model="test",
            )
            store.publish(version)
        assert len(store.load()) == 5
        assert store.rollback() == 1
        assert len(store.load()) == 2

    def test_prune_keeps_the_configured_number_of_versions(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path, keep_versions=2)
        for _ in range(4):
            version, _ = store.build(
                _chunks(2), np.eye(2, 8, dtype=np.float32), embedding_model="test"
            )
            store.publish(version)
        assert len(store.versions()) <= 2
        assert store.current_version() in store.versions()

    def test_search_ranks_by_cosine_similarity(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        vectors = np.eye(4, 8, dtype=np.float32)
        version, _ = store.build(_chunks(4), vectors, embedding_model="test")
        store.publish(version)
        index = store.load()

        query = np.zeros(8, dtype=np.float32)
        query[2] = 1.0
        hits = index.search(query, k=2)
        assert hits[0].chunk.ordinal == 2
        assert hits[0].score == pytest.approx(1.0)

    def test_dimension_mismatch_is_rejected_loudly(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        version, _ = store.build(
            _chunks(2), np.eye(2, 8, dtype=np.float32), embedding_model="test"
        )
        store.publish(version)
        with pytest.raises(VectorStoreError, match="dimensional"):
            store.load().search(np.ones(4, dtype=np.float32), k=1)

    def test_mismatched_chunk_and_vector_counts_are_rejected(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path)
        with pytest.raises(VectorStoreError):
            store.build(_chunks(3), np.eye(2, 8, dtype=np.float32), embedding_model="t")


class TestCatalog:
    def _document(self, doc_id: str = "doc1") -> TransformedDocument:
        return TransformedDocument(
            doc_id=doc_id, source_id="src1", filename="a.txt",
            chunks=_chunks(2, doc_id), redactions={"email": 2},
        )

    def test_records_lineage_from_source_to_chunk(self, tmp_path: Path) -> None:
        catalog = Catalog(tmp_path / "catalog.sqlite3")
        source = SourceFile(
            source_id="src1", filename="a.txt", path="/tmp/a.txt", extension=".txt",
            size_bytes=10, content_sha256="abc123",
        )
        catalog.record_source(source)
        catalog.record_document(self._document(), "run1", char_count=99)

        stats = catalog.stats()
        assert stats["documents"] == 1 and stats["chunks"] == 2
        assert catalog.list_documents()[0]["redactions"] == {"email": 2}

    def test_identical_content_is_recognised_as_already_ingested(
        self, tmp_path: Path
    ) -> None:
        catalog = Catalog(tmp_path / "catalog.sqlite3")
        catalog.record_source(SourceFile(
            source_id="src1", filename="a.txt", path="/tmp/a.txt", extension=".txt",
            size_bytes=10, content_sha256="abc123",
        ))
        catalog.record_document(self._document(), "run1")
        assert catalog.is_already_ingested("abc123")
        assert not catalog.is_already_ingested("different")

    def test_soft_delete_excludes_the_document_everywhere(self, tmp_path: Path) -> None:
        catalog = Catalog(tmp_path / "catalog.sqlite3")
        catalog.record_source(SourceFile(
            source_id="src1", filename="a.txt", path="/tmp/a.txt", extension=".txt",
            size_bytes=10, content_sha256="abc123",
        ))
        catalog.record_document(self._document(), "run1")

        assert catalog.delete_document("doc1")
        assert catalog.active_chunk_count() == 0
        assert catalog.list_documents() == []
        assert catalog.stats()["documents_deleted"] == 1
        # A deleted document must no longer count as ingested, so the same file
        # can be re-uploaded later.
        assert not catalog.is_already_ingested("abc123")

    def test_rejections_are_retained_for_audit(self, tmp_path: Path) -> None:
        catalog = Catalog(tmp_path / "catalog.sqlite3")
        catalog.record_rejection("evil.exe", "hash", "executable content", 100)
        assert catalog.stats()["sources_rejected"] == 1

    def test_run_manifests_round_trip_with_issues(self, tmp_path: Path) -> None:
        from jjrag.models import (
            Severity,
            StageResult,
            ValidationIssue,
            ValidationReport,
        )

        catalog = Catalog(tmp_path / "catalog.sqlite3")
        report = ValidationReport(stage="extract", checked=1)
        report.issues.append(ValidationIssue(
            rule="extract.empty_text", severity=Severity.ERROR,
            message="no text", stage="extract",
        ))
        manifest = RunManifest(
            run_id="run1", status=RunStatus.FAILED, documents=1,
            stages=[StageResult(name="extract", report=report)],
        )
        catalog.record_run(manifest)

        restored = catalog.get_run("run1")
        assert restored is not None and restored.status is RunStatus.FAILED
        assert catalog.issues_for_run("run1")[0]["rule"] == "extract.empty_text"
        assert catalog.list_runs()[0]["status"] == "failed"
