"""Pipeline orchestrator.

Runs the stages in order, gates each one, and writes an auditable manifest:

    scan ─▶ extract ─▶ transform ─▶ validate ─▶ embed ─▶ load ─▶ publish

Guarantees this orchestration provides:

* **Atomic publication.** A new index version is built alongside the live one
  and only becomes ``current`` after every gate has passed. A failed run leaves
  the previous index serving traffic, untouched.
* **Full-corpus rebuilds.** Each run re-indexes every active document in the
  catalog, not just the new files, so deletions and config changes take effect
  and the index can never drift from the catalog. Embeddings come from the
  cache, so this costs little after the first run.
* **Idempotency.** A file whose content hash is already ingested is skipped.
* **Traceability.** Every run writes ``manifest.json`` and rows in the catalog:
  what was admitted, rejected, redacted, checked and published.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import Settings
from ..models import (
    Chunk,
    RawDocument,
    RunManifest,
    RunStatus,
    SourceFile,
    StageResult,
    StageStatus,
    TransformedDocument,
)
from ..observability.audit import AuditLog
from ..security import scan
from ..store.catalog import Catalog
from ..store.vectorstore import VectorStore
from . import extract as extract_stage
from . import validate as validate_stage
from .embed import Embedder
from .transform import DedupState, transform_document

logger = logging.getLogger("jjrag.pipeline.runner")

ProgressCallback = Callable[[str, str], None]


class PipelineError(RuntimeError):
    def __init__(self, message: str, manifest: RunManifest) -> None:
        super().__init__(message)
        self.manifest = manifest


class Pipeline:
    """The ETL + validation pipeline. One instance per process is enough."""

    def __init__(
        self,
        settings: Settings,
        *,
        catalog: Catalog | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        settings.paths.ensure()
        self.catalog = catalog or Catalog(settings.paths.catalog_path)
        self.vector_store = vector_store or VectorStore(
            settings.paths.index_dir,
            backend=settings.vector_store.backend,
            metric=settings.vector_store.metric,
            keep_versions=settings.vector_store.keep_versions,
        )
        self._embedder = embedder
        self.audit = AuditLog(
            settings.paths.log_dir / "audit.jsonl",
            enabled=settings.security.audit_log_enabled,
        )

    @property
    def embedder(self) -> Embedder:
        """Built lazily — loading a model is expensive and not always needed."""
        if self._embedder is None:
            self._embedder = Embedder(
                self.settings.embedding,
                llm_host=self.settings.llm.host,
                cache_path=self.settings.paths.cache_path,
            )
        return self._embedder

    # ------------------------------------------------------------------
    def run(
        self,
        paths: Sequence[Path | str] | None = None,
        *,
        triggered_by: str = "cli",
        actor: str = "system",
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> RunManifest:
        """Ingest ``paths`` (default: everything in the inbox) and republish.

        Raises :class:`PipelineError` when a validation gate fails; the manifest
        on the exception explains exactly which rule stopped the run.
        """
        settings = self.settings
        manifest = RunManifest(
            triggered_by=triggered_by,
            status=RunStatus.RUNNING,
            config_snapshot=settings.describe_compliance(),
        )
        manifest.config_fingerprint = _fingerprint(settings)
        staging = settings.paths.staging_dir / manifest.run_id
        staging.mkdir(parents=True, exist_ok=True)

        def notify(stage: str, message: str) -> None:
            logger.info("[%s] %s", stage, message)
            if progress:
                progress(stage, message)

        self.catalog.record_run(manifest)
        self.audit.record(
            "ingest.start", actor=actor, subject_id=manifest.run_id,
            triggered_by=triggered_by,
        )

        try:
            candidates = self._collect_paths(paths)
            notify("scan", f"{len(candidates)} file(s) to consider")

            sources = self._stage_scan(manifest, candidates, force, notify)
            documents = self._stage_extract(manifest, sources, notify)
            transformed = self._stage_transform(manifest, documents, notify)
            self._persist_documents(transformed, manifest, documents)
            chunks = self._stage_assemble_corpus(manifest, transformed, notify)
            vectors = self._stage_embed(manifest, chunks, notify)
            self._stage_load(manifest, chunks, vectors, notify)

            manifest.status = RunStatus.SUCCEEDED
            manifest.finished_at = datetime.now(timezone.utc)
            self.audit.record(
                "ingest.succeed", actor=actor, subject_id=manifest.run_id,
                documents=manifest.documents, chunks=manifest.chunks,
                index_version=manifest.index_version,
            )
            notify("done", f"published index v{manifest.index_version}")
            return manifest

        except validate_stage.GateFailure as failure:
            manifest.status = RunStatus.FAILED
            manifest.error = str(failure)
            manifest.finished_at = datetime.now(timezone.utc)
            self.audit.record(
                "ingest.fail", actor=actor, subject_id=manifest.run_id,
                outcome="gate_failure", reason=str(failure),
            )
            logger.error("run %s failed validation: %s", manifest.run_id, failure)
            raise PipelineError(str(failure), manifest) from failure

        except Exception as exc:  # noqa: BLE001 - record then re-raise
            manifest.status = RunStatus.FAILED
            manifest.error = f"{type(exc).__name__}: {exc}"
            manifest.finished_at = datetime.now(timezone.utc)
            self.audit.record(
                "ingest.fail", actor=actor, subject_id=manifest.run_id,
                outcome="error", reason=type(exc).__name__,
            )
            logger.exception("run %s failed", manifest.run_id)
            raise PipelineError(manifest.error, manifest) from exc

        finally:
            manifest.finished_at = manifest.finished_at or datetime.now(timezone.utc)
            (staging / "manifest.json").write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            self.catalog.record_run(manifest)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def _collect_paths(self, paths: Sequence[Path | str] | None) -> list[Path]:
        if paths is None:
            roots: Iterable[Path] = [self.settings.paths.inbox_dir]
        else:
            roots = [Path(p) for p in paths]

        collected: list[Path] = []
        for root in roots:
            if root.is_dir():
                collected.extend(sorted(p for p in root.rglob("*") if p.is_file()))
            elif root.is_file():
                collected.append(root)
            else:
                logger.warning("path does not exist: %s", root)
        return collected[: self.settings.ingest.max_files_per_run]

    def _stage_scan(
        self, manifest: RunManifest, candidates: list[Path], force: bool,
        notify: ProgressCallback,
    ) -> list[SourceFile]:
        stage = _begin(manifest, "scan", len(candidates))
        settings = self.settings
        admitted: list[SourceFile] = []
        skipped = 0

        for path in candidates:
            result = scan.scan_file(
                path,
                allowed_extensions=settings.ingest.allowed_extensions,
                max_file_bytes=settings.ingest.max_file_bytes,
                enforce_content_type=settings.ingest.enforce_content_type,
                max_archive_expansion_ratio=settings.ingest.max_archive_expansion_ratio,
            )
            if not result.admitted:
                manifest.files_rejected += 1
                self.catalog.record_rejection(
                    path.name, result.sha256, result.reason or "rejected",
                    result.size_bytes,
                )
                scan.quarantine(
                    path, settings.paths.quarantine_dir, result.reason or "rejected"
                )
                notify("scan", f"rejected {path.name}: {result.reason}")
                continue

            if not force and self.catalog.is_already_ingested(result.sha256):
                skipped += 1
                logger.info("skipping %s — identical content already ingested",
                            path.name)
                continue

            source = result.source_file
            assert source is not None
            self.catalog.record_source(source)
            admitted.append(source)

        manifest.files_admitted = len(admitted)
        stage.metrics.update({
            "admitted": len(admitted),
            "rejected": manifest.files_rejected,
            "skipped_duplicates": skipped,
        })
        _end(stage, len(admitted))
        notify("scan", f"admitted {len(admitted)}, rejected "
                       f"{manifest.files_rejected}, skipped {skipped} duplicate(s)")
        return admitted

    def _stage_extract(
        self, manifest: RunManifest, sources: list[SourceFile],
        notify: ProgressCallback,
    ) -> list[RawDocument]:
        stage = _begin(manifest, "extract", len(sources))
        documents: list[RawDocument] = []
        failures: list[tuple[str, str]] = []

        for source in sources:
            try:
                documents.append(
                    extract_stage.extract(
                        source,
                        ocr=self.settings.ingest.ocr_enabled,
                        ocr_min_chars=self.settings.ingest.ocr_min_chars_per_page,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one file must not kill a run
                logger.error("extraction failed for %s: %s", source.filename, exc)
                failures.append((source.filename, str(exc)))

        report = validate_stage.validate_extraction(
            documents, self.settings.validation, failures
        )
        stage.report = report
        stage.metrics.update(report.metrics)
        _end(stage, len(documents))
        notify("extract",
               f"{len(documents)} document(s), {len(failures)} failure(s)")
        validate_stage.enforce_gates(report, self.settings.validation)
        return documents

    def _stage_transform(
        self, manifest: RunManifest, documents: list[RawDocument],
        notify: ProgressCallback,
    ) -> list[TransformedDocument]:
        stage = _begin(manifest, "transform", len(documents))
        dedup = DedupState.new(
            self.settings.transform.near_duplicate_threshold
            if self.settings.transform.drop_duplicate_chunks else 0.0
        )
        transformed = [
            transform_document(
                document, self.settings.transform, self.settings.privacy, dedup
            )
            for document in documents
        ]

        report = validate_stage.validate_transform(
            transformed, self.settings.validation, self.settings.privacy
        )
        stage.report = report
        stage.metrics.update(report.metrics)
        manifest.documents = len(transformed)
        manifest.redactions = report.metrics.get("redactions", {}) or {}
        _end(stage, sum(len(d.chunks) for d in transformed))
        notify("transform",
               f"{sum(len(d.chunks) for d in transformed)} chunk(s) from "
               f"{len(transformed)} document(s)")
        validate_stage.enforce_gates(report, self.settings.validation)
        return transformed

    def _persist_documents(
        self, transformed: list[TransformedDocument], manifest: RunManifest,
        raw: list[RawDocument],
    ) -> None:
        char_counts = {d.doc_id: d.char_count for d in raw}
        for document in transformed:
            self.catalog.record_document(
                document, manifest.run_id, char_counts.get(document.doc_id, 0)
            )
        # Chunk text lives in the index, not the catalog; keep a copy in staging
        # so a rebuild never has to re-parse the original files.
        corpus_path = (
            self.settings.paths.staging_dir / manifest.run_id / "chunks.jsonl"
        )
        with corpus_path.open("w", encoding="utf-8") as fh:
            for document in transformed:
                for chunk in document.chunks:
                    fh.write(chunk.model_dump_json() + "\n")

    def _stage_assemble_corpus(
        self, manifest: RunManifest, transformed: list[TransformedDocument],
        notify: ProgressCallback,
    ) -> list[Chunk]:
        """Combine this run's chunks with the still-active existing corpus.

        Rebuilding from the whole corpus is what keeps the index, the catalog
        and the deletions in agreement. The embedding cache makes the repeated
        work cheap.
        """
        stage = _begin(manifest, "assemble", len(transformed))
        new_chunks = [c for d in transformed for c in d.chunks]
        new_ids = {c.chunk_id for c in new_chunks}
        new_doc_ids = {d.doc_id for d in transformed}

        carried: list[Chunk] = []
        existing = self.vector_store.load()
        if existing is not None:
            active = self.catalog.active_chunk_ids()
            carried = [
                chunk for chunk in existing.chunks
                if chunk.chunk_id in active
                and chunk.chunk_id not in new_ids
                and chunk.doc_id not in new_doc_ids
            ]

        corpus = carried + new_chunks
        stage.metrics.update({
            "new_chunks": len(new_chunks),
            "carried_chunks": len(carried),
            "corpus_chunks": len(corpus),
        })
        _end(stage, len(corpus))
        notify("assemble",
               f"corpus is {len(corpus)} chunk(s) "
               f"({len(carried)} carried forward)")
        return corpus

    def _stage_embed(
        self, manifest: RunManifest, chunks: list[Chunk], notify: ProgressCallback,
    ) -> np.ndarray:
        stage = _begin(manifest, "embed", len(chunks))
        embedder = self.embedder
        vectors = embedder.embed_texts(
            [c.text for c in chunks], [c.text_sha256 for c in chunks]
        )

        existing = self.vector_store.load()
        expected_dim = existing.dimensions if existing else None
        if expected_dim and expected_dim != embedder.dimensions:
            # The model changed: a mixed index would silently return nonsense,
            # so rebuild everything at the new dimensionality instead.
            logger.warning(
                "embedding dimensionality changed %d -> %d; rebuilding index",
                expected_dim, embedder.dimensions,
            )
            expected_dim = embedder.dimensions

        report = validate_stage.validate_embeddings(chunks, vectors, expected_dim)
        stage.report = report
        stage.metrics.update({**report.metrics, **embedder.stats})
        _end(stage, len(vectors))
        notify("embed",
               f"{len(vectors)} vector(s) at {embedder.dimensions}d "
               f"({embedder.stats['cache_hits']} from cache)")
        validate_stage.enforce_gates(report, self.settings.validation)
        return vectors

    def _stage_load(
        self, manifest: RunManifest, chunks: list[Chunk], vectors: np.ndarray,
        notify: ProgressCallback,
    ) -> None:
        stage = _begin(manifest, "load", len(chunks))
        version, _ = self.vector_store.build(
            chunks, vectors,
            embedding_model=self.embedder.model_id,
            run_id=manifest.run_id,
        )

        report = validate_stage.validate_index(
            index_count=len(chunks),
            chunk_count=len(chunks),
            catalog_count=self.catalog.active_chunk_count(),
            settings=self.settings.validation,
        )
        stage.report = report
        stage.metrics.update(report.metrics)

        try:
            validate_stage.enforce_gates(report, self.settings.validation)
        except validate_stage.GateFailure:
            # Never publish an index that failed its gates. Discard the build.
            shutil.rmtree(self.vector_store.version_dir(version), ignore_errors=True)
            _end(stage, 0, status=StageStatus.FAILED)
            raise

        self.vector_store.publish(version)
        manifest.index_version = version
        manifest.chunks = len(chunks)
        manifest.vectors = int(vectors.shape[0])
        _end(stage, len(chunks))
        notify("load", f"published index v{version} with {len(chunks)} vector(s)")

    # ------------------------------------------------------------------
    def rebuild(self, *, triggered_by: str = "rebuild",
                actor: str = "system") -> RunManifest:
        """Re-index the active corpus without ingesting new files.

        Used after a deletion, a retention sweep, or a chunking/embedding
        config change.
        """
        return self.run([], triggered_by=triggered_by, actor=actor)

    def delete_document(self, doc_id: str, *, actor: str = "system",
                        rebuild: bool = True) -> bool:
        """Erase a document and remove it from the served index."""
        deleted = self.catalog.delete_document(doc_id)
        self.audit.record(
            "document.delete", actor=actor, subject_id=doc_id,
            outcome="ok" if deleted else "not_found",
        )
        if deleted and rebuild:
            try:
                self.rebuild(triggered_by="deletion", actor=actor)
            except PipelineError as exc:
                # An empty corpus fails the "empty index" gate; that is correct
                # behaviour for ingestion but expected after deleting the last
                # document, so the deletion itself still stands.
                logger.warning("rebuild after deletion did not publish: %s", exc)
        return deleted

    def apply_retention(self, *, actor: str = "system") -> list[str]:
        """Delete documents older than ``security.retention_days``."""
        days = self.settings.security.retention_days
        if not days:
            return []
        stale = self.catalog.documents_older_than(days)
        for doc_id in stale:
            self.catalog.delete_document(doc_id)
            self.audit.record(
                "document.retention_delete", actor=actor, subject_id=doc_id,
                retention_days=days,
            )
        if stale:
            try:
                self.rebuild(triggered_by="retention", actor=actor)
            except PipelineError as exc:
                logger.warning("retention rebuild did not publish: %s", exc)
        return stale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _begin(manifest: RunManifest, name: str, records_in: int) -> StageResult:
    stage = StageResult(
        name=name, status=StageStatus.RUNNING, records_in=records_in,
        started_at=datetime.now(timezone.utc),
    )
    manifest.stages.append(stage)
    return stage


def _end(stage: StageResult, records_out: int,
         status: StageStatus = StageStatus.SUCCEEDED) -> None:
    stage.records_out = records_out
    stage.status = status
    stage.finished_at = datetime.now(timezone.utc)


def _fingerprint(settings: Settings) -> str:
    """Hash of the settings that affect index contents.

    If this changes, the index should be rebuilt — the pipeline surfaces it in
    the manifest so an operator can see *why* two runs produced different
    output.
    """
    import hashlib

    relevant = {
        "chunk_size": settings.transform.chunk_size,
        "chunk_overlap": settings.transform.chunk_overlap,
        "min_chunk_chars": settings.transform.min_chunk_chars,
        "normalize_unicode": settings.transform.normalize_unicode,
        "dehyphenate": settings.transform.dehyphenate,
        "strip_repeated_headers": settings.transform.strip_repeated_headers,
        "redact_pii": settings.privacy.redact_pii,
        "pii_types": sorted(settings.privacy.pii_types),
        "embedding_backend": settings.embedding.backend,
        "embedding_model": settings.embedding.model,
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
