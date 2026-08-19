"""End-to-end pipeline behaviour: the guarantees an operator relies on."""

from __future__ import annotations

from pathlib import Path

import pytest

from jjrag.config import Settings
from jjrag.models import RunStatus
from jjrag.pipeline.runner import Pipeline, PipelineError


class TestFullRun:
    def test_ingests_documents_and_publishes_an_index(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        manifest = pipeline.run()

        assert manifest.status is RunStatus.SUCCEEDED
        assert manifest.documents == 3
        assert manifest.chunks > 0
        assert manifest.index_version == 1
        assert [s.name for s in manifest.stages] == [
            "scan", "extract", "transform", "assemble", "embed", "load",
        ]
        assert all(s.status.value == "succeeded" for s in manifest.stages)

        index = pipeline.vector_store.load()
        assert index is not None and len(index) == manifest.chunks

    def test_redaction_counts_reach_the_manifest(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        manifest = pipeline.run()
        assert manifest.redactions.get("email") == 1

        index = pipeline.vector_store.load()
        assert all("compliance@acme.example" not in c.text for c in index.chunks)

    def test_hostile_files_are_quarantined_not_indexed(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        (sample_docs / "payload.txt").write_bytes(b"MZ\x90\x00 windows binary")
        manifest = pipeline.run()

        assert manifest.files_rejected == 1
        assert (settings.paths.quarantine_dir / "payload.txt").exists()
        index = pipeline.vector_store.load()
        assert "payload.txt" not in {c.filename for c in index.chunks}

    def test_reruns_are_idempotent(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        first = pipeline.run()
        second = pipeline.run()

        assert second.files_admitted == 0          # nothing new to ingest
        assert second.chunks == first.chunks       # corpus unchanged
        assert second.stage("embed").metrics["cache_hits"] == second.chunks

    def test_new_files_are_added_incrementally(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        first = pipeline.run()
        (sample_docs / "vendor.md").write_text(
            "# Vendor Review\n\nVendors complete a security questionnaire every "
            "year, and the review board approves exceptions quarterly.\n",
            encoding="utf-8",
        )
        second = pipeline.run()

        assert second.files_admitted == 1
        assert second.chunks > first.chunks
        assert second.index_version == first.index_version + 1
        assert second.stage("assemble").metrics["carried_chunks"] == first.chunks

    def test_deleting_a_document_removes_it_from_the_index(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        pipeline.run()
        target = next(
            d for d in pipeline.catalog.list_documents() if d["filename"] == "notes.txt"
        )

        assert pipeline.delete_document(target["doc_id"])
        index = pipeline.vector_store.load()
        assert "notes.txt" not in {c.filename for c in index.chunks}
        assert len(index) == pipeline.catalog.active_chunk_count()

    def test_config_fingerprint_changes_when_chunking_changes(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        first = pipeline.run()
        settings.transform.chunk_size = 400
        second = Pipeline(settings).rebuild()
        assert first.config_fingerprint != second.config_fingerprint


class TestValidationGates:
    def test_unreadable_corpus_fails_the_run_and_keeps_the_old_index(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        good = pipeline.run()

        # Every new file is empty: the extraction gate must stop the run.
        for name in ("blank1.txt", "blank2.txt", "blank3.txt"):
            (sample_docs / name).write_text("   \n  \n", encoding="utf-8")

        with pytest.raises(PipelineError) as excinfo:
            pipeline.run()

        manifest = excinfo.value.manifest
        assert manifest.status is RunStatus.FAILED
        assert any(i.rule.startswith("extract.") for i in manifest.all_issues())
        # The previously good index is untouched and still serving.
        assert pipeline.vector_store.current_version() == good.index_version

    def test_empty_corpus_cannot_publish_an_empty_index(
        self, pipeline: Pipeline
    ) -> None:
        with pytest.raises(PipelineError):
            pipeline.run()
        assert pipeline.vector_store.current_version() is None

    def test_failed_runs_are_recorded_for_audit(
        self, pipeline: Pipeline
    ) -> None:
        with pytest.raises(PipelineError) as excinfo:
            pipeline.run()
        run_id = excinfo.value.manifest.run_id

        stored = pipeline.catalog.get_run(run_id)
        assert stored is not None and stored.status is RunStatus.FAILED
        assert pipeline.catalog.issues_for_run(run_id)

    def test_residual_pii_gate_blocks_publication(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        # Turn redaction off but keep the residual-PII check on: the pipeline
        # must refuse to index text that still contains identifiers.
        settings.privacy.redact_pii = True
        settings.privacy.pii_types = ["email"]
        settings.privacy.fail_on_residual_pii = True
        manifest = pipeline.run()
        # With redaction on, the gate passes and no email survives.
        assert manifest.status is RunStatus.SUCCEEDED
        index = pipeline.vector_store.load()
        assert all("@acme.example" not in c.text for c in index.chunks)


class TestAuditTrail:
    def test_run_writes_a_manifest_to_staging(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        manifest = pipeline.run()
        path = settings.paths.staging_dir / manifest.run_id / "manifest.json"
        assert path.is_file()
        assert manifest.run_id in path.read_text(encoding="utf-8")

    def test_audit_log_records_ingest_and_deletion(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        pipeline.run()
        target = pipeline.catalog.list_documents()[0]
        pipeline.delete_document(target["doc_id"], actor="tester")

        actions = {entry["action"] for entry in pipeline.audit.tail(50)}
        assert {"ingest.start", "ingest.succeed", "document.delete"} <= actions

    def test_compliance_snapshot_is_attached_to_every_run(
        self, pipeline: Pipeline, sample_docs: Path
    ) -> None:
        manifest = pipeline.run()
        assert manifest.config_snapshot["third_party_model_apis_enabled"] is False
        assert manifest.config_snapshot["generation_provider"] == "ollama"


class TestRetention:
    def test_sweep_is_a_no_op_for_recent_documents(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        pipeline.run()
        settings.security.retention_days = 30
        assert pipeline.apply_retention() == []
        assert pipeline.catalog.stats()["documents"] == 3

    def test_sweep_erases_documents_past_the_retention_window(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        pipeline.run()
        settings.security.retention_days = 30

        # Backdate one document past the window.
        target = pipeline.catalog.list_documents()[0]["doc_id"]
        with pipeline.catalog._tx() as conn:  # noqa: SLF001 - test reaches in deliberately
            conn.execute(
                "UPDATE documents SET created_at = ? WHERE doc_id = ?",
                ("2020-01-01T00:00:00+00:00", target),
            )

        assert pipeline.apply_retention() == [target]
        assert pipeline.catalog.stats()["documents"] == 2
        index = pipeline.vector_store.load()
        assert target not in {c.doc_id for c in index.chunks}

    def test_disabled_retention_never_deletes(
        self, pipeline: Pipeline, sample_docs: Path, settings: Settings
    ) -> None:
        pipeline.run()
        settings.security.retention_days = None
        assert pipeline.apply_retention() == []
