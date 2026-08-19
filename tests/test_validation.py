"""Validation rules and the gates that stop a bad run from publishing."""

from __future__ import annotations

import numpy as np
import pytest

from jjrag.config import PrivacySettings, ValidationSettings
from jjrag.models import (
    Chunk,
    RawDocument,
    Segment,
    Severity,
    SourceFile,
    TransformedDocument,
    ValidationReport,
)
from jjrag.pipeline.validate import (
    GateFailure,
    alpha_ratio,
    enforce_gates,
    garbled_ratio,
    repetition_ratio,
    validate_embeddings,
    validate_extraction,
    validate_index,
    validate_transform,
)


def _raw(text: str, filename: str = "doc.txt") -> RawDocument:
    source = SourceFile(
        filename=filename, path=f"/tmp/{filename}", extension=".txt",
        size_bytes=max(len(text), 1), content_sha256="hash",
    )
    return RawDocument(
        source=source, segments=[Segment(ordinal=0, text=text, kind="body")]
    )


def _chunk(text: str, ordinal: int = 0) -> Chunk:
    return Chunk(
        doc_id="doc", source_id="src", ordinal=ordinal, text=text,
        filename="doc.txt",
    ).finalize()


class TestTextMetrics:
    def test_garbled_ratio_flags_binary_noise(self) -> None:
        assert garbled_ratio("clean text") == 0.0
        assert garbled_ratio("\x01\x02\x03text") > 0.3

    def test_alpha_ratio_distinguishes_prose_from_numbers(self) -> None:
        assert alpha_ratio("plain words") > 0.8
        assert alpha_ratio("1234 5678 9012") < 0.1

    def test_repetition_ratio_catches_stuck_output(self) -> None:
        assert repetition_ratio("a b c d e f g h i j k") < 0.2
        assert repetition_ratio("same " * 40) > 0.9


class TestExtractionRules:
    def test_empty_document_is_an_error(self) -> None:
        report = validate_extraction([_raw("")], ValidationSettings())
        assert not report.passed
        assert any(i.rule == "extract.empty_text" for i in report.errors)

    def test_garbled_document_is_an_error(self) -> None:
        report = validate_extraction([_raw("\x01\x02\x03\x04\x05")], ValidationSettings())
        assert not report.passed

    def test_failed_files_are_reported(self) -> None:
        report = validate_extraction(
            [_raw("fine text")], ValidationSettings(),
            failed_files=[("broken.pdf", "password protected")],
        )
        assert any(i.rule == "extract.failed" for i in report.errors)

    def test_low_yield_across_the_batch_fails(self) -> None:
        documents = [_raw("good text")] + [_raw("") for _ in range(4)]
        report = validate_extraction(documents, ValidationSettings())
        assert any(i.rule == "extract.low_yield" for i in report.errors)

    def test_healthy_batch_passes(self) -> None:
        report = validate_extraction(
            [_raw("Perfectly ordinary policy text.")], ValidationSettings()
        )
        assert report.passed
        assert report.metrics["extraction_yield"] == 1.0


class TestTransformRules:
    def test_document_with_no_chunks_is_an_error(self) -> None:
        document = TransformedDocument(
            doc_id="d", source_id="s", filename="empty.txt", chunks=[]
        )
        report = validate_transform(
            [document], ValidationSettings(), PrivacySettings(redact_pii=False)
        )
        assert any(i.rule == "transform.no_chunks" for i in report.errors)

    def test_oversized_chunk_is_an_error(self) -> None:
        document = TransformedDocument(
            doc_id="d", source_id="s", filename="big.txt",
            chunks=[_chunk("x " * 3000)],
        )
        report = validate_transform(
            [document], ValidationSettings(), PrivacySettings(redact_pii=False)
        )
        assert any(i.rule == "transform.chunk_too_long" for i in report.errors)

    def test_residual_pii_fails_the_stage(self) -> None:
        document = TransformedDocument(
            doc_id="d", source_id="s", filename="leak.txt",
            chunks=[_chunk("please email ada@example.com for the policy document")],
        )
        report = validate_transform(
            [document], ValidationSettings(),
            PrivacySettings(redact_pii=True, fail_on_residual_pii=True),
        )
        assert any(i.rule == "transform.residual_pii" for i in report.errors)

    def test_clean_chunks_pass_and_report_metrics(self) -> None:
        document = TransformedDocument(
            doc_id="d", source_id="s", filename="ok.txt",
            chunks=[_chunk("A perfectly reasonable paragraph of policy text. " * 3)],
        )
        report = validate_transform(
            [document], ValidationSettings(), PrivacySettings(redact_pii=True)
        )
        assert report.passed
        assert report.metrics["chunks"] == 1


class TestEmbeddingRules:
    def test_count_mismatch_is_an_error(self) -> None:
        report = validate_embeddings([_chunk("a")], np.zeros((2, 4)) + 1)
        assert any(i.rule == "embed.count_mismatch" for i in report.errors)

    def test_zero_vector_is_an_error(self) -> None:
        report = validate_embeddings([_chunk("a")], np.zeros((1, 4)))
        assert any(i.rule == "embed.zero_vector" for i in report.errors)

    def test_non_finite_values_are_an_error(self) -> None:
        vectors = np.array([[np.nan, 1.0, 0.0, 0.0]])
        report = validate_embeddings([_chunk("a")], vectors)
        assert any(i.rule == "embed.non_finite" for i in report.errors)

    def test_dimension_change_is_an_error(self) -> None:
        report = validate_embeddings([_chunk("a")], np.ones((1, 8)), expected_dim=4)
        assert any(i.rule == "embed.unexpected_dimensions" for i in report.errors)

    def test_valid_embeddings_pass(self) -> None:
        report = validate_embeddings([_chunk("a")], np.ones((1, 4)), expected_dim=4)
        assert report.passed


class TestIndexRules:
    def test_parity_violation_is_an_error(self) -> None:
        report = validate_index(10, 12, 12, ValidationSettings())
        assert any(i.rule == "load.index_parity" for i in report.errors)

    def test_catalog_drift_is_an_error(self) -> None:
        report = validate_index(10, 10, 7, ValidationSettings())
        assert any(i.rule == "load.catalog_drift" for i in report.errors)

    def test_empty_index_is_an_error(self) -> None:
        report = validate_index(0, 0, 0, ValidationSettings())
        assert any(i.rule == "load.empty_index" for i in report.errors)

    def test_consistent_index_passes(self) -> None:
        assert validate_index(10, 10, 10, ValidationSettings()).passed


class TestGates:
    def test_errors_breach_the_gate(self) -> None:
        report = validate_index(0, 0, 0, ValidationSettings())
        with pytest.raises(GateFailure):
            enforce_gates(report, ValidationSettings())

    def test_too_many_warnings_breach_the_gate(self) -> None:
        from jjrag.models import ValidationIssue

        report = ValidationReport(stage="transform", checked=4)
        report.issues.extend(
            ValidationIssue(
                rule="transform.chunk_too_short", severity=Severity.WARNING,
                message="chunk is short", stage="transform",
            )
            for _ in range(3)
        )
        with pytest.raises(GateFailure):
            enforce_gates(report, ValidationSettings(max_warning_ratio=0.25))

    def test_disabled_validation_never_raises(self) -> None:
        report = validate_index(0, 0, 0, ValidationSettings())
        enforce_gates(report, ValidationSettings(enabled=False))
