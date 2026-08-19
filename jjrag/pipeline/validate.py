"""Validation stage — quality gates between every other stage.

A RAG system fails quietly: nobody sees a broken pipeline, they just get worse
answers. This module makes failure loud. Rules are registered against a stage,
run automatically at that stage's boundary, and produce
:class:`~jjrag.models.ValidationIssue` records with a severity. The run fails
when a gate is breached (configurable via ``validation.*``).

Rule catalogue
--------------
``extract``  file admitted but produced no text; garbled/binary text; very low
             yield across the batch; extractor warnings
``transform`` chunk too short/long; low alphabetic ratio (tables, OCR noise);
             residual PII after redaction; whole documents that produced zero
             chunks; schema conformance
``embed``    wrong dimensionality; non-finite values; zero vectors; count parity
``load``     index/chunk parity; catalog/index drift; empty index
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..config import PrivacySettings, ValidationSettings
from ..models import (
    Chunk,
    RawDocument,
    Severity,
    TransformedDocument,
    ValidationIssue,
    ValidationReport,
)
from ..security import pii

logger = logging.getLogger("jjrag.pipeline.validate")

_PRINTABLE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e -￿]")
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------
# Text quality metrics
# ---------------------------------------------------------------------------
def garbled_ratio(text: str) -> float:
    """Share of characters that no ordinary document should contain."""
    if not text:
        return 0.0
    return len(_PRINTABLE.findall(text)) / len(text)


def alpha_ratio(text: str) -> float:
    """Share of letters. Very low means a table, a number dump, or OCR noise."""
    if not text:
        return 0.0
    return len(_ALPHA.findall(text)) / len(text)


def repetition_ratio(text: str) -> float:
    """Share of tokens that are repeats — catches stuck-extractor output."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 10:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
Rule = Callable[..., Iterable[ValidationIssue]]
_RULES: dict[str, list[tuple[str, Rule]]] = {}


def rule(stage: str, name: str) -> Callable[[Rule], Rule]:
    def decorator(fn: Rule) -> Rule:
        _RULES.setdefault(stage, []).append((name, fn))
        return fn
    return decorator


def registered_rules() -> dict[str, list[str]]:
    return {stage: [name for name, _ in rules] for stage, rules in _RULES.items()}


def _issue(
    rule_name: str, severity: Severity, message: str, stage: str,
    subject_type: str = "document", subject_id: str | None = None,
    **details: Any,
) -> ValidationIssue:
    return ValidationIssue(
        rule=rule_name, severity=severity, message=message, stage=stage,
        subject_type=subject_type, subject_id=subject_id, details=details,
    )


# ---------------------------------------------------------------------------
# Extract-stage rules
# ---------------------------------------------------------------------------
def validate_extraction(
    documents: Sequence[RawDocument],
    settings: ValidationSettings,
    failed_files: Sequence[tuple[str, str]] = (),
) -> ValidationReport:
    report = ValidationReport(stage="extract", checked=len(documents))
    productive = 0

    for document in documents:
        text = document.text
        if not text.strip():
            report.issues.append(_issue(
                "extract.empty_text", Severity.ERROR,
                f"{document.source.filename} produced no extractable text "
                "(scanned PDF, empty file, or unsupported internal format)",
                "extract", "document", document.doc_id,
                filename=document.source.filename,
            ))
            continue
        productive += 1

        ratio = garbled_ratio(text)
        if ratio > settings.max_garbled_ratio:
            report.issues.append(_issue(
                "extract.garbled_text", Severity.ERROR,
                f"{document.source.filename} is {ratio:.0%} non-printable "
                "characters — extraction likely produced binary noise",
                "extract", "document", document.doc_id, garbled_ratio=ratio,
            ))

        repetition = repetition_ratio(text)
        if repetition > 0.95:
            report.issues.append(_issue(
                "extract.degenerate_text", Severity.WARNING,
                f"{document.source.filename} is {repetition:.0%} repeated tokens",
                "extract", "document", document.doc_id, repetition_ratio=repetition,
            ))

        for warning in document.warnings:
            report.issues.append(_issue(
                "extract.extractor_warning", Severity.WARNING,
                f"{document.source.filename}: {warning}",
                "extract", "document", document.doc_id,
            ))

    for filename, reason in failed_files:
        report.issues.append(_issue(
            "extract.failed", Severity.ERROR,
            f"{filename} could not be extracted: {reason}",
            "extract", "file", filename,
        ))

    total = len(documents) + len(failed_files)
    yield_ratio = productive / total if total else 1.0
    report.metrics.update({
        "documents": len(documents),
        "documents_with_text": productive,
        "failed_files": len(failed_files),
        "extraction_yield": round(yield_ratio, 4),
        "total_chars": sum(d.char_count for d in documents),
    })

    if total and yield_ratio < settings.min_extraction_char_ratio:
        report.issues.append(_issue(
            "extract.low_yield", Severity.ERROR,
            f"only {productive}/{total} files produced text "
            f"({yield_ratio:.0%}, gate is {settings.min_extraction_char_ratio:.0%})",
            "extract", "run",
        ))
    return report


# ---------------------------------------------------------------------------
# Transform-stage rules
# ---------------------------------------------------------------------------
def validate_transform(
    documents: Sequence[TransformedDocument],
    settings: ValidationSettings,
    privacy: PrivacySettings,
) -> ValidationReport:
    all_chunks = [c for d in documents for c in d.chunks]
    report = ValidationReport(stage="transform", checked=len(all_chunks))

    for document in documents:
        if not document.chunks:
            report.issues.append(_issue(
                "transform.no_chunks", Severity.ERROR,
                f"{document.filename} produced no chunks after cleaning "
                "(all content was boilerplate, duplicate, or below the "
                "minimum chunk length)",
                "transform", "document", document.doc_id,
            ))

    short = long = low_alpha = 0
    for chunk in all_chunks:
        if chunk.char_count < settings.min_chunk_chars:
            short += 1
            report.issues.append(_issue(
                "transform.chunk_too_short", Severity.WARNING,
                f"chunk {chunk.ordinal} of {chunk.filename} is "
                f"{chunk.char_count} chars (minimum {settings.min_chunk_chars})",
                "transform", "chunk", chunk.chunk_id,
            ))
        if chunk.char_count > settings.max_chunk_chars:
            long += 1
            report.issues.append(_issue(
                "transform.chunk_too_long", Severity.ERROR,
                f"chunk {chunk.ordinal} of {chunk.filename} is "
                f"{chunk.char_count} chars (maximum {settings.max_chunk_chars}) — "
                "it may not fit the embedding model's context window",
                "transform", "chunk", chunk.chunk_id,
            ))
        if alpha_ratio(chunk.text) < settings.min_alpha_ratio:
            low_alpha += 1
            report.issues.append(_issue(
                "transform.low_alpha_ratio", Severity.WARNING,
                f"chunk {chunk.ordinal} of {chunk.filename} is mostly "
                "non-alphabetic (table, figures, or OCR noise)",
                "transform", "chunk", chunk.chunk_id,
            ))
        if not chunk.text_sha256 or chunk.char_count != len(chunk.text):
            report.issues.append(_issue(
                "transform.schema_violation", Severity.ERROR,
                f"chunk {chunk.chunk_id} has inconsistent derived fields "
                "(finalize() was not called)",
                "transform", "chunk", chunk.chunk_id,
            ))

    # Defence in depth: prove redaction actually worked.
    if privacy.redact_pii:
        residual: dict[str, int] = {}
        for chunk in all_chunks:
            for kind, count in pii.residual_pii(chunk.text, privacy.pii_types).items():
                residual[kind] = residual.get(kind, 0) + count
        if residual:
            severity = (
                Severity.ERROR if privacy.fail_on_residual_pii else Severity.WARNING
            )
            report.issues.append(_issue(
                "transform.residual_pii", severity,
                "PII survived redaction: "
                + ", ".join(f"{k}×{v}" for k, v in sorted(residual.items())),
                "transform", "run", details_types=list(residual),
            ))
        report.metrics["residual_pii"] = residual

    lengths = [c.char_count for c in all_chunks]
    report.metrics.update({
        "chunks": len(all_chunks),
        "documents": len(documents),
        "chunks_too_short": short,
        "chunks_too_long": long,
        "chunks_low_alpha": low_alpha,
        "duplicate_chunks": sum(d.duplicate_chunks for d in documents),
        "dropped_chunks": sum(d.dropped_chunks for d in documents),
        "mean_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "redactions": {
            k: sum(d.redactions.get(k, 0) for d in documents)
            for k in {k for d in documents for k in d.redactions}
        },
    })
    return report


# ---------------------------------------------------------------------------
# Embed-stage rules
# ---------------------------------------------------------------------------
def validate_embeddings(
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    expected_dim: int | None = None,
) -> ValidationReport:
    report = ValidationReport(stage="embed", checked=len(vectors))

    if len(vectors) != len(chunks):
        report.issues.append(_issue(
            "embed.count_mismatch", Severity.ERROR,
            f"{len(vectors)} vectors for {len(chunks)} chunks — "
            "embeddings and chunks are out of sync",
            "embed", "run",
        ))

    dims = {len(v) for v in vectors}
    if len(dims) > 1:
        report.issues.append(_issue(
            "embed.inconsistent_dimensions", Severity.ERROR,
            f"vectors have mixed dimensionality: {sorted(dims)}",
            "embed", "run",
        ))
    elif dims and expected_dim and next(iter(dims)) != expected_dim:
        report.issues.append(_issue(
            "embed.unexpected_dimensions", Severity.ERROR,
            f"vectors are {next(iter(dims))}-dimensional, index expects "
            f"{expected_dim} — the embedding model changed; rebuild the index",
            "embed", "index",
        ))

    degenerate = 0
    non_finite = 0
    for i, vector in enumerate(vectors):
        if any(not math.isfinite(x) for x in vector):
            non_finite += 1
            chunk_id = chunks[i].chunk_id if i < len(chunks) else None
            report.issues.append(_issue(
                "embed.non_finite", Severity.ERROR,
                "embedding contains NaN or infinity",
                "embed", "chunk", chunk_id,
            ))
        elif all(x == 0.0 for x in vector):
            degenerate += 1
            chunk_id = chunks[i].chunk_id if i < len(chunks) else None
            report.issues.append(_issue(
                "embed.zero_vector", Severity.ERROR,
                "embedding is the zero vector — this chunk can never be retrieved",
                "embed", "chunk", chunk_id,
            ))

    report.metrics.update({
        "vectors": len(vectors),
        "dimensions": next(iter(dims), 0),
        "zero_vectors": degenerate,
        "non_finite_vectors": non_finite,
    })
    return report


# ---------------------------------------------------------------------------
# Load-stage rules
# ---------------------------------------------------------------------------
def validate_index(
    index_count: int, chunk_count: int, catalog_count: int,
    settings: ValidationSettings,
) -> ValidationReport:
    report = ValidationReport(stage="load", checked=index_count)

    if settings.require_index_parity and index_count != chunk_count:
        report.issues.append(_issue(
            "load.index_parity", Severity.ERROR,
            f"index holds {index_count} vectors but the run produced "
            f"{chunk_count} chunks",
            "load", "index",
        ))
    if catalog_count != index_count:
        report.issues.append(_issue(
            "load.catalog_drift", Severity.ERROR,
            f"catalog lists {catalog_count} chunks but the index holds "
            f"{index_count} vectors — they have drifted apart",
            "load", "index",
        ))
    if index_count == 0:
        report.issues.append(_issue(
            "load.empty_index", Severity.ERROR,
            "the index is empty — no question can be answered from it",
            "load", "index",
        ))

    report.metrics.update({
        "index_vectors": index_count,
        "run_chunks": chunk_count,
        "catalog_chunks": catalog_count,
    })
    return report


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
class GateFailure(RuntimeError):
    """Raised when a validation gate is breached and the run must stop."""

    def __init__(self, report: ValidationReport, message: str) -> None:
        super().__init__(message)
        self.report = report


def enforce_gates(report: ValidationReport, settings: ValidationSettings) -> None:
    """Raise :class:`GateFailure` if the report breaches configured thresholds."""
    if not settings.enabled:
        return

    errors = report.errors
    if len(errors) > settings.max_error_issues:
        preview = "; ".join(i.message for i in errors[:5])
        raise GateFailure(
            report,
            f"{report.stage} stage failed validation: {len(errors)} error(s) "
            f"(limit {settings.max_error_issues}). {preview}",
        )

    if report.checked:
        warning_ratio = len(report.warnings) / report.checked
        if warning_ratio > settings.max_warning_ratio:
            raise GateFailure(
                report,
                f"{report.stage} stage failed validation: "
                f"{warning_ratio:.0%} of records raised warnings "
                f"(limit {settings.max_warning_ratio:.0%})",
            )
