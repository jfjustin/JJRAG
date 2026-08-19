"""Typed records that flow between pipeline stages.

Each stage consumes one record type and emits another, and every record keeps
its provenance (``source_id`` / ``doc_id`` / ``chunk_id``) so any answer the app
gives can be traced back to the exact bytes that were uploaded.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
class SourceFile(BaseModel):
    """A file that has been admitted into the pipeline by the scanner."""

    source_id: str = Field(default_factory=lambda: new_id("src"))
    filename: str
    path: str
    extension: str
    media_type: str | None = None
    size_bytes: int
    content_sha256: str
    uploaded_at: datetime = Field(default_factory=utcnow)
    uploaded_by: str | None = None
    tags: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    """One addressable piece of an extracted document (a page, slide, sheet…)."""

    ordinal: int
    text: str
    kind: Literal["page", "slide", "section", "row_group", "message", "body"] = "page"
    label: str | None = None          # e.g. "p. 4", "Slide 2", "Sheet1!A1:D50"
    extraction_method: str = "text"   # "text" | "ocr" | "table" | "structured"


class RawDocument(BaseModel):
    """Output of the extract stage — text plus everything we know about it."""

    doc_id: str = Field(default_factory=lambda: new_id("doc"))
    source: SourceFile
    title: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extractor: str = "unknown"
    extracted_at: datetime = Field(default_factory=utcnow)
    warnings: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.segments)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """A retrieval unit. This is what gets embedded, stored, and cited."""

    chunk_id: str = Field(default_factory=lambda: new_id("chk"))
    doc_id: str
    source_id: str
    ordinal: int
    text: str
    text_sha256: str = ""
    char_count: int = 0
    token_estimate: int = 0
    segment_ordinal: int | None = None
    segment_label: str | None = None
    filename: str = ""
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    start_char: int | None = None
    end_char: int | None = None
    redactions: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def finalize(self) -> Chunk:
        """Fill derived fields. Called once the text is final."""
        self.text_sha256 = sha256_text(self.text)
        self.char_count = len(self.text)
        # ~4 chars per token is close enough for budgeting; no tokenizer needed.
        self.token_estimate = max(1, self.char_count // 4)
        return self

    def citation(self) -> str:
        return f"{self.filename}" + (
            f" ({self.segment_label})" if self.segment_label else ""
        )


class TransformedDocument(BaseModel):
    """Output of the transform stage."""

    doc_id: str
    source_id: str
    filename: str
    title: str | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    redactions: dict[str, int] = Field(default_factory=dict)
    dropped_chunks: int = 0
    duplicate_chunks: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    rule: str
    severity: Severity
    message: str
    stage: str
    subject_type: Literal["file", "document", "chunk", "index", "run"] = "document"
    subject_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    stage: str
    checked: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def merge(self, other: ValidationReport) -> ValidationReport:
        self.checked += other.checked
        self.issues.extend(other.issues)
        self.metrics.update(other.metrics)
        return self


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    records_in: int = 0
    records_out: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    report: ValidationReport | None = None
    error: str | None = None

    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class RunManifest(BaseModel):
    """The audit record for one pipeline execution.

    Written to ``data/staging/<run_id>/manifest.json`` and mirrored into the
    catalog. It answers: what went in, what came out, what was checked, what
    was rejected, and with which configuration.
    """

    run_id: str = Field(default_factory=lambda: new_id("run"))
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    triggered_by: str = "cli"
    stages: list[StageResult] = Field(default_factory=list)
    files_admitted: int = 0
    files_rejected: int = 0
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    index_version: int | None = None
    redactions: dict[str, int] = Field(default_factory=dict)
    config_fingerprint: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def duration_s(self) -> float | None:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)

    def all_issues(self) -> list[ValidationIssue]:
        out: list[ValidationIssue] = []
        for s in self.stages:
            if s.report:
                out.extend(s.report.issues)
        return out


# ---------------------------------------------------------------------------
# Retrieval / answering
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    rank: int = 0


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    index_version: int | None = None
