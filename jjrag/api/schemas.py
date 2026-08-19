"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadedFileInfo(BaseModel):
    filename: str
    size_bytes: int
    accepted: bool
    reason: str | None = None
    source_id: str | None = None


class UploadResponse(BaseModel):
    accepted: list[UploadedFileInfo] = Field(default_factory=list)
    rejected: list[UploadedFileInfo] = Field(default_factory=list)
    message: str


class IngestRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Re-ingest files whose content has already been indexed.",
    )


class StageSummary(BaseModel):
    name: str
    status: str
    duration_s: float | None = None
    records_in: int = 0
    records_out: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: int = 0
    warnings: int = 0


class IssueSummary(BaseModel):
    stage: str
    rule: str
    severity: str
    message: str
    subject_type: str | None = None
    subject_id: str | None = None


class RunResponse(BaseModel):
    run_id: str
    status: str
    started_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    files_admitted: int = 0
    files_rejected: int = 0
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    index_version: int | None = None
    redactions: dict[str, int] = Field(default_factory=dict)
    stages: list[StageSummary] = Field(default_factory=list)
    issues: list[IssueSummary] = Field(default_factory=list)
    error: str | None = None


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    title: str | None = None
    chunk_count: int = 0
    char_count: int = 0
    size_bytes: int | None = None
    media_type: str | None = None
    created_at: str | None = None
    redactions: dict[str, int] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    model: str | None = None


class CitationOut(BaseModel):
    rank: int
    filename: str
    segment_label: str | None = None
    text: str
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    doc_id: str
    chunk_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    model: str
    latency_ms: int
    index_version: int | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    index_version: int | None = None
    indexed_chunks: int = 0
    indexed_documents: int = 0
    local_model_available: bool = False
    local_models: list[str] = Field(default_factory=list)
    embedding_backend: str = ""
    embedding_dimensions: int | None = None


class ComplianceResponse(BaseModel):
    posture: dict[str, Any]
    blocked_egress_attempts: list[dict[str, Any]] = Field(default_factory=list)
    egress_guard_active: bool = False
    catalog: dict[str, Any] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str
