"""Central configuration for the JJRAG pipeline.

Every knob lives here so a deployment can be described by one YAML file plus
environment variables — which is what an auditor will ask for. Nothing in this
module reaches the network; it only decides what the rest of the system is
*allowed* to do.

Resolution order (lowest to highest precedence):

1. defaults defined below
2. ``config/jjrag.yaml`` (or the file named by ``JJRAG_CONFIG_FILE``)
3. environment variables prefixed ``JJRAG_`` (nested via ``__``)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_FILE = "config/jjrag.yaml"


class PathSettings(BaseModel):
    """Where the pipeline keeps its data. All paths are local to the host."""

    data_dir: Path = Path("data")
    inbox_dir: Path = Path("data/inbox")          # raw uploads land here
    quarantine_dir: Path = Path("data/quarantine")  # rejected by the scanner
    staging_dir: Path = Path("data/staging")      # per-run intermediate artifacts
    index_dir: Path = Path("data/index")          # versioned vector indexes
    catalog_path: Path = Path("data/catalog.sqlite3")
    cache_path: Path = Path("data/embedding_cache.sqlite3")
    log_dir: Path = Path("data/logs")

    def ensure(self) -> None:
        for p in (
            self.data_dir,
            self.inbox_dir,
            self.quarantine_dir,
            self.staging_dir,
            self.index_dir,
            self.log_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)


class IngestSettings(BaseModel):
    """Admission control for incoming files (the E of ETL)."""

    max_file_bytes: int = 100 * 1024 * 1024
    max_files_per_run: int = 2_000
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".txt", ".md", ".markdown",
            ".html", ".htm", ".csv", ".tsv", ".json", ".jsonl", ".eml", ".pptx",
        ]
    )
    # Reject files whose magic bytes disagree with their extension.
    enforce_content_type: bool = True
    # Refuse archives/office files that expand beyond this ratio (zip bombs).
    max_archive_expansion_ratio: float = 120.0
    ocr_enabled: bool = False          # requires tesseract on the host
    ocr_min_chars_per_page: int = 40   # below this, a PDF page looks scanned


class TransformSettings(BaseModel):
    """Text normalisation and chunking (the T of ETL)."""

    chunk_size: int = 1_000
    chunk_overlap: int = 150
    min_chunk_chars: int = 80
    normalize_unicode: bool = True
    dehyphenate: bool = True
    strip_repeated_headers: bool = True
    drop_duplicate_chunks: bool = True
    near_duplicate_threshold: float = 0.92  # 0 disables near-dup detection
    language_hint: str | None = None

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info: Any) -> int:
        size = info.data.get("chunk_size", 1_000)
        if v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


class PrivacySettings(BaseModel):
    """Compliance controls applied to document text before it is stored."""

    redact_pii: bool = True
    pii_types: list[str] = Field(
        default_factory=lambda: [
            "email", "phone", "ssn", "credit_card", "iban", "ip_address", "secret",
        ]
    )
    # Fail the run if PII survives redaction (defence in depth).
    fail_on_residual_pii: bool = True
    # Keep a hashed record of what was redacted so audits can prove coverage.
    record_redaction_counts: bool = True


class EmbeddingSettings(BaseModel):
    """Local embedding backend. No hosted embedding services are supported."""

    backend: Literal["sentence-transformers", "ollama", "hashing"] = (
        "sentence-transformers"
    )
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "nomic-embed-text"
    dimensions: int | None = None  # inferred at runtime; enforced once known
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    device: str = "cpu"

    @property
    def active_model(self) -> str:
        """The model identifier this backend will actually use."""
        if self.backend == "ollama":
            return self.ollama_model
        if self.backend == "hashing":
            return f"hashing-{self.dimensions or 384}"
        return self.model


class VectorStoreSettings(BaseModel):
    backend: Literal["numpy", "faiss"] = "numpy"
    metric: Literal["cosine", "l2"] = "cosine"
    # Keep this many previous index versions for rollback.
    keep_versions: int = 3


class RetrievalSettings(BaseModel):
    top_k: int = 5
    candidate_k: int = 20
    hybrid: bool = True           # dense + BM25 fused with reciprocal rank fusion
    rrf_k: int = 60
    mmr_lambda: float = 0.7       # 1.0 = pure relevance, 0.0 = pure diversity
    min_score: float = 0.0
    max_context_chars: int = 12_000


class LLMSettings(BaseModel):
    """Local generation only — this project deliberately has no cloud path."""

    provider: Literal["ollama"] = "ollama"
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    num_ctx: int = 8_192
    max_tokens: int = 1_024
    request_timeout_s: int = 600
    system_prompt: str = (
        "You are a careful document analyst. Answer using ONLY the provided "
        "excerpts. Cite the excerpt numbers you relied on, like [1] or [2]. "
        "If the excerpts do not contain the answer, say so plainly instead of "
        "guessing. Never invent sources, figures, or quotations."
    )


class ValidationSettings(BaseModel):
    """Quality gates. A run fails when a gate is breached."""

    enabled: bool = True
    max_error_issues: int = 0            # any error-severity issue fails the run
    max_warning_ratio: float = 0.25      # warnings / records
    min_extraction_char_ratio: float = 0.5   # docs yielding text / docs admitted
    min_chunk_chars: int = 80
    max_chunk_chars: int = 4_000
    max_garbled_ratio: float = 0.30      # non-printable / total chars in a doc
    min_alpha_ratio: float = 0.35        # letters / total chars in a chunk
    require_index_parity: bool = True    # #vectors must equal #chunks
    fail_fast: bool = False


class SecuritySettings(BaseModel):
    """Runtime posture. Defaults are the compliance-safe ones."""

    # Blocks every outbound socket except loopback + the configured Ollama host.
    enforce_local_only: bool = True
    extra_allowed_hosts: list[str] = Field(default_factory=list)
    # HTTP layer
    api_token: str | None = None          # bearer token for the write endpoints
    allow_anonymous_read: bool = True
    cors_allow_origins: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 60
    upload_rate_limit_per_minute: int = 12
    max_request_bytes: int = 100 * 1024 * 1024
    trusted_proxy_header: str | None = "cf-connecting-ip"  # set by Cloudflare
    audit_log_enabled: bool = True
    retention_days: int | None = None     # None = keep until explicitly deleted


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"   # bind loopback; expose via tunnel/reverse proxy
    port: int = 8000
    root_path: str = ""
    public_base_url: str | None = None


class Settings(BaseSettings):
    """Top-level settings object used by every component."""

    model_config = SettingsConfigDict(
        env_prefix="JJRAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "JJRAG"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    paths: PathSettings = Field(default_factory=PathSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    transform: TransformSettings = Field(default_factory=TransformSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def load(cls, config_file: str | Path | None = None) -> Settings:
        """Build settings from YAML + environment."""
        path = Path(
            config_file
            or os.getenv("JJRAG_CONFIG_FILE")
            or DEFAULT_CONFIG_FILE
        )
        file_values: dict[str, Any] = {}
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            file_values = loaded
        return cls(**file_values)

    def describe_compliance(self) -> dict[str, Any]:
        """Machine-readable attestation of the privacy-relevant posture.

        Served by ``GET /api/compliance`` so anyone — including an auditor with
        no shell access — can check how the running instance is configured.
        """
        return {
            "generation_provider": self.llm.provider,
            "generation_model": self.llm.model,
            "generation_host": self.llm.host,
            "embedding_backend": self.embedding.backend,
            "embedding_model": self.embedding.active_model,
            "third_party_model_apis_enabled": False,
            "egress_restricted_to_localhost": self.security.enforce_local_only,
            "extra_allowed_hosts": self.security.extra_allowed_hosts,
            "pii_redaction_enabled": self.privacy.redact_pii,
            "pii_types_redacted": self.privacy.pii_types,
            "validation_gates_enabled": self.validation.enabled,
            "audit_log_enabled": self.security.audit_log_enabled,
            "retention_days": self.security.retention_days,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings.load()


def reset_settings_cache() -> None:
    """Drop the cached singleton — used by tests and by ``jjrag serve --reload``."""
    get_settings.cache_clear()
