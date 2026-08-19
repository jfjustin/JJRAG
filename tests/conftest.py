"""Shared fixtures.

Tests use the ``hashing`` embedding backend so the suite needs no model
weights, no network and no GPU — it runs identically on a laptop and in CI.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from jjrag.config import (
    EmbeddingSettings,
    PathSettings,
    SecuritySettings,
    Settings,
)
from jjrag.pipeline.runner import Pipeline


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = tmp_path / "data"
    configured = Settings(
        environment="dev",
        log_level="ERROR",
        paths=PathSettings(
            data_dir=base,
            inbox_dir=base / "inbox",
            quarantine_dir=base / "quarantine",
            staging_dir=base / "staging",
            index_dir=base / "index",
            catalog_path=base / "catalog.sqlite3",
            cache_path=base / "cache.sqlite3",
            log_dir=base / "logs",
        ),
        embedding=EmbeddingSettings(backend="hashing", dimensions=256),
        security=SecuritySettings(enforce_local_only=False),
    )
    configured.paths.ensure()
    return configured


@pytest.fixture
def pipeline(settings: Settings) -> Pipeline:
    return Pipeline(settings)


@pytest.fixture
def sample_docs(settings: Settings) -> Path:
    inbox = settings.paths.inbox_dir
    (inbox / "policy.md").write_text(
        "# Data Retention Policy\n\n"
        "Customer records are retained for seven years from the date of last "
        "activity. Requests for exceptions go to compliance@acme.example and "
        "must be approved under procedure DR-14 before any deletion occurs.\n\n"
        "## Access Control\n\n"
        "Production access requires MFA and an approved change ticket. "
        "Contractors are never granted standing access to customer data.\n",
        encoding="utf-8",
    )
    (inbox / "notes.txt").write_text(
        "Meeting notes. Legal confirmed the seven year retention window on "
        "2024-03-02 and asked that procedure DR-14 be referenced in the policy "
        "document itself. Follow up on the contractor access question raised by "
        "the audit team before the next quarterly review.\n",
        encoding="utf-8",
    )
    (inbox / "figures.csv").write_text(
        "region,revenue,quarter\nEMEA,120000,Q1\nAPAC,98000,Q1\nAMER,150000,Q1\n",
        encoding="utf-8",
    )
    return inbox


@pytest.fixture
def docx_bytes() -> bytes:
    """A minimal but structurally valid .docx."""
    document_xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        "<w:r><w:t>Vendor Security Requirements</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>All vendors complete an annual security questionnaire "
        "covering encryption, access control and incident response.</w:t></w:r></w:p>"
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Control</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Frequency</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("[Content_Types].xml", "<Types/>")
    return buffer.getvalue()
