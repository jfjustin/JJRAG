"""Extractors: every supported format, plus the failure modes that matter."""

from __future__ import annotations

from pathlib import Path

import pytest

from jjrag.models import SourceFile
from jjrag.pipeline import extract


def source_for(path: Path) -> SourceFile:
    return SourceFile(
        filename=path.name, path=str(path), extension=path.suffix.lower(),
        size_bytes=path.stat().st_size, content_sha256="test",
    )


def test_registry_covers_the_documented_formats() -> None:
    assert {".pdf", ".docx", ".pptx", ".txt", ".md", ".html", ".csv", ".json",
            ".eml"} <= set(extract.supported_extensions())


def test_text_extraction(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello world", encoding="utf-8")
    document = extract.extract(source_for(path))
    assert document.text == "hello world"
    assert document.extractor == "text"


def test_text_extraction_survives_bad_encoding(tmp_path: Path) -> None:
    path = tmp_path / "latin.txt"
    path.write_bytes("café costs €3".encode("cp1252"))
    document = extract.extract(source_for(path))
    assert "caf" in document.text


def test_markdown_splits_on_headings(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nIntro.\n\n## Second\n\nBody.\n", encoding="utf-8")
    document = extract.extract(source_for(path))
    assert document.title == "Title"
    assert [segment.label for segment in document.segments] == ["Title", "Second"]


def test_html_strips_scripts_and_reads_title(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Report</title></head><body>"
        "<p>Visible text</p><script>steal()</script></body></html>",
        encoding="utf-8",
    )
    document = extract.extract(source_for(path))
    assert document.title == "Report"
    assert "Visible text" in document.text
    assert "steal" not in document.text


def test_csv_repeats_headers_so_chunks_are_self_describing(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,amount\nAda,100\nGrace,250\n", encoding="utf-8")
    document = extract.extract(source_for(path))
    assert document.metadata["row_count"] == 2
    assert "name: Ada" in document.text and "amount: 250" in document.text


def test_csv_detects_semicolon_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "eu.csv"
    path.write_text("name;amount\nAda;100\nGrace;250\n", encoding="utf-8")
    document = extract.extract(source_for(path))
    assert "name: Ada" in document.text


def test_json_is_flattened_to_readable_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"policy": {"retention_years": 7}}', encoding="utf-8")
    document = extract.extract(source_for(path))
    assert "policy.retention_years: 7" in document.text


def test_jsonl_records_become_segments(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    document = extract.extract(source_for(path))
    assert len(document.segments) == 2


def test_docx_reads_headings_paragraphs_and_tables(
    tmp_path: Path, docx_bytes: bytes
) -> None:
    path = tmp_path / "vendor.docx"
    path.write_bytes(docx_bytes)
    document = extract.extract(source_for(path))
    assert document.title == "Vendor Security Requirements"
    assert "annual security questionnaire" in document.text
    assert "Control | Frequency" in document.text


def test_pptx_makes_one_segment_per_slide(tmp_path: Path) -> None:
    import io
    import zipfile

    namespace = "http://schemas.openxmlformats.org/drawingml/2006/main"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i, text in enumerate(["First slide", "Second slide"], start=1):
            archive.writestr(
                f"ppt/slides/slide{i}.xml",
                f'<?xml version="1.0"?><p xmlns:a="{namespace}">'
                f"<a:t xmlns:a='{namespace}'>{text}</a:t></p>",
            )
    path = tmp_path / "deck.pptx"
    path.write_bytes(buffer.getvalue())
    document = extract.extract(source_for(path))
    assert [segment.label for segment in document.segments] == ["Slide 1", "Slide 2"]
    assert "Second slide" in document.text


def test_eml_keeps_headers_as_context(tmp_path: Path) -> None:
    path = tmp_path / "mail.eml"
    path.write_text(
        "From: ada@example.com\nTo: grace@example.com\n"
        "Subject: Retention question\n\nHow long do we keep records?\n",
        encoding="utf-8",
    )
    document = extract.extract(source_for(path))
    assert document.title == "Retention question"
    assert "How long do we keep records?" in document.text


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")
    with pytest.raises(extract.ExtractionError):
        extract.extract(source_for(path))


def test_corrupt_docx_is_rejected(tmp_path: Path) -> None:
    import zipfile

    path = tmp_path / "broken.docx"
    path.write_bytes(b"PK\x03\x04 not really a docx")
    with pytest.raises((extract.ExtractionError, zipfile.BadZipFile)):
        extract.extract(source_for(path))


def test_invalid_json_raises_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(extract.ExtractionError):
        extract.extract(source_for(path))


def test_pdf_extraction_when_pypdf_is_available(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as handle:
        writer.write(handle)

    document = extract.extract(source_for(path))
    assert document.extractor == "pypdf"
    assert document.metadata["page_count"] == 1
    # A blank page yields no text — the extractor must say so, not fail silently.
    assert document.warnings
