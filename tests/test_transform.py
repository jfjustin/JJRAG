"""Normalisation, chunking and deduplication."""

from __future__ import annotations

import pytest

from jjrag.config import PrivacySettings, TransformSettings
from jjrag.models import RawDocument, Segment, SourceFile
from jjrag.pipeline.transform import (
    DedupState,
    find_repeated_lines,
    jaccard,
    normalize_text,
    split_text,
    strip_boilerplate,
    transform_document,
)


class TestNormalisation:
    def test_joins_words_hyphenated_across_lines(self) -> None:
        assert normalize_text("compli-\nance matters") == "compliance matters"

    def test_folds_ligatures_and_smart_quotes(self) -> None:
        assert normalize_text("ﬁle “quoted”") == 'file "quoted"'

    def test_removes_control_characters(self) -> None:
        assert "\x00" not in normalize_text("bad\x00text")

    def test_collapses_runaway_whitespace(self) -> None:
        assert normalize_text("a   b\n\n\n\nc") == "a b\n\nc"

    def test_is_idempotent(self) -> None:
        once = normalize_text("ﬁle  “x”\n\n\ny")
        assert normalize_text(once) == once


class TestBoilerplate:
    def test_detects_lines_repeated_across_pages(self) -> None:
        pages = [f"ACME Confidential\nbody {i}\nPage {i}" for i in range(6)]
        assert "ACME Confidential" in find_repeated_lines(pages)

    def test_leaves_short_documents_alone(self) -> None:
        assert find_repeated_lines(["header\nbody"]) == set()

    def test_strips_detected_lines_and_page_numbers(self) -> None:
        text = "ACME Confidential\nreal content\n12"
        assert strip_boilerplate(text, {"ACME Confidential"}) == "real content"


class TestChunking:
    def test_short_text_stays_one_chunk(self) -> None:
        assert split_text("short text", 1000, 100) == ["short text"]

    def test_respects_the_size_budget(self) -> None:
        chunks = split_text("word " * 2000, chunk_size=500, chunk_overlap=50)
        assert all(len(c) <= 550 for c in chunks)
        assert len(chunks) > 1

    def test_overlap_carries_context_between_chunks(self) -> None:
        chunks = split_text("A" * 300 + " " + "B" * 300, chunk_size=320, chunk_overlap=60)
        assert len(chunks) >= 2

    def test_prefers_paragraph_boundaries(self) -> None:
        text = "First paragraph here.\n\nSecond paragraph here."
        assert split_text(text, 30, 5)[0].startswith("First paragraph")

    def test_rejects_overlap_larger_than_chunk(self) -> None:
        with pytest.raises(ValueError):
            split_text("text", 100, 100)

    def test_empty_input_returns_no_chunks(self) -> None:
        assert split_text("", 100, 10) == []


class TestDeduplication:
    def test_jaccard_bounds(self) -> None:
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_exact_duplicates_are_detected(self) -> None:
        from jjrag.models import Chunk

        state = DedupState.new(0.9)
        first = Chunk(doc_id="d", source_id="s", ordinal=0, text="same text").finalize()
        second = Chunk(doc_id="d", source_id="s", ordinal=1, text="same text").finalize()
        assert not state.is_duplicate(first)
        state.remember(first)
        assert state.is_duplicate(second)


def _document(text: str, filename: str = "doc.txt") -> RawDocument:
    source = SourceFile(
        filename=filename, path=f"/tmp/{filename}", extension=".txt",
        size_bytes=len(text), content_sha256="hash",
    )
    return RawDocument(
        source=source, title="Doc",
        segments=[Segment(ordinal=0, text=text, kind="body", label="body")],
    )


class TestTransformDocument:
    def test_produces_finalised_chunks_with_provenance(self) -> None:
        document = _document("Retention is seven years. " * 40)
        result = transform_document(
            document, TransformSettings(), PrivacySettings(redact_pii=False)
        )
        assert result.chunks
        for chunk in result.chunks:
            assert chunk.text_sha256 and chunk.char_count == len(chunk.text)
            assert chunk.filename == "doc.txt" and chunk.doc_id == document.doc_id

    def test_redacts_pii_before_chunking(self) -> None:
        document = _document(
            "Contact ada@example.com about the policy. " * 10
        )
        result = transform_document(
            document, TransformSettings(), PrivacySettings(redact_pii=True)
        )
        assert result.redactions.get("email") == 1
        assert all("ada@example.com" not in c.text for c in result.chunks)

    def test_drops_chunks_below_the_minimum_length(self) -> None:
        document = _document("tiny")
        result = transform_document(
            document, TransformSettings(min_chunk_chars=80),
            PrivacySettings(redact_pii=False),
        )
        assert result.chunks == [] and result.dropped_chunks == 1

    def test_duplicate_content_across_documents_is_indexed_once(self) -> None:
        body = "Identical boilerplate paragraph that appears in both files. " * 5
        settings = TransformSettings()
        privacy = PrivacySettings(redact_pii=False)
        state = DedupState.new(settings.near_duplicate_threshold)

        first = transform_document(_document(body, "a.txt"), settings, privacy, state)
        second = transform_document(_document(body, "b.txt"), settings, privacy, state)
        assert first.chunks and not second.chunks
        assert second.duplicate_chunks > 0
