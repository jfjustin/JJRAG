"""Transform stage — raw text becomes clean, redacted, retrievable chunks.

This is where most of the retrieval quality is won or lost. The steps, in
order:

1. **Normalise** — Unicode NFKC, ligatures, smart quotes, control characters,
   whitespace. Two spellings of the same word must not become two vectors.
2. **De-hyphenate** — PDFs break words across lines ("compli-\\nance"); joining
   them back is the single highest-value cleanup for PDF corpora.
3. **Strip boilerplate** — headers/footers that repeat on every page are noise
   that crowds out real content in retrieval.
4. **Redact PII** — before anything is persisted (see :mod:`jjrag.security.pii`).
5. **Chunk** — recursively on natural boundaries, never mid-word, with overlap
   so an answer spanning a boundary is still retrievable.
6. **Deduplicate** — exact (hash) and near-duplicate (token-shingle Jaccard).
   Duplicated boilerplate otherwise dominates the top-k for generic questions.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ..config import PrivacySettings, TransformSettings
from ..models import Chunk, RawDocument, TransformedDocument
from ..security import pii

logger = logging.getLogger("jjrag.pipeline.transform")

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "—", "…": "...", " ": " ",
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
_MULTI_SPACE = re.compile(r"[ \t - ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE)

# Chunking boundaries, most-preferred first.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_text(
    text: str,
    *,
    unicode_normalize: bool = True,
    dehyphenate: bool = True,
) -> str:
    if not text:
        return ""
    if dehyphenate:
        # Do this before whitespace collapsing, while the newline is still there.
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    for src, dst in _QUOTES.items():
        text = text.replace(src, dst)
    if unicode_normalize:
        text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def find_repeated_lines(segments: list[str], min_pages: int = 3) -> set[str]:
    """Lines appearing on most pages are headers/footers, not content."""
    if len(segments) < min_pages:
        return set()
    counter: Counter[str] = Counter()
    for segment in segments:
        lines = [line.strip() for line in segment.split("\n") if line.strip()]
        # Only the top and bottom of a page can hold a running header/footer.
        for line in set(lines[:3] + lines[-3:]):
            if 3 <= len(line) <= 120:
                counter[line] += 1
    threshold = max(min_pages, int(len(segments) * 0.6))
    return {line for line, count in counter.items() if count >= threshold}


def strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    if not boilerplate and not text:
        return text
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in boilerplate:
            continue
        if _PAGE_NUMBER.fullmatch(stripped):
            continue
        kept.append(line)
    return _MULTI_NEWLINE.sub("\n\n", "\n".join(kept)).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def split_text(
    text: str, chunk_size: int, chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursive character splitter.

    Tries each separator in turn, falling back to a harder split only when a
    piece is still too long. Equivalent in spirit to LangChain's
    ``RecursiveCharacterTextSplitter`` but implemented here so the pipeline has
    no LangChain dependency and the behaviour is pinned by our own tests.

    Note: because overlap is prepended after splitting, a returned chunk can be
    up to ``chunk_size + chunk_overlap`` characters long. Validation's
    ``max_chunk_chars`` gate is set well above that.
    """
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    separators = separators or _SEPARATORS

    def _split(piece: str, seps: list[str]) -> list[str]:
        if len(piece) <= chunk_size:
            return [piece] if piece.strip() else []
        if not seps:
            # Hard wrap — only reached by text with no separators at all.
            return [
                piece[i:i + chunk_size]
                for i in range(0, len(piece), chunk_size - chunk_overlap)
            ]
        sep, rest = seps[0], seps[1:]
        if sep == "":
            return [
                piece[i:i + chunk_size]
                for i in range(0, len(piece), chunk_size - chunk_overlap)
            ]
        parts = piece.split(sep)
        out: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{sep}{part}" if buffer else part
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    out.extend(_split(buffer, rest) if len(buffer) > chunk_size
                               else [buffer])
                buffer = part
        if buffer:
            out.extend(_split(buffer, rest) if len(buffer) > chunk_size else [buffer])
        return [p for p in out if p.strip()]

    pieces = _split(text, separators)

    # Re-apply overlap between adjacent pieces: the recursive split above
    # produces disjoint pieces, and retrieval wants a sliding window.
    if chunk_overlap <= 0 or len(pieces) < 2:
        return pieces
    overlapped: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        previous = overlapped[-1]
        tail = previous[-chunk_overlap:]
        # Start the overlap at a word boundary so chunks stay readable.
        space = tail.find(" ")
        if space > 0:
            tail = tail[space + 1:]
        overlapped.append(f"{tail} {piece}".strip() if tail.strip() else piece)
    return overlapped


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _shingles(text: str, size: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)


@dataclass
class DedupState:
    """Tracks what has already been indexed, across documents within a run."""

    seen_hashes: set[str]
    seen_shingles: list[set[str]]
    threshold: float

    @classmethod
    def new(cls, threshold: float) -> DedupState:
        return cls(seen_hashes=set(), seen_shingles=[], threshold=threshold)

    def is_duplicate(self, chunk: Chunk) -> bool:
        if chunk.text_sha256 in self.seen_hashes:
            return True
        if self.threshold <= 0:
            return False
        shingles = _shingles(chunk.text)
        if not shingles:
            return False
        # Compare against a bounded window — near-dup boilerplate clusters
        # locally, and an all-pairs scan would be quadratic on large corpora.
        for previous in self.seen_shingles[-200:]:
            if jaccard(shingles, previous) >= self.threshold:
                return True
        self.seen_shingles.append(shingles)
        return False

    def remember(self, chunk: Chunk) -> None:
        self.seen_hashes.add(chunk.text_sha256)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def transform_document(
    document: RawDocument,
    settings: TransformSettings,
    privacy: PrivacySettings,
    dedup: DedupState | None = None,
) -> TransformedDocument:
    """Clean, redact and chunk one extracted document."""
    dedup = dedup or DedupState.new(
        settings.near_duplicate_threshold if settings.drop_duplicate_chunks else 0.0
    )

    normalized = [
        normalize_text(
            segment.text,
            unicode_normalize=settings.normalize_unicode,
            dehyphenate=settings.dehyphenate,
        )
        for segment in document.segments
    ]

    boilerplate: set[str] = set()
    if settings.strip_repeated_headers:
        boilerplate = find_repeated_lines(normalized)
        if boilerplate:
            logger.info(
                "doc %s: stripping %d repeated header/footer line(s)",
                document.doc_id, len(boilerplate),
            )

    result = TransformedDocument(
        doc_id=document.doc_id,
        source_id=document.source.source_id,
        filename=document.source.filename,
        title=document.title,
        metadata={
            **document.metadata,
            "extractor": document.extractor,
            "content_sha256": document.source.content_sha256,
        },
    )

    total_redactions: Counter[str] = Counter()
    ordinal = 0
    running_offset = 0

    for segment, text in zip(document.segments, normalized, strict=True):
        text = strip_boilerplate(text, boilerplate)
        if not text.strip():
            running_offset += len(segment.text)
            continue

        if privacy.redact_pii:
            redaction = pii.redact(text, privacy.pii_types)
            text = redaction.text
            total_redactions.update(redaction.counts)

        for piece in split_text(text, settings.chunk_size, settings.chunk_overlap):
            piece = piece.strip()
            if len(piece) < settings.min_chunk_chars:
                result.dropped_chunks += 1
                continue

            chunk = Chunk(
                doc_id=document.doc_id,
                source_id=document.source.source_id,
                ordinal=ordinal,
                text=piece,
                segment_ordinal=segment.ordinal,
                segment_label=segment.label,
                filename=document.source.filename,
                title=document.title,
                section_path=[document.title] if document.title else [],
                start_char=running_offset,
                end_char=running_offset + len(piece),
                metadata={
                    "extraction_method": segment.extraction_method,
                    "segment_kind": segment.kind,
                },
            ).finalize()

            if settings.drop_duplicate_chunks and dedup.is_duplicate(chunk):
                result.duplicate_chunks += 1
                continue
            dedup.remember(chunk)

            result.chunks.append(chunk)
            ordinal += 1
            running_offset += len(piece)

    result.redactions = dict(total_redactions)
    if privacy.record_redaction_counts and total_redactions:
        logger.info(
            "doc %s: redacted %s", document.doc_id, dict(total_redactions)
        )
    logger.info(
        "doc %s: %d chunk(s), %d dropped (too short), %d duplicate(s)",
        document.doc_id, len(result.chunks), result.dropped_chunks,
        result.duplicate_chunks,
    )
    return result
