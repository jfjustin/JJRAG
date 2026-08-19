"""Extract stage — bytes on disk become :class:`RawDocument` records.

Design notes:

* One extractor per format, registered by extension, all returning the same
  record shape so downstream stages never branch on file type.
* Extraction is *segment-aware*: a PDF page, a slide, a CSV row-group and an
  email body all become :class:`Segment` objects with a human label. That label
  is what a citation shows the user ("report.pdf, p. 12"), which is the
  difference between a usable answer and an unverifiable one.
* Heavy parsers are imported lazily so a deployment that only handles text
  files does not need the PDF stack installed.
* A failure in one file never aborts the run — it is recorded as a warning and
  the file is reported in the validation stage.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from ..models import RawDocument, Segment, SourceFile

logger = logging.getLogger("jjrag.pipeline.extract")

Extractor = Callable[[Path, SourceFile], RawDocument]
_REGISTRY: dict[str, Extractor] = {}


class ExtractionError(RuntimeError):
    pass


def register(*extensions: str) -> Callable[[Extractor], Extractor]:
    def decorator(fn: Extractor) -> Extractor:
        for ext in extensions:
            _REGISTRY[ext.lower()] = fn
        return fn
    return decorator


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY)


def _base(source: SourceFile, extractor: str) -> RawDocument:
    return RawDocument(
        source=source,
        title=Path(source.filename).stem,
        extractor=extractor,
        metadata={"filename": source.filename, "extension": source.extension},
    )


# ---------------------------------------------------------------------------
# Plain text / markdown
# ---------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """Decode robustly: UTF-8, then UTF-8-sig, then latin-1 as a last resort."""
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("latin-1", errors="replace")


@register(".txt")
def extract_text(path: Path, source: SourceFile) -> RawDocument:
    doc = _base(source, "text")
    doc.segments = [Segment(ordinal=0, text=_read_text(path), kind="body",
                            label=source.filename)]
    return doc


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@register(".md", ".markdown")
def extract_markdown(path: Path, source: SourceFile) -> RawDocument:
    """Split markdown on headings so section context survives into chunks."""
    text = _read_text(path)
    doc = _base(source, "markdown")

    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        doc.segments = [Segment(ordinal=0, text=text, kind="body",
                                label=source.filename)]
        return doc

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            doc.segments.append(
                Segment(ordinal=0, text=preamble, kind="section", label="preamble")
            )

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[match.end():end].strip()
        if not body and not heading:
            continue
        doc.segments.append(
            Segment(
                ordinal=len(doc.segments),
                text=f"{heading}\n\n{body}".strip(),
                kind="section",
                label=heading[:80] or f"section {i + 1}",
            )
        )
    if matches:
        doc.title = matches[0].group(2).strip() or doc.title
    return doc


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_TAG = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLOCK_END = re.compile(
    r"</(p|div|section|article|h[1-6]|li|tr|table|blockquote)>", re.IGNORECASE
)


@register(".html", ".htm")
def extract_html(path: Path, source: SourceFile) -> RawDocument:
    """Tag-strip without pulling in a parser dependency.

    Good enough for the documents this pipeline targets (exported reports,
    saved pages). If a deployment needs DOM-accurate extraction, install
    ``beautifulsoup4`` and swap this extractor out via :func:`register`.
    """
    raw = _read_text(path)
    doc = _base(source, "html")

    title_match = _TITLE.search(raw)
    if title_match:
        doc.title = html.unescape(_TAG.sub("", title_match.group(1))).strip() or doc.title

    body = _SCRIPT_STYLE.sub(" ", raw)
    body = _BLOCK_END.sub("\n\n", body)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = _TAG.sub(" ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    doc.segments = [Segment(ordinal=0, text=body, kind="body", label=source.filename)]
    return doc


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
@register(".pdf")
def extract_pdf(path: Path, source: SourceFile) -> RawDocument:
    """Page-by-page text extraction, with an optional OCR fallback.

    Scanned pages yield little or no text; those are flagged as warnings so the
    validation stage can tell the operator that a document needs OCR rather
    than silently indexing a handful of empty pages.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ExtractionError(
            "pypdf is required for PDF extraction (pip install pypdf)"
        ) from exc

    doc = _base(source, "pypdf")
    reader = PdfReader(str(path))

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # some PDFs are "encrypted" with an empty owner pw
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"PDF is password protected: {exc}") from exc

    info = getattr(reader, "metadata", None)
    if info:
        title = (info.get("/Title") or "").strip() if hasattr(info, "get") else ""
        if title:
            doc.title = title
        doc.metadata["pdf_author"] = str(info.get("/Author") or "") if hasattr(info, "get") else ""
    doc.metadata["page_count"] = len(reader.pages)

    empty_pages = 0
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't kill the doc
            doc.warnings.append(f"page {i + 1}: extraction failed ({exc})")
            text = ""
        if len(text.strip()) < 1:
            empty_pages += 1
        doc.segments.append(
            Segment(ordinal=i, text=text, kind="page", label=f"p. {i + 1}")
        )

    if empty_pages:
        doc.warnings.append(
            f"{empty_pages}/{len(reader.pages)} pages produced no text "
            "(likely scanned images — enable OCR to index them)"
        )
        doc.metadata["empty_pages"] = empty_pages
    return doc


def ocr_pdf_pages(path: Path, doc: RawDocument, min_chars: int) -> RawDocument:
    """Best-effort OCR for pages that produced no text.

    Optional and off by default: it needs ``pdf2image`` + ``pytesseract`` and a
    tesseract binary on the host. Runs entirely locally, like everything else.
    """
    try:
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        doc.warnings.append("OCR requested but pytesseract/pdf2image not installed")
        return doc

    targets = [s for s in doc.segments if len(s.text.strip()) < min_chars]
    if not targets:
        return doc

    for segment in targets:
        try:
            images = convert_from_path(
                str(path), first_page=segment.ordinal + 1,
                last_page=segment.ordinal + 1, dpi=200,
            )
            if not images:
                continue
            text = pytesseract.image_to_string(images[0]) or ""
            if len(text.strip()) > len(segment.text.strip()):
                segment.text = text
                segment.extraction_method = "ocr"
        except Exception as exc:  # noqa: BLE001
            doc.warnings.append(f"OCR failed on page {segment.ordinal + 1}: {exc}")
    doc.metadata["ocr_pages"] = sum(
        1 for s in doc.segments if s.extraction_method == "ocr"
    )
    return doc


# ---------------------------------------------------------------------------
# DOCX / PPTX (OOXML, parsed from the zip container — no extra dependency)
# ---------------------------------------------------------------------------
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


@register(".docx")
def extract_docx(path: Path, source: SourceFile) -> RawDocument:
    """Read paragraphs and tables straight out of ``word/document.xml``.

    Using the standard library instead of ``python-docx``/``docx2txt`` keeps the
    dependency surface small — which matters when every dependency is something
    a security review has to clear.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    doc = _base(source, "docx")
    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("word/document.xml")
        except KeyError as exc:
            raise ExtractionError("not a Word document (no word/document.xml)") from exc

    root = ET.fromstring(xml)
    body = root.find(f"{_W_NS}body")
    if body is None:
        raise ExtractionError("Word document has no body")

    blocks: list[str] = []
    headings: list[str] = []

    def paragraph_text(node: ET.Element) -> str:
        parts = [t.text or "" for t in node.iter(f"{_W_NS}t")]
        # <w:tab/> and <w:br/> carry layout meaning worth preserving.
        return "".join(parts).strip()

    def style_of(node: ET.Element) -> str:
        props = node.find(f"{_W_NS}pPr")
        if props is None:
            return ""
        style = props.find(f"{_W_NS}pStyle")
        return (style.get(f"{_W_NS}val") or "") if style is not None else ""

    for child in body:
        tag = child.tag
        if tag == f"{_W_NS}p":
            text = paragraph_text(child)
            if not text:
                continue
            style = style_of(child)
            if style.lower().startswith("heading"):
                headings.append(text)
                blocks.append(f"\n{text}\n")
            else:
                blocks.append(text)
        elif tag == f"{_W_NS}tbl":
            rows: list[str] = []
            for row in child.iter(f"{_W_NS}tr"):
                cells = [
                    " ".join(t.text or "" for t in cell.iter(f"{_W_NS}t")).strip()
                    for cell in row.findall(f"{_W_NS}tc")
                ]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))

    text = "\n\n".join(b for b in blocks if b.strip())
    if headings:
        doc.title = headings[0][:200]
        doc.metadata["headings"] = headings[:50]
    doc.segments = [Segment(ordinal=0, text=text, kind="body", label=source.filename)]
    return doc


@register(".pptx")
def extract_pptx(path: Path, source: SourceFile) -> RawDocument:
    """One segment per slide, including speaker notes."""
    import xml.etree.ElementTree as ET
    import zipfile

    doc = _base(source, "pptx")
    with zipfile.ZipFile(path) as zf:
        slide_names = sorted(
            (n for n in zf.namelist()
             if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", Path(n).stem).group(1)),
        )
        if not slide_names:
            raise ExtractionError("no slides found in presentation")

        for i, name in enumerate(slide_names):
            root = ET.fromstring(zf.read(name))
            texts = [t.text or "" for t in root.iter(f"{_A_NS}t")]
            body = "\n".join(t for t in texts if t.strip())

            notes_name = name.replace("ppt/slides/", "ppt/notesSlides/notesSlide")
            notes_name = re.sub(r"slide(\d+)\.xml$", r"\1.xml", notes_name)
            if notes_name in zf.namelist():
                notes_root = ET.fromstring(zf.read(notes_name))
                notes = "\n".join(
                    t.text or "" for t in notes_root.iter(f"{_A_NS}t") if t.text
                ).strip()
                if notes:
                    body += f"\n\n[Speaker notes]\n{notes}"

            doc.segments.append(
                Segment(ordinal=i, text=body, kind="slide", label=f"Slide {i + 1}")
            )
    return doc


# ---------------------------------------------------------------------------
# Tabular / structured
# ---------------------------------------------------------------------------
@register(".csv", ".tsv")
def extract_csv(path: Path, source: SourceFile) -> RawDocument:
    """Serialise rows as ``header: value`` lines, batched into row-groups.

    Naively dumping CSV into a chunker produces chunks where the header has
    scrolled out of view and the numbers are meaningless. Repeating the header
    per record keeps every chunk self-describing and retrievable.
    """
    doc = _base(source, "csv")
    text = _read_text(path)
    delimiter = "\t" if source.extension == ".tsv" else None
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;|\t").delimiter
        except csv.Error:
            delimiter = ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        doc.warnings.append("CSV file is empty")
        return doc

    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(header)]
    doc.metadata["columns"] = header

    rows_per_segment = 50
    buffer: list[str] = []
    segment_index = 0
    row_count = 0

    def flush(first_row: int, last_row: int) -> None:
        nonlocal buffer, segment_index
        if not buffer:
            return
        doc.segments.append(
            Segment(
                ordinal=segment_index,
                text="\n\n".join(buffer),
                kind="row_group",
                label=f"rows {first_row}–{last_row}",
                extraction_method="structured",
            )
        )
        segment_index += 1
        buffer = []

    first_in_batch = 1
    for row in reader:
        row_count += 1
        record = "; ".join(
            f"{header[i] if i < len(header) else f'column_{i + 1}'}: {value.strip()}"
            for i, value in enumerate(row)
            if value and value.strip()
        )
        if record:
            buffer.append(f"Row {row_count} — {record}")
        if len(buffer) >= rows_per_segment:
            flush(first_in_batch, row_count)
            first_in_batch = row_count + 1
    flush(first_in_batch, row_count)

    doc.metadata["row_count"] = row_count
    return doc


@register(".json", ".jsonl")
def extract_json(path: Path, source: SourceFile) -> RawDocument:
    """Flatten JSON into readable ``path: value`` lines."""
    doc = _base(source, "json")
    text = _read_text(path)

    records: list[object] = []
    if source.extension == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                doc.warnings.append(f"line {line_no}: invalid JSON ({exc.msg})")
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"invalid JSON: {exc.msg}") from exc
        records = parsed if isinstance(parsed, list) else [parsed]

    def flatten(value: object, prefix: str = "") -> Iterable[str]:
        if isinstance(value, dict):
            for key, item in value.items():
                yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from flatten(item, f"{prefix}[{i}]")
        else:
            if value not in (None, "", []):
                yield f"{prefix}: {value}"

    for i, record in enumerate(records):
        lines = list(flatten(record))
        if lines:
            doc.segments.append(
                Segment(
                    ordinal=i, text="\n".join(lines), kind="section",
                    label=f"record {i + 1}", extraction_method="structured",
                )
            )
    doc.metadata["record_count"] = len(records)
    return doc


@register(".eml")
def extract_eml(path: Path, source: SourceFile) -> RawDocument:
    """Email: headers as context, plain-text body preferred over HTML."""
    from email import policy
    from email.parser import BytesParser

    doc = _base(source, "eml")
    message = BytesParser(policy=policy.default).parse(path.open("rb"))

    headers = {
        key: str(message.get(key, ""))
        for key in ("From", "To", "Cc", "Subject", "Date")
        if message.get(key)
    }
    doc.title = headers.get("Subject") or doc.title
    doc.metadata.update({f"email_{k.lower()}": v for k, v in headers.items()})

    body = ""
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and not part.get_filename():
                body += part.get_content()
            elif ctype == "text/html" and not body and not part.get_filename():
                stripped = _TAG.sub(" ", part.get_content())
                body += html.unescape(re.sub(r"\s+", " ", stripped))
    else:
        body = message.get_content() if message.get_content_type() == "text/plain" else ""

    header_block = "\n".join(f"{k}: {v}" for k, v in headers.items())
    doc.segments = [
        Segment(
            ordinal=0, text=f"{header_block}\n\n{body}".strip(),
            kind="message", label=headers.get("Subject", source.filename)[:80],
        )
    ]
    return doc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def extract(source: SourceFile, *, ocr: bool = False,
            ocr_min_chars: int = 40) -> RawDocument:
    """Run the extractor registered for this file's extension."""
    path = Path(source.path)
    extractor = _REGISTRY.get(source.extension.lower())
    if extractor is None:
        raise ExtractionError(f"no extractor registered for {source.extension}")

    document = extractor(path, source)
    if ocr and source.extension.lower() == ".pdf":
        document = ocr_pdf_pages(path, document, ocr_min_chars)

    logger.info(
        "extracted %s via %s: %d segment(s), %d chars",
        source.filename, document.extractor, len(document.segments),
        document.char_count,
    )
    return document
