"""File admission control — the gate in front of the extract stage.

Anything a user can upload is hostile until checked. This module decides
whether a file is allowed into the pipeline at all, based on:

* extension allowlist
* declared vs. actual content type (magic-byte sniffing, no libmagic needed)
* size caps
* zip-bomb detection for the OOXML container formats (.docx / .pptx)
* embedded-macro detection (rejects .docm-style payloads renamed to .docx)

Rejected files are moved to the quarantine directory with a reason, never
silently dropped — an auditor needs to see what was refused and why.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..models import SourceFile, sha256_bytes

logger = logging.getLogger("jjrag.security.scan")

# (magic prefix, canonical media type, plausible extensions)
_MAGIC: list[tuple[bytes, str, set[str]]] = [
    (b"%PDF-", "application/pdf", {".pdf"}),
    (b"PK\x03\x04", "application/zip", {".docx", ".pptx", ".xlsx", ".zip"}),
    (b"\xd0\xcf\x11\xe0", "application/x-ole-storage", {".doc", ".ppt", ".xls"}),
    (b"\x1f\x8b", "application/gzip", {".gz"}),
    (b"\x7fELF", "application/x-elf", set()),
    (b"MZ", "application/x-msdownload", set()),
    (b"\x89PNG", "image/png", {".png"}),
    (b"\xff\xd8\xff", "image/jpeg", {".jpg", ".jpeg"}),
]

_TEXTUAL_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".tsv",
    ".json", ".jsonl", ".eml",
}

_EXECUTABLE_TYPES = {
    "application/x-elf", "application/x-msdownload", "application/gzip",
}


@dataclass
class ScanResult:
    path: Path
    admitted: bool
    reason: str | None = None
    media_type: str | None = None
    sha256: str = ""
    size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    source_file: SourceFile | None = None


def sniff_media_type(head: bytes) -> str | None:
    for magic, media_type, _ in _MAGIC:
        if head.startswith(magic):
            return media_type
    return None


def _looks_textual(head: bytes) -> bool:
    """Heuristic: mostly printable, decodes as UTF-8 or latin-1."""
    if not head:
        return True
    if b"\x00" in head:
        return False
    printable = sum(
        1 for b in head if 32 <= b < 127 or b in (9, 10, 13) or b >= 128
    )
    return printable / len(head) > 0.90


def _check_ooxml(path: Path, max_ratio: float) -> tuple[bool, str | None, list[str]]:
    """Validate a zip-container document: expansion ratio and macro payloads."""
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, f"corrupt archive member: {bad}", warnings
            compressed = path.stat().st_size or 1
            uncompressed = sum(i.file_size for i in zf.infolist())
            ratio = uncompressed / compressed
            if ratio > max_ratio:
                return (
                    False,
                    f"archive expansion ratio {ratio:.0f}x exceeds limit "
                    f"{max_ratio:.0f}x (possible zip bomb)",
                    warnings,
                )
            names = zf.namelist()
            if any(n.lower().endswith(".bin") and "vbaproject" in n.lower()
                   for n in names):
                return False, "document contains a VBA macro project", warnings
            for name in names:
                # Absolute paths / traversal inside the container.
                if name.startswith("/") or ".." in Path(name).parts:
                    return False, f"unsafe path inside archive: {name}", warnings
            if any("externalLink" in n for n in names):
                warnings.append("document declares external links")
    except zipfile.BadZipFile:
        return False, "not a readable OOXML/zip container", warnings
    return True, None, warnings


def scan_file(
    path: Path,
    *,
    allowed_extensions: set[str] | list[str],
    max_file_bytes: int,
    enforce_content_type: bool = True,
    max_archive_expansion_ratio: float = 120.0,
    uploaded_by: str | None = None,
    tags: list[str] | None = None,
) -> ScanResult:
    """Decide whether ``path`` may enter the pipeline."""
    allowed = {e.lower() for e in allowed_extensions}
    ext = path.suffix.lower()
    result = ScanResult(path=path, admitted=False)

    if not path.is_file():
        result.reason = "not a regular file"
        return result
    if path.is_symlink():
        result.reason = "symlinks are not accepted"
        return result

    size = path.stat().st_size
    result.size_bytes = size
    if size == 0:
        result.reason = "file is empty"
        return result
    if size > max_file_bytes:
        result.reason = f"file is {size} bytes, limit is {max_file_bytes}"
        return result
    if ext not in allowed:
        result.reason = f"extension {ext or '(none)'} is not in the allowlist"
        return result

    data = path.read_bytes()
    result.sha256 = sha256_bytes(data)
    head = data[:4096]
    sniffed = sniff_media_type(head)

    if sniffed in _EXECUTABLE_TYPES and ext not in {".gz"}:
        result.reason = f"content looks like {sniffed}, which is never accepted"
        return result

    if enforce_content_type:
        if sniffed is None:
            if ext not in _TEXTUAL_EXTENSIONS or not _looks_textual(head):
                result.reason = (
                    f"content does not match extension {ext} "
                    "(binary data in a text format)"
                )
                return result
            result.media_type = "text/plain"
        else:
            plausible = next(
                (exts for magic, mt, exts in _MAGIC if mt == sniffed), set()
            )
            if plausible and ext not in plausible:
                result.reason = (
                    f"content type {sniffed} does not match extension {ext}"
                )
                return result
            result.media_type = sniffed
    else:
        result.media_type = sniffed or "application/octet-stream"

    if ext in {".docx", ".pptx", ".xlsx"}:
        ok, reason, warnings = _check_ooxml(path, max_archive_expansion_ratio)
        result.warnings.extend(warnings)
        if not ok:
            result.reason = reason
            return result

    result.admitted = True
    result.source_file = SourceFile(
        filename=path.name,
        path=str(path),
        extension=ext,
        media_type=result.media_type,
        size_bytes=size,
        content_sha256=result.sha256,
        uploaded_by=uploaded_by,
        tags=tags or [],
    )
    return result


def quarantine(path: Path, quarantine_dir: Path, reason: str) -> Path:
    """Move a rejected file aside and drop a ``.reason.txt`` next to it."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / path.name
    counter = 1
    while target.exists():
        target = quarantine_dir / f"{path.stem}.{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), target)
    target.with_suffix(target.suffix + ".reason.txt").write_text(
        f"rejected: {reason}\n", encoding="utf-8"
    )
    logger.warning("quarantined %s: %s", path.name, reason)
    return target
