"""PII detection and redaction.

Applied during transform, before any text is written to the index, so the
persisted corpus never contains the raw identifiers. Detection is
deterministic and dependency-free: regexes plus structural validators (Luhn for
card numbers, mod-97 for IBANs) to keep the false-positive rate low enough that
redaction does not destroy the document's meaning.

Every redaction is replaced with a stable, typed placeholder — ``[EMAIL_1]``
rather than a black box — so the surrounding sentence still reads correctly and
retrieval still works. Counts per type are recorded for the audit trail; the
values themselves are never logged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from re import Pattern

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
PATTERNS: dict[str, Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}\b"
    ),
    "phone": re.compile(
        r"(?<![\d.])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])"
        r"\d{3,4}[\s.\-]\d{3,4}(?!\d)"
    ),
    "ssn": re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    ),
    "secret": re.compile(
        r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
        r"xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,})\b"
    ),
}

PLACEHOLDER = "[{kind}_{index}]"


# ---------------------------------------------------------------------------
# Structural validators — cut down false positives
# ---------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    checksum = 0
    parity = len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


def _iban_ok(value: str) -> bool:
    v = value.replace(" ", "").upper()
    if not 15 <= len(v) <= 34:
        return False
    rearranged = v[4:] + v[:4]
    digits = "".join(
        str(ord(c) - 55) if c.isalpha() else c for c in rearranged
    )
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def _not_version_or_date(value: str) -> bool:
    """Reject things like 192.168.0.1-looking version strings and 1.2.3.4 IDs."""
    return not re.fullmatch(r"(?:0|[1-9]\d{0,2})(?:\.\d{1,3}){3}", value) or (
        value.startswith(("10.", "172.", "192.168."))
    )


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "credit_card": _luhn_ok,
    "iban": _iban_ok,
    "ip_address": _not_version_or_date,
}


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    # Hashes let an auditor confirm coverage without the raw values existing
    # anywhere in the corpus or logs.
    matched_spans: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def detect(text: str, kinds: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Return the PII values found, grouped by type. Used by validation."""
    selected = list(kinds) if kinds is not None else list(PATTERNS)
    found: dict[str, list[str]] = {}
    for kind in selected:
        pattern = PATTERNS.get(kind)
        if pattern is None:
            continue
        validator = VALIDATORS.get(kind)
        hits = [
            m.group(0)
            for m in pattern.finditer(text)
            if validator is None or validator(m.group(0))
        ]
        if hits:
            found[kind] = hits
    return found


def redact(text: str, kinds: Iterable[str] | None = None) -> RedactionResult:
    """Replace detected PII with typed placeholders.

    Placeholders are numbered per type and *stable within a document*: the same
    email seen twice becomes the same ``[EMAIL_1]``, so co-reference survives
    redaction and retrieval still connects the two mentions.
    """
    selected = list(kinds) if kinds is not None else list(PATTERNS)
    counts: dict[str, int] = {}
    spans = 0
    out = text

    # Longest-first so a card number isn't half-eaten by the phone pattern.
    order = [
        k for k in ("secret", "credit_card", "iban", "ssn", "email", "phone",
                    "ip_address")
        if k in selected
    ]
    for kind in order:
        pattern = PATTERNS.get(kind)
        if pattern is None:
            continue
        validator = VALIDATORS.get(kind)
        seen: dict[str, str] = {}

        # Loop variables are bound as defaults: the closure must capture *this*
        # iteration's kind/validator/seen, not whatever they are when it runs.
        def _replace(
            match: re.Match[str],
            kind: str = kind,
            validator: Callable[[str], bool] | None = validator,
            seen: dict[str, str] = seen,
        ) -> str:
            nonlocal spans
            value = match.group(0)
            if validator is not None and not validator(value):
                return value
            key = re.sub(r"[\s\-()]", "", value).lower()
            if key not in seen:
                seen[key] = PLACEHOLDER.format(
                    kind=kind.upper(), index=len(seen) + 1
                )
            spans += 1
            return seen[key]

        out = pattern.sub(_replace, out)
        if seen:
            counts[kind] = len(seen)

    return RedactionResult(text=out, counts=counts, matched_spans=spans)


def residual_pii(text: str, kinds: Iterable[str] | None = None) -> dict[str, int]:
    """Count PII still present after redaction — a defence-in-depth check."""
    return {k: len(v) for k, v in detect(text, kinds).items()}
