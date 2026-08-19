"""Security and compliance behaviour — the guarantees this project sells."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from jjrag.security import egress, pii, scan


class TestEgressGuard:
    def setup_method(self) -> None:
        egress.uninstall()

    def teardown_method(self) -> None:
        egress.uninstall()

    def test_blocks_external_hosts(self) -> None:
        egress.install(["http://localhost:11434"])
        with pytest.raises(egress.EgressBlocked):
            socket.getaddrinfo("api.openai.com", 443)
        with pytest.raises(egress.EgressBlocked):
            socket.create_connection(("api.anthropic.com", 443), timeout=1)

    def test_allows_loopback(self) -> None:
        egress.install([])
        assert socket.getaddrinfo("127.0.0.1", 8000)
        assert egress.is_allowed("localhost")

    def test_records_blocked_attempts_for_audit(self) -> None:
        egress.install([])
        with pytest.raises(egress.EgressBlocked):
            socket.getaddrinfo("evil.example", 443)
        assert ("evil.example", 443) in egress.blocked_attempts()

    def test_explicit_allowlist_admits_named_host(self) -> None:
        egress.install(["http://models.internal:11434"])
        assert egress.is_allowed("models.internal")
        assert not egress.is_allowed("models.external")


class TestPIIRedaction:
    def test_redacts_every_configured_type(self) -> None:
        text = (
            "Reach ada@example.com or call +1 415 555 0132. "
            "Card 4111 1111 1111 1111, SSN 123-45-6789, "
            "token sk-abcdefghijklmnopqrst, host 10.0.0.7, "
            "IBAN GB82WEST12345698765432."
        )
        result = pii.redact(text)
        assert set(result.counts) == {
            "email", "phone", "credit_card", "ssn", "secret", "ip_address", "iban",
        }
        assert not pii.residual_pii(result.text)

    def test_placeholders_are_stable_within_a_document(self) -> None:
        result = pii.redact("ada@example.com wrote; reply to ada@example.com")
        assert result.text.count("[EMAIL_1]") == 2

    def test_luhn_check_rejects_non_card_digit_runs(self) -> None:
        assert "credit_card" not in pii.detect("order 1234 5678 9012 3457")

    def test_leaves_ordinary_prose_untouched(self) -> None:
        text = "The retention window is seven years under procedure DR-14."
        assert pii.redact(text).text == text


class TestFileAdmission:
    def _scan(self, path: Path, **overrides):
        options = {
            "allowed_extensions": [".txt", ".pdf", ".docx", ".md"],
            "max_file_bytes": 1024 * 1024,
        }
        options.update(overrides)
        return scan.scan_file(path, **options)

    def test_admits_a_plain_text_file(self, tmp_path: Path) -> None:
        target = tmp_path / "ok.txt"
        target.write_text("hello world")
        assert self._scan(target).admitted

    def test_rejects_executable_content(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.txt"
        target.write_bytes(b"MZ\x90\x00 this is a windows binary")
        result = self._scan(target)
        assert not result.admitted and "x-msdownload" in (result.reason or "")

    def test_rejects_extension_content_mismatch(self, tmp_path: Path) -> None:
        target = tmp_path / "fake.pdf"
        target.write_bytes(b"\x00\x01\x02 definitely not a pdf")
        assert not self._scan(target).admitted

    def test_rejects_disallowed_extension(self, tmp_path: Path) -> None:
        target = tmp_path / "script.sh"
        target.write_text("#!/bin/sh\nrm -rf /\n")
        assert not self._scan(target).admitted

    def test_rejects_oversized_file(self, tmp_path: Path) -> None:
        target = tmp_path / "big.txt"
        target.write_text("x" * 5000)
        assert not self._scan(target, max_file_bytes=1000).admitted

    def test_rejects_empty_file(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.txt"
        target.touch()
        assert not self._scan(target).admitted

    def test_rejects_zip_bomb_ratio(self, tmp_path: Path) -> None:
        import zipfile

        target = tmp_path / "bomb.docx"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "A" * 2_000_000)
        result = self._scan(target, allowed_extensions=[".docx"])
        assert not result.admitted and "expansion ratio" in (result.reason or "")

    def test_rejects_macro_enabled_document(self, tmp_path: Path) -> None:
        import zipfile

        target = tmp_path / "macro.docx"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("word/document.xml", "<w:document/>")
            archive.writestr("word/vbaProject.bin", "payload")
        result = self._scan(target, allowed_extensions=[".docx"])
        assert not result.admitted and "macro" in (result.reason or "")

    def test_quarantine_preserves_the_file_and_reason(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.txt"
        target.write_bytes(b"MZ\x90\x00")
        quarantine_dir = tmp_path / "quarantine"
        moved = scan.quarantine(target, quarantine_dir, "looks executable")
        assert moved.exists() and not target.exists()
        reason_file = moved.with_suffix(moved.suffix + ".reason.txt")
        assert "looks executable" in reason_file.read_text()


class TestLogRedaction:
    def test_log_records_are_scrubbed(self, caplog) -> None:
        import logging

        from jjrag.observability.logging import RedactingFilter

        logger = logging.getLogger("jjrag.test.redaction")
        logger.addFilter(RedactingFilter())
        with caplog.at_level(logging.INFO):
            logger.info("uploaded by ada@example.com")
        assert "ada@example.com" not in caplog.text
        assert "EMAIL_REDACTED" in caplog.text
