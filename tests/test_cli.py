"""The CLI — the interface cron jobs and CI use."""

from __future__ import annotations

from pathlib import Path

import pytest

from jjrag.cli import main
from jjrag.config import Settings


@pytest.fixture
def config_file(tmp_path: Path, settings: Settings) -> str:
    """Write the fixture settings to YAML so the CLI can load them."""
    import yaml

    payload = {
        "log_level": "ERROR",
        "paths": {
            key: str(value)
            for key, value in settings.paths.model_dump().items()
        },
        "embedding": {"backend": "hashing", "dimensions": 256},
        "security": {"enforce_local_only": False},
    }
    path = tmp_path / "jjrag.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_help_lists_every_command(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    output = capsys.readouterr().out
    for command in ("ingest", "query", "serve", "doctor", "rm", "runs", "compliance"):
        assert command in output


def test_ingest_then_docs_then_rm(
    config_file: str, sample_docs: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["--config", config_file, "ingest", str(sample_docs)]) == 0
    assert "succeeded" in capsys.readouterr().out

    assert main(["--config", config_file, "docs"]) == 0
    listing = capsys.readouterr().out
    assert "policy.md" in listing

    doc_id = next(
        line.split()[0] for line in listing.splitlines() if line.startswith("doc_")
    )
    assert main(["--config", config_file, "rm", doc_id]) == 0
    assert "erased" in capsys.readouterr().out


def test_ingest_of_an_empty_directory_fails_with_a_reason(
    config_file: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--config", config_file, "ingest", str(empty)]) == 1
    assert "failed" in capsys.readouterr().out


def test_runs_lists_history(
    config_file: str, sample_docs: Path, capsys: pytest.CaptureFixture
) -> None:
    main(["--config", config_file, "ingest", str(sample_docs)])
    capsys.readouterr()
    assert main(["--config", config_file, "runs"]) == 0
    assert "succeeded" in capsys.readouterr().out


def test_query_retrieve_only_needs_no_model(
    config_file: str, sample_docs: Path, capsys: pytest.CaptureFixture
) -> None:
    main(["--config", config_file, "ingest", str(sample_docs)])
    capsys.readouterr()
    assert main(["--config", config_file, "query", "retention period",
                 "--retrieve-only"]) == 0
    assert "seven years" in capsys.readouterr().out


def test_compliance_prints_the_attestation(
    config_file: str, capsys: pytest.CaptureFixture
) -> None:
    import json

    assert main(["--config", config_file, "compliance"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["posture"]["third_party_model_apis_enabled"] is False


def test_doctor_reports_a_missing_local_model(
    config_file: str, capsys: pytest.CaptureFixture
) -> None:
    # No Ollama in CI, so doctor must exit non-zero and say what to install.
    assert main(["--config", config_file, "doctor"]) == 1
    output = capsys.readouterr().out
    assert "ollama pull" in output.lower() or "local model" in output.lower()
