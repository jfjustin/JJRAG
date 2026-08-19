"""Catalog — the system of record for what has been ingested.

SQLite because it is a single file, needs no server, is trivially backed up,
and every compliance reviewer already knows how to read it.

It holds four things the vector index cannot:

* **Lineage** — file → document → chunks → run → index version.
* **Idempotency** — content hashes, so re-ingesting an unchanged file is a
  no-op and a changed file supersedes its previous version.
* **Erasure** — deleting a document by id, which a "right to be forgotten"
  request needs, together with the list of index versions that must be rebuilt.
* **History** — every run, its manifest and its validation issues.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import RunManifest, SourceFile, TransformedDocument

logger = logging.getLogger("jjrag.store.catalog")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    extension       TEXT,
    media_type      TEXT,
    size_bytes      INTEGER,
    content_sha256  TEXT NOT NULL,
    uploaded_at     TEXT NOT NULL,
    uploaded_by     TEXT,
    tags            TEXT,
    status          TEXT NOT NULL DEFAULT 'admitted',
    reject_reason   TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_sha ON sources(content_sha256);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    title        TEXT,
    run_id       TEXT,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    char_count   INTEGER NOT NULL DEFAULT 0,
    redactions   TEXT,
    metadata     TEXT,
    created_at   TEXT NOT NULL,
    deleted_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source_id     TEXT NOT NULL,
    ordinal       INTEGER NOT NULL,
    text_sha256   TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    segment_label TEXT,
    run_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    triggered_by  TEXT,
    files_admitted INTEGER DEFAULT 0,
    files_rejected INTEGER DEFAULT 0,
    documents     INTEGER DEFAULT 0,
    chunks        INTEGER DEFAULT 0,
    vectors       INTEGER DEFAULT 0,
    index_version INTEGER,
    error         TEXT,
    manifest      TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);

CREATE TABLE IF NOT EXISTS validation_issues (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    rule         TEXT NOT NULL,
    severity     TEXT NOT NULL,
    subject_type TEXT,
    subject_id   TEXT,
    message      TEXT NOT NULL,
    details      TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_issues_run ON validation_issues(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        if not self._conn.execute("SELECT version FROM schema_info").fetchone():
            self._conn.execute(
                "INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- sources ------------------------------------------------------------
    def record_source(self, source: SourceFile, status: str = "admitted",
                      reject_reason: str | None = None) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sources
                (source_id, filename, extension, media_type, size_bytes,
                 content_sha256, uploaded_at, uploaded_by, tags, status,
                 reject_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_id, source.filename, source.extension,
                    source.media_type, source.size_bytes, source.content_sha256,
                    source.uploaded_at.isoformat(), source.uploaded_by,
                    json.dumps(source.tags), status, reject_reason,
                ),
            )

    def record_rejection(self, filename: str, sha256: str, reason: str,
                         size_bytes: int = 0) -> None:
        from ..models import new_id

        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO sources
                (source_id, filename, extension, media_type, size_bytes,
                 content_sha256, uploaded_at, uploaded_by, tags, status,
                 reject_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'rejected', ?)
                """,
                (
                    new_id("src"), filename, Path(filename).suffix.lower(), None,
                    size_bytes, sha256 or "", _now(), None, "[]", reason,
                ),
            )

    def find_by_hash(self, content_sha256: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT s.*, d.doc_id FROM sources s "
                "LEFT JOIN documents d ON d.source_id = s.source_id "
                "WHERE s.content_sha256 = ? AND s.status = 'admitted' "
                "AND (d.deleted_at IS NULL OR d.deleted_at = '') LIMIT 1",
                (content_sha256,),
            ).fetchone()
        return dict(row) if row else None

    def is_already_ingested(self, content_sha256: str) -> bool:
        record = self.find_by_hash(content_sha256)
        return bool(record and record.get("doc_id"))

    # -- documents / chunks -------------------------------------------------
    def record_document(
        self, document: TransformedDocument, run_id: str, char_count: int = 0
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, source_id, filename, title, run_id, chunk_count,
                 char_count, redactions, metadata, created_at, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    document.doc_id, document.source_id, document.filename,
                    document.title, run_id, len(document.chunks), char_count,
                    json.dumps(document.redactions),
                    json.dumps(document.metadata, default=str), _now(),
                ),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, source_id, ordinal, text_sha256, char_count,
                 segment_label, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id, c.doc_id, c.source_id, c.ordinal,
                        c.text_sha256, c.char_count, c.segment_label, run_id,
                    )
                    for c in document.chunks
                ],
            )

    def list_documents(
        self, include_deleted: bool = False, limit: int = 500, offset: int = 0
    ) -> list[dict[str, Any]]:
        clause = "" if include_deleted else "WHERE d.deleted_at IS NULL"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT d.doc_id, d.source_id, d.filename, d.title, d.run_id,
                       d.chunk_count, d.char_count, d.redactions, d.created_at,
                       d.deleted_at, s.size_bytes, s.media_type,
                       s.content_sha256, s.uploaded_by
                FROM documents d
                LEFT JOIN sources s ON s.source_id = d.source_id
                {clause}
                ORDER BY d.created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        out = []
        for row in rows:
            record = dict(row)
            record["redactions"] = json.loads(record.get("redactions") or "{}")
            out.append(record)
        return out

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return dict(row) if row else None

    def active_chunk_count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM chunks c JOIN documents d "
                "ON d.doc_id = c.doc_id WHERE d.deleted_at IS NULL"
            ).fetchone()[0]

    def active_chunk_ids(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.chunk_id FROM chunks c JOIN documents d "
                "ON d.doc_id = c.doc_id WHERE d.deleted_at IS NULL"
            ).fetchall()
        return {r[0] for r in rows}

    def delete_document(self, doc_id: str, hard: bool = False) -> bool:
        """Erase a document.

        Soft delete marks it and excludes it from every future index build;
        hard delete removes the rows outright. Either way the caller must
        rebuild the index for the change to take effect in retrieval — the API
        does this automatically.
        """
        with self._tx() as conn:
            if hard:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
                )
            else:
                cursor = conn.execute(
                    "UPDATE documents SET deleted_at = ? WHERE doc_id = ? "
                    "AND deleted_at IS NULL",
                    (_now(), doc_id),
                )
            return cursor.rowcount > 0

    # -- runs ---------------------------------------------------------------
    def record_run(self, manifest: RunManifest) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, status, started_at, finished_at, triggered_by,
                 files_admitted, files_rejected, documents, chunks, vectors,
                 index_version, error, manifest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id, manifest.status.value,
                    manifest.started_at.isoformat(),
                    manifest.finished_at.isoformat() if manifest.finished_at else None,
                    manifest.triggered_by, manifest.files_admitted,
                    manifest.files_rejected, manifest.documents, manifest.chunks,
                    manifest.vectors, manifest.index_version, manifest.error,
                    manifest.model_dump_json(),
                ),
            )
            conn.execute(
                "DELETE FROM validation_issues WHERE run_id = ?", (manifest.run_id,)
            )
            issues = manifest.all_issues()
            if issues:
                conn.executemany(
                    """
                    INSERT INTO validation_issues
                    (run_id, stage, rule, severity, subject_type, subject_id,
                     message, details, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            manifest.run_id, issue.stage, issue.rule,
                            issue.severity.value, issue.subject_type,
                            issue.subject_id, issue.message,
                            json.dumps(issue.details, default=str), _now(),
                        )
                        for issue in issues
                    ],
                )

    def get_run(self, run_id: str) -> RunManifest | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT manifest FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunManifest.model_validate_json(row[0]) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, status, started_at, finished_at, triggered_by, "
                "files_admitted, files_rejected, documents, chunks, vectors, "
                "index_version, error FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def issues_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT stage, rule, severity, subject_type, subject_id, message "
                "FROM validation_issues WHERE run_id = ? ORDER BY "
                "CASE severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 "
                "ELSE 2 END, id",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- retention / stats --------------------------------------------------
    def documents_older_than(self, days: int) -> list[str]:
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86_400
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id, created_at FROM documents WHERE deleted_at IS NULL"
            ).fetchall()
        stale = []
        for row in rows:
            try:
                created = datetime.fromisoformat(row["created_at"]).timestamp()
            except ValueError:
                continue
            if created < cutoff:
                stale.append(row["doc_id"])
        return stale

    def stats(self) -> dict[str, Any]:
        with self._lock:
            def scalar(sql: str, *args: Any) -> int:
                return self._conn.execute(sql, args).fetchone()[0]

            return {
                "documents": scalar(
                    "SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL"
                ),
                "documents_deleted": scalar(
                    "SELECT COUNT(*) FROM documents WHERE deleted_at IS NOT NULL"
                ),
                "chunks": scalar(
                    "SELECT COUNT(*) FROM chunks c JOIN documents d "
                    "ON d.doc_id = c.doc_id WHERE d.deleted_at IS NULL"
                ),
                "sources_admitted": scalar(
                    "SELECT COUNT(*) FROM sources WHERE status = 'admitted'"
                ),
                "sources_rejected": scalar(
                    "SELECT COUNT(*) FROM sources WHERE status = 'rejected'"
                ),
                "runs": scalar("SELECT COUNT(*) FROM runs"),
                "runs_failed": scalar(
                    "SELECT COUNT(*) FROM runs WHERE status = 'failed'"
                ),
            }
