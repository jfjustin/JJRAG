"""Append-only audit log.

Compliance regimes ask "who did what, when, to which document". This writes one
JSON line per privileged action (upload, ingest, delete, query, config change)
into ``data/logs/audit.jsonl``. It records identifiers and outcomes only — never
document content and never the question text unless explicitly enabled.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


class AuditLog:
    def __init__(self, path: Path | str, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        action: str,
        *,
        actor: str = "anonymous",
        subject_id: str | None = None,
        outcome: str = "ok",
        **details: Any,
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "actor": actor,
            "subject_id": subject_id,
            "outcome": outcome,
            **details,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _LOCK, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
