"""Shared service state, auth and rate limiting for the API."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from ..config import Settings
from ..observability.audit import AuditLog
from ..pipeline.embed import Embedder
from ..pipeline.runner import Pipeline
from ..retrieval.answer import AnswerEngine
from ..retrieval.search import Retriever
from ..store.vectorstore import VectorIndex

logger = logging.getLogger("jjrag.api.deps")


class ServiceState:
    """Process-wide state: pipeline, index, retriever, answer engine.

    The retriever is rebuilt whenever the published index version changes, so a
    successful ingest is visible to queries immediately without a restart.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = Pipeline(settings)
        self.audit = AuditLog(
            settings.paths.log_dir / "audit.jsonl",
            enabled=settings.security.audit_log_enabled,
        )
        self._lock = threading.Lock()
        self._retriever: Retriever | None = None
        self._loaded_version: int | None = None
        self._engine: AnswerEngine | None = None
        self.ingest_lock = threading.Lock()
        self.last_run_id: str | None = None

    @property
    def embedder(self) -> Embedder:
        return self.pipeline.embedder

    @property
    def index(self) -> VectorIndex | None:
        retriever = self.retriever
        return retriever.index if retriever else None

    @property
    def retriever(self) -> Retriever | None:
        """Reload lazily when a new index version has been published."""
        with self._lock:
            current = self.pipeline.vector_store.current_version()
            if current is None:
                self._retriever = None
                self._loaded_version = None
                return None
            if self._retriever is None or self._loaded_version != current:
                index = self.pipeline.vector_store.load(current)
                if index is None:
                    return None
                self._retriever = Retriever(index, self.settings.retrieval)
                self._loaded_version = current
                self._engine = None
                logger.info(
                    "serving index v%d (%d chunks)", current, len(index)
                )
            return self._retriever

    def answer_engine(self) -> AnswerEngine:
        retriever = self.retriever
        if retriever is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No documents have been indexed yet. Upload documents "
                       "and run ingestion first.",
            )
        if self._engine is None:
            self._engine = AnswerEngine(self.settings, retriever, self.embedder)
        return self._engine

    def invalidate(self) -> None:
        with self._lock:
            self._retriever = None
            self._loaded_version = None
            self._engine = None


_state: ServiceState | None = None


def init_state(settings: Settings) -> ServiceState:
    global _state
    _state = ServiceState(settings)
    return _state


def get_state() -> ServiceState:
    if _state is None:  # pragma: no cover - set during app startup
        raise RuntimeError("service state has not been initialised")
    return _state


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def client_ip(request: Request, settings: Settings) -> str:
    """Client address, honouring a trusted proxy header when configured.

    Behind Cloudflare, the socket address is Cloudflare's; ``cf-connecting-ip``
    carries the real client. Only trusted when explicitly configured, because
    an attacker can otherwise forge it to defeat rate limiting.
    """
    header = settings.security.trusted_proxy_header
    if header:
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_write_auth(
    request: Request, authorization: str | None = Header(default=None)
) -> str:
    """Bearer-token auth for mutating endpoints.

    When no token is configured the service is single-tenant/local and writes
    are open — which is fine on a laptop and *not* fine on a public domain, so
    the deployment docs make setting a token a required step and
    ``/api/health`` reports whether one is set.
    """
    state = get_state()
    token = state.settings.security.api_token
    if not token:
        return "anonymous"
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = authorization.split(" ", 1)[1].strip()
    import hmac

    if not hmac.compare_digest(provided, token):
        state.audit.record(
            "auth.reject", actor=client_ip(request, state.settings),
            outcome="invalid_token", path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token."
        )
    return "token"


def require_read_auth(
    request: Request, authorization: str | None = Header(default=None)
) -> str:
    state = get_state()
    if state.settings.security.allow_anonymous_read:
        return "anonymous"
    return require_write_auth(request, authorization)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Fixed-window-per-key limiter, sufficient for a single-instance service."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_s: int = 60) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > window_s:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


rate_limiter = RateLimiter()


def enforce_rate_limit(request: Request, bucket: str, limit: int) -> None:
    state = get_state()
    key = f"{bucket}:{client_ip(request, state.settings)}"
    if not rate_limiter.check(key, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({limit} requests per minute).",
            headers={"Retry-After": "60"},
        )
