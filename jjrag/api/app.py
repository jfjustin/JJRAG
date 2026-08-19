"""FastAPI application — the hostable front door of the pipeline.

Endpoints
---------
``POST   /api/documents``      upload files (multipart) into the inbox
``POST   /api/ingest``         run the ETL pipeline; returns the run manifest
``GET    /api/runs``           run history
``GET    /api/runs/{id}``      one run, with its full validation report
``GET    /api/documents``      indexed documents
``DELETE /api/documents/{id}`` erase a document and rebuild the index
``POST   /api/query``          ask a question (JSON response)
``POST   /api/query/stream``   ask a question (SSE stream)
``GET    /api/health``         liveness + index/model status
``GET    /api/compliance``     machine-readable privacy attestation
``GET    /api/audit``          recent audit entries (auth required)

The static UI is mounted at ``/``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings, get_settings
from ..models import RunManifest, Severity
from ..observability.logging import configure_logging
from ..pipeline.runner import PipelineError
from ..security import egress, scan
from .deps import (
    ServiceState,
    client_ip,
    enforce_rate_limit,
    init_state,
    require_read_auth,
    require_write_auth,
)
from .schemas import (
    CitationOut,
    ComplianceResponse,
    DeleteResponse,
    DocumentSummary,
    HealthResponse,
    IngestRequest,
    IssueSummary,
    QueryRequest,
    QueryResponse,
    RunResponse,
    StageSummary,
    UploadedFileInfo,
    UploadResponse,
)

logger = logging.getLogger("jjrag.api")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.paths.log_dir, settings.log_json)
    settings.paths.ensure()

    # Install the egress guard before anything can open a socket. Everything
    # downstream — embeddings, generation — is then provably local.
    egress.apply_policy(settings)

    app = FastAPI(
        title="JJRAG",
        version=__version__,
        description=(
            "Local-only document Q&A pipeline. Documents are extracted, "
            "validated, redacted, indexed and answered entirely on this host."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    state = init_state(settings)

    if settings.security.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Lock the browser down: no third-party anything, ever."""
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > settings.security.max_request_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "Request body too large."},
                    )
            except ValueError:
                pass

        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        if settings.environment == "prod":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response

    _register_routes(app, state)

    if WEB_DIR.is_dir():
        app.mount(
            "/assets", StaticFiles(directory=WEB_DIR / "static"), name="assets"
        )

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def _register_routes(app: FastAPI, state: ServiceState) -> None:  # noqa: C901
    settings = state.settings

    # -- health / compliance ------------------------------------------------
    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        retriever = state.retriever
        index = retriever.index if retriever else None
        try:
            from ..llm.ollama import OllamaClient

            client = OllamaClient(
                settings.llm, settings.security.extra_allowed_hosts
            )
            models = client.model_names()
            available = bool(models)
        except Exception:  # noqa: BLE001 - health must never fail
            models, available = [], False

        return HealthResponse(
            status="ok",
            version=__version__,
            index_version=index.version if index else None,
            indexed_chunks=len(index) if index else 0,
            indexed_documents=index.stats()["documents"] if index else 0,
            local_model_available=available,
            local_models=models,
            embedding_backend=settings.embedding.backend,
            embedding_dimensions=index.dimensions if index else None,
        )

    @app.get("/api/compliance", response_model=ComplianceResponse)
    async def compliance() -> ComplianceResponse:
        return ComplianceResponse(
            posture=settings.describe_compliance(),
            egress_guard_active=egress.is_installed(),
            blocked_egress_attempts=[
                {"host": host, "port": port}
                for host, port in egress.blocked_attempts()
            ],
            catalog=state.pipeline.catalog.stats(),
        )

    # -- upload -------------------------------------------------------------
    @app.post("/api/documents", response_model=UploadResponse)
    async def upload(
        request: Request,
        files: list[UploadFile] = File(...),
        actor: str = Depends(require_write_auth),
    ) -> UploadResponse:
        enforce_rate_limit(
            request, "upload", settings.security.upload_rate_limit_per_minute
        )
        if len(files) > settings.ingest.max_files_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files in one request (limit "
                       f"{settings.ingest.max_files_per_run}).",
            )

        inbox = settings.paths.inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        accepted: list[UploadedFileInfo] = []
        rejected: list[UploadedFileInfo] = []
        who = client_ip(request, settings)

        for upload_file in files:
            # Never trust a client-supplied filename: it can contain path
            # traversal or shell-hostile characters.
            safe_name = _safe_filename(upload_file.filename or "upload.bin")
            target = _unique_path(inbox / safe_name)
            size = 0
            try:
                with target.open("wb") as handle:
                    while chunk := await upload_file.read(1024 * 1024):
                        size += len(chunk)
                        if size > settings.ingest.max_file_bytes:
                            raise ValueError(
                                f"file exceeds {settings.ingest.max_file_bytes} bytes"
                            )
                        handle.write(chunk)
            except ValueError as exc:
                target.unlink(missing_ok=True)
                rejected.append(UploadedFileInfo(
                    filename=safe_name, size_bytes=size, accepted=False,
                    reason=str(exc),
                ))
                continue
            finally:
                await upload_file.close()

            result = scan.scan_file(
                target,
                allowed_extensions=settings.ingest.allowed_extensions,
                max_file_bytes=settings.ingest.max_file_bytes,
                enforce_content_type=settings.ingest.enforce_content_type,
                max_archive_expansion_ratio=settings.ingest.max_archive_expansion_ratio,
                uploaded_by=who,
            )
            if not result.admitted:
                state.pipeline.catalog.record_rejection(
                    safe_name, result.sha256, result.reason or "rejected", size
                )
                scan.quarantine(
                    target, settings.paths.quarantine_dir,
                    result.reason or "rejected",
                )
                state.audit.record(
                    "upload.reject", actor=who, subject_id=safe_name,
                    outcome="rejected", reason=result.reason,
                )
                rejected.append(UploadedFileInfo(
                    filename=safe_name, size_bytes=size, accepted=False,
                    reason=result.reason,
                ))
                continue

            state.audit.record(
                "upload.accept", actor=who, subject_id=safe_name,
                size_bytes=size, sha256=result.sha256[:16],
            )
            accepted.append(UploadedFileInfo(
                filename=safe_name, size_bytes=size, accepted=True,
                source_id=result.source_file.source_id if result.source_file else None,
            ))

        return UploadResponse(
            accepted=accepted, rejected=rejected,
            message=f"{len(accepted)} file(s) accepted, {len(rejected)} rejected.",
        )

    # -- ingest -------------------------------------------------------------
    @app.post("/api/ingest", response_model=RunResponse)
    async def ingest(
        request: Request,
        body: IngestRequest | None = None,
        actor: str = Depends(require_write_auth),
    ) -> RunResponse:
        enforce_rate_limit(request, "ingest", settings.security.rate_limit_per_minute)
        if not state.ingest_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An ingestion run is already in progress.",
            )
        try:
            manifest = state.pipeline.run(
                triggered_by="api",
                actor=client_ip(request, settings),
                force=bool(body and body.force),
            )
        except PipelineError as exc:
            state.invalidate()
            return JSONResponse(  # type: ignore[return-value]
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=json.loads(_run_response(exc.manifest).model_dump_json()),
            )
        finally:
            state.ingest_lock.release()

        state.invalidate()
        state.last_run_id = manifest.run_id
        return _run_response(manifest)

    @app.get("/api/runs", response_model=list[dict])
    async def list_runs(
        limit: int = 20, actor: str = Depends(require_read_auth)
    ) -> list[dict]:
        return state.pipeline.catalog.list_runs(min(limit, 100))

    @app.get("/api/runs/{run_id}", response_model=RunResponse)
    async def get_run(
        run_id: str, actor: str = Depends(require_read_auth)
    ) -> RunResponse:
        manifest = state.pipeline.catalog.get_run(run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return _run_response(manifest)

    # -- documents ----------------------------------------------------------
    @app.get("/api/documents", response_model=list[DocumentSummary])
    async def list_documents(
        limit: int = 200, actor: str = Depends(require_read_auth)
    ) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                doc_id=row["doc_id"], filename=row["filename"], title=row["title"],
                chunk_count=row["chunk_count"] or 0,
                char_count=row["char_count"] or 0,
                size_bytes=row.get("size_bytes"),
                media_type=row.get("media_type"),
                created_at=row.get("created_at"),
                redactions=row.get("redactions") or {},
            )
            for row in state.pipeline.catalog.list_documents(limit=min(limit, 1000))
        ]

    @app.delete("/api/documents/{doc_id}", response_model=DeleteResponse)
    async def delete_document(
        doc_id: str, request: Request, background: BackgroundTasks,
        actor: str = Depends(require_write_auth),
    ) -> DeleteResponse:
        who = client_ip(request, settings)
        deleted = state.pipeline.catalog.delete_document(doc_id)
        state.audit.record(
            "document.delete", actor=who, subject_id=doc_id,
            outcome="ok" if deleted else "not_found",
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found.")

        # Rebuilding is the slow part; do it after responding so the caller is
        # not left waiting on a full re-index.
        def rebuild() -> None:
            try:
                state.pipeline.rebuild(triggered_by="deletion", actor=who)
            except PipelineError as exc:
                logger.warning("post-deletion rebuild did not publish: %s", exc)
            finally:
                state.invalidate()

        background.add_task(rebuild)
        return DeleteResponse(
            doc_id=doc_id, deleted=True,
            message="Document erased. The index is being rebuilt without it.",
        )

    # -- query --------------------------------------------------------------
    @app.post("/api/query", response_model=QueryResponse)
    async def query(
        body: QueryRequest, request: Request,
        actor: str = Depends(require_read_auth),
    ) -> QueryResponse:
        enforce_rate_limit(request, "query", settings.security.rate_limit_per_minute)
        engine = state.answer_engine()
        try:
            answer = engine.answer(body.question, top_k=body.top_k, model=body.model)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

        state.audit.record(
            "query", actor=client_ip(request, settings),
            citations=len(answer.citations), index_version=answer.index_version,
        )
        return QueryResponse(
            question=answer.question, answer=answer.answer,
            citations=[_citation(c) for c in answer.citations],
            model=answer.model, latency_ms=answer.latency_ms,
            index_version=answer.index_version,
        )

    @app.post("/api/query/stream")
    async def query_stream(
        body: QueryRequest, request: Request,
        actor: str = Depends(require_read_auth),
    ) -> StreamingResponse:
        enforce_rate_limit(request, "query", settings.security.rate_limit_per_minute)
        engine = state.answer_engine()
        who = client_ip(request, settings)

        def events() -> Iterator[str]:
            try:
                for event, payload in engine.stream_answer(
                    body.question, top_k=body.top_k, model=body.model
                ):
                    if event == "sources":
                        payload = [
                            _citation_dict(item) for item in payload  # type: ignore[union-attr]
                        ]
                    yield f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
            finally:
                state.audit.record("query.stream", actor=who)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- audit --------------------------------------------------------------
    @app.get("/api/audit")
    async def audit_tail(
        limit: int = 100, actor: str = Depends(require_write_auth)
    ) -> list[dict]:
        return state.audit.tail(min(limit, 1000))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_filename(name: str) -> str:
    import re

    cleaned = Path(name).name  # strips any directory component
    cleaned = re.sub(r"[^A-Za-z0-9._ \-()]+", "_", cleaned).strip(". ")
    return cleaned[:180] or "upload.bin"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}.{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _citation(retrieved) -> CitationOut:  # type: ignore[no-untyped-def]
    chunk = retrieved.chunk
    return CitationOut(
        rank=retrieved.rank, filename=chunk.filename,
        segment_label=chunk.segment_label, text=chunk.text,
        score=retrieved.score, dense_score=retrieved.dense_score,
        lexical_score=retrieved.lexical_score, doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
    )


def _citation_dict(item: dict) -> dict:
    chunk = item.get("chunk", {})
    return {
        "rank": item.get("rank", 0),
        "filename": chunk.get("filename", ""),
        "segment_label": chunk.get("segment_label"),
        "text": chunk.get("text", ""),
        "score": item.get("score", 0.0),
        "dense_score": item.get("dense_score"),
        "lexical_score": item.get("lexical_score"),
        "doc_id": chunk.get("doc_id", ""),
        "chunk_id": chunk.get("chunk_id", ""),
    }


def _run_response(manifest: RunManifest) -> RunResponse:
    return RunResponse(
        run_id=manifest.run_id,
        status=manifest.status.value,
        started_at=manifest.started_at.isoformat(),
        finished_at=manifest.finished_at.isoformat() if manifest.finished_at else None,
        duration_s=manifest.duration_s,
        files_admitted=manifest.files_admitted,
        files_rejected=manifest.files_rejected,
        documents=manifest.documents,
        chunks=manifest.chunks,
        vectors=manifest.vectors,
        index_version=manifest.index_version,
        redactions=manifest.redactions,
        stages=[
            StageSummary(
                name=stage.name, status=stage.status.value,
                duration_s=stage.duration_s, records_in=stage.records_in,
                records_out=stage.records_out, metrics=stage.metrics,
                errors=len(stage.report.errors) if stage.report else 0,
                warnings=len(stage.report.warnings) if stage.report else 0,
            )
            for stage in manifest.stages
        ],
        issues=[
            IssueSummary(
                stage=issue.stage, rule=issue.rule, severity=issue.severity.value,
                message=issue.message, subject_type=issue.subject_type,
                subject_id=issue.subject_id,
            )
            for issue in manifest.all_issues()
            if issue.severity in (Severity.ERROR, Severity.WARNING)
        ][:200],
        error=manifest.error,
    )


app = None  # populated by jjrag.api.main for uvicorn
