"""Command-line interface.

    jjrag doctor                 check the environment is ready
    jjrag ingest [PATHS...]      run the full ETL pipeline
    jjrag rebuild                re-index the active corpus
    jjrag query "question"       ask a question from the terminal
    jjrag docs                   list indexed documents
    jjrag rm DOC_ID              erase a document
    jjrag runs [RUN_ID]          show run history / one validation report
    jjrag serve                  start the web app
    jjrag prefetch               download the embedding model (needs network)
    jjrag compliance             print the privacy attestation

Everything the web UI can do is available here, so the pipeline can be driven
from cron, a Makefile, or CI without the HTTP layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, get_settings
from .observability.logging import configure_logging
from .pipeline.runner import Pipeline, PipelineError
from .security import egress

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)


def _settings(args: argparse.Namespace) -> Settings:
    settings = (
        Settings.load(args.config) if getattr(args, "config", None)
        else get_settings()
    )
    configure_logging(
        "DEBUG" if getattr(args, "verbose", False) else settings.log_level,
        settings.paths.log_dir, settings.log_json,
    )
    return settings


def _ok(message: str) -> None:
    print(f"{GREEN}✓{RESET} {message}")


def _warn(message: str) -> None:
    print(f"{YELLOW}!{RESET} {message}")


def _fail(message: str) -> None:
    print(f"{RED}✗{RESET} {message}")


# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    print(f"JJRAG environment check ({settings.environment})\n")
    problems = 0

    settings.paths.ensure()
    _ok(f"data directory: {settings.paths.data_dir.resolve()}")

    try:
        from .pipeline.embed import Embedder

        embedder = Embedder(settings.embedding, llm_host=settings.llm.host)
        _ok(f"embeddings: {embedder.model_id} ({embedder.dimensions}d, "
            f"{settings.embedding.backend})")
    except Exception as exc:  # noqa: BLE001
        problems += 1
        _fail(f"embeddings unavailable: {exc}")

    try:
        from .llm.ollama import OllamaClient

        client = OllamaClient(settings.llm, settings.security.extra_allowed_hosts)
        models = client.model_names()
        if not models:
            problems += 1
            _fail(
                f"no local model server at {settings.llm.host}. "
                "Install Ollama (https://ollama.com), then: "
                f"ollama pull {settings.llm.model}"
            )
        elif settings.llm.model in models or any(
            m.split(":")[0] == settings.llm.model.split(":")[0] for m in models
        ):
            _ok(f"local model: {settings.llm.model} (installed: {', '.join(models)})")
        else:
            problems += 1
            _fail(f"model {settings.llm.model} not installed. Available: "
                  f"{', '.join(models)}")
    except Exception as exc:  # noqa: BLE001
        problems += 1
        _fail(f"local model check failed: {exc}")

    pipeline = Pipeline(settings)
    index = pipeline.vector_store.load()
    if index:
        stats = index.stats()
        _ok(f"index v{stats['version']}: {stats['vectors']} chunks from "
            f"{stats['documents']} document(s)")
    else:
        _warn("no index published yet — run: jjrag ingest <path>")

    if settings.security.enforce_local_only:
        _ok("egress policy: local-only (external hosts blocked at the socket)")
    else:
        problems += 1
        _fail("egress policy: UNRESTRICTED — set security.enforce_local_only")

    if settings.environment == "prod" and not settings.security.api_token:
        problems += 1
        _fail("no API token set, but environment is 'prod' — set "
              "security.api_token before exposing this service")

    print()
    if problems:
        _fail(f"{problems} problem(s) found")
    else:
        _ok("ready")
    return 1 if problems else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = Pipeline(settings)
    paths = [Path(p) for p in args.paths] if args.paths else None

    def progress(stage: str, message: str) -> None:
        print(f"  {DIM}{stage:<10}{RESET} {message}")

    print("Running pipeline: scan → extract → transform → validate → embed → load\n")
    try:
        manifest = pipeline.run(
            paths, triggered_by="cli", actor="cli", progress=progress,
            force=args.force,
        )
    except PipelineError as exc:
        print()
        _fail(f"run {exc.manifest.run_id} failed: {exc}")
        for issue in exc.manifest.all_issues():
            if issue.severity.value == "error":
                print(f"    {RED}error{RESET} [{issue.rule}] {issue.message}")
        return 1

    print()
    _ok(f"run {manifest.run_id} succeeded in {manifest.duration_s:.1f}s")
    print(f"  documents: {manifest.documents}   chunks: {manifest.chunks}   "
          f"index: v{manifest.index_version}")
    if manifest.files_rejected:
        _warn(f"{manifest.files_rejected} file(s) rejected — see "
              f"{settings.paths.quarantine_dir}")
    if manifest.redactions:
        print("  redacted: " + ", ".join(
            f"{k}×{v}" for k, v in sorted(manifest.redactions.items())
        ))
    warnings = [i for i in manifest.all_issues() if i.severity.value == "warning"]
    for issue in warnings[:10]:
        print(f"    {YELLOW}warn{RESET}  [{issue.rule}] {issue.message}")
    if len(warnings) > 10:
        print(f"    {DIM}… and {len(warnings) - 10} more warning(s){RESET}")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = Pipeline(settings)
    try:
        manifest = pipeline.rebuild(triggered_by="cli", actor="cli")
    except PipelineError as exc:
        _fail(f"rebuild failed: {exc}")
        return 1
    _ok(f"rebuilt index v{manifest.index_version} with {manifest.chunks} chunk(s)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    settings = _settings(args)
    egress.apply_policy(settings)
    pipeline = Pipeline(settings)
    index = pipeline.vector_store.load()
    if index is None:
        _fail("no index published — run: jjrag ingest <path>")
        return 1

    from .retrieval.answer import AnswerEngine
    from .retrieval.search import Retriever

    retriever = Retriever(index, settings.retrieval)
    engine = AnswerEngine(settings, retriever, pipeline.embedder)

    if args.retrieve_only:
        for result in engine.retrieve(args.question, args.top_k):
            print(f"\n{GREEN}[{result.rank}]{RESET} {result.chunk.filename}"
                  f" — {result.chunk.segment_label or ''} "
                  f"{DIM}(score {result.score:.4f}){RESET}")
            print(result.chunk.text[:600])
        return 0

    try:
        for event, payload in engine.stream_answer(
            args.question, top_k=args.top_k, model=args.model
        ):
            if event == "token":
                sys.stdout.write(str(payload))
                sys.stdout.flush()
            elif event == "error":
                print(f"\n{RED}✗{RESET} {payload}")
                return 1
            elif event == "done":
                print("\n")
                citations = payload.get("citations", []) if isinstance(payload, dict) else []
                for citation in citations:
                    chunk = citation.get("chunk", {})
                    print(f"  {DIM}[{citation.get('rank')}] "
                          f"{chunk.get('filename')} "
                          f"{chunk.get('segment_label') or ''}{RESET}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
        return 1
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = Pipeline(settings)
    documents = pipeline.catalog.list_documents()
    if not documents:
        _warn("no documents indexed")
        return 0
    print(f"{'DOC ID':<24} {'CHUNKS':>7}  {'CHARS':>8}  FILENAME")
    for row in documents:
        print(f"{row['doc_id']:<24} {row['chunk_count'] or 0:>7}  "
              f"{row['char_count'] or 0:>8}  {row['filename']}")
    print(f"\n{len(documents)} document(s)")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = Pipeline(settings)
    if pipeline.delete_document(args.doc_id, actor="cli"):
        _ok(f"erased {args.doc_id} and rebuilt the index")
        return 0
    _fail(f"document {args.doc_id} not found")
    return 1


def cmd_runs(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = Pipeline(settings)

    if args.run_id:
        manifest = pipeline.catalog.get_run(args.run_id)
        if manifest is None:
            _fail(f"run {args.run_id} not found")
            return 1
        print(json.dumps(json.loads(manifest.model_dump_json()), indent=2))
        return 0

    runs = pipeline.catalog.list_runs(args.limit)
    if not runs:
        _warn("no runs recorded")
        return 0
    print(f"{'RUN ID':<24} {'STATUS':<10} {'DOCS':>5} {'CHUNKS':>7} {'IDX':>4}  STARTED")
    for run in runs:
        status = run["status"]
        colour = GREEN if status == "succeeded" else RED if status == "failed" else YELLOW
        print(f"{run['run_id']:<24} {colour}{status:<10}{RESET} "
              f"{run['documents']:>5} {run['chunks']:>7} "
              f"{str(run['index_version'] or '-'):>4}  {run['started_at'][:19]}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    settings = _settings(args)
    try:
        import uvicorn
    except ImportError:
        _fail("uvicorn is not installed (pip install 'uvicorn[standard]')")
        return 1

    host = args.host or settings.server.host
    port = args.port or settings.server.port
    if settings.environment == "prod" and not settings.security.api_token:
        _warn("no API token configured — uploads and deletions are unauthenticated")
    print(f"JJRAG serving on http://{host}:{port} "
          f"({DIM}local model: {settings.llm.model}{RESET})")
    uvicorn.run(
        "jjrag.api.main:app", host=host, port=port, reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Download model weights. Run this *before* enabling the egress guard."""
    settings = _settings(args)
    from .pipeline.embed import prefetch_model

    print(f"Downloading {settings.embedding.model} …")
    print(prefetch_model(settings.embedding))
    _ok("embedding model cached locally; the service can now run offline")
    return 0


def cmd_compliance(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = Pipeline(settings)
    payload = {
        "posture": settings.describe_compliance(),
        "catalog": pipeline.catalog.stats(),
        "config_fingerprint": None,
    }
    print(json.dumps(payload, indent=2))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jjrag",
        description="Local-only document Q&A pipeline (extract → transform → "
                    "validate → load → retrieve → answer).",
    )
    parser.add_argument("--config", help="path to a YAML config file")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the environment")
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest", help="run the ETL pipeline")
    ingest.add_argument("paths", nargs="*", help="files or directories "
                        "(default: the configured inbox)")
    ingest.add_argument("--force", action="store_true",
                        help="re-ingest files that are already indexed")
    ingest.set_defaults(func=cmd_ingest)

    rebuild = sub.add_parser("rebuild", help="re-index the active corpus")
    rebuild.set_defaults(func=cmd_rebuild)

    query = sub.add_parser("query", help="ask a question")
    query.add_argument("question")
    query.add_argument("-k", "--top-k", type=int, default=None)
    query.add_argument("--model", default=None, help="override the local model")
    query.add_argument("--retrieve-only", action="store_true",
                       help="show retrieved passages without generating")
    query.set_defaults(func=cmd_query)

    docs = sub.add_parser("docs", help="list indexed documents")
    docs.set_defaults(func=cmd_docs)

    remove = sub.add_parser("rm", help="erase a document and rebuild")
    remove.add_argument("doc_id")
    remove.set_defaults(func=cmd_rm)

    runs = sub.add_parser("runs", help="run history and validation reports")
    runs.add_argument("run_id", nargs="?")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=cmd_runs)

    serve = sub.add_parser("serve", help="start the web app")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    prefetch = sub.add_parser("prefetch", help="download the embedding model")
    prefetch.set_defaults(func=cmd_prefetch)

    compliance = sub.add_parser("compliance", help="print the privacy attestation")
    compliance.set_defaults(func=cmd_compliance)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
