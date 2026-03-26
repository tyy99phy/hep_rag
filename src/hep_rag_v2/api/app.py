from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hep_rag_v2 import paths
from hep_rag_v2.config import apply_runtime_config, default_config
from hep_rag_v2.pipeline import ask, fetch_online_candidates, ingest_online, reparse_cached_pdfs, retrieve
from hep_rag_v2.service.inspect import audit_document_payload, show_document_payload, show_graph_payload
from hep_rag_v2.service.workspace import workspace_status_payload

from .jobs import BackgroundJobManager


ConfigLoader = Callable[[], tuple[Path, dict[str, Any]]]
UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_name: str | None = None
    target: str | None = None
    limit: int | None = Field(default=None, ge=1)
    model: str | None = None


class AskRequest(QueryRequest):
    mode: str = "answer"


class FetchPapersRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1)


class GraphRequest(BaseModel):
    work_id: int | None = None
    id_type: str | None = None
    id_value: str | None = None
    edge_kind: str = "all"
    collection: str | None = None
    limit: int = Field(default=20, ge=1)
    model: str | None = None


class DocumentRequest(BaseModel):
    work_id: int | None = None
    id_type: str | None = None
    id_value: str | None = None
    limit: int = Field(default=20, ge=1)


class IngestOnlineJobRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_name: str | None = None
    limit: int = Field(default=20, ge=1)
    download_limit: int | None = Field(default=None, ge=1)
    parse_limit: int | None = Field(default=None, ge=0)
    replace_existing: bool = False
    skip_parse: bool = False
    skip_index: bool = False
    skip_graph: bool = False


class ReparseJobRequest(BaseModel):
    collection_name: str | None = None
    limit: int | None = Field(default=None, ge=1)
    work_ids: list[int] | None = None
    replace_existing: bool = False
    skip_index: bool = False
    skip_graph: bool = False


def create_app(
    *,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    config_loader: ConfigLoader | None = None,
    job_manager: BackgroundJobManager | None = None,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def _lifespan(app: FastAPI):
        _load_runtime_config_cached(app, force_reload=True)
        try:
            yield
        finally:
            manager = app.state.job_manager
            if app.state.owns_job_manager and manager is not None:
                manager.shutdown()

    app = FastAPI(title="hep-rag API", version="0.2.0", lifespan=_lifespan)
    app.state.config_loader = config_loader or _default_config_loader(
        config_path=config_path,
        workspace_root=workspace_root,
    )
    app.state.requested_config_path = str(config_path) if config_path is not None else None
    app.state.requested_workspace_root = str(workspace_root) if workspace_root is not None else None
    app.state.config_path = None
    app.state.workspace_root = None
    app.state.runtime_config = None
    app.state.job_manager = job_manager
    app.state.owns_job_manager = job_manager is None
    app.state.api_auth_token = None

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        if _is_public_path(request.url.path):
            return await call_next(request)
        _load_runtime_config(request.app)
        expected_token = request.app.state.api_auth_token
        if not expected_token:
            return await call_next(request)
        provided_token = _extract_api_token(request)
        if provided_token != expected_token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        if not _ui_enabled(request.app):
            raise HTTPException(status_code=404, detail="UI is disabled.")
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    @app.get("/ui", response_class=HTMLResponse)
    def ui(request: Request) -> HTMLResponse:
        if not _ui_enabled(request.app):
            raise HTTPException(status_code=404, detail="UI is disabled.")
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    @app.get("/auth/status")
    def auth_status(request: Request) -> dict[str, Any]:
        runtime_path, config = _load_runtime_config(request.app)
        return {
            "auth_enabled": bool(request.app.state.api_auth_token),
            "ui_enabled": bool((config.get("api") or {}).get("enable_ui", True)),
            "config_path": str(runtime_path),
            "workspace_root": request.app.state.workspace_root,
        }

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        runtime_path, config = _load_runtime_config(request.app)
        return {
            "ok": True,
            "config_path": str(runtime_path),
            "workspace_root": request.app.state.workspace_root,
            "auth_enabled": bool(request.app.state.api_auth_token),
            "ui_enabled": bool((config.get("api") or {}).get("enable_ui", True)),
        }

    @app.get("/workspace/status")
    def workspace_status(request: Request) -> dict[str, Any]:
        _load_runtime_config(request.app)
        return workspace_status_payload()

    @app.post("/retrieve")
    def retrieve_endpoint(request: Request, body: QueryRequest) -> dict[str, Any]:
        _, config = _load_runtime_config(request.app)
        return _wrap_service_call(
            lambda: retrieve(
                config,
                query=body.query,
                limit=body.limit,
                target=body.target,
                collection_name=body.collection_name,
                model=body.model,
            )
        )

    @app.post("/ask")
    def ask_endpoint(request: Request, body: AskRequest) -> dict[str, Any]:
        _, config = _load_runtime_config(request.app)
        return _wrap_service_call(
            lambda: ask(
                config,
                query=body.query,
                mode=body.mode,
                limit=body.limit,
                target=body.target,
                collection_name=body.collection_name,
                model=body.model,
            )
        )

    @app.post("/fetch-papers")
    def fetch_papers_endpoint(request: Request, body: FetchPapersRequest) -> dict[str, Any]:
        _, config = _load_runtime_config(request.app)
        return _wrap_service_call(
            lambda: fetch_online_candidates(
                config,
                query=body.query,
                limit=body.limit,
            )
        )

    @app.post("/graph/neighbors")
    def graph_neighbors_endpoint(request: Request, body: GraphRequest) -> dict[str, Any]:
        _load_runtime_config(request.app)
        return _wrap_service_call(
            lambda: show_graph_payload(
                work_id=body.work_id,
                id_type=body.id_type,
                id_value=body.id_value,
                edge_kind=body.edge_kind,
                collection=body.collection,
                limit=body.limit,
                similarity_model=body.model,
            )
        )

    @app.post("/documents/show")
    def show_document_endpoint(request: Request, body: DocumentRequest) -> dict[str, Any]:
        _load_runtime_config(request.app)
        return _wrap_service_call(
            lambda: show_document_payload(
                work_id=body.work_id,
                id_type=body.id_type,
                id_value=body.id_value,
                limit=body.limit,
            )
        )

    @app.post("/documents/audit")
    def audit_document_endpoint(request: Request, body: DocumentRequest) -> dict[str, Any]:
        _load_runtime_config(request.app)
        return _wrap_service_call(
            lambda: audit_document_payload(
                work_id=body.work_id,
                id_type=body.id_type,
                id_value=body.id_value,
                limit=body.limit,
            )
        )

    @app.get("/jobs")
    def list_jobs(request: Request) -> list[dict[str, Any]]:
        return _job_manager(request.app).list_jobs()

    @app.post("/jobs/ingest-online")
    def ingest_online_job(request: Request, body: IngestOnlineJobRequest) -> dict[str, Any]:
        job_manager = _job_manager(request.app)
        job_id = uuid4().hex

        def _run() -> dict[str, Any]:
            _, config = _load_runtime_config(request.app)
            return ingest_online(
                config,
                query=body.query,
                limit=body.limit,
                collection_name=body.collection_name,
                download_limit=body.download_limit,
                parse_limit=body.parse_limit,
                replace_existing=body.replace_existing,
                skip_parse=body.skip_parse,
                skip_index=body.skip_index,
                skip_graph=body.skip_graph,
                progress=job_manager.progress_callback(job_id),
            )

        try:
            return job_manager.submit(
                kind="ingest_online",
                job_id=job_id,
                fn=_run,
                request_payload=body.model_dump(),
            )
        except Exception as exc:
            raise _http_exception_from_error(exc) from exc

    @app.post("/jobs/reparse-pdfs")
    def reparse_pdfs_job(request: Request, body: ReparseJobRequest) -> dict[str, Any]:
        job_manager = _job_manager(request.app)
        job_id = uuid4().hex

        def _run() -> dict[str, Any]:
            _, config = _load_runtime_config(request.app)
            return reparse_cached_pdfs(
                config,
                collection_name=body.collection_name,
                limit=body.limit,
                work_ids=body.work_ids,
                replace_existing=body.replace_existing,
                skip_index=body.skip_index,
                skip_graph=body.skip_graph,
                progress=job_manager.progress_callback(job_id),
            )

        try:
            return job_manager.submit(
                kind="reparse_pdfs",
                job_id=job_id,
                fn=_run,
                request_payload=body.model_dump(),
            )
        except Exception as exc:
            raise _http_exception_from_error(exc) from exc

    @app.get("/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, Any]:
        try:
            return _job_manager(request.app).get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc

    @app.get("/jobs/{job_id}/events")
    def get_job_events(request: Request, job_id: str, after: int = 0) -> dict[str, Any]:
        try:
            events = _job_manager(request.app).events(job_id, after=after)
            return {"job_id": job_id, "events": events}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc

    @app.get("/jobs/{job_id}/events/stream")
    def stream_job_events(request: Request, job_id: str, after: int = 0) -> StreamingResponse:
        manager = _job_manager(request.app)
        try:
            manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc

        def _event_stream():
            last_seq = int(after)
            while True:
                events = manager.events(job_id, after=last_seq)
                for event in events:
                    last_seq = int(event["seq"])
                    yield _sse_event("message", event)
                snapshot = manager.get(job_id)
                if snapshot["status"] in {"succeeded", "failed"}:
                    yield _sse_event("job", snapshot)
                    break
                time.sleep(0.5)

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="hep-rag-api")
    parser.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    parser.add_argument("--workspace", default=None, help="Override workspace root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(config_path=args.config, workspace_root=args.workspace),
        host=args.host,
        port=args.port,
    )


def _default_config_loader(
    *,
    config_path: str | Path | None,
    workspace_root: str | Path | None,
) -> ConfigLoader:
    def _loader() -> tuple[Path, dict[str, Any]]:
        return apply_runtime_config(config_path=config_path, workspace_root=workspace_root)

    return _loader


def _load_runtime_config(app: FastAPI) -> tuple[Path, dict[str, Any]]:
    return _load_runtime_config_cached(app, force_reload=False)


def _load_runtime_config_cached(app: FastAPI, *, force_reload: bool) -> tuple[Path, dict[str, Any]]:
    cached_config = getattr(app.state, "runtime_config", None)
    cached_path = getattr(app.state, "config_path", None)
    if not force_reload and cached_config is not None and cached_path is not None:
        return Path(cached_path), cached_config
    try:
        config_path, loaded = app.state.config_loader()
    except Exception as exc:
        raise _http_exception_from_error(exc, default_status=500) from exc
    runtime_path = Path(config_path).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = runtime_path.resolve()
    config = _normalize_runtime_config(loaded)
    workspace_root = _resolve_runtime_workspace_root(
        config=config,
        config_path=runtime_path,
        workspace_root=getattr(app.state, "requested_workspace_root", None),
    )
    paths.set_workspace_root(workspace_root)
    config["workspace"]["root"] = str(workspace_root)
    app.state.runtime_config = config
    app.state.config_path = str(runtime_path)
    app.state.workspace_root = str(workspace_root)
    app.state.api_auth_token = _resolve_auth_token(config)
    return runtime_path, config


def _wrap_service_call(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        raise _http_exception_from_error(exc) from exc


def _http_exception_from_error(exc: Exception, *, default_status: int = 400) -> HTTPException:
    text = str(exc)
    if isinstance(exc, ValueError):
        status_code = 404 if text.startswith(("Unknown work", "Unknown job_id", "No document materialized")) else 400
    elif isinstance(exc, FileNotFoundError):
        status_code = 500
    else:
        status_code = default_status if default_status >= 500 else 500
    return HTTPException(status_code=status_code, detail=text)


def _sse_event(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _normalize_runtime_config(loaded: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        raise ValueError("Config loader must return a mapping.")
    config = default_config()
    _deep_merge(config, loaded)
    return config


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _resolve_runtime_workspace_root(
    *,
    config: dict[str, Any],
    config_path: Path,
    workspace_root: str | Path | None,
) -> Path:
    if workspace_root is not None:
        return Path(workspace_root).expanduser().resolve()
    configured = str((config.get("workspace") or {}).get("root") or "").strip()
    if not configured:
        return (config_path.parent / "workspace").resolve()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root


def _resolve_auth_token(config: dict[str, Any]) -> str | None:
    env_token = os.environ.get("HEP_RAG_API_TOKEN")
    if env_token is not None and env_token.strip():
        return env_token.strip()
    configured = str((config.get("api") or {}).get("auth_token") or "").strip()
    return configured or None


def _extract_api_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization") or ""
    scheme, _, credentials = auth_header.partition(" ")
    if scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    api_key = request.headers.get("x-api-key") or ""
    if api_key.strip():
        return api_key.strip()
    return None


def _is_public_path(path: str) -> bool:
    if path in {"/", "/ui", "/health", "/openapi.json", "/auth/status"}:
        return True
    return path.startswith("/docs") or path.startswith("/redoc")


def _ui_enabled(app: FastAPI) -> bool:
    _, config = _load_runtime_config_cached(app, force_reload=False)
    return bool((config.get("api") or {}).get("enable_ui", True))


def _job_manager(app: FastAPI) -> BackgroundJobManager:
    manager = getattr(app.state, "job_manager", None)
    if manager is not None:
        return manager
    _, config = _load_runtime_config_cached(app, force_reload=False)
    api_cfg = config.get("api") or {}
    manager = BackgroundJobManager(
        max_workers=int(api_cfg.get("job_max_workers") or 2),
        max_events=int(api_cfg.get("job_max_events") or 1000),
    )
    app.state.job_manager = manager
    return manager
