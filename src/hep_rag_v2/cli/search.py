from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.graph import rebuild_graph_edges
from hep_rag_v2.maintenance import (
    clear_dirty_work_ids,
    dirty_counts,
    finish_maintenance_job,
    select_dirty_work_ids,
    start_maintenance_job,
)
from hep_rag_v2.search import (
    rebuild_search_indices,
    search_index_counts,
    search_assets_bm25,
    search_chunks_bm25,
    search_formulas_bm25,
    search_works_bm25,
)
from hep_rag_v2.service.inspect import show_graph_payload
from hep_rag_v2.vector import (
    DEFAULT_VECTOR_MODEL,
    rebuild_vector_indices,
    route_query,
    search_chunks_hybrid,
    search_chunks_vector,
    search_chunks_vector_chroma,
    search_works_hybrid,
    search_works_vector,
    search_works_vector_chroma,
    sync_chroma_indices,
    vector_index_counts,
)

from ._common import emit_cli_status


def cmd_build_search_index(args: argparse.Namespace) -> None:
    ensure_db()

    with connect() as conn:
        emit_cli_status(f"building BM25 search indexes for target={args.target}...")
        summary = rebuild_search_indices(conn, target=args.target)
        conn.commit()
        summary.update(search_index_counts(conn))
        emit_cli_status("BM25 search indexes ready.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_search_bm25(args: argparse.Namespace) -> None:
    ensure_db()
    if not args.query.strip():
        raise SystemExit("Query cannot be empty.")
    with connect() as conn:
        search_fn = {
            "works": search_works_bm25,
            "chunks": search_chunks_bm25,
            "formulas": search_formulas_bm25,
            "assets": search_assets_bm25,
        }[args.target]
        rows = search_fn(conn, query=args.query, collection=args.collection, limit=args.limit)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def cmd_build_vector_index(args: argparse.Namespace) -> None:
    ensure_db()
    try:
        with connect() as conn:
            emit_cli_status(f"building vector indexes for target={args.target} with model={args.model}...")
            summary = rebuild_vector_indices(
                conn,
                target=args.target,
                model=args.model,
                dim=args.dim,
                progress=emit_cli_status,
            )
            conn.commit()
            summary.update(vector_index_counts(conn))
            emit_cli_status("vector index build finished.")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_search_vector(args: argparse.Namespace) -> None:
    ensure_db()
    if not args.query.strip():
        raise SystemExit("Query cannot be empty.")
    try:
        with connect() as conn:
            if args.backend == "chroma":
                search_fn = {
                    "works": search_works_vector_chroma,
                    "chunks": search_chunks_vector_chroma,
                }[args.target]
                rows = search_fn(
                    conn,
                    query=args.query,
                    collection=args.collection,
                    limit=args.limit,
                    model=args.model,
                    chroma_dir=Path(args.chroma_dir).expanduser().resolve() if args.chroma_dir else None,
                )
            else:
                search_fn = {
                    "works": search_works_vector,
                    "chunks": search_chunks_vector,
                }[args.target]
                rows = search_fn(
                    conn,
                    query=args.query,
                    collection=args.collection,
                    limit=args.limit,
                    model=args.model,
                )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def cmd_sync_chroma_index(args: argparse.Namespace) -> None:
    ensure_db()
    try:
        with connect() as conn:
            emit_cli_status(f"syncing Chroma index for target={args.target} with model={args.model}...")
            summary = sync_chroma_indices(
                conn,
                target=args.target,
                model=args.model,
                collection=args.collection,
                chroma_dir=Path(args.chroma_dir).expanduser().resolve() if args.chroma_dir else None,
                batch_size=args.batch_size,
            )
            emit_cli_status("Chroma sync finished.")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_search_hybrid(args: argparse.Namespace) -> None:
    ensure_db()
    if not args.query.strip():
        raise SystemExit("Query cannot be empty.")
    routing = route_query(args.query) if args.target == "auto" else {
        "target": args.target,
        "graph_expand": 0,
        "reasons": ["manual_target"],
    }
    target = str(routing["target"])
    graph_expand = args.graph_expand if args.graph_expand is not None else int(routing.get("graph_expand") or 0)

    try:
        with connect() as conn:
            if target == "works":
                rows = search_works_hybrid(
                    conn,
                    query=args.query,
                    collection=args.collection,
                    limit=args.limit,
                    model=args.model,
                    graph_expand=graph_expand,
                    seed_limit=args.seed_limit,
                )
            else:
                rows = search_chunks_hybrid(
                    conn,
                    query=args.query,
                    collection=args.collection,
                    limit=args.limit,
                    model=args.model,
                )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    payload = {
        "query": args.query,
        "requested_target": args.target,
        "routing": {
            "target": target,
            "graph_expand": graph_expand,
            "reasons": routing.get("reasons") or [],
        },
        "results": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_build_graph(args: argparse.Namespace) -> None:
    ensure_db()
    try:
        with connect() as conn:
            emit_cli_status(f"building graph edges for target={args.target}...")
            summary = rebuild_graph_edges(
                conn,
                target=args.target,
                collection=args.collection,
                min_shared=args.min_shared,
                similarity_model=args.model,
                similarity_top_k=args.top_k,
                similarity_min_score=args.min_score,
                progress=emit_cli_status,
            )
            conn.commit()
            emit_cli_status("graph build finished.")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_show_graph(args: argparse.Namespace) -> None:
    payload = show_graph_payload(
        work_id=args.work_id,
        id_type=args.id_type,
        id_value=args.id_value,
        edge_kind=args.edge_kind,
        collection=args.collection,
        limit=args.limit,
        similarity_model=args.model if args.edge_kind == "similarity" else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_sync_search(args: argparse.Namespace) -> None:
    summary = _run_sync_job(lane="search", args=args, rebuild=lambda conn: _sync_search_impl(conn, args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_sync_vectors(args: argparse.Namespace) -> None:
    summary = _run_sync_job(lane="vectors", args=args, rebuild=lambda conn: _sync_vectors_impl(conn, args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_sync_graph(args: argparse.Namespace) -> None:
    summary = _run_sync_job(lane="graph", args=args, rebuild=lambda conn: _sync_graph_impl(conn, args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _sync_search_impl(conn, args: argparse.Namespace) -> dict[str, Any]:
    emit_cli_status(
        f"syncing BM25 search indexes for target={args.target} (scope={_sync_scope(args)})..."
    )
    summary = rebuild_search_indices(conn, target=args.target)
    summary.update(search_index_counts(conn))
    emit_cli_status("BM25 search sync finished.")
    return summary


def _sync_vectors_impl(conn, args: argparse.Namespace) -> dict[str, Any]:
    emit_cli_status(
        f"syncing vector indexes for target={args.target} with model={args.model} (scope={_sync_scope(args)})..."
    )
    summary = rebuild_vector_indices(
        conn,
        target=args.target,
        model=args.model,
        dim=args.dim,
        progress=emit_cli_status,
    )
    summary.update(vector_index_counts(conn))
    emit_cli_status("vector sync finished.")
    return summary


def _sync_graph_impl(conn, args: argparse.Namespace) -> dict[str, Any]:
    emit_cli_status(
        f"syncing graph edges for target={args.target} (scope={_sync_scope(args)})..."
    )
    summary = rebuild_graph_edges(
        conn,
        target=args.target,
        collection=args.collection,
        min_shared=args.min_shared,
        similarity_model=args.model,
        similarity_top_k=args.top_k,
        similarity_min_score=args.min_score,
        progress=emit_cli_status,
    )
    emit_cli_status("graph sync finished.")
    return summary


def _run_sync_job(
    *,
    lane: str,
    args: argparse.Namespace,
    rebuild,
) -> dict[str, Any]:
    ensure_db()
    scope = _sync_scope(args)
    collection = _sync_collection(args)
    updated_since = _sync_updated_since(args)
    try:
        with connect() as conn:
            selected_dirty = (
                select_dirty_work_ids(
                    conn,
                    lane=lane,
                    collection=collection,
                    updated_since=updated_since,
                )
                if scope == "dirty"
                else []
            )
            dirty_before = dirty_counts(conn, collection=collection)
            job_id = start_maintenance_job(
                conn,
                lane=lane,
                scope=scope,
                collection_name=collection,
                updated_since=updated_since,
                details=_sync_job_details(args, selected_dirty),
            )

            if scope == "dirty" and not selected_dirty:
                summary = _sync_noop_summary(
                    lane=lane,
                    args=args,
                    scope=scope,
                    collection=collection,
                    updated_since=updated_since,
                    job_id=job_id,
                    dirty_before=dirty_before,
                )
                finish_maintenance_job(conn, job_id=job_id, status="completed", result=summary)
                conn.commit()
                return summary

            summary = rebuild(conn)
            cleared_dirty = clear_dirty_work_ids(
                conn,
                lane=lane,
                collection=collection,
                work_ids=selected_dirty if scope == "dirty" else None,
            )
            after_dirty = dirty_counts(conn, collection=collection)
            summary.update(
                {
                    "lane": lane,
                    "scope": scope,
                    "collection": collection,
                    "updated_since": updated_since,
                    "selected_dirty_count": len(selected_dirty),
                    "selected_dirty_work_ids": selected_dirty,
                    "cleared_dirty_count": cleared_dirty,
                    "dirty_before": dirty_before,
                    "dirty_after": after_dirty,
                    "maintenance_job_id": job_id,
                }
            )
            finish_maintenance_job(conn, job_id=job_id, status="completed", result=summary)
            conn.commit()
            return summary
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _sync_job_details(args: argparse.Namespace, selected_dirty: list[int]) -> dict[str, Any]:
    return {
        "target": getattr(args, "target", None),
        "model": getattr(args, "model", None),
        "dim": getattr(args, "dim", None),
        "collection": _sync_collection(args),
        "updated_since": _sync_updated_since(args),
        "selected_dirty_count": len(selected_dirty),
        "selected_dirty_work_ids": selected_dirty,
    }


def _sync_noop_summary(
    *,
    lane: str,
    args: argparse.Namespace,
    scope: str,
    collection: str | None,
    updated_since: str | None,
    job_id: int,
    dirty_before: dict[str, int],
) -> dict[str, Any]:
    return {
        "lane": lane,
        "target": getattr(args, "target", None),
        "scope": scope,
        "collection": collection,
        "updated_since": updated_since,
        "selected_dirty_count": 0,
        "selected_dirty_work_ids": [],
        "cleared_dirty_count": 0,
        "dirty_before": dirty_before,
        "dirty_after": dirty_before,
        "maintenance_job_id": job_id,
        "skipped": True,
        "reason": "no dirty work ids matched the requested scope",
    }


def _sync_scope(args: argparse.Namespace) -> str:
    return str(getattr(args, "scope", "dirty") or "dirty")


def _sync_collection(args: argparse.Namespace) -> str | None:
    value = getattr(args, "collection", None)
    text = str(value).strip() if value is not None else ""
    return text or None


def _sync_updated_since(args: argparse.Namespace) -> str | None:
    value = getattr(args, "updated_since", None)
    text = str(value).strip() if value is not None else ""
    return text or None
