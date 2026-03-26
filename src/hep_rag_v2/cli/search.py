from __future__ import annotations

import argparse
import json
from pathlib import Path

from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.graph import rebuild_graph_edges
from hep_rag_v2.search import (
    rebuild_search_indices,
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
    from hep_rag_v2.search import search_index_counts

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
