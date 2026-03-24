from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import urllib.parse
from contextlib import closing
from pathlib import Path
from typing import Any

import requests

from hep_rag_v2 import paths
from hep_rag_v2.config import apply_runtime_config, default_config, resolve_config_path, save_config
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.fulltext import import_mineru_source, materialize_mineru_document
from hep_rag_v2.graph import graph_neighbors, rebuild_graph_edges
from hep_rag_v2.metadata import (
    backfill_unresolved_citations,
    find_work_id,
    load_collection_config,
    upsert_collection,
    upsert_work_from_hit,
)
from hep_rag_v2.search import (
    rebuild_search_indices,
    search_assets_bm25,
    search_chunks_bm25,
    search_formulas_bm25,
    search_index_counts,
    search_works_bm25,
)
from hep_rag_v2.vector import (
    DEFAULT_VECTOR_MODEL,
    rebuild_vector_indices,
    route_query,
    search_chunks_vector_chroma,
    search_chunks_hybrid,
    search_chunks_vector,
    search_works_vector_chroma,
    search_works_hybrid,
    search_works_vector,
    sync_chroma_indices,
    vector_index_counts,
)
from hep_rag_v2.pipeline import ask, fetch_online_candidates, ingest_online, initialize_workspace, retrieve

INSPIRE_API = "https://inspirehep.net/api/literature"
AUDIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("latex_frac_literal", re.compile(r"\bfrac\b", re.IGNORECASE)),
    ("style_command_literal", re.compile(r"\b(?:mathnormal|boldsymbol|textcircled|operatorname)\b", re.IGNORECASE)),
    ("array_artifact", re.compile(r"\b(?:begin|end)\s+array\b", re.IGNORECASE)),
    ("arrow_split", re.compile(r"(?<=[A-Za-zΑ-Ωα-ω0-9])\s+-\s+>(?=\s*[A-Za-zΑ-Ωα-ω0-9(])")),
    ("double_punct", re.compile(r"(?:,\s*,|,,|(?<!\.)\.\s*\.(?!\.))")),
    (
        "orphan_citation_phrase",
        re.compile(r"\bdetailed\s+in(?:\s+(?:Ref(?:s)?\.?|reference(?:s)?))?(?=\s*(?:[.,;:]|$))", re.IGNORECASE),
    ),
]
READINESS_THRESHOLDS = {
    "max_retrievable_chunk_noise": 0,
    "max_retrievable_block_noise": 1,
    "min_structured_equation_ratio": 0.6,
}


def load_collection(name: str) -> dict[str, Any]:
    path = paths.COLLECTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Collection config not found: {path}")
    return load_collection_config(path)


def http_get_json(url: str, *, timeout: int = 60, retries: int = 3) -> dict[str, Any]:
    headers = {
        "User-Agent": "hep-rag-v2/0.1 (+local CLI)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Failed to fetch JSON: {last_error}")


def save_raw_payload(*, run_id: int, collection: str, shard_slug: str, page: int, payload: dict[str, Any]) -> Path:
    out_dir = paths.RAW_INSPIRE_DIR / collection / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{shard_slug}_page_{page:04d}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_year_buckets(text: str | None) -> list[tuple[int, int]]:
    default = [(2010, 2014), (2015, 2018), (2019, 2021), (2022, 2026)]
    if not text:
        return default
    buckets: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise SystemExit(f"Invalid year bucket format: {part}. Use e.g. 2010-2014,2015-2018")
        a, b = part.split("-", 1)
        buckets.append((int(a), int(b)))
    return buckets or default


def cmd_init(_: argparse.Namespace) -> None:
    ensure_db()
    print(f"Initialized DB at {paths.DB_PATH}")


def cmd_collections(_: argparse.Namespace) -> None:
    for path in sorted(paths.COLLECTIONS_DIR.glob("*.json")):
        print(path.stem)


def cmd_status(_: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        snapshot = dict(
            conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM collections) AS collections,
                  (SELECT COUNT(*) FROM works) AS works,
                  (SELECT COUNT(*) FROM authors) AS authors,
                  (SELECT COUNT(*) FROM venues) AS venues,
                  (SELECT COUNT(*) FROM topics) AS topics,
                  (SELECT COUNT(*) FROM citations) AS citations,
                  (SELECT COUNT(*) FROM citations WHERE dst_work_id IS NOT NULL) AS resolved_citations,
                  (SELECT COUNT(*) FROM similarity_edges) AS similarity_edges,
                  (SELECT COUNT(*) FROM bibliographic_coupling_edges) AS bibliographic_coupling_edges,
                  (SELECT COUNT(*) FROM co_citation_edges) AS co_citation_edges,
                  (SELECT COUNT(*) FROM graph_build_runs) AS graph_build_runs,
                  (SELECT COUNT(*) FROM documents) AS documents,
                  (SELECT COUNT(*) FROM formulas) AS formulas,
                  (SELECT COUNT(*) FROM assets) AS assets,
                  (SELECT COUNT(*) FROM chunks) AS chunks
                """
            ).fetchone()
        )
        by_collection = [
            dict(row)
            for row in conn.execute(
                """
                SELECT c.name AS collection, COUNT(cw.work_id) AS works
                FROM collections c
                LEFT JOIN collection_works cw ON cw.collection_id = c.collection_id
                GROUP BY c.collection_id, c.name
                ORDER BY c.name
                """
            )
        ]
        snapshot.update(search_index_counts(conn))
        snapshot.update(vector_index_counts(conn))
    print(json.dumps({"snapshot": snapshot, "collections": by_collection}, ensure_ascii=False, indent=2))


def cmd_init_config(args: argparse.Namespace) -> None:
    try:
        config_path = resolve_config_path(args.config)
        workspace_root = Path(args.workspace).expanduser().resolve() if args.workspace else (config_path.parent / "workspace")
        config = default_config(workspace_root=workspace_root)
        if args.collection:
            config["collection"]["name"] = args.collection
            config["collection"]["label"] = args.collection
        save_config(config, config_path, overwrite=args.force)
        _, loaded = apply_runtime_config(config_path=config_path, workspace_root=workspace_root)
        summary = initialize_workspace(loaded, collection_name=args.collection)
        summary["config_path"] = str(config_path)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_fetch_papers(args: argparse.Namespace) -> None:
    try:
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        payload = fetch_online_candidates(
            config,
            query=args.query,
            limit=args.limit,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_ingest_online(args: argparse.Namespace) -> None:
    try:
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        payload = ingest_online(
            config,
            query=args.query,
            limit=args.limit,
            collection_name=args.collection,
            download_limit=args.download_limit,
            parse_limit=args.parse_limit,
            replace_existing=args.replace_existing,
            skip_parse=args.skip_parse,
            skip_index=args.skip_index,
            skip_graph=args.skip_graph,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    try:
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        payload = retrieve(
            config,
            query=args.query,
            limit=args.limit,
            target=args.target,
            collection_name=args.collection,
            model=args.model,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    try:
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        payload = ask(
            config,
            query=args.query,
            mode=args.mode,
            limit=args.limit,
            target=args.target,
            collection_name=args.collection,
            model=args.model,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_ingest_metadata(args: argparse.Namespace) -> None:
    ensure_db()
    config = load_collection(args.collection)
    queries = config.get("queries", {}).get("inspire", [])
    if not queries:
        raise SystemExit(f"No INSPIRE queries configured for {args.collection}")

    year_buckets = parse_year_buckets(args.year_buckets)
    page_size = max(1, min(args.page_size, 50))
    fields = config.get("fields") or []

    with connect() as conn:
        collection_id = upsert_collection(conn, config)
        run_id = _start_ingest_run(
            conn,
            collection_id=collection_id,
            source="inspirehep",
            status="running",
            query_json=json.dumps({"queries": queries, "year_buckets": year_buckets}, ensure_ascii=False),
            page_size=page_size,
            limit_requested=args.limit,
            raw_dir=str(paths.RAW_INSPIRE_DIR / args.collection),
        )
        processed = 0
        works_created = 0
        works_updated = 0
        citations_written = 0
        total_works = _collection_work_count(conn, collection_id)
        initial_total = total_works
        total_tasks = len(queries) * len(year_buckets)
        task_idx = 0

        try:
            for query_idx, query in enumerate(queries, start=1):
                for y1, y2 in year_buckets:
                    task_idx += 1
                    shard_query = f"{query} and date {y1}->{y2}"
                    shard_slug = f"q{query_idx}_{y1}_{y2}"
                    print(f"[{task_idx}/{total_tasks}] INSPIRE shard: {shard_query}")
                    page = 1
                    while total_works < args.limit:
                        params = {
                            "q": shard_query,
                            "size": page_size,
                            "page": page,
                        }
                        if fields:
                            params["fields"] = ",".join(fields)
                        url = f"{INSPIRE_API}?{urllib.parse.urlencode(params)}"
                        payload = http_get_json(url, timeout=args.timeout, retries=args.retries)
                        save_raw_payload(
                            run_id=run_id,
                            collection=args.collection,
                            shard_slug=shard_slug,
                            page=page,
                            payload=payload,
                        )

                        hits = (((payload.get("hits") or {}).get("hits")) or [])
                        if not hits:
                            print("  no more hits in shard")
                            break

                        for hit in hits:
                            stats = upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                            works_created += int(stats["created"])
                            works_updated += int(stats["updated"])
                            citations_written += int(stats["citations_written"])
                            processed += 1

                        resolved_now = backfill_unresolved_citations(conn)
                        conn.commit()
                        total_works = _collection_work_count(conn, collection_id)
                        _update_ingest_run(
                            conn,
                            run_id=run_id,
                            processed_hits=processed,
                            works_created=works_created,
                            works_updated=works_updated,
                            citations_written=citations_written,
                            notes=f"latest_resolved={resolved_now}",
                        )
                        print(
                            f"  page={page} hits={len(hits)} works_total={total_works} "
                            f"created={works_created} updated={works_updated} citations={citations_written}"
                        )
                        page += 1
                        time.sleep(args.sleep)
                        if len(hits) < page_size:
                            break
                    if total_works >= args.limit:
                        break
                if total_works >= args.limit:
                    break
        except Exception as exc:
            _finish_ingest_run(conn, run_id=run_id, status="failed", notes=f"{type(exc).__name__}: {exc}")
            conn.commit()
            raise

        _finish_ingest_run(
            conn,
            run_id=run_id,
            status="completed",
            notes=f"added={total_works - initial_total}",
        )
        conn.commit()
    print(
        f"Done. collection={args.collection} works_total={total_works} "
        f"added={total_works - initial_total} processed_hits={processed}"
    )


def cmd_resolve_citations(_: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        resolved = backfill_unresolved_citations(conn)
        conn.commit()
    print(f"Resolved citations: {resolved}")


def cmd_search_works(args: argparse.Namespace) -> None:
    ensure_db()
    pattern = f"%{args.query.strip()}%"
    if not args.query.strip():
        raise SystemExit("Query cannot be empty.")
    with connect() as conn:
        if args.collection:
            rows = conn.execute(
                """
                SELECT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
                FROM works w
                JOIN collection_works cw ON cw.work_id = w.work_id
                JOIN collections c ON c.collection_id = cw.collection_id
                WHERE c.name = ?
                  AND (w.title LIKE ? OR COALESCE(w.abstract, '') LIKE ?)
                ORDER BY w.year DESC, w.work_id DESC
                LIMIT ?
                """,
                (args.collection, pattern, pattern, args.limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT work_id, title, year, canonical_source, canonical_id
                FROM works
                WHERE title LIKE ? OR COALESCE(abstract, '') LIKE ?
                ORDER BY year DESC, work_id DESC
                LIMIT ?
                """,
                (pattern, pattern, args.limit),
            ).fetchall()
    print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))


def cmd_build_search_index(args: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        summary = rebuild_search_indices(conn, target=args.target)
        conn.commit()
        summary.update(search_index_counts(conn))
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
            summary = rebuild_vector_indices(
                conn,
                target=args.target,
                model=args.model,
                dim=args.dim,
            )
            conn.commit()
            summary.update(vector_index_counts(conn))
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
            summary = sync_chroma_indices(
                conn,
                target=args.target,
                model=args.model,
                collection=args.collection,
                chroma_dir=Path(args.chroma_dir).expanduser().resolve() if args.chroma_dir else None,
                batch_size=args.batch_size,
            )
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
            summary = rebuild_graph_edges(
                conn,
                target=args.target,
                collection=args.collection,
                min_shared=args.min_shared,
                similarity_model=args.model,
                similarity_top_k=args.top_k,
                similarity_min_score=args.min_score,
            )
            conn.commit()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_enrich_inspire_metadata(args: argparse.Namespace) -> None:
    ensure_db()
    config = load_collection(args.collection)
    fields = _required_inspire_fields(config.get("fields") or [])

    with connect() as conn:
        collection_id = upsert_collection(conn, config)
        targets = _select_inspire_enrichment_targets(
            conn,
            collection=args.collection,
            limit=args.limit,
            force=args.force,
        )
        run_id = _start_ingest_run(
            conn,
            collection_id=collection_id,
            source="inspire_refresh",
            status="running",
            query_json=json.dumps(
                {
                    "mode": "enrich_inspire_metadata",
                    "collection": args.collection,
                    "force": bool(args.force),
                },
                ensure_ascii=False,
            ),
            page_size=1,
            limit_requested=args.limit if args.limit is not None else len(targets),
            raw_dir=str(paths.RAW_INSPIRE_DIR / args.collection),
        )
        conn.commit()

        summary = {
            "collection": args.collection,
            "targets": len(targets),
            "fetched": 0,
            "created": 0,
            "updated": 0,
            "citations_written": 0,
            "resolved_citations": 0,
            "skipped_existing_citations": 0,
            "skipped_missing_inspire_id": 0,
            "failed": 0,
            "errors": [],
            "search": None,
            "graph": None,
            "status": None,
        }
        processed = 0
        targets_with_inspire = 0

        try:
            for row in targets:
                work_id = int(row["work_id"])
                inspire_id = str(row["inspire_id"] or "").strip()
                has_citations = bool(row["has_citations"])
                if not inspire_id:
                    summary["skipped_missing_inspire_id"] += 1
                    continue
                if has_citations and not args.force:
                    summary["skipped_existing_citations"] += 1
                    continue

                targets_with_inspire += 1
                payload = http_get_json(
                    _inspire_literature_url(inspire_id, fields=fields),
                    timeout=args.timeout,
                    retries=args.retries,
                )
                save_raw_payload(
                    run_id=run_id,
                    collection=args.collection,
                    shard_slug=f"refresh_{inspire_id}",
                    page=1,
                    payload=payload,
                )
                stats = upsert_work_from_hit(conn, collection_id=collection_id, hit=payload)
                summary["fetched"] += 1
                summary["created"] += int(stats["created"])
                summary["updated"] += int(stats["updated"])
                summary["citations_written"] += int(stats["citations_written"])
                processed += 1
                _update_ingest_run(
                    conn,
                    run_id=run_id,
                    processed_hits=processed,
                    works_created=summary["created"],
                    works_updated=summary["updated"],
                    citations_written=summary["citations_written"],
                    notes=f"latest_inspire_id={inspire_id}",
                )
                conn.commit()
                time.sleep(args.sleep)

            summary["resolved_citations"] = backfill_unresolved_citations(conn)
            if not args.skip_search:
                summary["search"] = rebuild_search_indices(conn, target="works")
            if not args.skip_graph:
                summary["graph"] = rebuild_graph_edges(
                    conn,
                    target="all",
                    collection=args.collection,
                    min_shared=args.min_shared,
                )
            conn.commit()
            _finish_ingest_run(
                conn,
                run_id=run_id,
                status="completed",
                notes=f"targets={len(targets)} fetched={summary['fetched']} resolved={summary['resolved_citations']}",
            )
            conn.commit()
        except Exception as exc:
            summary["failed"] += 1
            summary["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
            _finish_ingest_run(conn, run_id=run_id, status="failed", notes=f"{type(exc).__name__}: {exc}")
            conn.commit()
            raise

        summary["status"] = dict(
            conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM works) AS works,
                  (SELECT COUNT(*) FROM citations) AS citations,
                  (SELECT COUNT(*) FROM citations WHERE dst_work_id IS NOT NULL) AS resolved_citations,
                  (SELECT COUNT(*) FROM bibliographic_coupling_edges) AS bibliographic_coupling_edges,
                  (SELECT COUNT(*) FROM co_citation_edges) AS co_citation_edges,
                  (SELECT COUNT(*) FROM work_search) AS work_search
                """
            ).fetchone()
        )
        summary["targets_with_inspire"] = targets_with_inspire
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_show_graph(args: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        work = dict(_resolve_work_row(
            conn,
            work_id=args.work_id,
            id_type=args.id_type,
            id_value=args.id_value,
        ))
        neighbors = graph_neighbors(
            conn,
            work_id=int(work["work_id"]),
            edge_kind=args.edge_kind,
            collection=args.collection,
            limit=args.limit,
            similarity_model=args.model if args.edge_kind == "similarity" else None,
        )
    payload = {
        "work": {
            "work_id": int(work["work_id"]),
            "canonical_source": work["canonical_source"],
            "canonical_id": work["canonical_id"],
            "title": work["title"],
            "year": work["year"],
        },
        "edge_kind": args.edge_kind,
        "collection": args.collection,
        "neighbors": neighbors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_import_mineru(args: argparse.Namespace) -> None:
    ensure_db()
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"Source not found: {source_path}")

    with connect() as conn:
        work = _resolve_work_row(
            conn,
            work_id=args.work_id,
            id_type=args.id_type,
            id_value=args.id_value,
        )
        collection_name = args.collection or _infer_collection_name(conn, int(work["work_id"])) or "default"
        dest_dir = _parsed_doc_dir(
            collection_name,
            _paper_storage_stem(conn, int(work["work_id"])),
        )
        import_mineru_source(source_path=source_path, dest_dir=dest_dir, replace=args.replace)
        summary = materialize_mineru_document(
            conn,
            work_id=int(work["work_id"]),
            manifest_path=dest_dir / "manifest.json",
            replace=args.replace,
            chunk_size=args.chunk_size,
            overlap_blocks=args.overlap_blocks,
            section_parent_char_limit=args.section_parent_char_limit,
        )
        conn.commit()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_bootstrap_legacy_corpus(args: argparse.Namespace) -> None:
    ensure_db()
    legacy_db = Path(args.legacy_db).expanduser().resolve()
    parsed_root = Path(args.parsed_root).expanduser().resolve()
    if not legacy_db.exists():
        raise SystemExit(f"Legacy DB not found: {legacy_db}")
    if not parsed_root.exists():
        raise SystemExit(f"Parsed root not found: {parsed_root}")

    config = load_collection(args.collection)
    with closing(sqlite3.connect(legacy_db)) as legacy_conn, connect() as conn:
        legacy_conn.row_factory = sqlite3.Row
        collection_id = upsert_collection(conn, config)

        legacy_rows = _load_legacy_papers(
            legacy_conn,
            collection=args.collection,
            limit=args.limit,
        )
        metadata_summary = {
            "legacy_papers": len(legacy_rows),
            "created": 0,
            "updated": 0,
            "citations_written": 0,
            "resolved_citations": 0,
        }
        selected_ids: set[tuple[str, str]] = set()
        for row in legacy_rows:
            hit = _legacy_hit_from_row(row)
            stats = upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
            metadata_summary["created"] += int(stats["created"])
            metadata_summary["updated"] += int(stats["updated"])
            metadata_summary["citations_written"] += int(stats["citations_written"])
            selected_ids.update(_legacy_row_identities(row))

        metadata_summary["resolved_citations"] = backfill_unresolved_citations(conn)
        conn.commit()

        manifests = _discover_manifests(parsed_root)
        import_summary = {
            "manifests_seen": len(manifests),
            "matched_manifests": 0,
            "materialized": 0,
            "skipped_missing_work": 0,
            "ready_documents": 0,
            "not_ready_documents": 0,
            "audited_documents": 0,
            "audit_samples": [],
        }
        if selected_ids:
            manifests = [
                path for path in manifests
                if any(candidate in selected_ids for candidate in _manifest_identity_candidates(path.parent.name))
            ]
            import_summary["matched_manifests"] = len(manifests)
        else:
            import_summary["matched_manifests"] = len(manifests)

        audit_targets: list[tuple[int, int]] = []
        for manifest_path in manifests:
            work_id = _resolve_manifest_work_id(conn, manifest_path)
            if work_id is None:
                import_summary["skipped_missing_work"] += 1
                continue
            collection_name = args.collection or _infer_collection_name(conn, work_id) or "default"
            dest_dir = _parsed_doc_dir(collection_name, _paper_storage_stem(conn, work_id))
            import_mineru_source(source_path=manifest_path, dest_dir=dest_dir, replace=args.replace)
            materialize_mineru_document(
                conn,
                work_id=work_id,
                manifest_path=dest_dir / "manifest.json",
                replace=args.replace,
                chunk_size=args.chunk_size,
                overlap_blocks=args.overlap_blocks,
                section_parent_char_limit=args.section_parent_char_limit,
            )
            import_summary["materialized"] += 1

            document_row = conn.execute(
                """
                SELECT document_id
                FROM documents
                WHERE work_id = ?
                ORDER BY document_id DESC
                LIMIT 1
                """,
                (work_id,),
            ).fetchone()
            if document_row is not None:
                audit_targets.append((work_id, int(document_row["document_id"])))

        search_summary = rebuild_search_indices(conn, target="all")
        if args.audit_limit != 0:
            for work_id, document_id in audit_targets:
                audit = _audit_document_payload(
                    conn,
                    document_id=document_id,
                    limit=max(1, args.audit_limit),
                )
                import_summary["audited_documents"] += 1
                if bool(audit["ready"]):
                    import_summary["ready_documents"] += 1
                else:
                    import_summary["not_ready_documents"] += 1
                if len(import_summary["audit_samples"]) < max(1, args.audit_limit):
                    import_summary["audit_samples"].append(
                        {
                            "work_id": work_id,
                            "document_id": document_id,
                            "ready": bool(audit["ready"]),
                            "recommendation": audit["recommendation"],
                        }
                    )
        graph_summary = None
        if not args.skip_graph:
            graph_summary = rebuild_graph_edges(
                conn,
                target="all",
                collection=args.collection,
                min_shared=args.min_shared,
            )
        conn.commit()

        status_snapshot = dict(
            conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM works) AS works,
                  (SELECT COUNT(*) FROM documents) AS documents,
                  (SELECT COUNT(*) FROM chunks) AS chunks,
                  (SELECT COUNT(*) FROM citations) AS citations,
                  (SELECT COUNT(*) FROM citations WHERE dst_work_id IS NOT NULL) AS resolved_citations,
                  (SELECT COUNT(*) FROM bibliographic_coupling_edges) AS bibliographic_coupling_edges,
                  (SELECT COUNT(*) FROM co_citation_edges) AS co_citation_edges
                """
            ).fetchone()
        )

    print(
        json.dumps(
            {
                "collection": args.collection,
                "legacy_db": str(legacy_db),
                "parsed_root": str(parsed_root),
                "metadata": metadata_summary,
                "import": import_summary,
                "search": search_summary,
                "graph": graph_summary,
                "status": status_snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_show_document(args: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        work = _resolve_work_row(
            conn,
            work_id=args.work_id,
            id_type=args.id_type,
            id_value=args.id_value,
        )
        document = conn.execute(
            """
            SELECT d.document_id, d.work_id, d.parser_name, d.parser_version, d.parse_status,
                   d.parsed_dir, d.manifest_path, w.title
            FROM documents d
            JOIN works w ON w.work_id = d.work_id
            WHERE d.work_id = ?
            """,
            (int(work["work_id"]),),
        ).fetchone()
        if document is None:
            raise SystemExit(f"No document materialized for work_id={int(work['work_id'])}")

        sections = [
            dict(row)
            for row in conn.execute(
                """
                SELECT section_id, parent_section_id, ordinal, title, clean_title, section_kind,
                       level, order_index, page_start, page_end, path_text
                FROM document_sections
                WHERE document_id = ?
                ORDER BY order_index, section_id
                LIMIT ?
                """,
                (int(document["document_id"]), args.limit),
            )
        ]
        blocks = [
            dict(row)
            for row in conn.execute(
                """
                SELECT block_id, section_id, block_type, page, order_index, block_role,
                       is_heading, is_retrievable, exclusion_reason, clean_text
                FROM blocks
                WHERE document_id = ?
                ORDER BY order_index, block_id
                LIMIT ?
                """,
                (int(document["document_id"]), args.limit),
            )
        ]
        chunks = [
            dict(row)
            for row in conn.execute(
                """
                SELECT chunk_id, section_id, chunk_role, page_hint, is_retrievable,
                       exclusion_reason, clean_text
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_id
                LIMIT ?
                """,
                (int(document["document_id"]), args.limit),
            )
        ]
        role_counts = dict(
            conn.execute(
                """
                SELECT block_role, COUNT(*) AS n
                FROM blocks
                WHERE document_id = ?
                GROUP BY block_role
                ORDER BY block_role
                """,
                (int(document["document_id"]),),
            ).fetchall()
        )
        chunk_counts = dict(
            conn.execute(
                """
                SELECT chunk_role, COUNT(*) AS n
                FROM chunks
                WHERE document_id = ?
                GROUP BY chunk_role
                ORDER BY chunk_role
                """,
                (int(document["document_id"]),),
            ).fetchall()
        )
    print(
        json.dumps(
            {
                "document": dict(document),
                "block_roles": role_counts,
                "chunk_roles": chunk_counts,
                "sections_sample": sections,
                "blocks_sample": blocks,
                "chunks_sample": chunks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_audit_document(args: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        work = _resolve_work_row(
            conn,
            work_id=args.work_id,
            id_type=args.id_type,
            id_value=args.id_value,
        )
        document = conn.execute(
            """
            SELECT d.document_id, d.work_id, d.parser_name, d.parser_version, d.parse_status,
                   d.parsed_dir, d.manifest_path, w.title
            FROM documents d
            JOIN works w ON w.work_id = d.work_id
            WHERE d.work_id = ?
            """,
            (int(work["work_id"]),),
        ).fetchone()
        if document is None:
            raise SystemExit(f"No document materialized for work_id={int(work['work_id'])}")

        payload = _audit_document_payload(conn, document_id=int(document["document_id"]), limit=args.limit)
        payload["document"] = dict(document)
        payload["work"] = {
            "work_id": int(work["work_id"]),
            "canonical_source": work["canonical_source"],
            "canonical_id": work["canonical_id"],
            "title": work["title"],
            "year": work["year"],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _collection_work_count(conn: sqlite3.Connection, collection_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM collection_works WHERE collection_id = ?",
        (collection_id,),
    ).fetchone()
    return int(row["n"] if row is not None else 0)


def _required_inspire_fields(fields: list[str]) -> list[str]:
    required = [
        "control_number",
        "titles",
        "abstracts",
        "authors",
        "collaborations",
        "publication_info",
        "arxiv_eprints",
        "dois",
        "citation_count",
        "references",
        "keywords",
        "inspire_categories",
        "documents",
        "accelerator_experiments",
        "preprint_date",
        "earliest_date",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for field in [*fields, *required]:
        name = str(field or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _inspire_literature_url(inspire_id: str, *, fields: list[str]) -> str:
    query = urllib.parse.urlencode({"fields": ",".join(fields)})
    return f"{INSPIRE_API}/{urllib.parse.quote(str(inspire_id).strip())}?{query}"


def _select_inspire_enrichment_targets(
    conn: sqlite3.Connection,
    *,
    collection: str,
    limit: int | None,
    force: bool,
) -> list[sqlite3.Row]:
    sql = """
        SELECT
          w.work_id,
          w.canonical_source,
          w.canonical_id,
          COALESCE(
            (SELECT wi.id_value FROM work_ids wi WHERE wi.work_id = w.work_id AND wi.id_type = 'inspire' LIMIT 1),
            CASE WHEN w.canonical_source = 'inspire' THEN w.canonical_id ELSE NULL END
          ) AS inspire_id,
          EXISTS(SELECT 1 FROM citations c WHERE c.src_work_id = w.work_id) AS has_citations
        FROM works w
        JOIN collection_works cw ON cw.work_id = w.work_id
        JOIN collections c ON c.collection_id = cw.collection_id
        WHERE c.name = ?
    """
    params: list[Any] = [collection]
    if not force:
        sql += " AND NOT EXISTS(SELECT 1 FROM citations c WHERE c.src_work_id = w.work_id)"
    sql += " ORDER BY w.year DESC, w.work_id DESC"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _load_legacy_papers(
    conn: sqlite3.Connection,
    *,
    collection: str,
    limit: int | None,
) -> list[sqlite3.Row]:
    sql = """
        SELECT *
        FROM papers
        WHERE collection = ?
        ORDER BY paper_id
    """
    params: list[Any] = [collection]
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _legacy_hit_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata: dict[str, Any]
    raw_metadata = str(row["raw_metadata_json"] or "").strip()
    if raw_metadata:
        metadata = json.loads(raw_metadata)
    else:
        metadata = {}

    if not metadata.get("control_number") and row["inspire_id"]:
        metadata["control_number"] = int(row["inspire_id"]) if str(row["inspire_id"]).isdigit() else str(row["inspire_id"])
    if row["title"] and not metadata.get("titles"):
        metadata["titles"] = [{"title": str(row["title"])}]
    if row["abstract"] and not metadata.get("abstracts"):
        metadata["abstracts"] = [{"value": str(row["abstract"])}]
    if row["year"] and not metadata.get("publication_info"):
        metadata["publication_info"] = [{"year": int(row["year"])}]
    if row["arxiv_id"] and not metadata.get("arxiv_eprints"):
        metadata["arxiv_eprints"] = [{"value": str(row["arxiv_id"])}]
    if row["doi"] and not metadata.get("dois"):
        metadata["dois"] = [{"value": str(row["doi"])}]
    if row["citation_count"] and metadata.get("citation_count") is None:
        metadata["citation_count"] = int(row["citation_count"])

    links: dict[str, Any] = {}
    if row["source_url"]:
        links["self"] = str(row["source_url"])
    return {"metadata": metadata, "links": links}


def _legacy_row_identities(row: sqlite3.Row) -> set[tuple[str, str]]:
    ids: set[tuple[str, str]] = set()
    if row["inspire_id"]:
        ids.add(("inspire", str(row["inspire_id"]).strip()))
    if row["arxiv_id"]:
        ids.add(("arxiv", str(row["arxiv_id"]).strip()))
    if row["doi"]:
        ids.add(("doi", str(row["doi"]).strip().lower()))
    return ids


def _discover_manifests(parsed_root: Path) -> list[Path]:
    return sorted(path.resolve() for path in parsed_root.rglob("manifest.json"))


def _manifest_identity_candidates(stem: str) -> list[tuple[str, str]]:
    value = stem.strip()
    candidates: list[tuple[str, str]] = []
    if not value:
        return candidates
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$", value):
        candidates.append(("arxiv", value))
    elif re.match(r"^[A-Za-z-]+_[0-9]{7}$", value):
        candidates.append(("arxiv", value.replace("_", "/", 1)))
    elif re.match(r"^[0-9]+$", value):
        candidates.append(("inspire", value))
    return candidates


def _resolve_manifest_work_id(conn: sqlite3.Connection, manifest_path: Path) -> int | None:
    for id_type, id_value in _manifest_identity_candidates(manifest_path.parent.name):
        work_id = find_work_id(conn, id_type=id_type, id_value=id_value)
        if work_id is not None:
            return work_id
    return None


def _audit_document_payload(conn: sqlite3.Connection, *, document_id: int, limit: int) -> dict[str, Any]:
    block_counts = dict(
        conn.execute(
            """
            SELECT
              COUNT(*) AS blocks_total,
              SUM(CASE WHEN is_retrievable = 1 THEN 1 ELSE 0 END) AS retrievable_blocks,
              SUM(CASE WHEN block_role = 'body' THEN 1 ELSE 0 END) AS body_blocks,
              SUM(CASE WHEN block_role = 'body' AND is_retrievable = 1 THEN 1 ELSE 0 END) AS retrievable_body_blocks
            FROM blocks
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    )
    chunk_counts = dict(
        conn.execute(
            """
            SELECT
              COUNT(*) AS chunks_total,
              SUM(CASE WHEN is_retrievable = 1 THEN 1 ELSE 0 END) AS retrievable_chunks,
              SUM(CASE WHEN chunk_role = 'section_child' THEN 1 ELSE 0 END) AS section_child_chunks,
              SUM(CASE WHEN chunk_role = 'formula_window' THEN 1 ELSE 0 END) AS formula_window_chunks,
              SUM(CASE WHEN chunk_role = 'asset_window' THEN 1 ELSE 0 END) AS asset_window_chunks
            FROM chunks
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    )
    search_counts = dict(
        conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM chunk_search cs JOIN chunks c ON c.chunk_id = cs.chunk_id WHERE c.document_id = ?) AS chunk_search_rows,
              (SELECT COUNT(*) FROM formula_search fs JOIN formulas f ON f.formula_id = fs.formula_id WHERE f.document_id = ?) AS formula_search_rows,
              (SELECT COUNT(*) FROM asset_search asearch JOIN assets a ON a.asset_id = asearch.asset_id WHERE a.document_id = ?) AS asset_search_rows
            """,
            (document_id, document_id, document_id),
        ).fetchone()
    )

    retrievable_blocks = _collect_noise_hits(
        conn,
        """
        SELECT block_id AS row_id, block_role AS role, clean_text
        FROM blocks
        WHERE document_id = ?
          AND is_retrievable = 1
          AND block_type != 'equation'
          AND clean_text IS NOT NULL
        ORDER BY block_id
        """,
        document_id=document_id,
        limit=limit,
    )
    retrievable_chunks = _collect_noise_hits(
        conn,
        """
        SELECT chunk_id AS row_id, chunk_role AS role, clean_text
        FROM chunks
        WHERE document_id = ?
          AND is_retrievable = 1
          AND clean_text IS NOT NULL
        ORDER BY chunk_id
        """,
        document_id=document_id,
        limit=limit,
    )
    all_blocks = _collect_noise_hits(
        conn,
        """
        SELECT block_id AS row_id, block_role AS role, clean_text
        FROM blocks
        WHERE document_id = ?
          AND clean_text IS NOT NULL
        ORDER BY block_id
        """,
        document_id=document_id,
        limit=limit,
    )

    equation_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT chunk_id, chunk_role, clean_text
            FROM chunks
            WHERE document_id = ?
              AND chunk_role IN ('section_child', 'section_parent')
              AND clean_text LIKE '%Equation:%'
            ORDER BY chunk_id
            """,
            (document_id,),
        ).fetchall()
    ]
    equation_with_structure = sum(
        int(("/" in str(row["clean_text"])) or (" x " in str(row["clean_text"])))
        for row in equation_rows
    )
    equation_with_frac_literal = sum(
        int(bool(re.search(r"\bfrac\b", str(row["clean_text"]), re.IGNORECASE)))
        for row in equation_rows
    )
    structured_ratio = (
        float(equation_with_structure) / float(len(equation_rows))
        if equation_rows
        else 1.0
    )

    readiness_checks = {
        "retrievable_chunk_noise_ok": retrievable_chunks["total_hits"] <= READINESS_THRESHOLDS["max_retrievable_chunk_noise"],
        "retrievable_block_noise_ok": retrievable_blocks["total_hits"] <= READINESS_THRESHOLDS["max_retrievable_block_noise"],
        "equation_structure_ok": structured_ratio >= READINESS_THRESHOLDS["min_structured_equation_ratio"],
        "chunk_index_complete": int(search_counts.get("chunk_search_rows") or 0) == int(chunk_counts.get("chunks_total") or 0),
    }
    ready = all(readiness_checks.values())
    return {
        "ready": ready,
        "readiness_checks": readiness_checks,
        "counts": {
            "blocks": block_counts,
            "chunks": chunk_counts,
            "search": search_counts,
        },
        "equation_placeholders": {
            "total": len(equation_rows),
            "with_structure": equation_with_structure,
            "with_frac_literal": equation_with_frac_literal,
            "structured_ratio": round(structured_ratio, 3),
            "samples": [
                {
                    "chunk_id": int(row["chunk_id"]),
                    "chunk_role": row["chunk_role"],
                    "clean_text": str(row["clean_text"])[:300],
                }
                for row in equation_rows[:limit]
            ],
        },
        "noise": {
            "retrievable_blocks": retrievable_blocks,
            "retrievable_chunks": retrievable_chunks,
            "all_blocks": all_blocks,
        },
        "recommendation": _readiness_recommendation(
            ready=ready,
            retrievable_chunks=retrievable_chunks["total_hits"],
            retrievable_blocks=retrievable_blocks["total_hits"],
            equation_ratio=structured_ratio,
        ),
    }


def _collect_noise_hits(
    conn: sqlite3.Connection,
    query: str,
    *,
    document_id: int,
    limit: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {name: 0 for name, _ in AUDIT_PATTERNS}
    samples: list[dict[str, Any]] = []
    total_hits = 0
    for row in conn.execute(query, (document_id,)).fetchall():
        text = str(row["clean_text"] or "")
        matched = [name for name, pattern in AUDIT_PATTERNS if pattern.search(text)]
        if not matched:
            continue
        total_hits += 1
        for name in matched:
            counts[name] += 1
        if len(samples) < limit:
            samples.append(
                {
                    "row_id": int(row["row_id"]),
                    "role": row["role"],
                    "patterns": matched,
                    "clean_text": text[:300],
                }
            )
    return {
        "total_hits": total_hits,
        "pattern_counts": counts,
        "samples": samples,
    }


def _readiness_recommendation(
    *,
    ready: bool,
    retrievable_chunks: int,
    retrievable_blocks: int,
    equation_ratio: float,
) -> str:
    if ready:
        return "ready_for_next_phase"
    if retrievable_chunks > 0:
        return "clean_retrieval_text_first"
    if equation_ratio < READINESS_THRESHOLDS["min_structured_equation_ratio"]:
        return "improve_equation_placeholders"
    if retrievable_blocks > READINESS_THRESHOLDS["max_retrievable_block_noise"]:
        return "reduce_parser_noise_in_blocks"
    return "needs_manual_review"


def _start_ingest_run(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    source: str,
    status: str,
    query_json: str,
    page_size: int,
    limit_requested: int,
    raw_dir: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingest_runs (
          collection_id, source, status, query_json, page_size, limit_requested, raw_dir
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (collection_id, source, status, query_json, page_size, limit_requested, raw_dir),
    )
    return int(cur.lastrowid)


def _update_ingest_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    processed_hits: int,
    works_created: int,
    works_updated: int,
    citations_written: int,
    notes: str | None,
) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET processed_hits = ?, works_created = ?, works_updated = ?, citations_written = ?, notes = ?
        WHERE run_id = ?
        """,
        (processed_hits, works_created, works_updated, citations_written, notes, run_id),
    )


def _finish_ingest_run(conn: sqlite3.Connection, *, run_id: int, status: str, notes: str | None) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET status = ?, notes = ?, finished_at = CURRENT_TIMESTAMP
        WHERE run_id = ?
        """,
        (status, notes, run_id),
    )


def _resolve_work_row(
    conn: sqlite3.Connection,
    *,
    work_id: int | None,
    id_type: str | None,
    id_value: str | None,
) -> sqlite3.Row:
    if work_id is not None:
        row = conn.execute(
            """
            SELECT work_id, title, year, canonical_source, canonical_id
            FROM works
            WHERE work_id = ?
            """,
            (work_id,),
        ).fetchone()
        if row is None:
            raise SystemExit(f"Unknown work_id: {work_id}")
        return row

    if id_type and id_value:
        row = conn.execute(
            """
            SELECT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
            FROM work_ids wi
            JOIN works w ON w.work_id = wi.work_id
            WHERE wi.id_type = ? AND wi.id_value = ?
            """,
            (id_type, id_value),
        ).fetchone()
        if row is None:
            raise SystemExit(f"Unknown work identity: {id_type}:{id_value}")
        return row

    raise SystemExit("Specify either --work-id or both --id-type and --id-value.")


def _infer_collection_name(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT c.name
        FROM collection_works cw
        JOIN collections c ON c.collection_id = cw.collection_id
        WHERE cw.work_id = ?
        ORDER BY c.name
        LIMIT 1
        """,
        (work_id,),
    ).fetchone()
    return str(row["name"]) if row is not None else None


def _paper_storage_stem(conn: sqlite3.Connection, work_id: int) -> str:
    id_rows = conn.execute(
        """
        SELECT id_type, id_value, is_primary
        FROM work_ids
        WHERE work_id = ?
        ORDER BY is_primary DESC, CASE id_type WHEN 'arxiv' THEN 0 WHEN 'inspire' THEN 1 ELSE 2 END, id_type
        """,
        (work_id,),
    ).fetchall()
    for row in id_rows:
        value = str(row["id_value"]).strip()
        if value:
            return _safe_stem(value)
    row = conn.execute(
        "SELECT canonical_id FROM works WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if row is not None and str(row["canonical_id"]).strip():
        return _safe_stem(str(row["canonical_id"]).strip())
    return str(work_id)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "paper"


def _parsed_doc_dir(collection: str, stem: str) -> Path:
    return paths.PARSED_DIR / collection / stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hep-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-config", help="Write a user config file and initialize an empty workspace")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Workspace root for DB/data/indexes")
    s.add_argument("--collection", default=None, help="Initial collection name")
    s.add_argument("--force", action="store_true", help="Overwrite existing config file")
    s.set_defaults(func=cmd_init_config)

    s = sub.add_parser("fetch-papers", help="Search INSPIRE online and show candidate papers before ingest")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_fetch_papers)

    s = sub.add_parser("ingest-online", help="Search online, download PDFs, optionally parse with MinerU, then build indices")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--download-limit", type=int, default=None, help="Maximum number of PDFs to download")
    s.add_argument("--parse-limit", type=int, default=None, help="Maximum number of PDFs to send to MinerU")
    s.add_argument("--replace-existing", action="store_true", help="Rebuild already materialized documents")
    s.add_argument("--skip-parse", action="store_true", help="Only ingest metadata and PDFs, skip MinerU parsing")
    s.add_argument("--skip-index", action="store_true", help="Skip rebuilding search/vector indices after ingest")
    s.add_argument("--skip-graph", action="store_true", help="Skip rebuilding graph edges after ingest")
    s.set_defaults(func=cmd_ingest_online)

    s = sub.add_parser("query", help="Run config-driven retrieval and return structured evidence")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--target", choices=["auto", "works", "chunks"], default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--model", default=None, help="Override embedding model")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("ask", help="Run retrieval and synthesize an answer with either an OpenAI-compatible API or a local Transformers model")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--target", choices=["auto", "works", "chunks"], default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--model", default=None, help="Override embedding model")
    s.add_argument("--mode", choices=["answer", "survey", "idea"], default="answer")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("init", help="Initialize local database and directories")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("collections", help="List available collection configs")
    s.set_defaults(func=cmd_collections)

    s = sub.add_parser("status", help="Show metadata graph snapshot")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("ingest-metadata", help="Ingest INSPIRE metadata into the graph schema")
    s.add_argument("--collection", default="cms_rare_decay")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--page-size", type=int, default=25)
    s.add_argument("--timeout", type=int, default=60)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--sleep", type=float, default=0.2)
    s.add_argument("--year-buckets", default=None)
    s.set_defaults(func=cmd_ingest_metadata)

    s = sub.add_parser("resolve-citations", help="Backfill unresolved citation targets")
    s.set_defaults(func=cmd_resolve_citations)

    s = sub.add_parser("search-works", help="Search titles and abstracts in the metadata graph")
    s.add_argument("query")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_search_works)

    s = sub.add_parser("build-search-index", help="Rebuild SQLite FTS5 BM25 search indices")
    s.add_argument("--target", choices=["all", "works", "chunks", "formulas", "assets", "structure"], default="all")
    s.set_defaults(func=cmd_build_search_index)

    s = sub.add_parser("search-bm25", help="Search works, chunks, formulas, or assets with SQLite FTS5 BM25")
    s.add_argument("query")
    s.add_argument("--target", choices=["works", "chunks", "formulas", "assets"], default="works")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_search_bm25)

    s = sub.add_parser("build-vector-index", help="Build local vector indices for works and/or chunks")
    s.add_argument("--target", choices=["all", "works", "chunks"], default="all")
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--dim", type=int, default=768)
    s.set_defaults(func=cmd_build_vector_index)

    s = sub.add_parser("sync-chroma-index", help="Mirror local vector indices into an optional Chroma vector store")
    s.add_argument("--target", choices=["all", "works", "chunks"], default="all")
    s.add_argument("--collection", default=None)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--chroma-dir", default=None)
    s.add_argument("--batch-size", type=int, default=256)
    s.set_defaults(func=cmd_sync_chroma_index)

    s = sub.add_parser("search-vector", help="Search works or chunks with the local vector index")
    s.add_argument("query")
    s.add_argument("--target", choices=["works", "chunks"], default="works")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--backend", choices=["local", "chroma"], default="local")
    s.add_argument("--chroma-dir", default=None)
    s.set_defaults(func=cmd_search_vector)

    s = sub.add_parser("search-hybrid", help="Search with BM25 plus vector retrieval; auto mode can route broad queries to works")
    s.add_argument("query")
    s.add_argument("--target", choices=["auto", "works", "chunks"], default="auto")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--graph-expand", type=int, default=None)
    s.add_argument("--seed-limit", type=int, default=5)
    s.set_defaults(func=cmd_search_hybrid)

    s = sub.add_parser("enrich-inspire-metadata", help="Refresh existing works from INSPIRE and backfill citation graph data")
    s.add_argument("--collection", default="cms_rare_decay")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--force", action="store_true", help="Refresh works even if citations already exist")
    s.add_argument("--timeout", type=int, default=60)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--sleep", type=float, default=0.1)
    s.add_argument("--skip-search", action="store_true")
    s.add_argument("--skip-graph", action="store_true")
    s.add_argument("--min-shared", type=int, default=2)
    s.set_defaults(func=cmd_enrich_inspire_metadata)

    s = sub.add_parser("build-graph", help="Build citation and embedding-based graph edges")
    s.add_argument("--target", choices=["all", "bibliographic-coupling", "co-citation", "similarity"], default="all")
    s.add_argument("--collection", default=None)
    s.add_argument("--min-shared", type=int, default=2)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--min-score", type=float, default=0.35)
    s.set_defaults(func=cmd_build_graph)

    s = sub.add_parser("show-graph", help="Inspect graph neighbors for a work")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--edge-kind", choices=["all", "bibliographic-coupling", "co-citation", "similarity"], default="all")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.set_defaults(func=cmd_show_graph)

    s = sub.add_parser("import-mineru", help="Import MinerU output and materialize a structured document")
    s.add_argument("--source", required=True, help="ZIP, raw MinerU output dir, manifest.json, or parsed dir")
    s.add_argument("--collection", default=None, help="Override target parsed collection dir")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--replace", action="store_true")
    s.add_argument("--chunk-size", type=int, default=2400)
    s.add_argument("--overlap-blocks", type=int, default=1)
    s.add_argument("--section-parent-char-limit", type=int, default=12000)
    s.set_defaults(func=cmd_import_mineru)

    s = sub.add_parser("bootstrap-legacy-corpus", help="Bootstrap metadata and MinerU parses from the legacy hep_rag corpus")
    s.add_argument("--legacy-db", required=True, help="Path to legacy hep_rag SQLite DB")
    s.add_argument("--parsed-root", required=True, help="Root directory containing legacy parsed MinerU manifests")
    s.add_argument("--collection", default="cms_rare_decay")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--replace", action="store_true")
    s.add_argument("--chunk-size", type=int, default=2400)
    s.add_argument("--overlap-blocks", type=int, default=1)
    s.add_argument("--section-parent-char-limit", type=int, default=12000)
    s.add_argument("--audit-limit", type=int, default=5)
    s.add_argument("--skip-graph", action="store_true")
    s.add_argument("--min-shared", type=int, default=2)
    s.set_defaults(func=cmd_bootstrap_legacy_corpus)

    s = sub.add_parser("show-document", help="Inspect a materialized document and sample clean chunks")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_show_document)

    s = sub.add_parser("audit-document", help="Audit parser noise and retrieval readiness for a materialized document")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_audit_document)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
