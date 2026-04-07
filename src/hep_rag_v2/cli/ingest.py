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

from hep_rag_v2 import paths
from hep_rag_v2.config import apply_runtime_config, default_config
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.fulltext import import_mineru_source, materialize_mineru_document
from hep_rag_v2.graph import rebuild_graph_edges
from hep_rag_v2.metadata import (
    backfill_unresolved_citations,
    upsert_collection,
    upsert_work_from_hit,
)
from hep_rag_v2.pdg import import_pdg_source
from hep_rag_v2.pipeline import ask, fetch_online_candidates, import_pdg, ingest_online, reparse_cached_pdfs, retrieve
from hep_rag_v2.search import rebuild_search_indices

from ._common import (
    INSPIRE_API,
    _collection_work_count,
    _discover_manifests,
    _finish_ingest_run,
    _infer_collection_name,
    _inspire_literature_url,
    _legacy_hit_from_row,
    _legacy_row_identities,
    _load_legacy_papers,
    _manifest_identity_candidates,
    _paper_storage_stem,
    _parsed_doc_dir,
    _required_inspire_fields,
    _resolve_manifest_work_id,
    _resolve_work_row,
    _select_inspire_enrichment_targets,
    _start_ingest_run,
    _update_ingest_run,
    emit_cli_status,
    http_get_json,
    load_collection,
    parse_year_buckets,
    save_raw_payload,
)
from .inspect import _audit_document_payload


def cmd_fetch_papers(args: argparse.Namespace) -> None:
    try:
        emit_cli_status("loading config...")
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        emit_cli_status("preparing online search...")
        payload = fetch_online_candidates(
            config,
            query=args.query,
            limit=args.limit,
            progress=emit_cli_status,
        )
        emit_cli_status(f"done. found {payload['effective_count']} candidate papers.")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_ingest_online(args: argparse.Namespace) -> None:
    try:
        emit_cli_status("loading config...")
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        emit_cli_status("starting online ingest...")
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
            progress=emit_cli_status,
        )
        emit_cli_status("online ingest finished.")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_reparse_pdfs(args: argparse.Namespace) -> None:
    try:
        emit_cli_status("loading config...")
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        emit_cli_status("scanning cached PDFs that still need MinerU...")
        payload = reparse_cached_pdfs(
            config,
            collection_name=args.collection,
            limit=args.limit,
            work_ids=args.work_id,
            replace_existing=args.replace_existing,
            skip_index=args.skip_index,
            skip_graph=args.skip_graph,
            progress=emit_cli_status,
        )
        emit_cli_status("cached PDF reparse finished.")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))



def cmd_import_pdg(args: argparse.Namespace) -> None:
    try:
        emit_cli_status("loading config...")
        if args.config is None and args.workspace is None:
            config = default_config(workspace_root=paths.workspace_root())
            config["workspace"]["root"] = str(paths.workspace_root())
        else:
            _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        if args.source:
            if not args.source_id or not args.title:
                raise ValueError("When using --source, both --source-id and --title are required.")
            emit_cli_status("importing local PDG parsed source...")
            ensure_db()
            with connect() as conn:
                payload = import_pdg_source(
                    conn,
                    source_path=args.source,
                    source_id=args.source_id,
                    title=args.title,
                )
                conn.commit()
        else:
            if not args.edition:
                raise ValueError("Either --source or --edition must be provided for import-pdg.")
            emit_cli_status("preparing PDG archival import...")
            payload = import_pdg(
                config,
                edition=args.edition,
                collection_name=args.collection,
                pdf_path=args.pdf,
                download=args.download,
                progress=emit_cli_status,
            )
        emit_cli_status("PDG archival import finished.")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def cmd_query(args: argparse.Namespace) -> None:
    try:
        emit_cli_status("loading config...")
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        emit_cli_status("running retrieval...")
        payload = retrieve(
            config,
            query=args.query,
            limit=args.limit,
            target=args.target,
            collection_name=args.collection,
            model=args.model,
            progress=emit_cli_status,
        )
        emit_cli_status("retrieval finished.")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    try:
        emit_cli_status("loading config...")
        _, config = apply_runtime_config(config_path=args.config, workspace_root=args.workspace)
        emit_cli_status("starting retrieval + answer generation...")
        payload = ask(
            config,
            query=args.query,
            mode=args.mode,
            limit=args.limit,
            target=args.target,
            collection_name=args.collection,
            model=args.model,
            progress=emit_cli_status,
        )
        emit_cli_status("answer generation finished.")
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

        summary: dict[str, Any] = {
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
        metadata_summary: dict[str, Any] = {
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
        import_summary: dict[str, Any] = {
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
