from __future__ import annotations

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from hep_rag_v2 import paths
from hep_rag_v2.config import runtime_collection_config
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.maintenance import clear_dirty_work_ids, mark_work_dirty
from hep_rag_v2.methods import build_method_objects
from hep_rag_v2.results import build_result_objects
from hep_rag_v2.structure import build_work_structures
from hep_rag_v2.transfer import build_transfer_candidates
from hep_rag_v2.metadata import (
    canonical_identity,
    first_arxiv_id,
    first_doi,
    upsert_collection,
    upsert_work_from_hit,
)
from hep_rag_v2.providers.inspire import summarize_hit
from hep_rag_v2.providers.pdg import resolve_pdg_reference, stage_pdg_pdf
from hep_rag_v2.records import infer_collection_name, paper_storage_stem, parsed_doc_dir

from hep_rag_v2.online_search import (  # noqa: F401 — re-export
    _search_online_hits,
    _emit_progress,
    _resolve_parallelism,
)
from hep_rag_v2.download import (  # noqa: F401 — re-export
    _prepare_download,
    _execute_download,
)
from hep_rag_v2.parse import (  # noqa: F401 — re-export
    _parse_with_mineru,
    _materialize_existing_manifest,
    _mark_document_parse_failed,
    _upsert_document_parse_record,
    _document_row,
)
from hep_rag_v2.rag import (  # noqa: F401 — re-export
    retrieve,
    ask,
    _build_answer_messages,
    _build_llm_client,
    _build_mineru_client,
    _supporting_chunks,
    _hydrate_works_from_chunks,
)

ProgressCallback = Callable[[str], None] | None


def _truncate_progress_text(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _paper_progress_label(
    *,
    index: int | None,
    total: int | None,
    work_id: int,
    title: str | None,
) -> str:
    parts: list[str] = []
    if index is not None and total is not None:
        parts.append(f"{index}/{total}")
    parts.append(f"work_id={work_id}")
    title_text = _truncate_progress_text(title, limit=72)
    if title_text:
        parts.append(f'title="{title_text}"')
    return " ".join(parts)


def _prefixed_progress(progress: ProgressCallback, prefix: str) -> ProgressCallback:
    if progress is None:
        return None
    label = str(prefix or "").strip()
    if not label:
        return progress

    def _callback(message: str) -> None:
        text = str(message or "").strip()
        if text:
            progress(f"{label}: {text}")

    return _callback


def _queue_derived_maintenance(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    collection_name: str,
    work_ids: list[int],
    reason: str,
) -> dict[str, Any]:
    queued = 0
    lanes = ("search", "vectors", "graph", "structure", "results", "methods", "transfer")
    for lane in lanes:
        mark_work_dirty(
            conn,
            work_ids=work_ids,
            lanes=[lane],
            collection_id=collection_id,
            reason=reason,
        )
        queued += len(work_ids)
    from hep_rag_v2.maintenance import dirty_counts
    dirty = dirty_counts(conn)
    return {
        "queued": queued,
        "dirty": dirty,
    }


def _sync_thinking_engine_extractions(
    conn: sqlite3.Connection,
    *,
    collection_name: str,
    work_ids: list[int],
) -> dict[str, Any]:
    if not work_ids:
        return {"structure": None, "results": None, "methods": None, "transfer": None}
    summaries = {
        "structure": build_work_structures(conn, work_ids=work_ids, collection=collection_name),
        "results": build_result_objects(conn, work_ids=work_ids, collection=collection_name),
        "methods": build_method_objects(conn, work_ids=work_ids, collection=collection_name),
        "transfer": build_transfer_candidates(conn, work_ids=work_ids, collection=collection_name),
    }
    clear_dirty_work_ids(conn, lane="structure", collection=collection_name, work_ids=work_ids)
    clear_dirty_work_ids(conn, lane="results", collection=collection_name, work_ids=work_ids)
    clear_dirty_work_ids(conn, lane="methods", collection=collection_name, work_ids=work_ids)
    clear_dirty_work_ids(conn, lane="transfer", collection=collection_name, work_ids=work_ids)
    return summaries


def _annotate_hits_with_local_status(
    conn: sqlite3.Connection,
    *,
    hits: list[dict[str, Any]],
    collection_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotated: list[dict[str, Any]] = []
    summary = {
        "known_works": 0,
        "pdf_cached": 0,
        "document_materialized": 0,
        "needs_mineru": 0,
    }
    for hit in hits:
        local = _hit_local_status(conn, hit=hit, collection_name=collection_name)
        item = copy.deepcopy(hit)
        item["local_status"] = local
        family_members = []
        for member in item.get("_family_members") or []:
            member_payload = copy.deepcopy(member)
            member_payload["local_status"] = _hit_local_status(conn, hit=member, collection_name=collection_name)
            family_members.append(member_payload)
        if family_members:
            item["_family_members"] = family_members
        annotated.append(item)
        if local["known_work"]:
            summary["known_works"] += 1
        if local["pdf_exists"]:
            summary["pdf_cached"] += 1
        if local["document_materialized"]:
            summary["document_materialized"] += 1
        if local["needs_mineru"]:
            summary["needs_mineru"] += 1
    return annotated, summary


def _hit_local_status(
    conn: sqlite3.Connection,
    *,
    hit: dict[str, Any],
    collection_name: str,
) -> dict[str, Any]:
    work_id = _find_work_id_for_hit(conn, hit)
    if work_id is None:
        return {
            "known_work": False,
            "work_id": None,
            "collection": collection_name,
            "pdf_exists": False,
            "pdf_path": None,
            "document_exists": False,
            "document_materialized": False,
            "parse_status": None,
            "parse_error": None,
            "manifest_exists": False,
            "manifest_path": None,
            "needs_mineru": False,
        }

    effective_collection = infer_collection_name(conn, work_id) or collection_name
    pdf_path = _pdf_path_for_work(conn, work_id=work_id, collection_name=effective_collection)
    document_row = _document_row(conn, work_id=work_id)
    manifest_path = Path(str(document_row["manifest_path"])).expanduser() if document_row and document_row["manifest_path"] else None
    manifest_exists = bool(manifest_path and manifest_path.exists())
    parse_status = str(document_row["parse_status"]) if document_row and document_row["parse_status"] else None
    document_materialized = bool(document_row and parse_status == "materialized" and manifest_exists)
    pdf_exists = pdf_path.exists()
    return {
        "known_work": True,
        "work_id": work_id,
        "collection": effective_collection,
        "pdf_exists": pdf_exists,
        "pdf_path": str(pdf_path) if pdf_exists else None,
        "document_exists": document_row is not None,
        "document_materialized": document_materialized,
        "parse_status": parse_status,
        "parse_error": str(document_row["parse_error"]) if document_row and document_row["parse_error"] else None,
        "manifest_exists": manifest_exists,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "needs_mineru": pdf_exists and not document_materialized,
    }


def initialize_workspace(config: dict[str, Any], *, collection_name: str | None = None) -> dict[str, Any]:
    ensure_db()
    collection_payload = runtime_collection_config(config, name=collection_name)
    collection_config_path = paths.COLLECTIONS_DIR / f"{collection_payload['name']}.json"
    collection_config_path.write_text(
        json.dumps(collection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with connect() as conn:
        collection_id = upsert_collection(conn, collection_payload)
        snapshot = _snapshot(conn)
    return {
        "workspace_root": str(paths.workspace_root()),
        "db_path": str(paths.DB_PATH),
        "collection": {
            "collection_id": collection_id,
            "name": collection_payload["name"],
            "label": collection_payload["label"],
            "config_path": str(collection_config_path),
        },
        "snapshot": snapshot,
    }


def fetch_online_candidates(
    config: dict[str, Any],
    *,
    query: str,
    limit: int,
    max_parallelism: int | None = None,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_db()
    collection_name = str((config.get("collection") or {}).get("name") or "default")
    hits, search_plan = _search_online_hits(
        config,
        query=query,
        limit=limit,
        max_parallelism=max_parallelism,
        progress=progress,
    )
    with connect() as conn:
        hits, local_summary = _annotate_hits_with_local_status(
            conn,
            hits=hits,
            collection_name=collection_name,
        )
    _emit_progress(progress, f"found {len(hits)} candidate papers.")
    raw_record_count = sum(1 + len(hit.get("_family_members") or []) for hit in hits)
    return {
        "query": query,
        "search_plan": search_plan,
        "local_summary": local_summary,
        "effective_count": len(hits),
        "raw_record_count": raw_record_count,
        "results": [summarize_hit(hit) for hit in hits],
    }


def ingest_online(
    config: dict[str, Any],
    *,
    query: str,
    limit: int,
    collection_name: str | None = None,
    max_parallelism: int | None = None,
    download_limit: int | None = None,
    parse_limit: int | None = None,
    replace_existing: bool = False,
    skip_parse: bool = False,
    skip_index: bool = False,
    skip_graph: bool = False,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_db()
    collection_payload = runtime_collection_config(config, name=collection_name)
    download_cfg = config.get("download") or {}

    hits, search_plan = _search_online_hits(
        config,
        query=query,
        limit=limit,
        max_parallelism=max_parallelism,
        progress=progress,
    )

    download_cap = max(0, download_limit if download_limit is not None else len(hits))
    parse_cap = 0 if skip_parse else max(0, parse_limit if parse_limit is not None else download_cap)
    max_workers = _resolve_parallelism(
        requested=max_parallelism,
        configured=download_cfg.get("max_download_workers"),
        fallback=4,
    )

    summary: dict[str, Any] = {
        "query": query,
        "workspace_root": str(paths.workspace_root()),
        "collection": collection_payload["name"],
        "search_plan": search_plan,
        "local_summary": {"known_works": 0, "pdf_cached": 0, "document_materialized": 0, "needs_mineru": 0},
        "metadata": {"hits_seen": len(hits), "raw_records_seen": sum(1 + len(hit.get("_family_members") or []) for hit in hits), "created": 0, "updated": 0, "citations_written": 0},
        "downloads": {"max_parallelism": max_workers, "attempted": 0, "ok": 0, "failed": 0, "items": []},
        "mineru": {"enabled": bool((config.get("mineru") or {}).get("enabled")), "attempted": 0, "materialized": 0, "failed": 0, "items": []},
        "search": None,
        "vectors": None,
        "graph": None,
        "thinking": None,
        "maintenance": None,
    }

    mineru_client = _build_mineru_client(config)

    with connect() as conn:
        hits, local_summary = _annotate_hits_with_local_status(conn, hits=hits, collection_name=collection_payload["name"])
        summary["local_summary"] = local_summary
        collection_id = upsert_collection(conn, collection_payload)
        run_id = _start_ingest_run(conn, collection_id=collection_id, source="online_search", query_json=json.dumps({"query": query, "limit": limit}, ensure_ascii=False), limit_requested=limit)

        try:
            _emit_progress(progress, f"ingesting metadata for {len(hits)} paper families...")
            download_tasks: list[dict[str, Any]] = []
            touched_work_ids: set[int] = set()
            for hit in hits:
                family_hits = [hit, *(hit.get("_family_members") or [])]
                for family_hit in family_hits:
                    stats = upsert_work_from_hit(conn, collection_id=collection_id, hit=family_hit)
                    summary["metadata"]["created"] += int(stats["created"])
                    summary["metadata"]["updated"] += int(stats["updated"])
                    summary["metadata"]["citations_written"] += int(stats["citations_written"])
                    family_work_id = _find_work_id_for_hit(conn, family_hit)
                    if family_work_id is not None:
                        touched_work_ids.add(int(family_work_id))

                work_id = _find_work_id_for_hit(conn, hit)
                if work_id is None:
                    continue
                touched_work_ids.add(int(work_id))

                if len(download_tasks) >= download_cap:
                    continue
                download_tasks.append(_prepare_download(conn, hit=hit, work_id=work_id, collection_name=collection_payload["name"]))

            dl_timeout = int(download_cfg.get("timeout_sec") or 120)
            dl_retries = int(download_cfg.get("retries") or 3)
            dl_verify_ssl = bool(download_cfg.get("verify_ssl", True))

            download_results: list[dict[str, Any]] = []
            if download_tasks:
                _emit_progress(progress, f"downloading PDFs for {len(download_tasks)} papers...")
                with ThreadPoolExecutor(max_workers=min(max_workers, len(download_tasks))) as pool:
                    futures = {pool.submit(_execute_download, task=task, timeout=dl_timeout, retries=dl_retries, verify_ssl=dl_verify_ssl): task for task in download_tasks}
                    for idx, future in enumerate(as_completed(futures), start=1):
                        task = futures[future]
                        pdf_info = future.result()
                        download_results.append(pdf_info)
                        label = _paper_progress_label(index=idx, total=len(download_tasks), work_id=int(task["work_id"]), title=str(pdf_info.get("title") or task.get("title") or ""))
                        if pdf_info["ok"]:
                            _emit_progress(progress, f"downloaded PDF {label} source={pdf_info.get('source') or 'downloaded'}")
                        else:
                            errors = pdf_info.get("errors") or []
                            _emit_progress(progress, f"failed PDF download {label}: {_truncate_progress_text(errors[0] if errors else 'no PDF candidate succeeded', limit=140)}")

            summary["downloads"]["attempted"] = len(download_results)
            for pdf_info in download_results:
                summary["downloads"]["items"].append(pdf_info)
                if pdf_info["ok"]:
                    summary["downloads"]["ok"] += 1
                else:
                    summary["downloads"]["failed"] += 1
            if download_tasks:
                _emit_progress(progress, f"PDF download phase finished: {summary['downloads']['ok']}/{summary['downloads']['attempted']} succeeded, {summary['downloads']['failed']} failed.")

            parse_queue = ([pdf_info for pdf_info in download_results if pdf_info["ok"]][:max(0, parse_cap)] if mineru_client is not None else [])
            if parse_cap > 0 and mineru_client is None:
                _emit_progress(progress, "MinerU is disabled; skipping PDF parsing.")
            elif parse_cap > 0 and parse_queue:
                _emit_progress(progress, f"parsing {len(parse_queue)} PDFs with MinerU...")
            for idx, pdf_info in enumerate(parse_queue, start=1):
                summary["mineru"]["attempted"] += 1
                label = _paper_progress_label(index=idx, total=len(parse_queue), work_id=int(pdf_info["work_id"]), title=str(pdf_info.get("title") or ""))
                try:
                    _emit_progress(progress, f"starting MinerU parse for {label}...")
                    parsed = _parse_with_mineru(conn, config=config, client=mineru_client, work_id=pdf_info["work_id"], pdf_path=Path(pdf_info["path"]), collection_name=collection_payload["name"], replace_existing=replace_existing, progress=_prefixed_progress(progress, f"MinerU {label}"))
                    summary["mineru"]["materialized"] += 1
                    summary["mineru"]["items"].append(parsed)
                    touched_work_ids.add(int(pdf_info["work_id"]))
                    _emit_progress(progress, f"finished MinerU parse for {label} state={parsed.get('state') or 'materialized'}")
                except Exception as exc:
                    summary["mineru"]["failed"] += 1
                    _mark_document_parse_failed(conn, work_id=pdf_info["work_id"], collection_name=collection_payload["name"], error=str(exc))
                    summary["mineru"]["items"].append({"ok": False, "work_id": pdf_info["work_id"], "title": _work_title(conn, pdf_info["work_id"]), "error": str(exc)})
                    _emit_progress(progress, f"MinerU parse failed for {label}: {_truncate_progress_text(exc, limit=160)}")
            if parse_cap > 0 and mineru_client is not None:
                _emit_progress(progress, f"MinerU phase finished: {summary['mineru']['materialized']}/{summary['mineru']['attempted']} succeeded, {summary['mineru']['failed']} failed.")

            touched_ids = sorted(touched_work_ids)
            summary["maintenance"] = _queue_derived_maintenance(conn, collection_id=collection_id, collection_name=collection_payload["name"], work_ids=touched_ids, reason="ingest_online")
            summary["thinking"] = _sync_thinking_engine_extractions(conn, collection_name=collection_payload["name"], work_ids=touched_ids)
            if summary["maintenance"]["queued"]:
                _emit_progress(progress, f"queued maintenance instead of auto rebuild: {summary['maintenance']['dirty']}")
            if summary["thinking"]:
                _emit_progress(progress, f"updated thinking-engine substrate for {len(touched_ids)} works.")
            if skip_index or skip_graph:
                _emit_progress(progress, "legacy skip-index/skip-graph flags are now no-op because ingest no longer auto rebuilds derived lanes.")

            _finish_ingest_run(conn, run_id=run_id, status="finished", processed_hits=len(hits), works_created=int(summary["metadata"]["created"]), works_updated=int(summary["metadata"]["updated"]), citations_written=int(summary["metadata"]["citations_written"]))
        except Exception:
            _finish_ingest_run(conn, run_id=run_id, status="failed", processed_hits=len(hits), works_created=int(summary["metadata"]["created"]), works_updated=int(summary["metadata"]["updated"]), citations_written=int(summary["metadata"]["citations_written"]))
            raise

        summary["snapshot"] = _snapshot(conn)
    return summary


def reparse_cached_pdfs(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    limit: int | None = None,
    work_ids: list[int] | None = None,
    replace_existing: bool = False,
    skip_index: bool = False,
    skip_graph: bool = False,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_db()
    collection_payload = runtime_collection_config(config, name=collection_name)
    mineru_client = _build_mineru_client(config)
    if mineru_client is None:
        raise ValueError("MinerU is disabled in config. Set mineru.enabled=true before reparsing PDFs.")

    summary: dict[str, Any] = {
        "workspace_root": str(paths.workspace_root()),
        "collection": collection_payload["name"],
        "selection": {"limit": limit, "work_ids": list(work_ids or []), "replace_existing": bool(replace_existing)},
        "candidates_seen": 0, "attempted": 0, "materialized": 0, "failed": 0, "skipped": 0, "items": [],
        "search": None, "vectors": None, "graph": None, "thinking": None, "maintenance": None,
    }

    with connect() as conn:
        collection_row = conn.execute("SELECT collection_id FROM collections WHERE name = ?", (collection_payload["name"],)).fetchone()
        collection_id = int(collection_row["collection_id"]) if collection_row is not None else upsert_collection(conn, collection_payload)
        candidates = _select_reparse_candidates(conn, collection_name=collection_payload["name"], limit=limit, work_ids=work_ids, replace_existing=replace_existing)
        summary["candidates_seen"] = len(candidates)
        touched_work_ids: set[int] = set()
        if candidates:
            _emit_progress(progress, f"reparsing {len(candidates)} cached PDFs with MinerU...")
        for idx, candidate in enumerate(candidates, start=1):
            summary["attempted"] += 1
            touched_work_ids.add(int(candidate["work_id"]))
            label = _paper_progress_label(index=idx, total=len(candidates), work_id=int(candidate["work_id"]), title=str(candidate["title"] or ""))
            try:
                _emit_progress(progress, f"starting MinerU reparse for {label}...")
                parsed = _parse_with_mineru(conn, config=config, client=mineru_client, work_id=int(candidate["work_id"]), pdf_path=Path(str(candidate["pdf_path"])), collection_name=collection_payload["name"], replace_existing=replace_existing, progress=_prefixed_progress(progress, f"MinerU {label}"))
                summary["materialized"] += 1
                summary["items"].append(parsed)
                _emit_progress(progress, f"finished MinerU reparse for {label} state={parsed.get('state') or 'materialized'}")
            except Exception as exc:
                summary["failed"] += 1
                _mark_document_parse_failed(conn, work_id=int(candidate["work_id"]), collection_name=collection_payload["name"], error=str(exc))
                summary["items"].append({"ok": False, "work_id": int(candidate["work_id"]), "title": str(candidate["title"]), "error": str(exc)})
                _emit_progress(progress, f"MinerU reparse failed for {label}: {_truncate_progress_text(exc, limit=160)}")
        if candidates:
            _emit_progress(progress, f"MinerU reparse phase finished: {summary['materialized']}/{summary['attempted']} succeeded, {summary['failed']} failed.")

        touched_ids = sorted(touched_work_ids)
        summary["maintenance"] = _queue_derived_maintenance(conn, collection_id=collection_id, collection_name=collection_payload["name"], work_ids=touched_ids, reason="reparse_cached_pdfs")
        summary["thinking"] = _sync_thinking_engine_extractions(conn, collection_name=collection_payload["name"], work_ids=touched_ids)
        if summary["maintenance"]["queued"]:
            _emit_progress(progress, f"queued maintenance instead of auto rebuild: {summary['maintenance']['dirty']}")
        if summary["thinking"]:
            _emit_progress(progress, f"updated thinking-engine substrate for {len(touched_ids)} works.")
        if skip_index or skip_graph:
            _emit_progress(progress, "legacy skip-index/skip-graph flags are now no-op because reparse no longer auto rebuilds derived lanes.")

        summary["snapshot"] = _snapshot(conn)
    return summary


def import_pdg(
    config: dict[str, Any],
    *,
    edition: str | int,
    collection_name: str | None = None,
    pdf_path: str | Path | None = None,
    download: bool = False,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_db()
    resolved_collection = str(collection_name or (config.get("collection") or {}).get("name") or "pdg").strip() or "pdg"
    workspace = initialize_workspace(config, collection_name=resolved_collection)
    reference = resolve_pdg_reference(edition=edition)
    _emit_progress(progress, f"preparing PDG archival import for {reference['canonical_id']}...")

    with connect() as conn:
        collection_payload = runtime_collection_config(config, name=resolved_collection)
        collection_id = upsert_collection(conn, collection_payload)
        work_id = _upsert_archival_work(conn, collection_id=collection_id, reference=reference)
        stem = paper_storage_stem(conn, work_id)
        output_path = paths.PDF_DIR / resolved_collection / f"{stem}.pdf"
        pdf = stage_pdg_pdf(reference, output_path=output_path, pdf_path=pdf_path, download=download)
        document = _upsert_archival_document_record(
            conn, work_id=work_id, collection_name=resolved_collection, stem=stem, parser_name="pdg", parser_version=str(reference["edition"]),
            parse_status="pdf_ready" if bool(pdf.get("ok")) else "awaiting_pdf",
            parse_error=None if bool(pdf.get("ok")) else "PDG PDF not staged yet",
        )

    return {
        "workspace": workspace,
        "collection": workspace["collection"],
        "reference": reference,
        "work": {"work_id": work_id, "canonical_source": reference["canonical_source"], "canonical_id": reference["canonical_id"], "title": reference["title"]},
        "pdf": pdf,
        "document": document,
    }


def _upsert_archival_work(conn: sqlite3.Connection, *, collection_id: int, reference: dict[str, Any]) -> int:
    row = conn.execute("SELECT work_id FROM works WHERE canonical_source = ? AND canonical_id = ?", (reference["canonical_source"], reference["canonical_id"])).fetchone()
    payload = (reference["canonical_source"], reference["canonical_id"], reference["title"], int(reference.get("year") or 0) or None, str(reference.get("landing_url") or "").strip() or None, str(reference.get("pdf_url") or "").strip() or None, json.dumps(reference, ensure_ascii=False))
    if row is None:
        cur = conn.execute("INSERT INTO works (canonical_source, canonical_id, title, year, primary_source_url, primary_pdf_url, raw_metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)", payload)
        work_id = int(cur.lastrowid)
    else:
        work_id = int(row["work_id"])
        conn.execute("UPDATE works SET title = ?, year = ?, primary_source_url = ?, primary_pdf_url = ?, raw_metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE work_id = ?", (reference["title"], int(reference.get("year") or 0) or None, str(reference.get("landing_url") or "").strip() or None, str(reference.get("pdf_url") or "").strip() or None, json.dumps(reference, ensure_ascii=False), work_id))
    conn.execute("INSERT OR IGNORE INTO work_ids (id_type, id_value, work_id, is_primary) VALUES (?, ?, ?, 1)", ("pdg", str(reference["canonical_id"]), work_id))
    conn.execute("INSERT OR IGNORE INTO collection_works (collection_id, work_id) VALUES (?, ?)", (collection_id, work_id))
    return work_id


def _upsert_archival_document_record(conn: sqlite3.Connection, *, work_id: int, collection_name: str, stem: str, parser_name: str, parser_version: str, parse_status: str, parse_error: str | None) -> dict[str, Any]:
    parsed_dir = parsed_doc_dir(collection_name, stem)
    manifest_path = parsed_dir / "manifest.json"
    conn.execute(
        """
        INSERT INTO documents (work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path, parse_error, last_parse_attempt_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(work_id) DO UPDATE SET
          parser_name = excluded.parser_name, parser_version = excluded.parser_version, parse_status = excluded.parse_status,
          parsed_dir = excluded.parsed_dir, manifest_path = excluded.manifest_path, parse_error = excluded.parse_error,
          last_parse_attempt_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        """,
        (work_id, parser_name, parser_version, parse_status, str(parsed_dir), str(manifest_path), parse_error),
    )
    return {"parser_name": parser_name, "parser_version": parser_version, "parse_status": parse_status, "parsed_dir": str(parsed_dir), "manifest_path": str(manifest_path), "parse_error": parse_error}


def _find_work_id_for_hit(conn: sqlite3.Connection, hit: dict[str, Any]) -> int | None:
    metadata = hit.get("metadata") or {}
    for id_type, id_value in (
        ("inspire", str(metadata.get("control_number") or "").strip() or None),
        ("arxiv", first_arxiv_id(metadata)),
        ("doi", first_doi(metadata)),
    ):
        if not id_value:
            continue
        row = conn.execute("SELECT work_id FROM work_ids WHERE id_type = ? AND id_value = ?", (id_type, str(id_value).strip())).fetchone()
        if row is not None:
            return int(row["work_id"])
    source, external_id = canonical_identity(metadata)
    row = conn.execute("SELECT work_id FROM works WHERE canonical_source = ? AND canonical_id = ?", (source, external_id)).fetchone()
    return int(row["work_id"]) if row is not None else None


def _pdf_path_for_work(conn: sqlite3.Connection, *, work_id: int, collection_name: str) -> Path:
    stem = paper_storage_stem(conn, work_id)
    return paths.PDF_DIR / collection_name / f"{stem}.pdf"


def _work_title(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute("SELECT title FROM works WHERE work_id = ?", (work_id,)).fetchone()
    return str(row["title"]) if row is not None else None


def _select_reparse_candidates(conn: sqlite3.Connection, *, collection_name: str, limit: int | None, work_ids: list[int] | None, replace_existing: bool) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT w.work_id, w.title FROM works w JOIN collection_works cw ON cw.work_id = w.work_id JOIN collections c ON c.collection_id = cw.collection_id WHERE c.name = ? ORDER BY w.work_id", (collection_name,)).fetchall()
    requested = {int(item) for item in (work_ids or [])}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        work_id = int(row["work_id"])
        if requested and work_id not in requested:
            continue
        pdf_path = _pdf_path_for_work(conn, work_id=work_id, collection_name=collection_name)
        if not pdf_path.exists():
            continue
        document_row = _document_row(conn, work_id=work_id)
        manifest_path = Path(str(document_row["manifest_path"])).expanduser() if document_row and document_row["manifest_path"] else None
        document_materialized = bool(document_row and str(document_row["parse_status"] or "") == "materialized" and manifest_path is not None and manifest_path.exists())
        if not replace_existing and document_materialized:
            continue
        candidates.append({"work_id": work_id, "title": str(row["title"]), "pdf_path": str(pdf_path), "parse_status": str(document_row["parse_status"]) if document_row and document_row["parse_status"] else None})
        if limit is not None and len(candidates) >= max(1, int(limit)):
            break
    return candidates


def _snapshot(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute("SELECT (SELECT COUNT(*) FROM collections) AS collections, (SELECT COUNT(*) FROM works) AS works, (SELECT COUNT(*) FROM documents) AS documents, (SELECT COUNT(*) FROM chunks) AS chunks, (SELECT COUNT(*) FROM citations) AS citations, (SELECT COUNT(*) FROM similarity_edges) AS similarity_edges").fetchone()
    return {key: int(row[key]) for key in row.keys()} if row is not None else {}


def _start_ingest_run(conn: sqlite3.Connection, *, collection_id: int, source: str, query_json: str, limit_requested: int) -> int:
    cur = conn.execute("INSERT INTO ingest_runs (collection_id, source, status, query_json, page_size, limit_requested, raw_dir) VALUES (?, ?, 'running', ?, NULL, ?, ?)", (collection_id, source, query_json, limit_requested, str(paths.RAW_DIR)))
    return int(cur.lastrowid)


def _finish_ingest_run(conn: sqlite3.Connection, *, run_id: int, status: str, processed_hits: int, works_created: int, works_updated: int, citations_written: int) -> None:
    conn.execute("UPDATE ingest_runs SET status = ?, processed_hits = ?, works_created = ?, works_updated = ?, citations_written = ?, finished_at = CURRENT_TIMESTAMP WHERE run_id = ?", (status, processed_hits, works_created, works_updated, citations_written, run_id))
