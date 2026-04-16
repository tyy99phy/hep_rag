from __future__ import annotations

import copy
import json
import re
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from hep_rag_v2.community import rebuild_community_summaries
from hep_rag_v2 import paths
from hep_rag_v2.config import runtime_collection_config
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.maintenance import clear_dirty_work_ids, mark_work_dirty
from hep_rag_v2.methods import build_method_objects
from hep_rag_v2.ontology import rebuild_ontology_summaries
from hep_rag_v2.physics import build_physics_substrate
from hep_rag_v2.results import build_result_objects
from hep_rag_v2.structure import build_work_structures
from hep_rag_v2.transfer import build_transfer_candidates
from hep_rag_v2.pdg import import_pdg_website_source, register_pdg_artifact
from hep_rag_v2.metadata import (
    canonical_identity,
    first_arxiv_id,
    first_doi,
    upsert_collection,
    upsert_work_from_hit,
)
from hep_rag_v2.providers.inspire import summarize_hit
from hep_rag_v2.providers.pdg import (
    normalize_pdg_artifact,
    normalize_pdg_sqlite_variant,
    resolve_pdg_references,
    stage_pdg_artifact,
)
from hep_rag_v2.records import infer_collection_name, paper_storage_stem, parsed_doc_dir, safe_stem

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
        return {
            "structure": None,
            "results": None,
            "methods": None,
            "transfer": None,
            "physics": None,
            "ontology": None,
            "community": None,
        }
    summaries = {
        "structure": build_work_structures(conn, work_ids=work_ids, collection=collection_name),
        "results": build_result_objects(conn, work_ids=work_ids, collection=collection_name),
        "methods": build_method_objects(conn, work_ids=work_ids, collection=collection_name),
        "transfer": build_transfer_candidates(conn, work_ids=work_ids, collection=collection_name),
        "physics": build_physics_substrate(conn, work_ids=work_ids, collection=collection_name),
        "ontology": {
            "collection": rebuild_ontology_summaries(conn, collection=collection_name),
            "all": rebuild_ontology_summaries(conn, collection=None),
        },
        "community": {
            "collection": rebuild_community_summaries(conn, collection=collection_name),
            "all": rebuild_community_summaries(conn, collection=None),
        },
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
            metadata_total = len(hits)
            if metadata_total >= 1000:
                metadata_report_every = 100
            elif metadata_total >= 200:
                metadata_report_every = 50
            else:
                metadata_report_every = 10
            for idx, hit in enumerate(hits, start=1):
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
                    if idx % metadata_report_every == 0 or idx == metadata_total:
                        _emit_progress(
                            progress,
                            "metadata ingest "
                            f"{idx}/{metadata_total} families complete; "
                            f"created={summary['metadata']['created']} "
                            f"updated={summary['metadata']['updated']} "
                            f"citations_written={summary['metadata']['citations_written']}",
                        )
                    continue
                touched_work_ids.add(int(work_id))

                if idx % metadata_report_every == 0 or idx == metadata_total:
                    _emit_progress(
                        progress,
                        "metadata ingest "
                        f"{idx}/{metadata_total} families complete; "
                        f"created={summary['metadata']['created']} "
                        f"updated={summary['metadata']['updated']} "
                        f"citations_written={summary['metadata']['citations_written']}",
                    )

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
    parser_name: str | None = None,
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
        "selection": {
            "limit": limit,
            "work_ids": list(work_ids or []),
            "parser_name": str(parser_name).strip() or None,
            "replace_existing": bool(replace_existing),
        },
        "candidates_seen": 0, "attempted": 0, "materialized": 0, "failed": 0, "skipped": 0, "items": [],
        "search": None, "vectors": None, "graph": None, "thinking": None, "maintenance": None,
    }

    with connect() as conn:
        collection_row = conn.execute("SELECT collection_id FROM collections WHERE name = ?", (collection_payload["name"],)).fetchone()
        collection_id = int(collection_row["collection_id"]) if collection_row is not None else upsert_collection(conn, collection_payload)
        candidates = _select_reparse_candidates(
            conn,
            collection_name=collection_payload["name"],
            limit=limit,
            work_ids=work_ids,
            parser_name=parser_name,
            replace_existing=replace_existing,
        )
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
    artifact: str | None = None,
    source_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    download: bool = False,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_db()
    resolved_collection = str(collection_name or (config.get("collection") or {}).get("name") or "pdg").strip() or "pdg"
    workspace = initialize_workspace(config, collection_name=resolved_collection)
    pdg_cfg = config.get("pdg") or {}
    resolved_artifact = normalize_pdg_artifact(str(artifact or pdg_cfg.get("default_artifact") or "full"))
    resolved_sqlite_variant = normalize_pdg_sqlite_variant(str(pdg_cfg.get("sqlite_variant") or "all"))
    references = resolve_pdg_references(
        edition=edition,
        artifact=resolved_artifact,
        sqlite_variant=resolved_sqlite_variant,
    )
    local_source = source_path or pdf_path
    if local_source is not None and len(references) != 1:
        raise ValueError("A local --source/--pdf path can only be used when importing a single PDG artifact.")

    _emit_progress(progress, f"preparing PDG import for artifact={resolved_artifact}...")
    summary = {
        "workspace": workspace,
        "collection": workspace["collection"],
        "edition": str(edition),
        "artifact": resolved_artifact,
        "references": references,
        "staged_artifacts": [],
        "website_import": None,
        "registered_primary_pdfs": [],
        "registered_embedded_pdfs": None,
    }
    with connect() as conn:
        collection_id = int(workspace["collection"]["collection_id"])
        for reference in references:
            artifact_kind = str(reference.get("artifact_kind") or "").strip()
            output_path = _pdg_artifact_output_path(reference=reference, collection_name=resolved_collection)
            _emit_progress(progress, f"staging PDG {artifact_kind} artifact {reference['file_name']}...")
            staged = stage_pdg_artifact(
                reference,
                output_path=output_path,
                source_path=local_source if len(references) == 1 else None,
                download=download,
                progress=_prefixed_progress(progress, f"pdg {artifact_kind}"),
            )
            registration = register_pdg_artifact(
                conn,
                source_id=str(reference["canonical_id"]),
                artifact_kind=artifact_kind,
                edition=str(reference.get("edition") or ""),
                title=str(reference.get("title") or ""),
                local_path=staged.get("path"),
                source_url=str(reference.get("download_url") or ""),
                file_name=str(reference.get("file_name") or output_path.name),
                metadata={
                    "state": staged.get("state"),
                    "landing_url": str(reference.get("landing_url") or ""),
                    "artifact_kind": artifact_kind,
                },
            )
            item = {
                "reference": reference,
                "artifact": staged,
                "registration": registration,
            }
            if artifact_kind == "website" and bool(staged.get("ok")):
                _emit_progress(progress, "importing PDG website corpus into pdg_sections...")
                website_import = import_pdg_website_source(
                    conn,
                    source_path=staged["path"],
                    source_id=str(reference["canonical_id"]),
                    title=str(reference["title"]),
                    replace=True,
                    max_capsule_chars=int(pdg_cfg.get("max_capsule_chars") or 1200),
                    progress=_prefixed_progress(progress, "pdg website"),
                )
                item["import"] = website_import
                summary["website_import"] = website_import
                if bool(pdg_cfg.get("register_embedded_pdfs", True)):
                    pdf_registration = _register_pdg_embedded_pdfs_from_website(
                        conn,
                        collection_id=collection_id,
                        collection_name=resolved_collection,
                        edition=str(reference.get("edition") or edition),
                        site_root=Path(str(website_import["import_manifest"]["site_root"])),
                        progress=progress,
                    )
                    item["embedded_pdfs"] = pdf_registration
                    summary["registered_embedded_pdfs"] = pdf_registration
            elif artifact_kind in {"book_pdf", "booklet_pdf"} and bool(staged.get("ok")):
                pdf_registration = _register_pdg_primary_pdf_artifact(
                    conn,
                    collection_id=collection_id,
                    collection_name=resolved_collection,
                    reference=reference,
                    source_path=Path(str(staged["path"])),
                    progress=progress,
                )
                item["primary_pdf"] = pdf_registration
                summary["registered_primary_pdfs"].append(pdf_registration)
            summary["staged_artifacts"].append(item)
        summary["snapshot"] = _snapshot(conn)
    return summary


def _pdg_artifact_output_path(*, reference: dict[str, Any], collection_name: str) -> Path:
    artifact_kind = str(reference.get("artifact_kind") or "").strip()
    file_name = str(reference.get("file_name") or "").strip() or str(reference.get("canonical_id") or "pdg-artifact")
    return paths.RAW_DIR / "pdg" / artifact_kind / file_name


def _register_pdg_primary_pdf_artifact(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    collection_name: str,
    reference: dict[str, Any],
    source_path: Path,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    artifact_kind = str(reference.get("artifact_kind") or "").strip() or "pdg_pdf"
    work_id = _upsert_archival_work(conn, collection_id=collection_id, reference=reference)
    stem = paper_storage_stem(conn, work_id)
    output_path = paths.PDF_DIR / collection_name / f"{stem}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_source = source_path.expanduser().resolve()
    if output_path.resolve() != resolved_source:
        shutil.copy2(resolved_source, output_path)
    document = _upsert_archival_document_record(
        conn,
        work_id=work_id,
        collection_name=collection_name,
        stem=stem,
        parser_name=f"pdg_{artifact_kind}",
        parser_version=str(reference.get("edition") or ""),
        parse_status="pdf_ready",
        parse_error=None,
    )
    _emit_progress(progress, f"registered PDG {artifact_kind} parse candidate: work_id={work_id} path={output_path.name}")
    return {
        "work_id": work_id,
        "canonical_id": str(reference.get("canonical_id") or ""),
        "title": str(reference.get("title") or ""),
        "artifact_kind": artifact_kind,
        "path": str(output_path),
        "document": document,
    }


def _register_pdg_embedded_pdfs_from_website(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    collection_name: str,
    edition: str,
    site_root: Path,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    pdf_candidates: list[Path] = []
    for pdf_path in sorted(site_root.rglob("*.pdf")):
        rel_path = pdf_path.relative_to(site_root)
        if not rel_path.parts or rel_path.parts[0] not in {"reviews", "tables", "listings"}:
            continue
        pdf_candidates.append(pdf_path)

    if not pdf_candidates:
        return {"registered": 0, "categories": {}, "sample": []}

    categories: dict[str, int] = {}
    sample: list[dict[str, Any]] = []
    _emit_progress(progress, f"registering {len(pdf_candidates)} embedded PDG PDFs for later parsing...")
    for index, pdf_path in enumerate(pdf_candidates, start=1):
        rel_path = pdf_path.relative_to(site_root)
        category = str(rel_path.parts[0])
        categories[category] = categories.get(category, 0) + 1
        reference = _pdg_embedded_pdf_reference(edition=edition, rel_path=rel_path)
        work_id = _upsert_archival_work(conn, collection_id=collection_id, reference=reference)
        stem = paper_storage_stem(conn, work_id)
        output_path = paths.PDF_DIR / collection_name / f"{stem}.pdf"
        legacy_output_path = _legacy_pdg_embedded_pdf_output_path(collection_name=collection_name, edition=edition, rel_path=rel_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            shutil.copy2(pdf_path, output_path)
        if legacy_output_path != output_path and legacy_output_path.exists():
            legacy_output_path.unlink(missing_ok=True)
        document = _upsert_archival_document_record(
            conn,
            work_id=work_id,
            collection_name=collection_name,
            stem=stem,
            parser_name="pdg_website_pdf",
            parser_version=str(edition),
            parse_status="pdf_ready",
            parse_error=None,
        )
        if len(sample) < 20:
            sample.append(
                {
                    "work_id": work_id,
                    "canonical_id": reference["canonical_id"],
                    "title": reference["title"],
                    "category": category,
                    "path": str(output_path),
                    "document": document,
                }
            )
        if index % 100 == 0 or index == len(pdf_candidates):
            _emit_progress(progress, f"registered embedded PDG PDFs: {index}/{len(pdf_candidates)}")
    return {
        "registered": len(pdf_candidates),
        "categories": categories,
        "sample": sample,
    }


def _pdg_embedded_pdf_reference(*, edition: str, rel_path: Path) -> dict[str, Any]:
    rel_posix = rel_path.as_posix()
    rel_without_suffix = rel_path.with_suffix("").as_posix()
    stem = safe_stem(rel_without_suffix)
    category = str(rel_path.parts[0]) if rel_path.parts else "pdg"
    title = _humanize_pdg_embedded_pdf_title(rel_path)
    return {
        "canonical_source": "pdg",
        "canonical_id": f"pdg-{edition}-{category}-{stem}",
        "title": title,
        "year": int(edition),
        "landing_url": f"https://pdg.lbl.gov/{edition}/{rel_posix}",
        "pdf_url": f"https://pdg.lbl.gov/{edition}/{rel_posix}",
        "raw_metadata_json": {
            "edition": edition,
            "category": category,
            "relative_path": rel_posix,
        },
    }


def _legacy_pdg_embedded_pdf_canonical_id(*, edition: str, rel_path: str, category: str) -> str:
    return f"pdg-{edition}-{category}-{safe_stem(rel_path)}"


def _legacy_pdg_embedded_pdf_output_path(*, collection_name: str, edition: str, rel_path: Path) -> Path:
    category = str(rel_path.parts[0]) if rel_path.parts else "pdg"
    legacy_canonical_id = _legacy_pdg_embedded_pdf_canonical_id(
        edition=edition,
        rel_path=rel_path.as_posix(),
        category=category,
    )
    return paths.PDF_DIR / collection_name / f"{safe_stem(legacy_canonical_id)}.pdf"


def _humanize_pdg_embedded_pdf_title(rel_path: Path) -> str:
    stem = rel_path.stem
    stem = re.sub(r"^rpp\d{4}-(rev|tab|sum|qtab|list)-", "", stem)
    stem = stem.replace("-", " ")
    title = " ".join(part for part in stem.split() if part)
    return title.title() or rel_path.name


def _upsert_archival_work(conn: sqlite3.Connection, *, collection_id: int, reference: dict[str, Any]) -> int:
    work_id = _find_existing_archival_work_id(conn, reference=reference)
    payload = (
        reference["canonical_source"],
        reference["canonical_id"],
        reference["title"],
        int(reference.get("year") or 0) or None,
        str(reference.get("landing_url") or "").strip() or None,
        str(reference.get("pdf_url") or "").strip() or None,
        json.dumps(reference, ensure_ascii=False),
    )
    if work_id is None:
        cur = conn.execute(
            "INSERT INTO works (canonical_source, canonical_id, title, year, primary_source_url, primary_pdf_url, raw_metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
        work_id = int(cur.lastrowid)
    else:
        conn.execute(
            """
            UPDATE works
            SET canonical_source = ?, canonical_id = ?, title = ?, year = ?, primary_source_url = ?, primary_pdf_url = ?, raw_metadata_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE work_id = ?
            """,
            (
                reference["canonical_source"],
                reference["canonical_id"],
                reference["title"],
                int(reference.get("year") or 0) or None,
                str(reference.get("landing_url") or "").strip() or None,
                str(reference.get("pdf_url") or "").strip() or None,
                json.dumps(reference, ensure_ascii=False),
                work_id,
            ),
        )
    canonical_id = str(reference["canonical_id"])
    if str(reference.get("canonical_source") or "") == "pdg":
        conn.execute("DELETE FROM work_ids WHERE work_id = ? AND id_type = 'pdg' AND id_value <> ?", (work_id, canonical_id))
        conn.execute(
            "INSERT OR IGNORE INTO work_ids (id_type, id_value, work_id, is_primary) VALUES (?, ?, ?, 1)",
            ("pdg", canonical_id, work_id),
        )
        conn.execute(
            "UPDATE work_ids SET is_primary = CASE WHEN id_value = ? THEN 1 ELSE 0 END WHERE work_id = ? AND id_type = 'pdg'",
            (canonical_id, work_id),
        )
    else:
        conn.execute("INSERT OR IGNORE INTO work_ids (id_type, id_value, work_id, is_primary) VALUES (?, ?, ?, 1)", ("pdg", canonical_id, work_id))
    conn.execute("INSERT OR IGNORE INTO collection_works (collection_id, work_id) VALUES (?, ?)", (collection_id, work_id))
    return work_id


def _find_existing_archival_work_id(conn: sqlite3.Connection, *, reference: dict[str, Any]) -> int | None:
    row = conn.execute(
        "SELECT work_id FROM works WHERE canonical_source = ? AND canonical_id = ?",
        (reference["canonical_source"], reference["canonical_id"]),
    ).fetchone()
    if row is not None:
        return int(row["work_id"])
    if str(reference.get("canonical_source") or "") != "pdg":
        return None

    metadata = reference.get("raw_metadata_json") or {}
    rel_path = str(metadata.get("relative_path") or "").strip()
    category = str(metadata.get("category") or "").strip()
    edition = str(metadata.get("edition") or reference.get("year") or "").strip()
    legacy_candidates: list[str] = []
    if rel_path and category and edition:
        legacy_candidates.append(_legacy_pdg_embedded_pdf_canonical_id(edition=edition, rel_path=rel_path, category=category))
    for legacy_id in legacy_candidates:
        row = conn.execute(
            "SELECT work_id FROM works WHERE canonical_source = 'pdg' AND canonical_id = ?",
            (legacy_id,),
        ).fetchone()
        if row is not None:
            return int(row["work_id"])

    if not rel_path:
        return None
    rows = conn.execute(
        "SELECT work_id, raw_metadata_json FROM works WHERE canonical_source = 'pdg'"
    ).fetchall()
    for candidate in rows:
        payload = json.loads(str(candidate["raw_metadata_json"] or "{}") or "{}")
        candidate_meta = payload.get("raw_metadata_json") or {}
        if str(candidate_meta.get("relative_path") or "").strip() == rel_path:
            return int(candidate["work_id"])
    return None


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


def _select_reparse_candidates(
    conn: sqlite3.Connection,
    *,
    collection_name: str,
    limit: int | None,
    work_ids: list[int] | None,
    parser_name: str | None,
    replace_existing: bool,
) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT w.work_id, w.title FROM works w JOIN collection_works cw ON cw.work_id = w.work_id JOIN collections c ON c.collection_id = cw.collection_id WHERE c.name = ? ORDER BY w.work_id", (collection_name,)).fetchall()
    requested = {int(item) for item in (work_ids or [])}
    parser_filter = str(parser_name or "").strip()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        work_id = int(row["work_id"])
        if requested and work_id not in requested:
            continue
        pdf_path = _pdf_path_for_work(conn, work_id=work_id, collection_name=collection_name)
        if not pdf_path.exists():
            continue
        document_row = _document_row(conn, work_id=work_id)
        if parser_filter and (document_row is None or str(document_row["parser_name"] or "") != parser_filter):
            continue
        manifest_path = Path(str(document_row["manifest_path"])).expanduser() if document_row and document_row["manifest_path"] else None
        document_materialized = bool(document_row and str(document_row["parse_status"] or "") == "materialized" and manifest_path is not None and manifest_path.exists())
        if not replace_existing and document_materialized:
            continue
        candidates.append(
            {
                "work_id": work_id,
                "title": str(row["title"]),
                "pdf_path": str(pdf_path),
                "parse_status": str(document_row["parse_status"]) if document_row and document_row["parse_status"] else None,
                "parser_name": str(document_row["parser_name"]) if document_row and document_row["parser_name"] else None,
            }
        )
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
