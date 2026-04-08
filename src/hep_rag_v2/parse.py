from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from hep_rag_v2 import paths
from hep_rag_v2.fulltext import import_mineru_source, materialize_mineru_document
from hep_rag_v2.records import paper_storage_stem, parsed_doc_dir

ProgressCallback = Callable[[str], None] | None


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is not None:
        progress(message)


def _parse_with_mineru(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    client: Any,
    work_id: int,
    pdf_path: Path,
    collection_name: str,
    replace_existing: bool,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    stem = paper_storage_stem(conn, work_id)
    dest_dir = parsed_doc_dir(collection_name, stem)
    manifest_path = dest_dir / "manifest.json"
    document_row = _document_row(conn, work_id=work_id)
    if manifest_path.exists() and not replace_existing:
        if document_row is not None and str(document_row["parse_status"] or "") == "materialized":
            _emit_progress(progress, "using cached parsed document; skipping MinerU submission.")
            return {
                "ok": True,
                "work_id": work_id,
                "title": _work_title(conn, work_id),
                "state": "cached",
                "parsed_dir": str(dest_dir),
            }
        _emit_progress(progress, "found cached manifest; materializing into the database...")
        return _materialize_existing_manifest(
            conn,
            config=config,
            work_id=work_id,
            manifest_path=manifest_path,
            replace_existing=document_row is not None,
        )

    raw_zip_dir = paths.RAW_DIR / "mineru" / collection_name
    raw_zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_zip_dir / f"{stem}.zip"
    _upsert_document_parse_record(
        conn,
        work_id=work_id,
        parse_status="submitted",
        parsed_dir=dest_dir,
        manifest_path=manifest_path,
        parse_error=None,
    )
    _emit_progress(progress, f"submitting {pdf_path.name} to MinerU...")
    task = client.submit_local_pdf(pdf_path, data_id=stem, progress=progress)
    client.download_result_zip(task, output_path=zip_path, progress=progress)
    _emit_progress(progress, f"importing parsed bundle into {dest_dir.name}...")
    import_mineru_source(source_path=zip_path, dest_dir=dest_dir, replace=replace_existing)

    _emit_progress(progress, "materializing parsed manifest into SQLite...")
    result = _materialize_existing_manifest(
        conn,
        config=config,
        work_id=work_id,
        manifest_path=manifest_path,
        replace_existing=True,
    )
    result["state"] = task.state
    result["zip_path"] = str(zip_path)
    _emit_progress(progress, "materialized parsed document.")
    return result


def _materialize_existing_manifest(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    work_id: int,
    manifest_path: Path,
    replace_existing: bool,
) -> dict[str, Any]:
    ingest_cfg = config.get("ingest") or {}
    summary = materialize_mineru_document(
        conn,
        work_id=work_id,
        manifest_path=manifest_path,
        replace=replace_existing,
        chunk_size=int(ingest_cfg.get("chunk_size") or 2400),
        overlap_blocks=int(ingest_cfg.get("overlap_blocks") or 1),
        section_parent_char_limit=int(ingest_cfg.get("section_parent_char_limit") or 12000),
    )
    _upsert_document_parse_record(
        conn,
        work_id=work_id,
        parse_status="materialized",
        parsed_dir=manifest_path.parent,
        manifest_path=manifest_path,
        parse_error=None,
    )
    return {
        "ok": True,
        "work_id": work_id,
        "title": _work_title(conn, work_id),
        "state": "materialized",
        "parsed_dir": str(manifest_path.parent),
        "document": summary,
    }


def _document_row(conn: sqlite3.Connection, *, work_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT document_id, parse_status, parsed_dir, manifest_path, parse_error
        FROM documents
        WHERE work_id = ?
        ORDER BY document_id DESC
        LIMIT 1
        """,
        (work_id,),
    ).fetchone()


def _upsert_document_parse_record(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    parse_status: str,
    parsed_dir: Path | None,
    manifest_path: Path | None,
    parse_error: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO documents (
          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path, parse_error, last_parse_attempt_at, updated_at
        ) VALUES (?, 'mineru', 'v2-contract', ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(work_id) DO UPDATE SET
          parser_name = excluded.parser_name,
          parser_version = excluded.parser_version,
          parse_status = excluded.parse_status,
          parsed_dir = excluded.parsed_dir,
          manifest_path = excluded.manifest_path,
          parse_error = excluded.parse_error,
          last_parse_attempt_at = CURRENT_TIMESTAMP,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            work_id,
            parse_status,
            str(parsed_dir) if parsed_dir is not None else None,
            str(manifest_path) if manifest_path is not None else None,
            str(parse_error) if parse_error else None,
        ),
    )


def _mark_document_parse_failed(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_name: str,
    error: str,
) -> None:
    dest_dir = parsed_doc_dir(collection_name, paper_storage_stem(conn, work_id))
    manifest_path = dest_dir / "manifest.json"
    _upsert_document_parse_record(
        conn,
        work_id=work_id,
        parse_status="failed",
        parsed_dir=dest_dir,
        manifest_path=manifest_path if manifest_path.exists() else None,
        parse_error=error,
    )


def _work_title(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute("SELECT title FROM works WHERE work_id = ?", (work_id,)).fetchone()
    return str(row["title"]) if row is not None else None
