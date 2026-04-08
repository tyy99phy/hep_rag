from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from hep_rag_v2 import paths
from hep_rag_v2.providers.inspire import content_addressed_name, download_pdf_candidates, list_pdf_candidates
from hep_rag_v2.records import paper_storage_stem


def _prepare_download(
    conn: sqlite3.Connection,
    *,
    hit: dict[str, Any],
    work_id: int,
    collection_name: str,
) -> dict[str, Any]:
    stem = paper_storage_stem(conn, work_id)
    output_path = paths.PDF_DIR / collection_name / f"{stem}.pdf"
    candidates = list_pdf_candidates(
        hit,
        resolve_arxiv_from_doi=True,
        timeout=30,
        retries=3,
    )
    return {
        "work_id": work_id,
        "title": _work_title(conn, work_id),
        "output_path": str(output_path),
        "candidates": candidates,
        "content_addressed_name": content_addressed_name(hit),
    }


def _execute_download(
    *,
    task: dict[str, Any],
    timeout: int,
    retries: int,
    verify_ssl: bool,
) -> dict[str, Any]:
    output_path = Path(task["output_path"])
    if output_path.exists():
        return {
            "ok": True,
            "work_id": task["work_id"],
            "title": task["title"],
            "path": str(output_path),
            "source": "cached",
        }

    result = download_pdf_candidates(
        task["candidates"],
        output_path=output_path,
        timeout=timeout,
        retries=retries,
        verify_ssl=verify_ssl,
    )
    result.update(
        {
            "work_id": task["work_id"],
            "title": task["title"],
            "candidate_name": task["content_addressed_name"],
        }
    )
    return result


def _work_title(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute("SELECT title FROM works WHERE work_id = ?", (work_id,)).fetchone()
    return str(row["title"]) if row is not None else None
