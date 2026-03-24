from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from hep_rag_v2 import paths


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "paper"


def parsed_doc_dir(collection: str, stem: str) -> Path:
    return paths.PARSED_DIR / collection / stem


def resolve_work_row(
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
            raise ValueError(f"Unknown work_id: {work_id}")
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
            raise ValueError(f"Unknown work identity: {id_type}:{id_value}")
        return row

    raise ValueError("Specify either work_id or both id_type and id_value.")


def infer_collection_name(conn: sqlite3.Connection, work_id: int) -> str | None:
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


def paper_storage_stem(conn: sqlite3.Connection, work_id: int) -> str:
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
            return safe_stem(value)

    row = conn.execute(
        "SELECT canonical_id FROM works WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if row is not None and str(row["canonical_id"]).strip():
        return safe_stem(str(row["canonical_id"]).strip())
    return str(work_id)
