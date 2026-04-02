from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from typing import Any


DERIVED_LANES = ("search", "vectors", "graph")


def ensure_maintenance_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dirty_objects (
          dirty_id INTEGER PRIMARY KEY AUTOINCREMENT,
          lane TEXT NOT NULL,
          object_kind TEXT NOT NULL,
          object_id INTEGER NOT NULL,
          collection_id INTEGER,
          reason TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(lane, object_kind, object_id),
          FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS maintenance_jobs (
          job_id INTEGER PRIMARY KEY AUTOINCREMENT,
          lane TEXT NOT NULL,
          status TEXT NOT NULL,
          scope TEXT,
          collection_name TEXT,
          updated_since TEXT,
          details_json TEXT,
          requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
          started_at TEXT DEFAULT CURRENT_TIMESTAMP,
          finished_at TEXT,
          result_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_dirty_objects_lane ON dirty_objects(lane, object_kind, updated_at);
        CREATE INDEX IF NOT EXISTS idx_dirty_objects_collection ON dirty_objects(collection_id, lane);
        CREATE INDEX IF NOT EXISTS idx_maintenance_jobs_lane ON maintenance_jobs(lane, status, requested_at);
        """
    )


def mark_work_dirty(
    conn: sqlite3.Connection,
    *,
    work_ids: Iterable[int],
    lanes: Iterable[str] = DERIVED_LANES,
    collection_id: int | None = None,
    reason: str | None = None,
) -> dict[str, int]:
    ensure_maintenance_schema(conn)
    unique_work_ids = sorted({int(work_id) for work_id in work_ids})
    summary: dict[str, int] = {}
    for lane in lanes:
        normalized_lane = str(lane).strip()
        if not normalized_lane:
            continue
        inserted = 0
        for work_id in unique_work_ids:
            conn.execute(
                """
                INSERT INTO dirty_objects (lane, object_kind, object_id, collection_id, reason)
                VALUES (?, 'work', ?, ?, ?)
                ON CONFLICT(lane, object_kind, object_id)
                DO UPDATE SET
                  collection_id = excluded.collection_id,
                  reason = excluded.reason,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (normalized_lane, work_id, collection_id, reason),
            )
            inserted += 1
        summary[normalized_lane] = inserted
    return summary


def dirty_counts(
    conn: sqlite3.Connection,
    *,
    collection: str | None = None,
) -> dict[str, int]:
    ensure_maintenance_schema(conn)
    params: list[Any] = []
    collection_sql = ""
    if collection is not None:
        collection_sql = """
            AND collection_id = (
              SELECT collection_id FROM collections WHERE name = ?
            )
        """
        params.append(collection)

    rows = conn.execute(
        f"""
        SELECT lane, COUNT(*) AS n
        FROM dirty_objects
        WHERE 1=1
        {collection_sql}
        GROUP BY lane
        ORDER BY lane
        """,
        params,
    ).fetchall()
    counts = {lane: 0 for lane in DERIVED_LANES}
    counts.update({str(row["lane"]): int(row["n"]) for row in rows})
    return counts


def select_dirty_work_ids(
    conn: sqlite3.Connection,
    *,
    lane: str,
    collection: str | None = None,
    updated_since: str | None = None,
    work_ids: Iterable[int] | None = None,
) -> list[int]:
    ensure_maintenance_schema(conn)
    explicit_ids = sorted({int(work_id) for work_id in (work_ids or [])})
    params: list[Any] = [lane]
    where: list[str] = ["lane = ?", "object_kind = 'work'"]
    if collection is not None:
        where.append(
            """
            collection_id = (
              SELECT collection_id FROM collections WHERE name = ?
            )
            """
        )
        params.append(collection)
    if updated_since is not None:
        where.append("updated_at >= ?")
        params.append(updated_since)
    if explicit_ids:
        placeholders = ", ".join("?" for _ in explicit_ids)
        where.append(f"object_id IN ({placeholders})")
        params.extend(explicit_ids)

    rows = conn.execute(
        f"""
        SELECT object_id
        FROM dirty_objects
        WHERE {' AND '.join(where)}
        ORDER BY object_id
        """,
        params,
    ).fetchall()
    return [int(row["object_id"]) for row in rows]


def clear_dirty_work_ids(
    conn: sqlite3.Connection,
    *,
    lane: str,
    collection: str | None = None,
    work_ids: Iterable[int] | None = None,
) -> int:
    ensure_maintenance_schema(conn)
    params: list[Any] = [lane]
    where: list[str] = ["lane = ?", "object_kind = 'work'"]
    if collection is not None:
        where.append(
            """
            collection_id = (
              SELECT collection_id FROM collections WHERE name = ?
            )
            """
        )
        params.append(collection)
    selected_work_ids = sorted({int(work_id) for work_id in (work_ids or [])})
    if selected_work_ids:
        placeholders = ", ".join("?" for _ in selected_work_ids)
        where.append(f"object_id IN ({placeholders})")
        params.extend(selected_work_ids)

    cursor = conn.execute(
        f"DELETE FROM dirty_objects WHERE {' AND '.join(where)}",
        params,
    )
    return int(cursor.rowcount or 0)


def start_maintenance_job(
    conn: sqlite3.Connection,
    *,
    lane: str,
    scope: str | None = None,
    collection_name: str | None = None,
    updated_since: str | None = None,
    details: dict[str, Any] | None = None,
) -> int:
    ensure_maintenance_schema(conn)
    cur = conn.execute(
        """
        INSERT INTO maintenance_jobs (
          lane, status, scope, collection_name, updated_since, details_json
        ) VALUES (?, 'running', ?, ?, ?, ?)
        """,
        (
            lane,
            scope,
            collection_name,
            updated_since,
            json.dumps(details or {}, ensure_ascii=False),
        ),
    )
    return int(cur.lastrowid)


def finish_maintenance_job(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    status: str,
    result: dict[str, Any] | None = None,
) -> None:
    ensure_maintenance_schema(conn)
    conn.execute(
        """
        UPDATE maintenance_jobs
        SET status = ?, finished_at = CURRENT_TIMESTAMP, result_json = ?
        WHERE job_id = ?
        """,
        (status, json.dumps(result or {}, ensure_ascii=False), job_id),
    )
