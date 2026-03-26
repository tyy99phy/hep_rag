from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from hep_rag_v2.db import connect, ensure_db


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundJobManager:
    def __init__(self, *, max_workers: int = 2, max_events: int = 1000) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="hep-rag-job")
        self._max_events = max(100, int(max_events))
        self._lock = Lock()
        ensure_db()

    def submit(
        self,
        *,
        kind: str,
        fn: Callable[..., Any],
        job_id: str | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ensure_db()
        job_id = str(job_id or uuid4().hex)
        created_at = _utcnow()
        with self._lock, connect() as conn:
            conn.execute(
                """
                INSERT INTO api_jobs (
                  job_id, kind, status, request_json, result_json, error,
                  created_at, updated_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, NULL, NULL)
                """,
                (
                    job_id,
                    str(kind or "job"),
                    "queued",
                    _json_dumps(request_payload),
                    created_at,
                    created_at,
                ),
            )
            conn.execute("DELETE FROM api_job_events WHERE job_id = ?", (job_id,))
        self._executor.submit(self._run_job, job_id, fn, args or (), kwargs or {})
        return self.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        ensure_db()
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  j.*,
                  COALESCE((SELECT COUNT(*) FROM api_job_events e WHERE e.job_id = j.job_id), 0) AS event_count,
                  COALESCE((SELECT MAX(seq) FROM api_job_events e WHERE e.job_id = j.job_id), 0) AS last_event_seq
                FROM api_jobs j
                ORDER BY j.created_at DESC, j.job_id DESC
                """
            ).fetchall()
        return [_snapshot_row(row) for row in rows]

    def get(self, job_id: str) -> dict[str, Any]:
        ensure_db()
        with connect() as conn:
            row = conn.execute(
                """
                SELECT
                  j.*,
                  COALESCE((SELECT COUNT(*) FROM api_job_events e WHERE e.job_id = j.job_id), 0) AS event_count,
                  COALESCE((SELECT MAX(seq) FROM api_job_events e WHERE e.job_id = j.job_id), 0) AS last_event_seq
                FROM api_jobs j
                WHERE j.job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _snapshot_row(row)

    def events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        ensure_db()
        with connect() as conn:
            exists = conn.execute("SELECT 1 FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if exists is None:
                raise KeyError(job_id)
            rows = conn.execute(
                """
                SELECT seq, event_type, level, message, payload_json, created_at
                FROM api_job_events
                WHERE job_id = ?
                  AND seq > ?
                ORDER BY seq
                """,
                (job_id, int(after)),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "type": row["event_type"],
                "level": row["level"],
                "message": row["message"],
                "payload": _json_loads(row["payload_json"]),
                "timestamp": row["created_at"],
            }
            for row in rows
        ]

    def progress_callback(self, job_id: str) -> Callable[[str], None]:
        def _callback(message: str) -> None:
            self.append_event(job_id, message, event_type="progress")

        return _callback

    def append_event(
        self,
        job_id: str,
        message: str,
        *,
        event_type: str = "progress",
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        text = str(message or "").strip()
        if not text:
            return
        ensure_db()
        with self._lock, connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM api_job_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return
            seq = int(row["max_seq"] or 0) + 1
            timestamp = _utcnow()
            conn.execute(
                """
                INSERT INTO api_job_events (
                  job_id, seq, event_type, level, message, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    seq,
                    str(event_type or "progress"),
                    str(level or "info"),
                    text,
                    _json_dumps(payload),
                    timestamp,
                ),
            )
            conn.execute(
                "UPDATE api_jobs SET updated_at = ? WHERE job_id = ?",
                (timestamp, job_id),
            )
            self._prune_events(conn, job_id=job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_job(self, job_id: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        started_at = _utcnow()
        with self._lock, connect() as conn:
            conn.execute(
                """
                UPDATE api_jobs
                SET status = ?, started_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("running", started_at, started_at, job_id),
            )
        self.append_event(job_id, "job started", event_type="lifecycle")
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            finished_at = _utcnow()
            with self._lock, connect() as conn:
                conn.execute(
                    """
                    UPDATE api_jobs
                    SET status = ?, error = ?, finished_at = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    ("failed", str(exc), finished_at, finished_at, job_id),
                )
            self.append_event(job_id, str(exc), event_type="error", level="error")
            self.append_event(job_id, "job finished with failure", event_type="lifecycle", level="error")
            return

        finished_at = _utcnow()
        with self._lock, connect() as conn:
            conn.execute(
                """
                UPDATE api_jobs
                SET status = ?, result_json = ?, error = NULL, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("succeeded", _json_dumps(result), finished_at, finished_at, job_id),
            )
        self.append_event(job_id, "job finished successfully", event_type="lifecycle")

    def _prune_events(self, conn, *, job_id: str) -> None:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM api_job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        count = int(row["n"] or 0) if row is not None else 0
        overflow = count - self._max_events
        if overflow <= 0:
            return
        conn.execute(
            """
            DELETE FROM api_job_events
            WHERE job_id = ?
              AND seq IN (
                SELECT seq
                FROM api_job_events
                WHERE job_id = ?
                ORDER BY seq ASC
                LIMIT ?
              )
            """,
            (job_id, job_id, overflow),
        )


def _snapshot_row(row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "kind": row["kind"],
        "status": row["status"],
        "request": _json_loads(row["request_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
        "result": _json_loads(row["result_json"]),
        "event_count": int(row["event_count"] or 0),
        "last_event_seq": int(row["last_event_seq"] or 0),
    }


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value
