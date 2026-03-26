from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from hep_rag_v2 import paths


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


API_JOB_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS api_jobs (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  request_json TEXT,
  result_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS api_job_events (
  job_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  level TEXT NOT NULL,
  message TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (job_id, seq),
  FOREIGN KEY (job_id) REFERENCES api_jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_api_jobs_status ON api_jobs(status);
CREATE INDEX IF NOT EXISTS idx_api_job_events_job ON api_job_events(job_id, seq);
"""


class BackgroundJobManager:
    def __init__(self, *, max_workers: int = 2, max_events: int = 1000) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="hep-rag-job")
        self._max_events = max(100, int(max_events))
        self._lock = Lock()
        self._pending_events: dict[str, list[dict[str, Any]]] = {}
        self._next_seq: dict[str, int] = {}
        ensure_api_job_db()

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
        job_id = str(job_id or uuid4().hex)
        created_at = _utcnow()
        with self._lock, _job_db_connect() as conn:
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
            self._pending_events[job_id] = []
            self._next_seq[job_id] = 1
        self._executor.submit(self._run_job, job_id, fn, args or (), kwargs or {})
        return self.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with _job_db_connect() as conn:
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
        snapshots = [_snapshot_row(row) for row in rows]
        with self._lock:
            return [self._overlay_pending_locked(snapshot) for snapshot in snapshots]

    def get(self, job_id: str) -> dict[str, Any]:
        with _job_db_connect() as conn:
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
        snapshot = _snapshot_row(row)
        with self._lock:
            return self._overlay_pending_locked(snapshot)

    def events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        with _job_db_connect() as conn:
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
        persisted = [
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
        with self._lock:
            pending = [
                dict(item)
                for item in self._pending_events.get(job_id, [])
                if int(item["seq"]) > int(after)
            ]
        merged = {int(item["seq"]): item for item in persisted}
        for item in pending:
            merged.setdefault(int(item["seq"]), item)
        return [merged[key] for key in sorted(merged)]

    def progress_callback(self, job_id: str) -> Callable[[str], None]:
        def _callback(message: str) -> None:
            try:
                self.append_event(job_id, message, event_type="progress")
            except Exception:
                return

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
        event = {
            "seq": 0,
            "type": str(event_type or "progress"),
            "level": str(level or "info"),
            "message": text,
            "payload": payload,
            "timestamp": _utcnow(),
        }
        with self._lock:
            event["seq"] = self._reserve_seq_locked(job_id)
            self._pending_events.setdefault(job_id, []).append(event)
            self._flush_pending_locked(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_job(self, job_id: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        started_at = _utcnow()
        with self._lock, _job_db_connect() as conn:
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
            with self._lock, _job_db_connect() as conn:
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
            with self._lock:
                self._flush_pending_locked(job_id)
            return

        finished_at = _utcnow()
        with self._lock, _job_db_connect() as conn:
            conn.execute(
                """
                UPDATE api_jobs
                SET status = ?, result_json = ?, error = NULL, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("succeeded", _json_dumps(result), finished_at, finished_at, job_id),
            )
        self.append_event(job_id, "job finished successfully", event_type="lifecycle")
        with self._lock:
            self._flush_pending_locked(job_id)

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

    def _overlay_pending_locked(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        pending = self._pending_events.get(snapshot["job_id"], [])
        if not pending:
            return snapshot
        merged = dict(snapshot)
        merged["event_count"] = int(snapshot["event_count"]) + len(pending)
        merged["last_event_seq"] = max(
            int(snapshot["last_event_seq"]),
            max(int(item["seq"]) for item in pending),
        )
        latest_timestamp = max(str(item["timestamp"]) for item in pending)
        if not merged.get("updated_at") or str(merged["updated_at"]) < latest_timestamp:
            merged["updated_at"] = latest_timestamp
        return merged

    def _reserve_seq_locked(self, job_id: str) -> int:
        next_seq = self._next_seq.get(job_id)
        if next_seq is None:
            with _job_db_connect() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM api_job_events WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
            next_seq = int(row["max_seq"] or 0) + 1 if row is not None else 1
        self._next_seq[job_id] = int(next_seq) + 1
        return int(next_seq)

    def _flush_pending_locked(self, job_id: str) -> None:
        pending = list(self._pending_events.get(job_id, []))
        if not pending:
            return
        try:
            with _job_db_connect(timeout_sec=0.05) as conn:
                for item in pending:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO api_job_events (
                          job_id, seq, event_type, level, message, payload_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job_id,
                            int(item["seq"]),
                            str(item["type"]),
                            str(item["level"]),
                            str(item["message"]),
                            _json_dumps(item.get("payload")),
                            str(item["timestamp"]),
                        ),
                    )
                conn.execute(
                    "UPDATE api_jobs SET updated_at = ? WHERE job_id = ?",
                    (str(pending[-1]["timestamp"]), job_id),
                )
                self._prune_events(conn, job_id=job_id)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                return
            raise
        self._pending_events[job_id] = []


def ensure_api_job_db() -> None:
    paths.DB_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(paths.API_DB_PATH)) as conn:
        conn.executescript(API_JOB_SCHEMA)
        conn.commit()


@contextmanager
def _job_db_connect(timeout_sec: float = 30.0):
    ensure_api_job_db()
    conn = sqlite3.connect(paths.API_DB_PATH, timeout=float(timeout_sec))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {max(1, int(float(timeout_sec) * 1000))}")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
