from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

RESULT_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("measurement", "measurement", re.compile(r"\b(measure(?:ment|d|s)?|determination|branching fraction|branching ratio|cross section)\b", re.IGNORECASE)),
    ("upper_limit", "upper limit", re.compile(r"\b(upper limit|95%\s*cl|limit at 95% cl|set limits?)\b", re.IGNORECASE)),
    ("significance", "significance", re.compile(r"\b(significance|significant excess|observed excess)\b", re.IGNORECASE)),
    ("exclusion", "exclusion", re.compile(r"\b(exclude(?:d|s|ion)|exclusion limit)\b", re.IGNORECASE)),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS result_objects (
  result_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  collection_id INTEGER,
  object_key TEXT,
  label TEXT NOT NULL,
  result_kind TEXT,
  summary_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'extracted',
  confidence REAL,
  signature_json TEXT NOT NULL DEFAULT '[]',
  evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL,
  UNIQUE(work_id, object_key)
);
CREATE TABLE IF NOT EXISTS result_values (
  result_value_id INTEGER PRIMARY KEY AUTOINCREMENT,
  result_object_id INTEGER NOT NULL,
  value_label TEXT NOT NULL,
  value_text TEXT,
  numeric_value REAL,
  unit_text TEXT,
  comparator TEXT,
  uncertainty_text TEXT,
  context_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS result_context (
  result_context_id INTEGER PRIMARY KEY AUTOINCREMENT,
  result_object_id INTEGER NOT NULL,
  section_hint TEXT,
  dataset_hint TEXT,
  selection_hint TEXT,
  raw_context_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_result_values_object ON result_values(result_object_id, value_label);
"""


def ensure_result_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def build_result_objects(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int] | None = None,
    collection: str | None = None,
) -> dict[str, Any]:
    ensure_result_schema(conn)
    selected_work_ids = _select_work_ids(conn, work_ids=work_ids, collection=collection)
    summary: dict[str, Any] = {"processed": 0, "ready": 0, "partial": 0, "failed": 0, "items": []}
    for work_id in selected_work_ids:
        try:
            work = _load_work_row(conn, work_id)
            if work is None:
                continue
            text_blob = _work_text_blob(conn, work_id=work_id, title=str(work["title"] or ""), abstract=str(work["abstract"] or ""))
            values = _extract_values(text_blob)
            has_structured_text = _has_structured_text(conn, work_id=work_id)
            if values:
                status = "ready"
                summary["ready"] += 1
            elif text_blob.strip():
                status = "partial"
                summary["partial"] += 1
            else:
                status = "failed"
                summary["failed"] += 1
            result_object_id = _upsert_result_object(
                conn,
                work_id=work_id,
                collection_id=work["collection_id"],
                status=status,
                summary_text=_build_summary(title=str(work["title"] or ""), values=values, has_structured_text=has_structured_text),
                values=values,
                label=str(work["title"] or "").strip() or f"work-{work_id}-result",
            )
            _replace_result_values(conn, result_object_id=result_object_id, values=values)
            _replace_result_context(
                conn,
                result_object_id=result_object_id,
                contexts=[
                    {"context_kind": "source", "content": "chunks" if has_structured_text else "metadata_only"},
                    {"context_kind": "title", "content": str(work["title"] or "")},
                ],
            )
            summary["processed"] += 1
            summary["items"].append({"work_id": work_id, "status": status, "value_count": len(values)})
        except Exception as exc:
            _upsert_result_object(
                conn,
                work_id=work_id,
                collection_id=None,
                status="failed",
                summary_text=None,
                values=[],
                label=f"work-{work_id}-result",
            )
            summary["processed"] += 1
            summary["failed"] += 1
            summary["items"].append({"work_id": work_id, "status": "failed", "error": str(exc)})
    return summary


def _select_work_ids(conn: sqlite3.Connection, *, work_ids: list[int] | None, collection: str | None) -> list[int]:
    explicit = sorted({int(work_id) for work_id in (work_ids or [])})
    if explicit:
        return explicit
    if collection:
        rows = conn.execute(
            """
            SELECT w.work_id
            FROM works w
            JOIN collection_works cw ON cw.work_id = w.work_id
            JOIN collections c ON c.collection_id = cw.collection_id
            WHERE c.name = ?
            ORDER BY w.work_id
            """,
            (collection,),
        ).fetchall()
        return [int(row["work_id"]) for row in rows]
    rows = conn.execute("SELECT work_id FROM works ORDER BY work_id").fetchall()
    return [int(row["work_id"]) for row in rows]


def _load_work_row(conn: sqlite3.Connection, work_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT w.work_id, w.title, w.abstract, cw.collection_id
        FROM works w
        LEFT JOIN collection_works cw ON cw.work_id = w.work_id
        WHERE w.work_id = ?
        ORDER BY cw.collection_id
        LIMIT 1
        """,
        (work_id,),
    ).fetchone()


def _work_text_blob(conn: sqlite3.Connection, *, work_id: int, title: str, abstract: str) -> str:
    parts = [title.strip(), abstract.strip()]
    rows = conn.execute(
        """
        SELECT COALESCE(clean_text, text, '') AS value
        FROM chunks
        WHERE work_id = ? AND is_retrievable = 1
        ORDER BY chunk_id
        LIMIT 12
        """,
        (work_id,),
    ).fetchall()
    parts.extend(str(row["value"] or "").strip() for row in rows)
    return "\n\n".join(part for part in parts if part)


def _has_structured_text(conn: sqlite3.Connection, *, work_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM chunks WHERE work_id = ? LIMIT 1", (work_id,)).fetchone()
    return row is not None


def _extract_values(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, label, pattern in RESULT_PATTERNS:
        match = pattern.search(text)
        if match is None or (kind, label) in seen:
            continue
        seen.add((kind, label))
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 120)
        values.append(
            {
                "value_kind": kind,
                "label": label,
                "evidence_text": " ".join(text[start:end].split()),
                "confidence": 0.65,
            }
        )
    return values


def _build_summary(*, title: str, values: list[dict[str, Any]], has_structured_text: bool) -> str:
    parts = [title.strip()]
    if values:
        parts.append("result signatures: " + ", ".join(item["label"] for item in values))
    parts.append(f"source={'chunks' if has_structured_text else 'metadata_only'}")
    return " | ".join(part for part in parts if part)


def _upsert_result_object(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_id: Any,
    status: str,
    summary_text: str | None,
    values: list[dict[str, Any]],
    label: str,
) -> int:
    signature_json = json.dumps(
        [{"value_kind": value["value_kind"], "label": value["label"]} for value in values],
        ensure_ascii=False,
    )
    evidence_json = json.dumps(
        [{"label": value["label"], "evidence_text": value["evidence_text"]} for value in values],
        ensure_ascii=False,
    )
    result_kind = values[0]["value_kind"] if values else None
    confidence = max((float(value.get("confidence") or 0.0) for value in values), default=None)
    conn.execute(
        """
        INSERT INTO result_objects (
          work_id, collection_id, object_key, label, result_kind, summary_text,
          status, confidence, signature_json, evidence_json
        )
        VALUES (?, ?, 'default', ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(work_id, object_key) DO UPDATE SET
          collection_id = excluded.collection_id,
          label = excluded.label,
          result_kind = excluded.result_kind,
          status = excluded.status,
          summary_text = excluded.summary_text,
          confidence = excluded.confidence,
          signature_json = excluded.signature_json,
          evidence_json = excluded.evidence_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            work_id,
            collection_id,
            label,
            result_kind,
            summary_text or "",
            status,
            confidence,
            signature_json,
            evidence_json,
        ),
    )
    row = conn.execute(
        "SELECT result_object_id FROM result_objects WHERE work_id = ? AND object_key = 'default'",
        (work_id,),
    ).fetchone()
    return int(row["result_object_id"])


def _replace_result_values(conn: sqlite3.Connection, *, result_object_id: int, values: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM result_values WHERE result_object_id = ?", (result_object_id,))
    for value in values:
        conn.execute(
            """
            INSERT INTO result_values (
              result_object_id, value_label, value_text, numeric_value, unit_text,
              comparator, uncertainty_text, context_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_object_id,
                value["label"],
                value["evidence_text"],
                None,
                None,
                None,
                None,
                json.dumps(
                    {
                        "value_kind": value["value_kind"],
                        "confidence": value["confidence"],
                    },
                    ensure_ascii=False,
                ),
            ),
        )


def _replace_result_context(conn: sqlite3.Connection, *, result_object_id: int, contexts: list[dict[str, str]]) -> None:
    conn.execute("DELETE FROM result_context WHERE result_object_id = ?", (result_object_id,))
    source = next((item["content"] for item in contexts if item.get("context_kind") == "source"), None)
    title = next((item["content"] for item in contexts if item.get("context_kind") == "title"), None)
    if source is None and title is None:
        return
    conn.execute(
        """
        INSERT INTO result_context (
          result_object_id, section_hint, dataset_hint, selection_hint, raw_context_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            result_object_id,
            source,
            None,
            title,
            json.dumps(contexts, ensure_ascii=False),
        ),
    )
