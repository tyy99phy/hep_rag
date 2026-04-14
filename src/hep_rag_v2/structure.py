from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

REVIEW_DOC_TYPES = {"review", "review article"}

RESULT_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("measurement", "measurement", re.compile(r"\b(measure(?:ment|d|s)?|determination|branching fraction|branching ratio|cross section)\b", re.IGNORECASE)),
    ("upper_limit", "upper limit", re.compile(r"\b(upper limit|95%\s*cl|limit at 95% cl|set limits?)\b", re.IGNORECASE)),
    ("significance", "significance", re.compile(r"\b(significance|significant excess|observed excess)\b", re.IGNORECASE)),
    ("exclusion", "exclusion", re.compile(r"\b(exclude(?:d|s|ion)|exclusion limit)\b", re.IGNORECASE)),
)

METHOD_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("statistical_fit", "profile likelihood", re.compile(r"\b(profile likelihood|likelihood fit|template fit|maximum likelihood)\b", re.IGNORECASE)),
    ("multivariate", "multivariate", re.compile(r"\b(boosted decision tree|\bbdt\b|neural network|deep neural network|multivariate)\b", re.IGNORECASE)),
    ("background_estimation", "background estimation", re.compile(r"\b(background estimation|data[- ]driven|control region)\b", re.IGNORECASE)),
    ("reconstruction", "reconstruction", re.compile(r"\b(reconstruction|unfolding|matrix element|tagger)\b", re.IGNORECASE)),
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS work_capsules (
  capsule_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL UNIQUE,
  collection_id INTEGER,
  profile TEXT NOT NULL DEFAULT 'default',
  builder TEXT,
  is_review INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  capsule_text TEXT NOT NULL,
  result_signature_json TEXT NOT NULL DEFAULT '[]',
  method_signature_json TEXT NOT NULL DEFAULT '[]',
  anomaly_code TEXT,
  anomaly_detail TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_work_capsules_collection ON work_capsules(collection_id, status);
"""


def ensure_structure_schema(conn: sqlite3.Connection) -> None:
    for statement in SCHEMA.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)


def build_work_structures(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int] | None = None,
    collection: str | None = None,
    profile: str = "default",
    builder: str = "heuristic-v1",
    require_default_signatures: bool = True,
) -> dict[str, Any]:
    ensure_structure_schema(conn)
    selected_work_ids = _select_work_ids(conn, work_ids=work_ids, collection=collection)
    summary: dict[str, Any] = {
        "processed": 0,
        "ready": 0,
        "partial": 0,
        "needs_review": 0,
        "failed": 0,
        "items": [],
    }
    for work_id in selected_work_ids:
        work_row = conn.execute(
            """
            SELECT w.work_id, w.title, w.abstract, w.raw_metadata_json, cw.collection_id
            FROM works w
            LEFT JOIN collection_works cw ON cw.work_id = w.work_id
            WHERE w.work_id = ?
            ORDER BY cw.collection_id
            LIMIT 1
            """,
            (work_id,),
        ).fetchone()
        if work_row is None:
            continue
        metadata = _load_json(work_row["raw_metadata_json"])
        is_review = _is_review_work(metadata)
        text_blob = _work_text_blob(conn, work_id=work_id, title=str(work_row["title"] or ""), abstract=str(work_row["abstract"] or ""))
        result_signatures = _extract_signatures(text_blob, RESULT_PATTERNS)
        method_signatures = _extract_signatures(text_blob, METHOD_PATTERNS)
        missing: list[str] = []
        if require_default_signatures and not is_review:
            if not result_signatures:
                missing.append("result")
            if not method_signatures:
                missing.append("method")

        if is_review:
            status = "partial"
            anomaly_code = None
            anomaly_detail = None
            summary["partial"] += 1
        elif missing:
            status = "needs_review"
            anomaly_code = "missing_required_signatures"
            anomaly_detail = f"missing: {', '.join(missing)}"
            summary["needs_review"] += 1
        else:
            status = "ready"
            anomaly_code = None
            anomaly_detail = None
            summary["ready"] += 1

        capsule_text = _build_capsule_text(
            title=str(work_row["title"] or ""),
            abstract=str(work_row["abstract"] or ""),
            result_signatures=result_signatures,
            method_signatures=method_signatures,
            is_review=is_review,
        )
        _upsert_work_capsule(
            conn,
            work_id=int(work_id),
            collection_id=work_row["collection_id"],
            profile=profile,
            builder=builder,
            is_review=is_review,
            status=status,
            capsule_text=capsule_text,
            result_signatures=result_signatures,
            method_signatures=method_signatures,
            anomaly_code=anomaly_code,
            anomaly_detail=anomaly_detail,
        )
        summary["processed"] += 1
        summary["items"].append(
            {
                "work_id": int(work_id),
                "status": status,
                "is_review": is_review,
                "result_signatures": len(result_signatures),
                "method_signatures": len(method_signatures),
                "anomaly_code": anomaly_code,
            }
        )
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


def _load_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_review_work(metadata: dict[str, Any]) -> bool:
    doc_types = {str(item).strip().casefold() for item in (metadata.get("document_type") or []) if str(item).strip()}
    return bool(doc_types & REVIEW_DOC_TYPES)


def _work_text_blob(conn: sqlite3.Connection, *, work_id: int, title: str, abstract: str) -> str:
    parts = [title.strip(), abstract.strip()]
    rows = conn.execute(
        """
        SELECT COALESCE(clean_text, text, '') AS value
        FROM chunks
        WHERE work_id = ?
          AND is_retrievable = 1
        ORDER BY chunk_id
        LIMIT 12
        """,
        (work_id,),
    ).fetchall()
    parts.extend(str(row["value"] or "").strip() for row in rows)
    return "\n\n".join(part for part in parts if part)


def _extract_signatures(text: str, patterns: tuple[tuple[str, str, re.Pattern[str]], ...]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, label, pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        key = (kind, label)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 120)
        evidence = " ".join(text[start:end].split())
        out.append(
            {
                "kind": kind,
                "label": label,
                "evidence": evidence,
            }
        )
    return out


def _build_capsule_text(
    *,
    title: str,
    abstract: str,
    result_signatures: list[dict[str, str]],
    method_signatures: list[dict[str, str]],
    is_review: bool,
) -> str:
    parts = [title.strip()]
    if abstract.strip():
        parts.append(abstract.strip())
    if result_signatures:
        parts.append("结果签名: " + ", ".join(item["label"] for item in result_signatures))
    if method_signatures:
        parts.append("方法签名: " + ", ".join(item["label"] for item in method_signatures))
    if is_review:
        parts.append("文档类型: review")
    return "\n".join(part for part in parts if part)


def _upsert_work_capsule(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_id: Any,
    profile: str,
    builder: str,
    is_review: bool,
    status: str,
    capsule_text: str,
    result_signatures: list[dict[str, str]],
    method_signatures: list[dict[str, str]],
    anomaly_code: str | None,
    anomaly_detail: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO work_capsules (
          work_id, collection_id, profile, builder, is_review, status, capsule_text,
          result_signature_json, method_signature_json, anomaly_code, anomaly_detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(work_id) DO UPDATE SET
          collection_id = excluded.collection_id,
          profile = excluded.profile,
          builder = excluded.builder,
          is_review = excluded.is_review,
          status = excluded.status,
          capsule_text = excluded.capsule_text,
          result_signature_json = excluded.result_signature_json,
          method_signature_json = excluded.method_signature_json,
          anomaly_code = excluded.anomaly_code,
          anomaly_detail = excluded.anomaly_detail,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            work_id,
            collection_id,
            profile,
            builder,
            int(is_review),
            status,
            capsule_text,
            json.dumps(result_signatures, ensure_ascii=False),
            json.dumps(method_signatures, ensure_ascii=False),
            anomaly_code,
            anomaly_detail,
        ),
    )
