from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from hep_rag_v2.object_contracts import ALLOWED_STATUSES

METHOD_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("statistical_fit", "profile likelihood", re.compile(r"\b(profile likelihood|likelihood fit|template fit|maximum likelihood)\b", re.IGNORECASE)),
    ("multivariate", "multivariate", re.compile(r"\b(boosted decision tree|\bbdt\b|neural network|deep neural network|multivariate)\b", re.IGNORECASE)),
    ("background_estimation", "background estimation", re.compile(r"\b(background estimation|data[- ]driven|control region)\b", re.IGNORECASE)),
    ("reconstruction", "reconstruction", re.compile(r"\b(reconstruction|unfolding|matrix element|tagger)\b", re.IGNORECASE)),
)
OBJECT_CONTRACT_VERSION = "v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS method_objects (
  method_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  collection_id INTEGER,
  object_key TEXT,
  name TEXT NOT NULL,
  method_family TEXT,
  summary_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'extracted',
  signature_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL,
  UNIQUE(work_id, object_key)
);
CREATE TABLE IF NOT EXISTS method_signatures (
  method_signature_id INTEGER PRIMARY KEY AUTOINCREMENT,
  method_object_id INTEGER NOT NULL,
  signature_kind TEXT NOT NULL,
  signature_text TEXT NOT NULL,
  normalized_text TEXT,
  raw_signature_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (method_object_id) REFERENCES method_objects(method_object_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS method_application_links (
  method_application_link_id INTEGER PRIMARY KEY AUTOINCREMENT,
  method_object_id INTEGER NOT NULL,
  result_object_id INTEGER,
  target_work_id INTEGER,
  relation_kind TEXT NOT NULL DEFAULT 'applied_to',
  confidence REAL,
  notes TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (method_object_id) REFERENCES method_objects(method_object_id) ON DELETE CASCADE,
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE SET NULL,
  FOREIGN KEY (target_work_id) REFERENCES works(work_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_method_signatures_object ON method_signatures(method_object_id, signature_kind);
"""


def ensure_method_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def build_method_objects(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int] | None = None,
    collection: str | None = None,
) -> dict[str, Any]:
    ensure_method_schema(conn)
    selected_work_ids = _select_work_ids(conn, work_ids=work_ids, collection=collection)
    summary: dict[str, Any] = {"processed": 0, "ready": 0, "partial": 0, "needs_review": 0, "failed": 0, "items": []}
    for work_id in selected_work_ids:
        try:
            work = _load_work_row(conn, work_id)
            if work is None:
                continue
            text_blob = _work_text_blob(conn, work_id=work_id, title=str(work["title"] or ""), abstract=str(work["abstract"] or ""))
            structure_snapshot = _load_structure_method_snapshot(conn, work_id=work_id)
            use_structure_snapshot = bool(structure_snapshot and structure_snapshot["use_structure_payload"])
            signatures = structure_snapshot["signatures"] if use_structure_snapshot else _extract_signatures(text_blob)
            content_source = _resolve_content_source(conn, work_id=work_id, use_structure_snapshot=use_structure_snapshot)
            if structure_snapshot is not None and structure_snapshot["status"] in {"needs_review", "failed"}:
                status = str(structure_snapshot["status"])
                summary[status] += 1
            elif signatures:
                status = "ready"
                summary["ready"] += 1
            elif text_blob.strip():
                status = "partial"
                summary["partial"] += 1
            else:
                status = "failed"
                summary["failed"] += 1
            method_object_id = _upsert_method_object(
                conn,
                work_id=work_id,
                collection_id=work["collection_id"],
                status=status,
                summary_text=_build_summary(title=str(work["title"] or ""), signatures=signatures, content_source=content_source),
                signatures=signatures,
                name=str(work["title"] or "").strip() or f"work-{work_id}-method",
                source_kind="extraction",
                content_source=content_source,
            )
            _replace_method_signatures(conn, method_object_id=method_object_id, signatures=signatures)
            _replace_method_applications(conn, method_object_id=method_object_id, work_id=work_id, signatures=signatures)
            summary["processed"] += 1
            summary["items"].append({"work_id": work_id, "status": status, "signature_count": len(signatures)})
        except Exception as exc:
            _upsert_method_object(
                conn,
                work_id=work_id,
                collection_id=None,
                status="failed",
                summary_text=None,
                signatures=[],
                name=f"work-{work_id}-method",
                source_kind="extraction",
                content_source="metadata_only",
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


def _load_structure_method_snapshot(conn: sqlite3.Connection, *, work_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT status, builder, method_signature_json FROM work_capsules WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(str(row["method_signature_json"] or "[]"))
    except Exception:
        payload = []
    signatures = [
        {
            "method_kind": str(item.get("kind") or "unknown"),
            "label": str(item.get("label") or item.get("kind") or "unknown"),
            "evidence_text": str(item.get("evidence") or item.get("summary_text") or item.get("label") or ""),
            "confidence": float(item.get("confidence") or 0.65),
        }
        for item in payload
        if isinstance(item, dict)
    ]
    return {
        "status": _normalize_structure_status(row["status"]),
        "signatures": signatures,
        "use_structure_payload": bool(signatures),
    }


def _normalize_structure_status(value: Any) -> str:
    status = str(value or "").strip()
    if not status:
        return "failed"
    return status if status in ALLOWED_STATUSES else "failed"


def _resolve_content_source(conn: sqlite3.Connection, *, work_id: int, use_structure_snapshot: bool) -> str:
    if use_structure_snapshot:
        return "structure_capsule"
    return "chunks" if _has_structured_text(conn, work_id=work_id) else "metadata_only"


def _extract_signatures(text: str) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, label, pattern in METHOD_PATTERNS:
        match = pattern.search(text)
        if match is None or (kind, label) in seen:
            continue
        seen.add((kind, label))
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 120)
        signatures.append(
            {
                "method_kind": kind,
                "label": label,
                "evidence_text": " ".join(text[start:end].split()),
                "confidence": 0.65,
            }
        )
    return signatures


def _build_summary(*, title: str, signatures: list[dict[str, Any]], content_source: str) -> str:
    parts = [title.strip()]
    if signatures:
        parts.append("method signatures: " + ", ".join(item["label"] for item in signatures))
    parts.append(f"source={content_source}")
    return " | ".join(part for part in parts if part)


def _build_evidence_bundle(
    *,
    work_id: int,
    label: str,
    evidence_text: str,
    source_kind: str,
    status: str,
    confidence: float | None,
    ref_suffix: str,
    content_source: str,
) -> dict[str, Any]:
    return {
        "contract_version": OBJECT_CONTRACT_VERSION,
        "object_type": "evidence_bundle",
        "object_id": f"evidence_bundle:{work_id}:{ref_suffix}",
        "label": label,
        "source_kind": source_kind,
        "status": status,
        "source_refs": [f"work:{work_id}", f"evidence:{ref_suffix}", f"content_source:{content_source}"],
        "derivation": "extracted",
        "content_source": content_source,
        "confidence": confidence,
        "items": [
            {
                "kind": "text_span",
                "ref": f"work:{work_id}",
                "text": evidence_text,
            }
        ],
    }


def _build_method_signature(
    signature: dict[str, Any], *, work_id: int, source_kind: str, content_source: str, status: str, index: int
) -> dict[str, Any]:
    evidence_bundle = _build_evidence_bundle(
        work_id=work_id,
        label=str(signature["label"]),
        evidence_text=str(signature["evidence_text"]),
        source_kind=source_kind,
        status=status,
        confidence=float(signature.get("confidence") or 0.0),
        ref_suffix=f"method:{index}",
        content_source=content_source,
    )
    return {
        "contract_version": OBJECT_CONTRACT_VERSION,
        "object_type": "method_signature",
        "object_id": f"method_signature:{work_id}:{index}",
        "work_id": work_id,
        "source_kind": source_kind,
        "status": status,
        "source_refs": [f"work:{work_id}", f"method_signature:{index}", f"content_source:{content_source}"],
        "derivation": "normalized",
        "content_source": content_source,
        "method_kind": signature["method_kind"],
        "label": signature["label"],
        "confidence": signature["confidence"],
        "summary_text": signature["evidence_text"],
        "normalized_text": signature["label"].strip().lower(),
        "evidence_bundle": evidence_bundle,
    }


def _upsert_method_object(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_id: Any,
    status: str,
    summary_text: str | None,
    signatures: list[dict[str, Any]],
    name: str,
    source_kind: str,
    content_source: str,
) -> int:
    signature_json = json.dumps(
        [
            _build_method_signature(
                item,
                work_id=work_id,
                source_kind=source_kind,
                content_source=content_source,
                status=status,
                index=index,
            )
            for index, item in enumerate(signatures, start=1)
        ],
        ensure_ascii=False,
    )
    method_family = signatures[0]["method_kind"] if signatures else None
    conn.execute(
        """
        INSERT INTO method_objects (
          work_id, collection_id, object_key, name, method_family, summary_text, status, signature_json
        )
        VALUES (?, ?, 'default', ?, ?, ?, ?, ?)
        ON CONFLICT(work_id, object_key) DO UPDATE SET
          collection_id = excluded.collection_id,
          name = excluded.name,
          method_family = excluded.method_family,
          status = excluded.status,
          summary_text = excluded.summary_text,
          signature_json = excluded.signature_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            work_id,
            collection_id,
            name,
            method_family,
            summary_text or "",
            status,
            signature_json,
        ),
    )
    row = conn.execute(
        "SELECT method_object_id FROM method_objects WHERE work_id = ? AND object_key = 'default'",
        (work_id,),
    ).fetchone()
    return int(row["method_object_id"])


def _replace_method_signatures(conn: sqlite3.Connection, *, method_object_id: int, signatures: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM method_signatures WHERE method_object_id = ?", (method_object_id,))
    for item in signatures:
        conn.execute(
            """
            INSERT INTO method_signatures (method_object_id, signature_kind, signature_text, normalized_text, raw_signature_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                method_object_id,
                item["method_kind"],
                item["label"],
                item["label"].strip().lower(),
                json.dumps(item, ensure_ascii=False),
            ),
        )


def _replace_method_applications(conn: sqlite3.Connection, *, method_object_id: int, work_id: int, signatures: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM method_application_links WHERE method_object_id = ?", (method_object_id,))
    result_row = conn.execute(
        "SELECT result_object_id FROM result_objects WHERE work_id = ? AND object_key = 'default'",
        (work_id,),
    ).fetchone()
    result_object_id = int(result_row["result_object_id"]) if result_row is not None else None
    for item in signatures:
        conn.execute(
            """
            INSERT INTO method_application_links (
              method_object_id, result_object_id, target_work_id, relation_kind, confidence, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                method_object_id,
                result_object_id,
                work_id,
                item["method_kind"],
                item["confidence"],
                item["label"],
            ),
        )
