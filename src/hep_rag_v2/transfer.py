from __future__ import annotations

import json
import sqlite3
from typing import Any

from hep_rag_v2.methods import ensure_method_schema
from hep_rag_v2.object_contracts import ALLOWED_STATUSES
from hep_rag_v2.results import ensure_result_schema

OBJECT_CONTRACT_VERSION = "v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transfer_candidates (
  transfer_candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_method_object_id INTEGER,
  source_result_object_id INTEGER,
  target_work_id INTEGER,
  target_context_json TEXT NOT NULL DEFAULT '{}',
  rationale_text TEXT,
  status TEXT NOT NULL DEFAULT 'proposed',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (source_method_object_id) REFERENCES method_objects(method_object_id) ON DELETE SET NULL,
  FOREIGN KEY (source_result_object_id) REFERENCES result_objects(result_object_id) ON DELETE SET NULL,
  FOREIGN KEY (target_work_id) REFERENCES works(work_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS transfer_edges (
  transfer_edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
  transfer_candidate_id INTEGER NOT NULL,
  src_method_object_id INTEGER,
  dst_work_id INTEGER,
  edge_kind TEXT NOT NULL DEFAULT 'candidate',
  score REAL,
  evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (transfer_candidate_id) REFERENCES transfer_candidates(transfer_candidate_id) ON DELETE CASCADE,
  FOREIGN KEY (src_method_object_id) REFERENCES method_objects(method_object_id) ON DELETE SET NULL,
  FOREIGN KEY (dst_work_id) REFERENCES works(work_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_transfer_candidates_target ON transfer_candidates(target_work_id, status);
"""


def ensure_transfer_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def build_transfer_candidates(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int] | None = None,
    collection: str | None = None,
) -> dict[str, Any]:
    ensure_method_schema(conn)
    ensure_result_schema(conn)
    ensure_transfer_schema(conn)
    selected_work_ids = _select_work_ids(conn, work_ids=work_ids, collection=collection)
    summary: dict[str, Any] = {"processed": 0, "ready": 0, "partial": 0, "needs_review": 0, "failed": 0, "items": []}
    for work_id in selected_work_ids:
        try:
            structure_status = _load_structure_status(conn, work_id=work_id)
            candidates = _candidate_rows(conn, work_id=work_id, collection=collection)
            source_method_object_id = _lookup_method_object_id(conn, work_id=work_id)
            source_result_object_id = _lookup_result_object_id(conn, work_id=work_id)
            if source_method_object_id is not None:
                _delete_transfer_candidates_for_source(conn, source_method_object_id=source_method_object_id)
            if structure_status in {"needs_review", "failed"}:
                status = str(structure_status)
                summary[status] += 1
                candidates = []
            elif candidates:
                status = "ready"
                summary["ready"] += 1
            else:
                status = "partial"
                summary["partial"] += 1
            count = 0
            for candidate in candidates:
                transfer_candidate_id = _upsert_transfer_candidate(
                    conn,
                    source_method_object_id=source_method_object_id,
                    source_result_object_id=source_result_object_id,
                    target_work_id=int(candidate["target_work_id"]),
                    target_context_json={
                        "shared_method_label": str(candidate["label"]),
                        "support_count": int(candidate["support_count"]),
                        "score": float(candidate["score"]),
                    },
                    status="ready",
                    rationale_text=str(candidate["rationale"]),
                )
                _insert_transfer_edge(
                    conn,
                    transfer_candidate_id=transfer_candidate_id,
                    src_method_object_id=source_method_object_id,
                    dst_work_id=int(candidate["target_work_id"]),
                    score=float(candidate["score"]),
                    evidence_json=[{"shared_method_label": str(candidate["label"]), "support_count": int(candidate["support_count"])}],
                )
                count += 1
            summary["processed"] += 1
            summary["items"].append({"work_id": work_id, "status": status, "candidate_count": count})
        except Exception as exc:
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


def _load_structure_status(conn: sqlite3.Connection, *, work_id: int) -> str | None:
    row = conn.execute("SELECT status FROM work_capsules WHERE work_id = ?", (work_id,)).fetchone()
    if row is None:
        return None
    status = str(row["status"] or "").strip()
    if not status:
        return "failed"
    return status if status in ALLOWED_STATUSES else "failed"


def _candidate_rows(conn: sqlite3.Connection, *, work_id: int, collection: str | None) -> list[dict[str, Any]]:
    where = ["mo.work_id = ?"]
    params: list[Any] = [work_id]
    collection_sql = ""
    if collection:
        collection_sql = """
          JOIN collection_works cw1 ON cw1.work_id = mo.work_id
          JOIN collections c1 ON c1.collection_id = cw1.collection_id
          JOIN collection_works cw2 ON cw2.work_id = other.work_id
          JOIN collections c2 ON c2.collection_id = cw2.collection_id
        """
        where.extend(["c1.name = ?", "c2.name = ?"])
        params.extend([collection, collection])
    rows = conn.execute(
        f"""
        SELECT
          other.work_id AS target_work_id,
          w.title AS target_title,
          ms.signature_text AS label,
          COUNT(*) AS support_count,
          MAX(CASE WHEN rv.result_value_id IS NOT NULL THEN 1 ELSE 0 END) AS has_result_support
        FROM method_objects mo
        JOIN method_signatures ms ON ms.method_object_id = mo.method_object_id
        JOIN method_signatures other_ms
          ON COALESCE(other_ms.normalized_text, lower(other_ms.signature_text))
           = COALESCE(ms.normalized_text, lower(ms.signature_text))
        JOIN method_objects other ON other.method_object_id = other_ms.method_object_id
        JOIN works w ON w.work_id = other.work_id
        LEFT JOIN result_objects ro ON ro.work_id = other.work_id
        LEFT JOIN result_values rv ON rv.result_object_id = ro.result_object_id
        {collection_sql}
        WHERE {' AND '.join(where)}
          AND other.work_id != mo.work_id
        GROUP BY other.work_id, w.title, ms.signature_text
        ORDER BY has_result_support DESC, support_count DESC, other.work_id
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        support = int(row["support_count"] or 0)
        has_result_support = bool(row["has_result_support"])
        out.append(
            {
                "target_work_id": int(row["target_work_id"]),
                "target_title": str(row["target_title"] or "").strip() or f"work-{int(row['target_work_id'])}",
                "label": str(row["label"]),
                "support_count": support,
                "score": min(0.99, 0.35 + 0.2 * support + (0.15 if has_result_support else 0.0)),
                "rationale": f"shared method signature '{row['label']}'" + (" with result support" if has_result_support else ""),
            }
        )
    return out


def _lookup_method_object_id(conn: sqlite3.Connection, *, work_id: int) -> int | None:
    row = conn.execute(
        "SELECT method_object_id FROM method_objects WHERE work_id = ? AND object_key = 'default'",
        (work_id,),
    ).fetchone()
    return int(row["method_object_id"]) if row is not None else None


def _lookup_result_object_id(conn: sqlite3.Connection, *, work_id: int) -> int | None:
    row = conn.execute(
        "SELECT result_object_id FROM result_objects WHERE work_id = ? AND object_key = 'default'",
        (work_id,),
    ).fetchone()
    return int(row["result_object_id"]) if row is not None else None


def _delete_transfer_candidates_for_source(conn: sqlite3.Connection, *, source_method_object_id: int) -> None:
    conn.execute("DELETE FROM transfer_candidates WHERE source_method_object_id = ?", (source_method_object_id,))


def _build_evidence_bundle(*, target_work_id: int, label: str, support_count: int, score: float, ref_suffix: str) -> dict[str, Any]:
    return {
        "contract_version": OBJECT_CONTRACT_VERSION,
        "object_type": "evidence_bundle",
        "object_id": f"evidence_bundle:{target_work_id}:{ref_suffix}",
        "source_kind": "extraction",
        "status": "ready",
        "source_refs": [f"work:{target_work_id}", f"transfer:{ref_suffix}"],
        "derivation": "aggregated",
        "label": label,
        "confidence": score,
        "items": [
            {
                "kind": "shared_method",
                "ref": f"work:{target_work_id}",
                "text": label,
                "support_count": support_count,
            }
        ],
    }


def _build_trace_step(*, label: str, support_count: int, score: float, source_method_object_id: int | None, target_work_id: int) -> dict[str, Any]:
    return {
        "contract_version": OBJECT_CONTRACT_VERSION,
        "object_type": "trace_step",
        "object_id": f"trace_step:{source_method_object_id or 'unknown'}:{target_work_id}:{label.strip().lower().replace(' ', '_')}",
        "source_kind": "extraction",
        "status": "ready",
        "source_refs": [
            f"method_object:{source_method_object_id}" if source_method_object_id is not None else "method_object:unknown",
            f"work:{target_work_id}",
        ],
        "derivation": "inferred",
        "step_kind": "shared_method_transfer",
        "target_work_id": target_work_id,
        "summary_text": f"shared method signature '{label}'",
        "confidence": score,
        "evidence_refs": [f"evidence_bundle:{target_work_id}:transfer"],
        "evidence_bundle": _build_evidence_bundle(
            target_work_id=target_work_id,
            label=label,
            support_count=support_count,
            score=score,
            ref_suffix="transfer",
        ),
    }


def _build_work_capsule(
    *,
    label: str,
    support_count: int,
    score: float,
    source_method_object_id: int | None,
    target_work_id: int,
    target_title: str,
) -> dict[str, Any]:
    return {
        "contract_version": OBJECT_CONTRACT_VERSION,
        "object_type": "work_capsule",
        "object_id": f"work_capsule:{target_work_id}",
        "work_id": target_work_id,
        "title": target_title,
        "source_kind": "extraction",
        "status": "ready",
        "source_refs": [f"work:{target_work_id}", f"method_object:{source_method_object_id or 'unknown'}"],
        "derivation": "aggregated",
        "shared_method_label": label,
        "support_count": support_count,
        "score": score,
        "trace_step": _build_trace_step(
            label=label,
            support_count=support_count,
            score=score,
            source_method_object_id=source_method_object_id,
            target_work_id=target_work_id,
        ),
    }


def _upsert_transfer_candidate(
    conn: sqlite3.Connection,
    *,
    source_method_object_id: int | None,
    source_result_object_id: int | None,
    target_work_id: int,
    target_context_json: dict[str, Any],
    status: str,
    rationale_text: str,
) -> int:
    conn.execute(
        """
        INSERT INTO transfer_candidates (
          source_method_object_id, source_result_object_id, target_work_id, target_context_json, rationale_text, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            source_method_object_id,
            source_result_object_id,
            target_work_id,
            json.dumps(
                _build_work_capsule(
                    label=str(target_context_json["shared_method_label"]),
                    support_count=int(target_context_json["support_count"]),
                    score=float(target_context_json["score"]),
                    source_method_object_id=source_method_object_id,
                    target_work_id=target_work_id,
                    target_title=str(target_context_json.get("target_title") or f"work-{target_work_id}"),
                ),
                ensure_ascii=False,
            ),
            rationale_text,
            status,
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_transfer_edge(
    conn: sqlite3.Connection,
    *,
    transfer_candidate_id: int,
    src_method_object_id: int | None,
    dst_work_id: int,
    score: float,
    evidence_json: list[dict[str, Any]],
) -> None:
    conn.execute(
        """
        INSERT INTO transfer_edges (
          transfer_candidate_id, src_method_object_id, dst_work_id, edge_kind, score, evidence_json
        )
        VALUES (?, ?, ?, 'shared_method', ?, ?)
        """,
        (
            transfer_candidate_id,
            src_method_object_id,
            dst_work_id,
            score,
            json.dumps(
                [
                    _build_evidence_bundle(
                        target_work_id=dst_work_id,
                        label=str(item["shared_method_label"]),
                        support_count=int(item["support_count"]),
                        score=score,
                        ref_suffix="transfer",
                    )
                    for item in evidence_json
                ],
                ensure_ascii=False,
            ),
        ),
    )
