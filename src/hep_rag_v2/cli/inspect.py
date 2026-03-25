from __future__ import annotations

import argparse
import json
import re
import sqlite3
from typing import Any

from hep_rag_v2.db import connect, ensure_db

from ._common import (
    AUDIT_PATTERNS,
    READINESS_THRESHOLDS,
    _resolve_work_row,
)


def cmd_show_document(args: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        work = _resolve_work_row(
            conn,
            work_id=args.work_id,
            id_type=args.id_type,
            id_value=args.id_value,
        )
        document = conn.execute(
            """
            SELECT d.document_id, d.work_id, d.parser_name, d.parser_version, d.parse_status,
                   d.parsed_dir, d.manifest_path, w.title
            FROM documents d
            JOIN works w ON w.work_id = d.work_id
            WHERE d.work_id = ?
            """,
            (int(work["work_id"]),),
        ).fetchone()
        if document is None:
            raise SystemExit(f"No document materialized for work_id={int(work['work_id'])}")

        sections = [
            dict(row)
            for row in conn.execute(
                """
                SELECT section_id, parent_section_id, ordinal, title, clean_title, section_kind,
                       level, order_index, page_start, page_end, path_text
                FROM document_sections
                WHERE document_id = ?
                ORDER BY order_index, section_id
                LIMIT ?
                """,
                (int(document["document_id"]), args.limit),
            )
        ]
        blocks = [
            dict(row)
            for row in conn.execute(
                """
                SELECT block_id, section_id, block_type, page, order_index, block_role,
                       is_heading, is_retrievable, exclusion_reason, clean_text
                FROM blocks
                WHERE document_id = ?
                ORDER BY order_index, block_id
                LIMIT ?
                """,
                (int(document["document_id"]), args.limit),
            )
        ]
        chunks = [
            dict(row)
            for row in conn.execute(
                """
                SELECT chunk_id, section_id, chunk_role, page_hint, is_retrievable,
                       exclusion_reason, clean_text
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_id
                LIMIT ?
                """,
                (int(document["document_id"]), args.limit),
            )
        ]
        role_counts = dict(
            conn.execute(
                """
                SELECT block_role, COUNT(*) AS n
                FROM blocks
                WHERE document_id = ?
                GROUP BY block_role
                ORDER BY block_role
                """,
                (int(document["document_id"]),),
            ).fetchall()
        )
        chunk_counts = dict(
            conn.execute(
                """
                SELECT chunk_role, COUNT(*) AS n
                FROM chunks
                WHERE document_id = ?
                GROUP BY chunk_role
                ORDER BY chunk_role
                """,
                (int(document["document_id"]),),
            ).fetchall()
        )
    print(
        json.dumps(
            {
                "document": dict(document),
                "block_roles": role_counts,
                "chunk_roles": chunk_counts,
                "sections_sample": sections,
                "blocks_sample": blocks,
                "chunks_sample": chunks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_audit_document(args: argparse.Namespace) -> None:
    ensure_db()
    with connect() as conn:
        work = _resolve_work_row(
            conn,
            work_id=args.work_id,
            id_type=args.id_type,
            id_value=args.id_value,
        )
        document = conn.execute(
            """
            SELECT d.document_id, d.work_id, d.parser_name, d.parser_version, d.parse_status,
                   d.parsed_dir, d.manifest_path, w.title
            FROM documents d
            JOIN works w ON w.work_id = d.work_id
            WHERE d.work_id = ?
            """,
            (int(work["work_id"]),),
        ).fetchone()
        if document is None:
            raise SystemExit(f"No document materialized for work_id={int(work['work_id'])}")

        payload = _audit_document_payload(conn, document_id=int(document["document_id"]), limit=args.limit)
        payload["document"] = dict(document)
        payload["work"] = {
            "work_id": int(work["work_id"]),
            "canonical_source": work["canonical_source"],
            "canonical_id": work["canonical_id"],
            "title": work["title"],
            "year": work["year"],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Internal audit helpers
# ---------------------------------------------------------------------------


def _audit_document_payload(conn: sqlite3.Connection, *, document_id: int, limit: int) -> dict[str, Any]:
    block_counts = dict(
        conn.execute(
            """
            SELECT
              COUNT(*) AS blocks_total,
              SUM(CASE WHEN is_retrievable = 1 THEN 1 ELSE 0 END) AS retrievable_blocks,
              SUM(CASE WHEN block_role = 'body' THEN 1 ELSE 0 END) AS body_blocks,
              SUM(CASE WHEN block_role = 'body' AND is_retrievable = 1 THEN 1 ELSE 0 END) AS retrievable_body_blocks
            FROM blocks
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    )
    chunk_counts = dict(
        conn.execute(
            """
            SELECT
              COUNT(*) AS chunks_total,
              SUM(CASE WHEN is_retrievable = 1 THEN 1 ELSE 0 END) AS retrievable_chunks,
              SUM(CASE WHEN chunk_role = 'section_child' THEN 1 ELSE 0 END) AS section_child_chunks,
              SUM(CASE WHEN chunk_role = 'formula_window' THEN 1 ELSE 0 END) AS formula_window_chunks,
              SUM(CASE WHEN chunk_role = 'asset_window' THEN 1 ELSE 0 END) AS asset_window_chunks
            FROM chunks
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    )
    search_counts = dict(
        conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM chunk_search cs JOIN chunks c ON c.chunk_id = cs.chunk_id WHERE c.document_id = ?) AS chunk_search_rows,
              (SELECT COUNT(*) FROM formula_search fs JOIN formulas f ON f.formula_id = fs.formula_id WHERE f.document_id = ?) AS formula_search_rows,
              (SELECT COUNT(*) FROM asset_search asearch JOIN assets a ON a.asset_id = asearch.asset_id WHERE a.document_id = ?) AS asset_search_rows
            """,
            (document_id, document_id, document_id),
        ).fetchone()
    )

    retrievable_blocks = _collect_noise_hits(
        conn,
        """
        SELECT block_id AS row_id, block_role AS role, clean_text
        FROM blocks
        WHERE document_id = ?
          AND is_retrievable = 1
          AND block_type != 'equation'
          AND clean_text IS NOT NULL
        ORDER BY block_id
        """,
        document_id=document_id,
        limit=limit,
    )
    retrievable_chunks = _collect_noise_hits(
        conn,
        """
        SELECT chunk_id AS row_id, chunk_role AS role, clean_text
        FROM chunks
        WHERE document_id = ?
          AND is_retrievable = 1
          AND clean_text IS NOT NULL
        ORDER BY chunk_id
        """,
        document_id=document_id,
        limit=limit,
    )
    all_blocks = _collect_noise_hits(
        conn,
        """
        SELECT block_id AS row_id, block_role AS role, clean_text
        FROM blocks
        WHERE document_id = ?
          AND clean_text IS NOT NULL
        ORDER BY block_id
        """,
        document_id=document_id,
        limit=limit,
    )

    equation_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT chunk_id, chunk_role, clean_text
            FROM chunks
            WHERE document_id = ?
              AND chunk_role IN ('section_child', 'section_parent')
              AND clean_text LIKE '%Equation:%'
            ORDER BY chunk_id
            """,
            (document_id,),
        ).fetchall()
    ]
    equation_with_structure = sum(
        int(("/" in str(row["clean_text"])) or (" x " in str(row["clean_text"])))
        for row in equation_rows
    )
    equation_with_frac_literal = sum(
        int(bool(re.search(r"\bfrac\b", str(row["clean_text"]), re.IGNORECASE)))
        for row in equation_rows
    )
    structured_ratio = (
        float(equation_with_structure) / float(len(equation_rows))
        if equation_rows
        else 1.0
    )

    readiness_checks = {
        "retrievable_chunk_noise_ok": retrievable_chunks["total_hits"] <= READINESS_THRESHOLDS["max_retrievable_chunk_noise"],
        "retrievable_block_noise_ok": retrievable_blocks["total_hits"] <= READINESS_THRESHOLDS["max_retrievable_block_noise"],
        "equation_structure_ok": structured_ratio >= READINESS_THRESHOLDS["min_structured_equation_ratio"],
        "chunk_index_complete": int(search_counts.get("chunk_search_rows") or 0) == int(chunk_counts.get("chunks_total") or 0),
    }
    ready = all(readiness_checks.values())
    return {
        "ready": ready,
        "readiness_checks": readiness_checks,
        "counts": {
            "blocks": block_counts,
            "chunks": chunk_counts,
            "search": search_counts,
        },
        "equation_placeholders": {
            "total": len(equation_rows),
            "with_structure": equation_with_structure,
            "with_frac_literal": equation_with_frac_literal,
            "structured_ratio": round(structured_ratio, 3),
            "samples": [
                {
                    "chunk_id": int(row["chunk_id"]),
                    "chunk_role": row["chunk_role"],
                    "clean_text": str(row["clean_text"])[:300],
                }
                for row in equation_rows[:limit]
            ],
        },
        "noise": {
            "retrievable_blocks": retrievable_blocks,
            "retrievable_chunks": retrievable_chunks,
            "all_blocks": all_blocks,
        },
        "recommendation": _readiness_recommendation(
            ready=ready,
            retrievable_chunks=retrievable_chunks["total_hits"],
            retrievable_blocks=retrievable_blocks["total_hits"],
            equation_ratio=structured_ratio,
        ),
    }


def _collect_noise_hits(
    conn: sqlite3.Connection,
    query: str,
    *,
    document_id: int,
    limit: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {name: 0 for name, _ in AUDIT_PATTERNS}
    samples: list[dict[str, Any]] = []
    total_hits = 0
    for row in conn.execute(query, (document_id,)).fetchall():
        text = str(row["clean_text"] or "")
        matched = [name for name, pattern in AUDIT_PATTERNS if pattern.search(text)]
        if not matched:
            continue
        total_hits += 1
        for name in matched:
            counts[name] += 1
        if len(samples) < limit:
            samples.append(
                {
                    "row_id": int(row["row_id"]),
                    "role": row["role"],
                    "patterns": matched,
                    "clean_text": text[:300],
                }
            )
    return {
        "total_hits": total_hits,
        "pattern_counts": counts,
        "samples": samples,
    }


def _readiness_recommendation(
    *,
    ready: bool,
    retrievable_chunks: int,
    retrievable_blocks: int,
    equation_ratio: float,
) -> str:
    if ready:
        return "ready_for_next_phase"
    if retrievable_chunks > 0:
        return "clean_retrieval_text_first"
    if equation_ratio < READINESS_THRESHOLDS["min_structured_equation_ratio"]:
        return "improve_equation_placeholders"
    if retrievable_blocks > READINESS_THRESHOLDS["max_retrievable_block_noise"]:
        return "reduce_parser_noise_in_blocks"
    return "needs_manual_review"
