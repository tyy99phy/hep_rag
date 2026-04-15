from __future__ import annotations

import sqlite3
from typing import Any

from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.search_scope import available_search_scopes
from hep_rag_v2.search import search_index_counts
from hep_rag_v2.vector import vector_index_counts


def workspace_status_payload() -> dict[str, Any]:
    ensure_db()
    with connect() as conn:
        snapshot = _snapshot(conn)
        by_collection = _collections_payload(conn)
        snapshot.update(search_index_counts(conn))
        snapshot.update(vector_index_counts(conn))
        search_scopes = available_search_scopes(
            conn,
            snapshot=snapshot,
            collections=by_collection,
        )
    return {
        "snapshot": snapshot,
        "collections": by_collection,
        "search_scopes": search_scopes,
    }


def _snapshot(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM collections) AS collections,
          (SELECT COUNT(*) FROM works) AS works,
          (SELECT COUNT(*) FROM work_families) AS work_families,
          (SELECT COUNT(*) FROM authors) AS authors,
          (SELECT COUNT(*) FROM venues) AS venues,
          (SELECT COUNT(*) FROM topics) AS topics,
          (SELECT COUNT(*) FROM citations) AS citations,
          (SELECT COUNT(*) FROM citations WHERE dst_work_id IS NOT NULL) AS resolved_citations,
          (SELECT COUNT(*) FROM similarity_edges) AS similarity_edges,
          (SELECT COUNT(*) FROM bibliographic_coupling_edges) AS bibliographic_coupling_edges,
          (SELECT COUNT(*) FROM co_citation_edges) AS co_citation_edges,
          (SELECT COUNT(*) FROM graph_build_runs) AS graph_build_runs,
          (SELECT COUNT(*) FROM documents) AS documents,
          (SELECT COUNT(*) FROM formulas) AS formulas,
          (SELECT COUNT(*) FROM assets) AS assets,
          (SELECT COUNT(*) FROM chunks) AS chunks
        """
    ).fetchone()
    return {key: int(row[key]) for key in row.keys()} if row is not None else {}


def _collections_payload(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT c.name AS collection, COUNT(cw.work_id) AS works
            FROM collections c
            LEFT JOIN collection_works cw ON cw.collection_id = c.collection_id
            GROUP BY c.collection_id, c.name
            ORDER BY c.name
            """
        )
    ]
