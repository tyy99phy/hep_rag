from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Callable

import numpy as np


DEFAULT_SIMILARITY_MODEL = "hash-idf-v1"
ProgressCallback = Callable[[str], None] | None


def rebuild_graph_edges(
    conn: sqlite3.Connection,
    *,
    target: str = "all",
    collection: str | None = None,
    min_shared: int = 2,
    similarity_model: str = DEFAULT_SIMILARITY_MODEL,
    similarity_top_k: int = 10,
    similarity_min_score: float = 0.35,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    collection_id = _collection_id(conn, collection) if collection else None
    summary = {
        "target": target,
        "collection": collection,
        "min_shared": min_shared,
        "bibliographic_coupling_edges": 0,
        "co_citation_edges": 0,
        "similarity_model": similarity_model,
        "similarity_top_k": similarity_top_k,
        "similarity_min_score": similarity_min_score,
        "similarity_edges": 0,
        "build_ids": [],
    }

    if target in {"all", "bibliographic-coupling"}:
        _emit_progress(progress, "building bibliographic coupling edges...")
        build_id = _start_graph_build(
            conn,
            build_kind="bibliographic_coupling",
            notes=json.dumps({"collection": collection, "min_shared": min_shared}, ensure_ascii=False),
        )
        try:
            count = rebuild_bibliographic_coupling_edges(
                conn,
                build_id=build_id,
                collection_id=collection_id,
                min_shared=min_shared,
            )
            _finish_graph_build(conn, build_id=build_id, status="completed", notes=f"edges={count}")
            summary["bibliographic_coupling_edges"] = count
            summary["build_ids"].append(build_id)
            _emit_progress(progress, f"bibliographic coupling edges ready: {count}")
        except Exception as exc:
            _finish_graph_build(conn, build_id=build_id, status="failed", notes=f"{type(exc).__name__}: {exc}")
            raise

    if target in {"all", "co-citation"}:
        _emit_progress(progress, "building co-citation edges...")
        build_id = _start_graph_build(
            conn,
            build_kind="co_citation",
            notes=json.dumps({"collection": collection, "min_shared": min_shared}, ensure_ascii=False),
        )
        try:
            count = rebuild_co_citation_edges(
                conn,
                build_id=build_id,
                collection_id=collection_id,
                min_shared=min_shared,
            )
            _finish_graph_build(conn, build_id=build_id, status="completed", notes=f"edges={count}")
            summary["co_citation_edges"] = count
            summary["build_ids"].append(build_id)
            _emit_progress(progress, f"co-citation edges ready: {count}")
        except Exception as exc:
            _finish_graph_build(conn, build_id=build_id, status="failed", notes=f"{type(exc).__name__}: {exc}")
            raise

    if target in {"all", "similarity"} and (target == "similarity" or _has_work_vectors(conn, model=similarity_model)):
        _emit_progress(
            progress,
            f"building similarity edges with model={similarity_model}, top_k={similarity_top_k}, min_score={similarity_min_score}...",
        )
        build_id = _start_graph_build(
            conn,
            build_kind="embedding_similarity",
            notes=json.dumps(
                {
                    "collection": collection,
                    "model": similarity_model,
                    "top_k": similarity_top_k,
                    "min_score": similarity_min_score,
                },
                ensure_ascii=False,
            ),
        )
        try:
            count = rebuild_similarity_edges(
                conn,
                build_id=build_id,
                collection_id=collection_id,
                model=similarity_model,
                top_k=similarity_top_k,
                min_score=similarity_min_score,
            )
            _finish_graph_build(conn, build_id=build_id, status="completed", notes=f"edges={count}")
            summary["similarity_edges"] = count
            summary["build_ids"].append(build_id)
            _emit_progress(progress, f"similarity edges ready: {count}")
        except Exception as exc:
            _finish_graph_build(conn, build_id=build_id, status="failed", notes=f"{type(exc).__name__}: {exc}")
            raise
    elif target in {"all", "similarity"}:
        summary["similarity_skipped"] = f"work embedding index not found for model={similarity_model}"
        _emit_progress(progress, summary["similarity_skipped"])
    return summary


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is None:
        return
    text = str(message or "").strip()
    if text:
        progress(text)


def graph_neighbors(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    edge_kind: str = "all",
    collection: str | None = None,
    limit: int = 20,
    similarity_model: str | None = None,
) -> list[dict[str, Any]]:
    collection_id = _collection_id(conn, collection) if collection else None
    rows: list[dict[str, Any]] = []

    if edge_kind in {"all", "bibliographic-coupling"}:
        rows.extend(
            _edge_neighbors(
                conn,
                work_id=work_id,
                collection_id=collection_id,
                table="bibliographic_coupling_edges",
                edge_kind="bibliographic_coupling",
                shared_column="shared_reference_count",
            )
        )
    if edge_kind in {"all", "co-citation"}:
        rows.extend(
            _edge_neighbors(
                conn,
                work_id=work_id,
                collection_id=collection_id,
                table="co_citation_edges",
                edge_kind="co_citation",
                shared_column="shared_citer_count",
            )
        )
    if edge_kind == "similarity":
        rows.extend(
            _similarity_neighbors(
                conn,
                work_id=work_id,
                collection_id=collection_id,
                similarity_model=similarity_model,
            )
        )

    rows.sort(
        key=lambda row: (
            -int(row["shared_count"]),
            -(float(row["score"]) if row["score"] is not None else float("-inf")),
            row["neighbor_work_id"],
        )
    )
    return rows[:limit]


def rebuild_bibliographic_coupling_edges(
    conn: sqlite3.Connection,
    *,
    build_id: int,
    collection_id: int | None,
    min_shared: int,
) -> int:
    _delete_bibliographic_edges(conn, collection_id=collection_id)
    out_degree = _resolved_out_degree(conn)
    rows = conn.execute(
        f"""
        WITH resolved AS (
          SELECT DISTINCT src_work_id, dst_work_id
          FROM citations
          WHERE dst_work_id IS NOT NULL
        )
        SELECT
          r1.src_work_id AS src_work_id,
          r2.src_work_id AS dst_work_id,
          COUNT(*) AS shared_reference_count
        FROM resolved r1
        JOIN resolved r2
          ON r1.dst_work_id = r2.dst_work_id
         AND r1.src_work_id < r2.src_work_id
        {_where_collection("r1.src_work_id", "r2.src_work_id", collection_id)}
        GROUP BY r1.src_work_id, r2.src_work_id
        HAVING COUNT(*) >= ?
        ORDER BY shared_reference_count DESC, src_work_id, dst_work_id
        """,
        _where_params(collection_id) + [min_shared],
    ).fetchall()

    inserted = 0
    for row in rows:
        src_work_id = int(row["src_work_id"])
        dst_work_id = int(row["dst_work_id"])
        shared = int(row["shared_reference_count"])
        score = _association_score(shared, out_degree.get(src_work_id, 0), out_degree.get(dst_work_id, 0))
        conn.execute(
            """
            INSERT INTO bibliographic_coupling_edges (
              src_work_id, dst_work_id, shared_reference_count, score, build_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (src_work_id, dst_work_id, shared, score, build_id),
        )
        inserted += 1
    return inserted


def rebuild_co_citation_edges(
    conn: sqlite3.Connection,
    *,
    build_id: int,
    collection_id: int | None,
    min_shared: int,
) -> int:
    _delete_co_citation_edges(conn, collection_id=collection_id)
    in_degree = _resolved_in_degree(conn)
    rows = conn.execute(
        f"""
        WITH resolved AS (
          SELECT DISTINCT src_work_id, dst_work_id
          FROM citations
          WHERE dst_work_id IS NOT NULL
        )
        SELECT
          r1.dst_work_id AS src_work_id,
          r2.dst_work_id AS dst_work_id,
          COUNT(*) AS shared_citer_count
        FROM resolved r1
        JOIN resolved r2
          ON r1.src_work_id = r2.src_work_id
         AND r1.dst_work_id < r2.dst_work_id
        {_where_collection("r1.dst_work_id", "r2.dst_work_id", collection_id)}
        GROUP BY r1.dst_work_id, r2.dst_work_id
        HAVING COUNT(*) >= ?
        ORDER BY shared_citer_count DESC, src_work_id, dst_work_id
        """,
        _where_params(collection_id) + [min_shared],
    ).fetchall()

    inserted = 0
    for row in rows:
        src_work_id = int(row["src_work_id"])
        dst_work_id = int(row["dst_work_id"])
        shared = int(row["shared_citer_count"])
        score = _association_score(shared, in_degree.get(src_work_id, 0), in_degree.get(dst_work_id, 0))
        conn.execute(
            """
            INSERT INTO co_citation_edges (
              src_work_id, dst_work_id, shared_citer_count, score, build_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (src_work_id, dst_work_id, shared, score, build_id),
        )
        inserted += 1
    return inserted


def rebuild_similarity_edges(
    conn: sqlite3.Connection,
    *,
    build_id: int,
    collection_id: int | None,
    model: str,
    top_k: int,
    min_score: float,
    batch_size: int = 256,
) -> int:
    metric = _similarity_metric(model)
    _delete_similarity_edges(conn, collection_id=collection_id, metric=metric)
    ids, vectors = _load_work_vectors(conn, model=model, collection_id=collection_id)
    if ids.size == 0 or vectors.size == 0:
        return 0
    if top_k <= 0:
        return 0

    edge_scores: dict[tuple[int, int], float] = {}
    n = int(ids.shape[0])
    step = max(1, min(batch_size, n))
    for start in range(0, n, step):
        stop = min(start + step, n)
        block = vectors[start:stop]
        scores = block @ vectors.T
        for offset in range(stop - start):
            row_idx = start + offset
            scores[offset, row_idx] = -1.0
            candidate_mask = scores[offset] >= float(min_score)
            candidate_idx = np.flatnonzero(candidate_mask)
            if candidate_idx.size == 0:
                continue
            if candidate_idx.size > top_k:
                local_scores = scores[offset, candidate_idx]
                keep = np.argpartition(-local_scores, top_k - 1)[:top_k]
                candidate_idx = candidate_idx[keep]
            candidate_idx = candidate_idx[np.argsort(-scores[offset, candidate_idx], kind="stable")]
            for neighbor_idx in candidate_idx.tolist():
                src_work_id = int(ids[row_idx])
                dst_work_id = int(ids[int(neighbor_idx)])
                left, right = sorted((src_work_id, dst_work_id))
                if left == right:
                    continue
                edge_scores[(left, right)] = max(
                    float(edge_scores.get((left, right), float("-inf"))),
                    float(scores[offset, int(neighbor_idx)]),
                )

    rows = sorted(
        (
            (src_work_id, dst_work_id, metric, score, build_id)
            for (src_work_id, dst_work_id), score in edge_scores.items()
            if score >= float(min_score)
        ),
        key=lambda item: (-float(item[3]), item[0], item[1]),
    )
    conn.executemany(
        """
        INSERT INTO similarity_edges (
          src_work_id, dst_work_id, metric, score, build_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _resolved_out_degree(conn: sqlite3.Connection) -> dict[int, int]:
    return {
        int(row["src_work_id"]): int(row["n"])
        for row in conn.execute(
            """
            SELECT src_work_id, COUNT(DISTINCT dst_work_id) AS n
            FROM citations
            WHERE dst_work_id IS NOT NULL
            GROUP BY src_work_id
            """
        ).fetchall()
    }


def _resolved_in_degree(conn: sqlite3.Connection) -> dict[int, int]:
    return {
        int(row["dst_work_id"]): int(row["n"])
        for row in conn.execute(
            """
            SELECT dst_work_id, COUNT(DISTINCT src_work_id) AS n
            FROM citations
            WHERE dst_work_id IS NOT NULL
            GROUP BY dst_work_id
            """
        ).fetchall()
    }


def _association_score(shared: int, left_total: int, right_total: int) -> float | None:
    if left_total <= 0 or right_total <= 0:
        return None
    return shared / math.sqrt(left_total * right_total)


def _collection_id(conn: sqlite3.Connection, collection: str) -> int:
    row = conn.execute(
        "SELECT collection_id FROM collections WHERE name = ?",
        (collection,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown collection: {collection}")
    return int(row["collection_id"])


def _start_graph_build(conn: sqlite3.Connection, *, build_kind: str, notes: str | None) -> int:
    cur = conn.execute(
        """
        INSERT INTO graph_build_runs (build_kind, status, notes)
        VALUES (?, 'running', ?)
        """,
        (build_kind, notes),
    )
    return int(cur.lastrowid)


def _finish_graph_build(conn: sqlite3.Connection, *, build_id: int, status: str, notes: str | None) -> None:
    conn.execute(
        """
        UPDATE graph_build_runs
        SET status = ?, notes = ?, finished_at = CURRENT_TIMESTAMP
        WHERE build_id = ?
        """,
        (status, notes, build_id),
    )


def _delete_bibliographic_edges(conn: sqlite3.Connection, *, collection_id: int | None) -> None:
    if collection_id is None:
        conn.execute("DELETE FROM bibliographic_coupling_edges")
        return
    conn.execute(
        """
        DELETE FROM bibliographic_coupling_edges
        WHERE src_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
          AND dst_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
        """,
        (collection_id, collection_id),
    )


def _delete_co_citation_edges(conn: sqlite3.Connection, *, collection_id: int | None) -> None:
    if collection_id is None:
        conn.execute("DELETE FROM co_citation_edges")
        return
    conn.execute(
        """
        DELETE FROM co_citation_edges
        WHERE src_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
          AND dst_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
        """,
        (collection_id, collection_id),
    )


def _delete_similarity_edges(conn: sqlite3.Connection, *, collection_id: int | None, metric: str) -> None:
    if collection_id is None:
        conn.execute(
            "DELETE FROM similarity_edges WHERE metric = ?",
            (metric,),
        )
        return
    conn.execute(
        """
        DELETE FROM similarity_edges
        WHERE metric = ?
          AND src_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
          AND dst_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
        """,
        (metric, collection_id, collection_id),
    )


def _where_collection(left_col: str, right_col: str, collection_id: int | None) -> str:
    if collection_id is None:
        return "WHERE 1=1"
    return (
        "WHERE "
        f"{left_col} IN (SELECT work_id FROM collection_works WHERE collection_id = ?) "
        "AND "
        f"{right_col} IN (SELECT work_id FROM collection_works WHERE collection_id = ?)"
    )


def _where_params(collection_id: int | None) -> list[Any]:
    if collection_id is None:
        return []
    return [collection_id, collection_id]


def _edge_neighbors(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_id: int | None,
    table: str,
    edge_kind: str,
    shared_column: str,
) -> list[dict[str, Any]]:
    collection_sql = ""
    if collection_id is not None:
        collection_sql = """
          AND neighbor.work_id IN (
            SELECT work_id FROM collection_works WHERE collection_id = ?
          )
        """

    rows = conn.execute(
        f"""
        SELECT
          CASE WHEN e.src_work_id = ? THEN e.dst_work_id ELSE e.src_work_id END AS neighbor_work_id,
          neighbor.canonical_source,
          neighbor.canonical_id,
          neighbor.title,
          neighbor.year,
          e.{shared_column} AS shared_count,
          e.score
        FROM {table} e
        JOIN works neighbor
          ON neighbor.work_id = CASE WHEN e.src_work_id = ? THEN e.dst_work_id ELSE e.src_work_id END
        WHERE e.src_work_id = ? OR e.dst_work_id = ?
        {collection_sql}
        ORDER BY e.{shared_column} DESC, e.score DESC, neighbor.work_id
        """,
        [work_id, work_id, work_id, work_id, *([] if collection_id is None else [collection_id])],
    ).fetchall()

    return [
        {
            "edge_kind": edge_kind,
            "neighbor_work_id": int(row["neighbor_work_id"]),
            "canonical_source": row["canonical_source"],
            "canonical_id": row["canonical_id"],
            "title": row["title"],
            "year": row["year"],
            "shared_count": int(row["shared_count"]),
            "score": row["score"],
        }
        for row in rows
    ]


def _similarity_neighbors(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_id: int | None,
    similarity_model: str | None,
) -> list[dict[str, Any]]:
    collection_sql = ""
    metric_sql = ""
    params: list[Any] = [work_id, work_id, work_id, work_id]
    if similarity_model:
        metric_sql = " AND e.metric = ?"
        params.append(_similarity_metric(similarity_model))
    if collection_id is not None:
        collection_sql = """
          AND neighbor.work_id IN (
            SELECT work_id FROM collection_works WHERE collection_id = ?
          )
        """
        params.append(collection_id)

    rows = conn.execute(
        f"""
        SELECT
          CASE WHEN e.src_work_id = ? THEN e.dst_work_id ELSE e.src_work_id END AS neighbor_work_id,
          neighbor.canonical_source,
          neighbor.canonical_id,
          neighbor.title,
          neighbor.year,
          MAX(e.score) AS score
        FROM similarity_edges e
        JOIN works neighbor
          ON neighbor.work_id = CASE WHEN e.src_work_id = ? THEN e.dst_work_id ELSE e.src_work_id END
        WHERE e.src_work_id = ? OR e.dst_work_id = ?
        {metric_sql}
        {collection_sql}
        GROUP BY neighbor_work_id, neighbor.canonical_source, neighbor.canonical_id, neighbor.title, neighbor.year
        ORDER BY score DESC, neighbor_work_id
        """,
        params,
    ).fetchall()
    return [
        {
            "edge_kind": "similarity",
            "neighbor_work_id": int(row["neighbor_work_id"]),
            "canonical_source": row["canonical_source"],
            "canonical_id": row["canonical_id"],
            "title": row["title"],
            "year": row["year"],
            "shared_count": 0,
            "score": float(row["score"]) if row["score"] is not None else None,
        }
        for row in rows
    ]


def _load_work_vectors(
    conn: sqlite3.Connection,
    *,
    model: str,
    collection_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    row = conn.execute(
        """
        SELECT vector_path
        FROM work_embeddings
        WHERE embedding_model = ?
          AND vector_path IS NOT NULL
        LIMIT 1
        """,
        (model,),
    ).fetchone()
    if row is None or not str(row["vector_path"] or "").strip():
        raise ValueError(f"Work embedding index not found for model={model}. Build vector index first.")

    with np.load(str(row["vector_path"]), allow_pickle=False) as payload:
        ids = np.asarray(payload["ids"], dtype=np.int64)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)

    if collection_id is not None:
        allowed_rows = conn.execute(
            "SELECT work_id FROM collection_works WHERE collection_id = ? ORDER BY work_id",
            (collection_id,),
        ).fetchall()
        allowed_ids = np.asarray([int(item["work_id"]) for item in allowed_rows], dtype=np.int64)
        if allowed_ids.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=np.float32),
            )
        mask = np.isin(ids, allowed_ids)
        ids = ids[mask]
        vectors = vectors[mask]
    return (ids, vectors)


def _similarity_metric(model: str) -> str:
    return f"cosine::{model}"


def _has_work_vectors(conn: sqlite3.Connection, *, model: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM work_embeddings
        WHERE embedding_model = ?
          AND vector_path IS NOT NULL
        LIMIT 1
        """,
        (model,),
    ).fetchone()
    return row is not None
