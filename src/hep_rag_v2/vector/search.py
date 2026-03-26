from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from hep_rag_v2.graph import graph_neighbors
from hep_rag_v2.query import analyze_query, is_relation_query, is_result_query, query_match_stats
from hep_rag_v2.search import search_chunks_bm25, search_works_bm25
from .embedding import (
    DEFAULT_VECTOR_MODEL,
    LocalIndex,
    _score_query,
)
from .index import _load_local_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RRF_K = 60


# ---------------------------------------------------------------------------
# DB fetch helpers
# ---------------------------------------------------------------------------

def _allowed_work_ids(conn: sqlite3.Connection, *, collection: str | None) -> set[int] | None:
    if not collection:
        return None
    rows = conn.execute(
        """
        SELECT cw.work_id
        FROM collection_works cw
        JOIN collections c ON c.collection_id = cw.collection_id
        WHERE c.name = ?
        """,
        (collection,),
    ).fetchall()
    return {int(row["work_id"]) for row in rows}


def _allowed_chunk_ids(conn: sqlite3.Connection, *, collection: str | None) -> set[int] | None:
    if not collection:
        return None
    rows = conn.execute(
        """
        SELECT c.chunk_id
        FROM chunks c
        JOIN collection_works cw ON cw.work_id = c.work_id
        JOIN collections col ON col.collection_id = cw.collection_id
        WHERE col.name = ?
        """,
        (collection,),
    ).fetchall()
    return {int(row["chunk_id"]) for row in rows}


def _fetch_work_rows(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT w.work_id, w.title AS raw_title, w.abstract, w.year, w.canonical_source, w.canonical_id
        FROM works w
        WHERE w.work_id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return {int(row["work_id"]): dict(row) for row in rows}


def _fetch_chunk_rows(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT
          c.chunk_id,
          c.work_id,
          c.chunk_role,
          c.section_hint,
          c.page_hint,
          c.clean_text,
          w.title AS raw_title
        FROM chunks c
        JOIN works w ON w.work_id = c.work_id
        WHERE c.chunk_id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return {int(row["chunk_id"]): dict(row) for row in rows}


# ---------------------------------------------------------------------------
# Scoring / ranking
# ---------------------------------------------------------------------------

def _rank_scores(
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    limit: int,
    allowed_ids: set[int] | None,
) -> tuple[list[int], list[float]]:
    if ids.size == 0:
        return [], []
    mask = np.ones(ids.shape[0], dtype=bool)
    if allowed_ids is not None:
        allowed = np.fromiter((int(item) for item in allowed_ids), dtype=np.int64, count=len(allowed_ids))
        mask &= np.isin(ids, allowed)
    if not np.any(mask):
        return [], []

    filtered_ids = ids[mask]
    filtered_scores = scores[mask]
    positive_mask = filtered_scores > 0
    if np.any(positive_mask):
        filtered_ids = filtered_ids[positive_mask]
        filtered_scores = filtered_scores[positive_mask]
    if filtered_ids.size == 0:
        return [], []

    take = min(limit, int(filtered_ids.size))
    order = np.argsort(-filtered_scores, kind="stable")[:take]
    ranked_ids = [int(item) for item in filtered_ids[order].tolist()]
    ranked_scores = [float(item) for item in filtered_scores[order].tolist()]
    return ranked_ids, ranked_scores


def _annotate_query_agreement(
    rows: list[dict[str, Any]],
    *,
    profile: Any,
    text_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        combined_text = "\n".join(
            str(item.get(field) or "").strip()
            for field in text_fields
            if str(item.get(field) or "").strip()
        )
        hits, coverage = query_match_stats(combined_text, profile)
        item["query_group_hits"] = hits
        item["query_group_coverage"] = coverage
        out.append(item)
    return out


def _filter_vector_rows(
    rows: list[dict[str, Any]],
    *,
    profile: Any,
    fallback_if_empty: bool,
) -> list[dict[str, Any]]:
    if not rows or not getattr(profile, "match_groups", None):
        return rows

    required_hits = max(1, min(2, len(profile.match_groups)))
    filtered = [
        row
        for row in rows
        if int(row.get("query_group_hits") or 0) >= required_hits
        or float(row.get("query_group_coverage") or 0.0) >= 0.75
    ]
    if filtered or not fallback_if_empty:
        return filtered
    return rows


def _postprocess_vector_rows(rows: list[dict[str, Any]], *, profile: Any) -> list[dict[str, Any]]:
    filtered = _filter_vector_rows(rows, profile=profile, fallback_if_empty=False)
    if filtered:
        rows = filtered
    else:
        positive_match_rows = [
            row
            for row in rows
            if int(row.get("query_group_hits") or 0) > 0
            or float(row.get("query_group_coverage") or 0.0) > 0.0
        ]
        if positive_match_rows:
            rows = positive_match_rows
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row.get("query_group_hits") or 0),
            -float(row.get("query_group_coverage") or 0.0),
            -float(row.get("score") or 0.0),
            int(row.get("rank") or 0),
        ),
    )
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(
    *,
    key_field: str,
    named_rows: list[tuple[str, list[dict[str, Any]], float]],
) -> dict[int, dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}
    for source_name, rows, weight in named_rows:
        for rank, row in enumerate(rows, start=1):
            item_id = int(row[key_field])
            entry = fused.setdefault(
                item_id,
                {
                    "id": item_id,
                    "hybrid_score": 0.0,
                },
            )
            entry["hybrid_score"] += float(weight) / float(RRF_K + rank)
            entry[f"{source_name}_rank"] = rank
            entry["query_group_hits"] = max(
                int(entry.get("query_group_hits") or 0),
                int(row.get("query_group_hits") or 0),
            )
            entry["query_group_coverage"] = max(
                float(entry.get("query_group_coverage") or 0.0),
                float(row.get("query_group_coverage") or 0.0),
            )
    return fused


# ---------------------------------------------------------------------------
# Graph injection
# ---------------------------------------------------------------------------

def _allow_graph_seed(seed: dict[str, Any], *, profile: Any) -> bool:
    if seed.get("bm25_rank") is not None:
        return True
    groups = getattr(profile, "match_groups", ())
    if not groups:
        return True
    required_hits = max(1, min(2, len(groups)))
    hits = int(seed.get("query_group_hits") or 0)
    coverage = float(seed.get("query_group_coverage") or 0.0)
    return hits >= required_hits or coverage >= 0.75


def _inject_graph_support(
    conn: sqlite3.Connection,
    fused: dict[int, dict[str, Any]],
    *,
    profile: Any,
    collection: str | None,
    graph_expand: int,
    seed_limit: int,
    model: str,
) -> None:
    seeds = sorted(
        fused.values(),
        key=lambda item: (-float(item["hybrid_score"]), item["id"]),
    )[: max(1, seed_limit)]
    for seed_rank, seed in enumerate(seeds, start=1):
        if not _allow_graph_seed(seed, profile=profile):
            continue
        neighbors = graph_neighbors(
            conn,
            work_id=int(seed["id"]),
            edge_kind="all",
            collection=collection,
            limit=max(1, graph_expand),
        )
        if len(neighbors) < graph_expand:
            seen_neighbor_ids = {int(item["neighbor_work_id"]) for item in neighbors}
            similarity_neighbors = graph_neighbors(
                conn,
                work_id=int(seed["id"]),
                edge_kind="similarity",
                collection=collection,
                limit=max(1, graph_expand * 2),
                similarity_model=model,
            )
            for row in similarity_neighbors:
                neighbor_id = int(row["neighbor_work_id"])
                if neighbor_id in seen_neighbor_ids:
                    continue
                neighbors.append(row)
                seen_neighbor_ids.add(neighbor_id)
                if len(neighbors) >= graph_expand:
                    break
        for neighbor_rank, row in enumerate(neighbors, start=1):
            neighbor_id = int(row["neighbor_work_id"])
            entry = fused.setdefault(
                neighbor_id,
                {
                    "id": neighbor_id,
                    "hybrid_score": 0.0,
                },
            )
            support_scale = 1.0 if seed.get("bm25_rank") is not None else max(
                0.25,
                float(seed.get("query_group_coverage") or 0.0),
            )
            edge_scale = 1.0
            if str(row.get("edge_kind") or "") == "similarity":
                edge_scale = max(0.2, float(row.get("score") or 0.0))
            boost = support_scale * edge_scale * 0.5 / float(RRF_K + seed_rank + neighbor_rank)
            entry["hybrid_score"] += boost
            entry["graph_score"] = float(entry.get("graph_score") or 0.0) + boost
            entry["graph_votes"] = int(entry.get("graph_votes") or 0) + 1
            entry["graph_shared_max"] = max(
                int(entry.get("graph_shared_max") or 0),
                int(row.get("shared_count") or 0),
            )


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------

def search_works_vector(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_VECTOR_MODEL,
) -> list[dict[str, Any]]:
    profile = analyze_query(query)
    index = _load_local_index(target="works", model=model)
    if index.ids.size == 0:
        return []

    scores = _score_query(query, index=index, model=model)
    ranked_ids, ranked_scores = _rank_scores(
        ids=index.ids,
        scores=scores,
        limit=limit,
        allowed_ids=_allowed_work_ids(conn, collection=collection),
    )
    if not ranked_ids:
        return []

    row_map = _fetch_work_rows(conn, ranked_ids)
    out: list[dict[str, Any]] = []
    for rank, (work_id, score) in enumerate(zip(ranked_ids, ranked_scores), start=1):
        row = row_map.get(int(work_id))
        if row is None:
            continue
        item = dict(row)
        item["score"] = float(score)
        item["rank"] = rank
        item["search_type"] = "vector"
        out.append(item)
    out = _annotate_query_agreement(out, profile=profile, text_fields=("raw_title", "abstract"))
    out = _postprocess_vector_rows(out, profile=profile)
    return out


def search_chunks_vector(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_VECTOR_MODEL,
) -> list[dict[str, Any]]:
    profile = analyze_query(query)
    index = _load_local_index(target="chunks", model=model)
    if index.ids.size == 0:
        return []

    scores = _score_query(query, index=index, model=model)
    ranked_ids, ranked_scores = _rank_scores(
        ids=index.ids,
        scores=scores,
        limit=limit,
        allowed_ids=_allowed_chunk_ids(conn, collection=collection),
    )
    if not ranked_ids:
        return []

    row_map = _fetch_chunk_rows(conn, ranked_ids)
    out: list[dict[str, Any]] = []
    for rank, (chunk_id, score) in enumerate(zip(ranked_ids, ranked_scores), start=1):
        row = row_map.get(int(chunk_id))
        if row is None:
            continue
        item = dict(row)
        item["score"] = float(score)
        item["rank"] = rank
        item["search_type"] = "vector"
        out.append(item)
    out = _annotate_query_agreement(out, profile=profile, text_fields=("raw_title", "section_hint", "clean_text"))
    out = _postprocess_vector_rows(out, profile=profile)
    return out


def search_works_hybrid(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_VECTOR_MODEL,
    graph_expand: int = 0,
    seed_limit: int = 5,
) -> list[dict[str, Any]]:
    profile = analyze_query(query)
    bm25_rows = search_works_bm25(conn, query=query, collection=collection, limit=max(limit * 3, 50))
    bm25_rows = _annotate_query_agreement(
        bm25_rows,
        profile=profile,
        text_fields=("raw_title", "indexed_abstract", "indexed_collaborations"),
    )
    vector_rows = search_works_vector(conn, query=query, collection=collection, limit=max(limit * 3, 50), model=model)
    vector_rows = _annotate_query_agreement(
        vector_rows,
        profile=profile,
        text_fields=("raw_title", "abstract"),
    )
    vector_rows = _filter_vector_rows(
        vector_rows,
        profile=profile,
        fallback_if_empty=not bm25_rows,
    )
    fused = _rrf_merge(
        key_field="work_id",
        named_rows=[
            ("bm25", bm25_rows, 1.25 if bm25_rows else 1.0),
            ("vector", vector_rows, 0.75 if bm25_rows else 1.0),
        ],
    )
    if graph_expand > 0 and fused:
        _inject_graph_support(
            conn,
            fused,
            profile=profile,
            collection=collection,
            graph_expand=graph_expand,
            seed_limit=seed_limit,
            model=model,
        )

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -int(item.get("query_group_hits") or 0),
            -float(item.get("query_group_coverage") or 0.0),
            -float(item["hybrid_score"]),
            -float(item.get("graph_score") or 0.0),
            item["id"],
        ),
    )[:limit]
    row_map = _fetch_work_rows(conn, [int(item["id"]) for item in ordered])
    out: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered, start=1):
        row = row_map.get(int(item["id"]))
        if row is None:
            continue
        payload = dict(row)
        payload.update(
            {
                "hybrid_score": float(item["hybrid_score"]),
                "bm25_rank": item.get("bm25_rank"),
                "vector_rank": item.get("vector_rank"),
                "graph_score": float(item.get("graph_score") or 0.0),
                "graph_votes": int(item.get("graph_votes") or 0),
                "graph_shared_max": int(item.get("graph_shared_max") or 0),
                "query_group_hits": int(item.get("query_group_hits") or 0),
                "query_group_coverage": float(item.get("query_group_coverage") or 0.0),
                "rank": rank,
                "search_type": "hybrid",
            }
        )
        out.append(payload)
    return out


def search_chunks_hybrid(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_VECTOR_MODEL,
) -> list[dict[str, Any]]:
    profile = analyze_query(query)
    bm25_rows = search_chunks_bm25(conn, query=query, collection=collection, limit=max(limit * 3, 50))
    bm25_rows = _annotate_query_agreement(
        bm25_rows,
        profile=profile,
        text_fields=("raw_title", "section_hint", "clean_text"),
    )
    vector_rows = search_chunks_vector(conn, query=query, collection=collection, limit=max(limit * 3, 50), model=model)
    vector_rows = _annotate_query_agreement(
        vector_rows,
        profile=profile,
        text_fields=("raw_title", "section_hint", "clean_text"),
    )
    vector_rows = _filter_vector_rows(
        vector_rows,
        profile=profile,
        fallback_if_empty=not bm25_rows,
    )
    fused = _rrf_merge(
        key_field="chunk_id",
        named_rows=[
            ("bm25", bm25_rows, 1.15 if bm25_rows else 1.0),
            ("vector", vector_rows, 0.85 if bm25_rows else 1.0),
        ],
    )
    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["hybrid_score"]),
            -int(item.get("query_group_hits") or 0),
            -float(item.get("query_group_coverage") or 0.0),
            item["id"],
        ),
    )[:limit]
    row_map = _fetch_chunk_rows(conn, [int(item["id"]) for item in ordered])
    out: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered, start=1):
        row = row_map.get(int(item["id"]))
        if row is None:
            continue
        payload = dict(row)
        payload.update(
            {
                "hybrid_score": float(item["hybrid_score"]),
                "bm25_rank": item.get("bm25_rank"),
                "vector_rank": item.get("vector_rank"),
                "query_group_hits": int(item.get("query_group_hits") or 0),
                "query_group_coverage": float(item.get("query_group_coverage") or 0.0),
                "rank": rank,
                "search_type": "hybrid",
            }
        )
        out.append(payload)
    return out


def route_query(query: str) -> dict[str, Any]:
    profile = analyze_query(query)
    if is_relation_query(query):
        return {
            "target": "works",
            "graph_expand": 5,
            "reasons": [f"matched:{pattern}" for pattern in profile.relation_patterns[:4]],
        }
    hep_concepts = set(profile.concept_names) & {"cms", "atlas", "vbs", "same_sign_ww", "higgs", "pseudoscalar"}
    if is_result_query(query) and (hep_concepts or len(profile.content_tokens) >= 2):
        return {
            "target": "works",
            "graph_expand": 3,
            "reasons": ["matched:result_summary", *[f"concept:{name}" for name in sorted(hep_concepts)[:3]]],
        }
    return {
        "target": "chunks",
        "graph_expand": 0,
        "reasons": ["default:evidence_lookup"],
    }
