from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hep_rag_v2 import paths
from hep_rag_v2.graph import graph_neighbors
from hep_rag_v2.query import analyze_query, is_relation_query, query_match_stats, rewrite_query_for_embedding
from hep_rag_v2.search import search_chunks_bm25, search_works_bm25
from hep_rag_v2.textnorm import normalize_search_text


HASH_VECTOR_MODEL = "hash-v1"
HASH_IDF_VECTOR_MODEL = "hash-idf-v1"
DEFAULT_VECTOR_MODEL = HASH_IDF_VECTOR_MODEL
DEFAULT_VECTOR_DIM = 768
RRF_K = 60
EMBEDDING_STOPWORDS = {
    "search",
    "searches",
    "observation",
    "observations",
    "measurement",
    "measurements",
    "result",
    "results",
    "study",
    "studies",
    "paper",
    "presents",
    "presented",
    "reported",
    "report",
    "collision",
    "collisions",
    "proton",
    "pp",
    "tev",
    "sqrt",
    "final",
    "state",
    "states",
}


@dataclass(frozen=True)
class LocalIndex:
    ids: np.ndarray
    vectors: np.ndarray
    extras: dict[str, np.ndarray]


def vector_index_counts(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(DISTINCT work_id) FROM work_embeddings) AS work_embeddings,
          (SELECT COUNT(*) FROM work_embeddings) AS work_embedding_rows,
          (SELECT COUNT(DISTINCT chunk_id) FROM chunk_embeddings) AS chunk_embeddings,
          (SELECT COUNT(*) FROM chunk_embeddings) AS chunk_embedding_rows
        """
    ).fetchone()
    return {
        "work_embeddings": int(row["work_embeddings"] if row is not None else 0),
        "work_embedding_rows": int(row["work_embedding_rows"] if row is not None else 0),
        "chunk_embeddings": int(row["chunk_embeddings"] if row is not None else 0),
        "chunk_embedding_rows": int(row["chunk_embedding_rows"] if row is not None else 0),
    }


def rebuild_vector_indices(
    conn: sqlite3.Connection,
    *,
    target: str = "all",
    model: str = DEFAULT_VECTOR_MODEL,
    dim: int = DEFAULT_VECTOR_DIM,
) -> dict[str, Any]:
    _validate_model(model)
    if dim <= 0:
        raise ValueError(f"Vector dim must be positive, got: {dim}")

    summary: dict[str, Any] = {
        "target": target,
        "model": model,
        "dim": dim,
        "works": 0,
        "chunks": 0,
        "work_vector_path": None,
        "chunk_vector_path": None,
    }

    paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_dir = paths.INDEX_DIR / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    if target in {"all", "works"}:
        work_path, work_count, work_dim = _rebuild_work_vector_index(conn, model=model, dim=dim)
        summary["works"] = work_count
        summary["dim"] = work_dim
        summary["work_vector_path"] = str(work_path)
    if target in {"all", "chunks"}:
        chunk_path, chunk_count, chunk_dim = _rebuild_chunk_vector_index(conn, model=model, dim=dim)
        summary["chunks"] = chunk_count
        summary["dim"] = chunk_dim
        summary["chunk_vector_path"] = str(chunk_path)
    return summary


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
    out = _annotate_query_agreement(out, profile=profile, text_fields=("raw_title",))
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


def sync_chroma_indices(
    conn: sqlite3.Connection,
    *,
    target: str = "all",
    model: str = DEFAULT_VECTOR_MODEL,
    collection: str | None = None,
    chroma_dir: Path | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    _validate_model(model)
    out_dir = chroma_dir or (paths.INDEX_DIR / "chroma")
    summary: dict[str, Any] = {
        "target": target,
        "model": model,
        "collection": collection,
        "chroma_dir": str(out_dir),
        "works": 0,
        "chunks": 0,
        "collections": [],
    }
    if target in {"all", "works"}:
        info = _sync_chroma_target(
            conn,
            target="works",
            model=model,
            collection=collection,
            chroma_dir=out_dir,
            batch_size=batch_size,
        )
        summary["works"] = int(info["count"])
        summary["collections"].append(info["collection_name"])
    if target in {"all", "chunks"}:
        info = _sync_chroma_target(
            conn,
            target="chunks",
            model=model,
            collection=collection,
            chroma_dir=out_dir,
            batch_size=batch_size,
        )
        summary["chunks"] = int(info["count"])
        summary["collections"].append(info["collection_name"])
    return summary


def search_works_vector_chroma(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_VECTOR_MODEL,
    chroma_dir: Path | None = None,
) -> list[dict[str, Any]]:
    index = _load_local_index(target="works", model=model)
    return _search_vector_chroma(
        conn,
        target="works",
        query=query,
        collection=collection,
        limit=limit,
        model=model,
        chroma_dir=chroma_dir,
        index=index,
    )


def search_chunks_vector_chroma(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_VECTOR_MODEL,
    chroma_dir: Path | None = None,
) -> list[dict[str, Any]]:
    index = _load_local_index(target="chunks", model=model)
    return _search_vector_chroma(
        conn,
        target="chunks",
        query=query,
        collection=collection,
        limit=limit,
        model=model,
        chroma_dir=chroma_dir,
        index=index,
    )


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
        text_fields=("raw_title",),
    )
    vector_rows = search_works_vector(conn, query=query, collection=collection, limit=max(limit * 3, 50), model=model)
    vector_rows = _annotate_query_agreement(
        vector_rows,
        profile=profile,
        text_fields=("raw_title",),
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
            -float(item["hybrid_score"]),
            -int(item.get("query_group_hits") or 0),
            -float(item.get("query_group_coverage") or 0.0),
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
    return {
        "target": "chunks",
        "graph_expand": 0,
        "reasons": ["default:evidence_lookup"],
    }


def _rebuild_work_vector_index(conn: sqlite3.Connection, *, model: str, dim: int) -> tuple[Path, int, int]:
    topics = _aggregate_text_map(
        conn,
        """
        SELECT wt.work_id AS owner_id, t.label AS value
        FROM work_topics wt
        JOIN topics t ON t.topic_id = wt.topic_id
        ORDER BY wt.work_id, t.label
        """,
    )
    rows = conn.execute(
        """
        SELECT work_id, title, abstract, year
        FROM works
        ORDER BY work_id
        """
    ).fetchall()

    ids: list[int] = []
    texts: list[str] = []
    for row in rows:
        work_id = int(row["work_id"])
        ids.append(work_id)
        texts.append(
            "\n".join(
                part
                for part in (
                    str(row["title"] or ""),
                    str(row["title"] or ""),
                    str(row["abstract"] or ""),
                    topics.get(work_id, ""),
                )
                if str(part).strip()
            )
        )

    vectors, extras, actual_dim = _embed_corpus(texts, model=model, dim=dim)
    matrix_path = _write_local_index(
        target="works",
        model=model,
        ids=ids,
        vectors=vectors,
        dim=actual_dim,
        extras=extras,
    )
    conn.execute("DELETE FROM work_embeddings WHERE embedding_model = ?", (model,))
    conn.executemany(
        """
        INSERT INTO work_embeddings (work_id, embedding_model, vector_path, dim)
        VALUES (?, ?, ?, ?)
        """,
        [(work_id, model, str(matrix_path), actual_dim) for work_id in ids],
    )
    return matrix_path, len(ids), actual_dim


def _rebuild_chunk_vector_index(conn: sqlite3.Connection, *, model: str, dim: int) -> tuple[Path, int, int]:
    rows = conn.execute(
        """
        SELECT
          c.chunk_id,
          c.work_id,
          COALESCE(w.title, '') AS title,
          COALESCE(c.section_hint, '') AS section_hint,
          COALESCE(c.clean_text, '') AS clean_text
        FROM chunks c
        JOIN works w ON w.work_id = c.work_id
        WHERE c.is_retrievable = 1
          AND COALESCE(c.clean_text, '') <> ''
        ORDER BY c.chunk_id
        """
    ).fetchall()

    ids: list[int] = []
    texts: list[str] = []
    for row in rows:
        ids.append(int(row["chunk_id"]))
        texts.append(
            "\n".join(
                part
                for part in (
                    str(row["title"] or ""),
                    str(row["section_hint"] or ""),
                    str(row["clean_text"] or ""),
                )
                if str(part).strip()
            )
        )

    vectors, extras, actual_dim = _embed_corpus(texts, model=model, dim=dim)
    matrix_path = _write_local_index(
        target="chunks",
        model=model,
        ids=ids,
        vectors=vectors,
        dim=actual_dim,
        extras=extras,
    )
    conn.execute("DELETE FROM chunk_embeddings WHERE embedding_model = ?", (model,))
    conn.executemany(
        """
        INSERT INTO chunk_embeddings (chunk_id, embedding_model, vector_path, dim)
        VALUES (?, ?, ?, ?)
        """,
        [(chunk_id, model, str(matrix_path), actual_dim) for chunk_id in ids],
    )
    return matrix_path, len(ids), actual_dim


def _embed_corpus(texts: list[str], *, model: str, dim: int) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if model == HASH_VECTOR_MODEL:
        vectors = _embed_corpus_hash(texts, dim=dim)
        return (vectors, {}, dim)
    if model == HASH_IDF_VECTOR_MODEL:
        vectors, bucket_idf = _embed_corpus_hash_with_idf(texts, dim=dim)
        return (vectors, {"bucket_idf": bucket_idf}, dim)
    if _is_sentence_transformer_model(model):
        vectors = _embed_corpus_sentence_transformers(texts, model=model)
        actual_dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0
        return (vectors, {}, actual_dim)
    raise ValueError(f"Unsupported vector model: {model}")


def _embed_corpus_hash(texts: list[str], *, dim: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts):
        matrix[idx] = _hash_embed_text(text, dim=dim)
    return matrix


def _embed_corpus_hash_with_idf(texts: list[str], *, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not texts:
        return (
            np.zeros((0, dim), dtype=np.float32),
            np.ones((dim,), dtype=np.float32),
        )

    doc_count = len(texts)
    doc_freq = np.zeros((dim,), dtype=np.float32)
    tokenized: list[list[str]] = []
    for text in texts:
        tokens = _embedding_tokens(text)
        tokenized.append(tokens)
        seen_buckets: set[int] = set()
        for feature in _hash_features(tokens):
            index, _sign = _hashed_feature(feature, dim=dim)
            seen_buckets.add(index)
        for index in seen_buckets:
            doc_freq[index] += 1.0

    bucket_idf = np.log((1.0 + float(doc_count)) / (1.0 + doc_freq)) + 1.0
    matrix = np.zeros((doc_count, dim), dtype=np.float32)
    for idx, tokens in enumerate(tokenized):
        matrix[idx] = _hash_embed_tokens(tokens, dim=dim, bucket_idf=bucket_idf)
    return (matrix, bucket_idf.astype(np.float32))


def _hash_embed_text(text: str, *, dim: int, bucket_idf: np.ndarray | None = None) -> np.ndarray:
    tokens = _embedding_tokens(text)
    return _hash_embed_tokens(tokens, dim=dim, bucket_idf=bucket_idf)


def _hash_embed_tokens(tokens: list[str], *, dim: int, bucket_idf: np.ndarray | None = None) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vector

    counts: dict[str, float] = defaultdict(float)
    for feature in _hash_features(tokens):
        counts[feature] += 1.0 if "__" not in feature else 0.5

    for feature, weight in counts.items():
        index, sign = _hashed_feature(feature, dim=dim)
        if bucket_idf is not None and bucket_idf.shape[0] == dim:
            weight *= float(bucket_idf[index])
        vector[index] += float(weight) * sign

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _embedding_tokens(text: str) -> list[str]:
    normalized = normalize_search_text(text)
    out: list[str] = []
    for token in re.findall(r"\w+", normalized, flags=re.UNICODE):
        lowered = token.casefold()
        if lowered in EMBEDDING_STOPWORDS:
            continue
        out.append(lowered)
    return out


def _hash_features(tokens: list[str]) -> list[str]:
    features = list(tokens)
    features.extend(f"{left}__{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def _hashed_feature(feature: str, *, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
    index = int.from_bytes(digest[:8], "little", signed=False) % dim
    sign = 1.0 if (digest[8] & 1) else -1.0
    return (index, sign)


def _score_query(query: str, *, index: LocalIndex, model: str) -> np.ndarray:
    if index.vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_vector = _query_vector(query, index=index, model=model)
    if not float(np.linalg.norm(query_vector)):
        return np.zeros((index.vectors.shape[0],), dtype=np.float32)
    return index.vectors @ query_vector


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


def _write_local_index(
    *,
    target: str,
    model: str,
    ids: list[int],
    vectors: np.ndarray,
    dim: int,
    extras: dict[str, np.ndarray] | None = None,
) -> Path:
    out_dir = paths.INDEX_DIR / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{target}_{_safe_stem(model)}"
    matrix_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.json"
    payload: dict[str, np.ndarray] = {
        "ids": np.asarray(ids, dtype=np.int64),
        "vectors": np.asarray(vectors, dtype=np.float32),
    }
    for key, value in (extras or {}).items():
        payload[key] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(matrix_path, **payload)
    meta_path.write_text(
        json.dumps(
            {
                "target": target,
                "model": model,
                "dim": dim,
                "count": len(ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return matrix_path


def _load_local_index(*, target: str, model: str) -> LocalIndex:
    matrix_path = paths.INDEX_DIR / "vectors" / f"{target}_{_safe_stem(model)}.npz"
    if not matrix_path.exists():
        return LocalIndex(
            ids=np.zeros((0,), dtype=np.int64),
            vectors=np.zeros((0, DEFAULT_VECTOR_DIM), dtype=np.float32),
            extras={},
        )
    with np.load(matrix_path, allow_pickle=False) as payload:
        ids = np.asarray(payload["ids"], dtype=np.int64)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        extras = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key not in {"ids", "vectors"}
        }
    return LocalIndex(ids=ids, vectors=vectors, extras=extras)


def _sync_chroma_target(
    conn: sqlite3.Connection,
    *,
    target: str,
    model: str,
    collection: str | None,
    chroma_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    index = _load_local_index(target=target, model=model)
    if index.ids.size == 0:
        raise RuntimeError(f"Local vector index not found for target={target} model={model}. Build it first.")

    if target == "works":
        allowed_ids = _allowed_work_ids(conn, collection=collection)
        row_map = _fetch_work_rows(conn, [int(item) for item in index.ids.tolist()])
    else:
        allowed_ids = _allowed_chunk_ids(conn, collection=collection)
        row_map = _fetch_chunk_rows(conn, [int(item) for item in index.ids.tolist()])

    mask = np.ones(index.ids.shape[0], dtype=bool)
    if allowed_ids is not None:
        allowed = np.fromiter((int(item) for item in allowed_ids), dtype=np.int64, count=len(allowed_ids))
        mask &= np.isin(index.ids, allowed)
    ids = index.ids[mask]
    vectors = index.vectors[mask]

    client = _get_chroma_client(chroma_dir)
    collection_name = _chroma_collection_name(target=target, model=model, collection=collection)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    chroma_collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, int(ids.shape[0]), max(1, batch_size)):
        end = start + max(1, batch_size)
        batch_ids = [int(item) for item in ids[start:end].tolist()]
        batch_vectors = vectors[start:end]
        metadatas, documents = _chroma_payload_rows(target=target, row_map=row_map, ids=batch_ids)
        chroma_collection.upsert(
            ids=[str(item) for item in batch_ids],
            embeddings=batch_vectors.tolist(),
            metadatas=metadatas,
            documents=documents,
        )

    return {
        "collection_name": collection_name,
        "count": int(ids.shape[0]),
    }


def _search_vector_chroma(
    conn: sqlite3.Connection,
    *,
    target: str,
    query: str,
    collection: str | None,
    limit: int,
    model: str,
    chroma_dir: Path | None,
    index: LocalIndex,
) -> list[dict[str, Any]]:
    if index.ids.size == 0:
        return []

    client = _get_chroma_client(chroma_dir or (paths.INDEX_DIR / "chroma"))
    collection_name = _chroma_collection_name(target=target, model=model, collection=collection)
    try:
        chroma_collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Chroma collection not found: {collection_name}. Run sync-chroma-index first."
        ) from exc

    query_vector = _query_vector(query, index=index, model=model)
    if not float(np.linalg.norm(query_vector)):
        return []

    raw = chroma_collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=limit,
        include=["distances"],
    )
    ids = [int(item) for item in ((raw.get("ids") or [[]])[0] or [])]
    distances = [float(item) for item in ((raw.get("distances") or [[]])[0] or [])]
    if not ids:
        return []

    if target == "works":
        row_map = _fetch_work_rows(conn, ids)
        rows = []
        for rank, (item_id, distance) in enumerate(zip(ids, distances), start=1):
            row = row_map.get(int(item_id))
            if row is None:
                continue
            payload = dict(row)
            payload["score"] = 1.0 - float(distance)
            payload["distance"] = float(distance)
            payload["rank"] = rank
            payload["search_type"] = "vector_chroma"
            rows.append(payload)
        profile = analyze_query(query)
        rows = _annotate_query_agreement(rows, profile=profile, text_fields=("raw_title",))
        return _postprocess_vector_rows(rows, profile=profile)

    row_map = _fetch_chunk_rows(conn, ids)
    rows = []
    for rank, (item_id, distance) in enumerate(zip(ids, distances), start=1):
        row = row_map.get(int(item_id))
        if row is None:
            continue
        payload = dict(row)
        payload["score"] = 1.0 - float(distance)
        payload["distance"] = float(distance)
        payload["rank"] = rank
        payload["search_type"] = "vector_chroma"
        rows.append(payload)
    profile = analyze_query(query)
    rows = _annotate_query_agreement(rows, profile=profile, text_fields=("raw_title", "section_hint", "clean_text"))
    return _postprocess_vector_rows(rows, profile=profile)


def _chroma_payload_rows(
    *,
    target: str,
    row_map: dict[int, dict[str, Any]],
    ids: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    metadatas: list[dict[str, Any]] = []
    documents: list[str] = []
    for item_id in ids:
        row = row_map.get(int(item_id)) or {}
        if target == "works":
            metadatas.append(
                {
                    "kind": "work",
                    "work_id": int(item_id),
                    "year": int(row.get("year") or 0),
                    "canonical_source": str(row.get("canonical_source") or ""),
                    "canonical_id": str(row.get("canonical_id") or ""),
                }
            )
            documents.append(str(row.get("raw_title") or ""))
        else:
            metadatas.append(
                {
                    "kind": "chunk",
                    "chunk_id": int(item_id),
                    "work_id": int(row.get("work_id") or 0),
                    "chunk_role": str(row.get("chunk_role") or ""),
                    "page_hint": str(row.get("page_hint") or ""),
                }
            )
            documents.append(str(row.get("clean_text") or ""))
    return (metadatas, documents)


def _chroma_collection_name(*, target: str, model: str, collection: str | None) -> str:
    scope = collection or "all"
    return f"hep_rag_v2__{_safe_stem(scope)}__{target}__{_safe_stem(model)}"


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
        SELECT w.work_id, w.title AS raw_title, w.year, w.canonical_source, w.canonical_id
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


def _aggregate_text_map(conn: sqlite3.Connection, query: str) -> dict[int, str]:
    mapping: dict[int, list[str]] = defaultdict(list)
    for row in conn.execute(query):
        owner_id = int(row["owner_id"])
        value = str(row["value"] or "").strip()
        if not value:
            continue
        if value in mapping[owner_id]:
            continue
        mapping[owner_id].append(value)
    return {
        owner_id: " ".join(values)
        for owner_id, values in mapping.items()
    }


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "vector"


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


def _validate_model(model: str) -> None:
    if model in {HASH_VECTOR_MODEL, HASH_IDF_VECTOR_MODEL}:
        return
    if _is_sentence_transformer_model(model):
        return
    raise ValueError(f"Unsupported vector model: {model}")


def _is_sentence_transformer_model(model: str) -> bool:
    return model.startswith("st:") or model.startswith("sentence-transformers:")


def _sentence_transformer_name(model: str) -> str:
    prefix, _, value = model.partition(":")
    if prefix not in {"st", "sentence-transformers"} or not value.strip():
        raise ValueError(f"Unsupported sentence-transformers model spec: {model}")
    return value.strip()


def _embed_corpus_sentence_transformers(texts: list[str], *, model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    encoder = _get_sentence_transformer(_sentence_transformer_name(model))
    vectors = encoder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _embed_query_sentence_transformers(query: str, *, model: str) -> np.ndarray:
    vectors = _embed_corpus_sentence_transformers([query], model=model)
    if vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(vectors[0], dtype=np.float32)


def _query_vector(query: str, *, index: LocalIndex, model: str) -> np.ndarray:
    rewritten_query = rewrite_query_for_embedding(query)
    if model == HASH_VECTOR_MODEL:
        return _hash_embed_text(rewritten_query, dim=int(index.vectors.shape[1]))
    if model == HASH_IDF_VECTOR_MODEL:
        return _hash_embed_text(
            rewritten_query,
            dim=int(index.vectors.shape[1]),
            bucket_idf=index.extras.get("bucket_idf"),
        )
    if _is_sentence_transformer_model(model):
        return _embed_query_sentence_transformers(rewritten_query, model=model)
    raise ValueError(f"Unsupported vector model: {model}")


@lru_cache(maxsize=2)
def _get_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -e .[embeddings]` to use this model."
        ) from exc
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _get_chroma_client(chroma_dir: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Run `pip install -e .[vectorstore]` to use the Chroma backend."
        ) from exc
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))
