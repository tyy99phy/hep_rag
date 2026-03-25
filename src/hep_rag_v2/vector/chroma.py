from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hep_rag_v2 import paths
from hep_rag_v2.query import analyze_query
from .embedding import (
    DEFAULT_VECTOR_MODEL,
    LocalIndex,
    _query_vector,
    _validate_model,
)
from .index import _load_local_index
from .search import (
    _allowed_chunk_ids,
    _allowed_work_ids,
    _annotate_query_agreement,
    _fetch_chunk_rows,
    _fetch_work_rows,
    _postprocess_vector_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "vector"


def _chroma_collection_name(*, target: str, model: str, collection: str | None) -> str:
    scope = collection or "all"
    return f"hep_rag_v2__{_safe_stem(scope)}__{target}__{_safe_stem(model)}"


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


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chroma search
# ---------------------------------------------------------------------------

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
