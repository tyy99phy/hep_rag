from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hep_rag_v2 import paths
from hep_rag_v2.physics import chunk_physics_text_map, work_physics_text_map
from .embedding import (
    DEFAULT_VECTOR_DIM,
    DEFAULT_VECTOR_MODEL,
    LocalIndex,
    _aggregate_text_map,
    _embed_corpus,
    _safe_stem,
    _validate_model,
)


ProgressCallback = Callable[[str], None] | None
_LOCAL_INDEX_CACHE_LOCK = threading.Lock()
_LOCAL_INDEX_CACHE: dict[tuple[str, int, int], LocalIndex] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    force: bool = True,
    progress: ProgressCallback = None,
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
        work_source_count = int(conn.execute("SELECT COUNT(*) FROM works").fetchone()[0])
        _emit_progress(progress, f"building work vector index for {work_source_count} works with model={model}...")
        work_path, work_count, work_dim = _rebuild_work_vector_index(conn, model=model, dim=dim, force=force)
        summary["works"] = work_count
        summary["dim"] = work_dim
        summary["work_vector_path"] = str(work_path)
        _emit_progress(progress, f"work vector index ready: {work_count} works -> {work_path.name}")
    if target in {"all", "chunks"}:
        chunk_source_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE is_retrievable = 1
                  AND COALESCE(clean_text, '') <> ''
                """
            ).fetchone()[0]
        )
        _emit_progress(progress, f"building chunk vector index for {chunk_source_count} chunks with model={model}...")
        chunk_path, chunk_count, chunk_dim = _rebuild_chunk_vector_index(conn, model=model, dim=dim, force=force)
        summary["chunks"] = chunk_count
        summary["dim"] = chunk_dim
        summary["chunk_vector_path"] = str(chunk_path)
        _emit_progress(progress, f"chunk vector index ready: {chunk_count} chunks -> {chunk_path.name}")
    return summary


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is None:
        return
    text = str(message or "").strip()
    if text:
        progress(text)


# ---------------------------------------------------------------------------
# Write / load index
# ---------------------------------------------------------------------------

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
    _invalidate_local_index_cache(matrix_path)
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
    stat = matrix_path.stat()
    cache_key = (str(matrix_path), int(stat.st_mtime_ns), int(stat.st_size))
    with _LOCAL_INDEX_CACHE_LOCK:
        cached = _LOCAL_INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached
    with np.load(matrix_path, allow_pickle=False) as payload:
        ids = np.asarray(payload["ids"], dtype=np.int64)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        extras = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key not in {"ids", "vectors"}
        }
    index = LocalIndex(ids=ids, vectors=vectors, extras=extras)
    with _LOCAL_INDEX_CACHE_LOCK:
        _LOCAL_INDEX_CACHE.pop(cache_key, None)
        _clear_stale_cache_entries(str(matrix_path))
        _LOCAL_INDEX_CACHE[cache_key] = index
    return index


def _invalidate_local_index_cache(matrix_path: Path) -> None:
    with _LOCAL_INDEX_CACHE_LOCK:
        _clear_stale_cache_entries(str(matrix_path))


def _clear_stale_cache_entries(path_str: str) -> None:
    stale_keys = [key for key in _LOCAL_INDEX_CACHE if key[0] == path_str]
    for key in stale_keys:
        _LOCAL_INDEX_CACHE.pop(key, None)


# ---------------------------------------------------------------------------
# Private rebuild helpers
# ---------------------------------------------------------------------------

def _rebuild_work_vector_index(conn: sqlite3.Connection, *, model: str, dim: int, force: bool = True) -> tuple[Path, int, int]:
    if not force:
        existing = conn.execute(
            "SELECT COUNT(DISTINCT work_id) FROM work_embeddings WHERE embedding_model = ?", (model,)
        ).fetchone()
        source = conn.execute("SELECT COUNT(*) FROM works").fetchone()
        if existing and source and int(existing[0]) == int(source[0]) and int(source[0]) > 0:
            row = conn.execute(
                "SELECT vector_path, dim FROM work_embeddings WHERE embedding_model = ? AND vector_path IS NOT NULL LIMIT 1",
                (model,),
            ).fetchone()
            if row and row["vector_path"]:
                return Path(row["vector_path"]), int(existing[0]), int(row["dim"] or dim)
    topics = _aggregate_text_map(
        conn,
        """
        SELECT wt.work_id AS owner_id, t.label AS value
        FROM work_topics wt
        JOIN topics t ON t.topic_id = wt.topic_id
        ORDER BY wt.work_id, t.label
        """,
    )
    physics = work_physics_text_map(conn)
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
                    physics.get(work_id, ""),
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


def _rebuild_chunk_vector_index(conn: sqlite3.Connection, *, model: str, dim: int, force: bool = True) -> tuple[Path, int, int]:
    if not force:
        existing = conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM chunk_embeddings WHERE embedding_model = ?", (model,)
        ).fetchone()
        source = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_retrievable = 1 AND COALESCE(clean_text, '') <> ''"
        ).fetchone()
        if existing and source and int(existing[0]) == int(source[0]) and int(source[0]) > 0:
            row = conn.execute(
                "SELECT vector_path, dim FROM chunk_embeddings WHERE embedding_model = ? AND vector_path IS NOT NULL LIMIT 1",
                (model,),
            ).fetchone()
            if row and row["vector_path"]:
                return Path(row["vector_path"]), int(existing[0]), int(row["dim"] or dim)
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
    physics = chunk_physics_text_map(conn)

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
                    physics.get(int(row["chunk_id"]), ""),
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
