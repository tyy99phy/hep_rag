from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from typing import Any

from hep_rag_v2.query import build_match_queries
from hep_rag_v2.textnorm import normalize_search_text


WORK_SEARCH_TABLE = "work_search"
CHUNK_SEARCH_TABLE = "chunk_search"
FORMULA_SEARCH_TABLE = "formula_search"
ASSET_SEARCH_TABLE = "asset_search"


def ensure_search_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {WORK_SEARCH_TABLE} USING fts5(
          work_id UNINDEXED,
          collections UNINDEXED,
          year UNINDEXED,
          canonical_source UNINDEXED,
          canonical_id UNINDEXED,
          title,
          abstract,
          topics,
          authors,
          collaborations,
          venues,
          identifiers,
          tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {CHUNK_SEARCH_TABLE} USING fts5(
          chunk_id UNINDEXED,
          work_id UNINDEXED,
          collections UNINDEXED,
          chunk_role UNINDEXED,
          section_hint UNINDEXED,
          page_hint UNINDEXED,
          title,
          section,
          body,
          tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {FORMULA_SEARCH_TABLE} USING fts5(
          formula_id UNINDEXED,
          work_id UNINDEXED,
          collections UNINDEXED,
          page UNINDEXED,
          section_hint UNINDEXED,
          title,
          section,
          latex,
          context,
          tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {ASSET_SEARCH_TABLE} USING fts5(
          asset_id UNINDEXED,
          work_id UNINDEXED,
          collections UNINDEXED,
          asset_type UNINDEXED,
          page UNINDEXED,
          section_hint UNINDEXED,
          title,
          section,
          caption,
          context,
          tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )


def search_index_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in (
        WORK_SEARCH_TABLE,
        CHUNK_SEARCH_TABLE,
        FORMULA_SEARCH_TABLE,
        ASSET_SEARCH_TABLE,
    ):
        if not _table_exists(conn, table):
            counts[table] = 0
            continue
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = int(row[0] if row is not None else 0)
    return counts


def rebuild_search_indices(conn: sqlite3.Connection, *, target: str = "all", force: bool = True) -> dict[str, int]:
    ensure_search_schema(conn)
    summary = {
        "works": 0,
        "chunks": 0,
        "formulas": 0,
        "assets": 0,
    }
    if target in {"all", "works"}:
        summary["works"] = rebuild_work_search_index(conn, force=force)
    if target in {"all", "chunks"}:
        summary["chunks"] = rebuild_chunk_search_index(conn, force=force)
    if target in {"all", "formulas", "structure"}:
        summary["formulas"] = rebuild_formula_search_index(conn, force=force)
    if target in {"all", "assets", "structure"}:
        summary["assets"] = rebuild_asset_search_index(conn, force=force)
    return summary


def rebuild_work_search_index(conn: sqlite3.Connection, *, force: bool = True) -> int:
    ensure_search_schema(conn)
    if not force:
        existing = conn.execute(f"SELECT COUNT(*) FROM {WORK_SEARCH_TABLE}").fetchone()
        source = conn.execute("SELECT COUNT(*) FROM works").fetchone()
        if existing and source and int(existing[0]) == int(source[0]) and int(source[0]) > 0:
            return int(existing[0])
    conn.execute(f"DELETE FROM {WORK_SEARCH_TABLE}")

    work_rows = conn.execute(
        """
        SELECT work_id, title, abstract, year, canonical_source, canonical_id
        FROM works
        ORDER BY work_id
        """
    ).fetchall()
    collections = _collections_by_work(conn)
    topics = _aggregate_text_map(
        conn,
        """
        SELECT wt.work_id AS owner_id, t.label AS value
        FROM work_topics wt
        JOIN topics t ON t.topic_id = wt.topic_id
        ORDER BY wt.work_id, t.label
        """,
    )
    authors = _aggregate_text_map(
        conn,
        """
        SELECT wa.work_id AS owner_id, a.display_name AS value
        FROM work_authors wa
        JOIN authors a ON a.author_id = wa.author_id
        ORDER BY wa.work_id, wa.author_position
        """,
    )
    collaborations = _aggregate_text_map(
        conn,
        """
        SELECT wc.work_id AS owner_id, c.name AS value
        FROM work_collaborations wc
        JOIN collaborations c ON c.collaboration_id = wc.collaboration_id
        ORDER BY wc.work_id, c.name
        """,
    )
    venues = _aggregate_text_map(
        conn,
        """
        SELECT wv.work_id AS owner_id, v.name AS value
        FROM work_venues wv
        JOIN venues v ON v.venue_id = wv.venue_id
        ORDER BY wv.work_id, v.name
        """,
    )
    identifiers = _aggregate_text_map(
        conn,
        """
        SELECT work_id AS owner_id, id_type || ':' || id_value AS value
        FROM work_ids
        ORDER BY work_id, is_primary DESC, id_type
        """,
    )

    inserted = 0
    for row in work_rows:
        work_id = int(row["work_id"])
        payload = (
            work_id,
            str(work_id),
            collections.get(work_id, ""),
            str(row["year"] or ""),
            str(row["canonical_source"] or ""),
            str(row["canonical_id"] or ""),
            normalize_search_text(str(row["title"] or "")),
            normalize_search_text(str(row["abstract"] or "")),
            normalize_search_text(topics.get(work_id, "")),
            normalize_search_text(authors.get(work_id, "")),
            normalize_search_text(collaborations.get(work_id, "")),
            normalize_search_text(venues.get(work_id, "")),
            normalize_search_text(identifiers.get(work_id, "")),
        )
        conn.execute(
            f"""
            INSERT INTO {WORK_SEARCH_TABLE} (
              rowid, work_id, collections, year, canonical_source, canonical_id,
              title, abstract, topics, authors, collaborations, venues, identifiers
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted += 1
    return inserted


def rebuild_chunk_search_index(conn: sqlite3.Connection, *, force: bool = True) -> int:
    ensure_search_schema(conn)
    if not force:
        existing = conn.execute(f"SELECT COUNT(*) FROM {CHUNK_SEARCH_TABLE}").fetchone()
        source = conn.execute("SELECT COUNT(*) FROM chunks WHERE is_retrievable = 1 AND COALESCE(clean_text, '') <> ''").fetchone()
        if existing and source and int(existing[0]) == int(source[0]) and int(source[0]) > 0:
            return int(existing[0])
    conn.execute(f"DELETE FROM {CHUNK_SEARCH_TABLE}")

    collections = _collections_by_work(conn)
    chunk_rows = conn.execute(
        """
        SELECT c.chunk_id, c.work_id, c.chunk_role, c.section_hint, c.page_hint, c.clean_text, w.title
        FROM chunks c
        JOIN works w ON w.work_id = c.work_id
        WHERE c.is_retrievable = 1
          AND COALESCE(c.clean_text, '') <> ''
        ORDER BY c.chunk_id
        """
    ).fetchall()

    inserted = 0
    for row in chunk_rows:
        work_id = int(row["work_id"])
        chunk_id = int(row["chunk_id"])
        payload = (
            chunk_id,
            str(chunk_id),
            str(work_id),
            collections.get(work_id, ""),
            str(row["chunk_role"] or ""),
            str(row["section_hint"] or ""),
            str(row["page_hint"] or ""),
            normalize_search_text(str(row["title"] or "")),
            normalize_search_text(str(row["section_hint"] or "")),
            normalize_search_text(str(row["clean_text"] or "")),
        )
        conn.execute(
            f"""
            INSERT INTO {CHUNK_SEARCH_TABLE} (
              rowid, chunk_id, work_id, collections, chunk_role, section_hint, page_hint,
              title, section, body
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted += 1
    return inserted


def rebuild_formula_search_index(conn: sqlite3.Connection, *, force: bool = True) -> int:
    ensure_search_schema(conn)
    if not force:
        existing = conn.execute(f"SELECT COUNT(*) FROM {FORMULA_SEARCH_TABLE}").fetchone()
        source = conn.execute("SELECT COUNT(*) FROM formulas").fetchone()
        if existing and source and int(existing[0]) == int(source[0]) and int(source[0]) > 0:
            return int(existing[0])
    conn.execute(f"DELETE FROM {FORMULA_SEARCH_TABLE}")

    collections = _collections_by_work(conn)
    blocks_by_document, block_positions = _load_retrievable_blocks(conn)
    rows = conn.execute(
        """
        SELECT
          f.formula_id,
          d.work_id,
          f.document_id,
          f.block_id,
          f.page,
          COALESCE(ds.path_text, '') AS section_hint,
          COALESCE(f.normalized_latex, f.latex, '') AS normalized_latex,
          w.title
        FROM formulas f
        JOIN documents d ON d.document_id = f.document_id
        JOIN works w ON w.work_id = d.work_id
        LEFT JOIN document_sections ds ON ds.section_id = f.section_id
        ORDER BY f.formula_id
        """
    ).fetchall()

    inserted = 0
    for row in rows:
        formula_id = int(row["formula_id"])
        work_id = int(row["work_id"])
        context = _block_context_text(
            blocks_by_document,
            block_positions,
            document_id=int(row["document_id"]),
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            include_self=False,
        )
        payload = (
            formula_id,
            str(formula_id),
            str(work_id),
            collections.get(work_id, ""),
            str(row["page"] or ""),
            str(row["section_hint"] or ""),
            normalize_search_text(str(row["title"] or "")),
            normalize_search_text(str(row["section_hint"] or "")),
            normalize_search_text(str(row["normalized_latex"] or "")),
            normalize_search_text(context),
        )
        conn.execute(
            f"""
            INSERT INTO {FORMULA_SEARCH_TABLE} (
              rowid, formula_id, work_id, collections, page, section_hint,
              title, section, latex, context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted += 1
    return inserted


def rebuild_asset_search_index(conn: sqlite3.Connection, *, force: bool = True) -> int:
    ensure_search_schema(conn)
    if not force:
        existing = conn.execute(f"SELECT COUNT(*) FROM {ASSET_SEARCH_TABLE}").fetchone()
        source = conn.execute("SELECT COUNT(*) FROM assets").fetchone()
        if existing and source and int(existing[0]) == int(source[0]) and int(source[0]) > 0:
            return int(existing[0])
    conn.execute(f"DELETE FROM {ASSET_SEARCH_TABLE}")

    collections = _collections_by_work(conn)
    blocks_by_document, block_positions = _load_retrievable_blocks(conn)
    rows = conn.execute(
        """
        SELECT
          a.asset_id,
          d.work_id,
          a.document_id,
          a.block_id,
          a.asset_type,
          a.page,
          COALESCE(a.caption, '') AS caption,
          COALESCE(ds.path_text, '') AS section_hint,
          w.title
        FROM assets a
        JOIN documents d ON d.document_id = a.document_id
        JOIN works w ON w.work_id = d.work_id
        LEFT JOIN document_sections ds ON ds.section_id = a.section_id
        ORDER BY a.asset_id
        """
    ).fetchall()

    inserted = 0
    for row in rows:
        asset_id = int(row["asset_id"])
        work_id = int(row["work_id"])
        context = _block_context_text(
            blocks_by_document,
            block_positions,
            document_id=int(row["document_id"]),
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            include_self=True,
        )
        payload = (
            asset_id,
            str(asset_id),
            str(work_id),
            collections.get(work_id, ""),
            str(row["asset_type"] or ""),
            str(row["page"] or ""),
            str(row["section_hint"] or ""),
            normalize_search_text(str(row["title"] or "")),
            normalize_search_text(str(row["section_hint"] or "")),
            normalize_search_text(str(row["caption"] or "")),
            normalize_search_text(context),
        )
        conn.execute(
            f"""
            INSERT INTO {ASSET_SEARCH_TABLE} (
              rowid, asset_id, work_id, collections, asset_type, page, section_hint,
              title, section, caption, context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted += 1
    return inserted


def search_works_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_search_schema(conn)
    match_queries = build_match_queries(query)
    if not match_queries:
        return []

    sql = f"""
        SELECT
          CAST(ws.work_id AS INTEGER) AS work_id,
          ws.collections,
          CAST(NULLIF(ws.year, '') AS INTEGER) AS year,
          w.canonical_source,
          w.canonical_id,
          w.title AS raw_title,
          ws.title AS indexed_title,
          ws.abstract AS indexed_abstract,
          bm25({WORK_SEARCH_TABLE}) AS score
        FROM {WORK_SEARCH_TABLE} ws
        JOIN works w ON w.work_id = CAST(ws.work_id AS INTEGER)
        WHERE {WORK_SEARCH_TABLE} MATCH ?
    """
    params_tail: list[Any] = []
    if collection:
        sql += " AND ws.collections LIKE ?"
        params_tail.append(f"%{collection}%")
    sql += " ORDER BY score ASC, year DESC, work_id DESC LIMIT ?"
    return _run_match_queries(
        conn,
        sql=sql,
        match_queries=match_queries,
        params_tail=params_tail,
        key_field="work_id",
        limit=limit,
    )


def search_chunks_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_search_schema(conn)
    match_queries = build_match_queries(query)
    if not match_queries:
        return []

    sql = f"""
        SELECT
          CAST(cs.chunk_id AS INTEGER) AS chunk_id,
          CAST(cs.work_id AS INTEGER) AS work_id,
          cs.collections,
          c.chunk_role,
          c.section_hint,
          c.page_hint,
          w.title AS raw_title,
          c.clean_text,
          bm25({CHUNK_SEARCH_TABLE}) AS score,
          CASE c.chunk_role
            WHEN 'asset_window' THEN 0.02
            WHEN 'formula_window' THEN 0.05
            ELSE 0.0
          END AS role_penalty
        FROM {CHUNK_SEARCH_TABLE} cs
        JOIN chunks c ON c.chunk_id = CAST(cs.chunk_id AS INTEGER)
        JOIN works w ON w.work_id = CAST(cs.work_id AS INTEGER)
        WHERE {CHUNK_SEARCH_TABLE} MATCH ?
    """
    params_tail: list[Any] = []
    if collection:
        sql += " AND cs.collections LIKE ?"
        params_tail.append(f"%{collection}%")
    sql += """
        ORDER BY
          score + role_penalty ASC,
          score ASC,
          CASE c.chunk_role
            WHEN 'abstract_chunk' THEN 0
            WHEN 'section_parent' THEN 1
            WHEN 'section_child' THEN 2
            WHEN 'asset_window' THEN 3
            WHEN 'formula_window' THEN 4
            ELSE 5
          END ASC,
          chunk_id ASC
        LIMIT ?
    """
    rows = _run_match_queries(
        conn,
        sql=sql,
        match_queries=match_queries,
        params_tail=params_tail,
        key_field="chunk_id",
        limit=limit,
    )
    return [{key: value for key, value in row.items() if key != "role_penalty"} for row in rows]


def search_formulas_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_search_schema(conn)
    match_queries = build_match_queries(query)
    if not match_queries:
        return []

    sql = f"""
        SELECT
          CAST(fs.formula_id AS INTEGER) AS formula_id,
          CAST(fs.work_id AS INTEGER) AS work_id,
          fs.collections,
          CAST(NULLIF(fs.page, '') AS INTEGER) AS page,
          fs.section_hint,
          w.title AS raw_title,
          COALESCE(f.normalized_latex, f.latex) AS normalized_latex,
          f.latex AS raw_latex,
          f.document_id,
          f.block_id,
          bm25({FORMULA_SEARCH_TABLE}) AS score
        FROM {FORMULA_SEARCH_TABLE} fs
        JOIN formulas f ON f.formula_id = CAST(fs.formula_id AS INTEGER)
        JOIN works w ON w.work_id = CAST(fs.work_id AS INTEGER)
        WHERE {FORMULA_SEARCH_TABLE} MATCH ?
    """
    params_tail: list[Any] = []
    if collection:
        sql += " AND fs.collections LIKE ?"
        params_tail.append(f"%{collection}%")
    sql += " ORDER BY score ASC, formula_id ASC LIMIT ?"
    rows = _run_match_queries(
        conn,
        sql=sql,
        match_queries=match_queries,
        params_tail=params_tail,
        key_field="formula_id",
        limit=limit,
    )

    blocks_by_document, block_positions = _load_retrievable_blocks(conn)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["display_context"] = _block_context_text(
            blocks_by_document,
            block_positions,
            document_id=int(item["document_id"]),
            block_id=int(item["block_id"]) if item["block_id"] is not None else None,
            include_self=False,
        )
        item.pop("document_id", None)
        item.pop("block_id", None)
        out.append(item)
    return out


def search_assets_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_search_schema(conn)
    match_queries = build_match_queries(query)
    if not match_queries:
        return []

    sql = f"""
        SELECT
          CAST(ats.asset_id AS INTEGER) AS asset_id,
          CAST(ats.work_id AS INTEGER) AS work_id,
          ats.collections,
          a.asset_type,
          CAST(NULLIF(ats.page, '') AS INTEGER) AS page,
          ats.section_hint,
          w.title AS raw_title,
          COALESCE(b.clean_text, a.caption, '') AS caption,
          a.asset_path,
          a.document_id,
          a.block_id,
          bm25({ASSET_SEARCH_TABLE}) AS score
        FROM {ASSET_SEARCH_TABLE} ats
        JOIN assets a ON a.asset_id = CAST(ats.asset_id AS INTEGER)
        JOIN works w ON w.work_id = CAST(ats.work_id AS INTEGER)
        LEFT JOIN blocks b ON b.block_id = a.block_id
        WHERE {ASSET_SEARCH_TABLE} MATCH ?
    """
    params_tail: list[Any] = []
    if collection:
        sql += " AND ats.collections LIKE ?"
        params_tail.append(f"%{collection}%")
    sql += " ORDER BY score ASC, asset_id ASC LIMIT ?"
    rows = _run_match_queries(
        conn,
        sql=sql,
        match_queries=match_queries,
        params_tail=params_tail,
        key_field="asset_id",
        limit=limit,
    )

    blocks_by_document, block_positions = _load_retrievable_blocks(conn)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["display_context"] = _block_context_text(
            blocks_by_document,
            block_positions,
            document_id=int(item["document_id"]),
            block_id=int(item["block_id"]) if item["block_id"] is not None else None,
            include_self=True,
        )
        item.pop("document_id", None)
        item.pop("block_id", None)
        out.append(item)
    return out


def _collections_by_work(conn: sqlite3.Connection) -> dict[int, str]:
    return _aggregate_text_map(
        conn,
        """
        SELECT cw.work_id AS owner_id, c.name AS value
        FROM collection_works cw
        JOIN collections c ON c.collection_id = cw.collection_id
        ORDER BY cw.work_id, c.name
        """,
    )


def _load_retrievable_blocks(
    conn: sqlite3.Connection,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, tuple[int, int]]]:
    blocks_by_document: dict[int, list[dict[str, Any]]] = defaultdict(list)
    block_positions: dict[int, tuple[int, int]] = {}
    rows = conn.execute(
        """
        SELECT document_id, block_id, block_type, clean_text
        FROM blocks
        WHERE is_retrievable = 1
          AND COALESCE(clean_text, '') <> ''
        ORDER BY document_id, order_index, block_id
        """
    ).fetchall()
    for row in rows:
        document_id = int(row["document_id"])
        block_id = int(row["block_id"])
        entry = {
            "block_id": block_id,
            "block_type": str(row["block_type"] or ""),
            "clean_text": str(row["clean_text"] or ""),
        }
        block_positions[block_id] = (document_id, len(blocks_by_document[document_id]))
        blocks_by_document[document_id].append(entry)
    return dict(blocks_by_document), block_positions


def _block_context_text(
    blocks_by_document: dict[int, list[dict[str, Any]]],
    block_positions: dict[int, tuple[int, int]],
    *,
    document_id: int,
    block_id: int | None,
    include_self: bool,
) -> str:
    if block_id is None:
        return ""
    position = block_positions.get(block_id)
    if position is None:
        return ""
    block_document_id, idx = position
    if block_document_id != document_id:
        return ""

    blocks = blocks_by_document.get(document_id, [])
    pieces: list[str] = []
    _append_unique_text(pieces, _nearest_block_text(blocks, start=idx - 1, step=-1))
    if include_self:
        candidate = blocks[idx]
        if candidate["block_type"] != "equation":
            _append_unique_text(pieces, candidate["clean_text"])
    _append_unique_text(pieces, _nearest_block_text(blocks, start=idx + 1, step=1))
    return "\n\n".join(pieces)


def _nearest_block_text(blocks: list[dict[str, Any]], *, start: int, step: int) -> str | None:
    idx = start
    while 0 <= idx < len(blocks):
        candidate = blocks[idx]
        if candidate["block_type"] != "equation" and str(candidate["clean_text"]).strip():
            return str(candidate["clean_text"]).strip()
        idx += step
    return None


def _append_unique_text(items: list[str], text: str | None) -> None:
    value = str(text or "").strip()
    if not value:
        return
    if value in items:
        return
    items.append(value)


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


def _run_match_queries(
    conn: sqlite3.Connection,
    *,
    sql: str,
    match_queries: list[str],
    params_tail: list[Any],
    key_field: str,
    limit: int,
) -> list[dict[str, Any]]:
    seen_ids: set[int] = set()
    out: list[dict[str, Any]] = []
    per_query_limit = max(limit * 3, 20)
    for match_query in match_queries:
        params: list[Any] = [match_query, *params_tail, per_query_limit]
        rows = conn.execute(sql, params).fetchall()
        for row in rows:
            item = dict(row)
            item_id = int(item[key_field])
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            out.append(item)
            if len(out) >= limit:
                return out
    return out


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
