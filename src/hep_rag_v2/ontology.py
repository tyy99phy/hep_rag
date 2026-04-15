from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from typing import Any

from hep_rag_v2.methods import ensure_method_schema
from hep_rag_v2.query import analyze_query, query_match_stats
from hep_rag_v2.results import ensure_result_schema
from hep_rag_v2.structure import ensure_structure_schema


SCHEMA = """
CREATE TABLE IF NOT EXISTS ontology_summaries (
  summary_id TEXT PRIMARY KEY,
  collection_id INTEGER,
  facet_kind TEXT NOT NULL,
  facet_key TEXT NOT NULL,
  label TEXT NOT NULL,
  summary_text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'ready',
  work_count INTEGER NOT NULL DEFAULT 0,
  signal_count INTEGER NOT NULL DEFAULT 0,
  representative_works_json TEXT NOT NULL DEFAULT '[]',
  source_refs_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(collection_id, facet_kind, facet_key),
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_ontology_summaries_scope
  ON ontology_summaries(collection_id, facet_kind, work_count);
"""


def ensure_ontology_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def rebuild_ontology_summaries(
    conn: sqlite3.Connection,
    *,
    collection: str | None = None,
    representative_limit: int = 4,
) -> dict[str, Any]:
    ensure_ontology_schema(conn)
    ensure_structure_schema(conn)
    ensure_result_schema(conn)
    ensure_method_schema(conn)

    collection_id = _collection_id(conn, collection) if collection else None
    scope_name = collection or "all"
    _clear_scope(conn, collection_id=collection_id)

    items: list[dict[str, Any]] = []
    builders = (
        _build_collaboration_summaries,
        _build_topic_summaries,
        _build_result_kind_summaries,
        _build_method_family_summaries,
    )
    for builder in builders:
        items.extend(
            builder(
                conn,
                collection_id=collection_id,
                scope_name=scope_name,
                representative_limit=representative_limit,
            )
        )
    for item in items:
        _upsert_ontology_summary(conn, item)

    by_kind = Counter(str(item["facet_kind"]) for item in items)
    return {
        "collection": scope_name,
        "total": len(items),
        "by_kind": dict(sorted(by_kind.items())),
    }


def search_ontology_summaries(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    ensure_ontology_schema(conn)
    if _scope_summary_count(conn, collection=collection) == 0:
        rebuild_ontology_summaries(conn, collection=collection)

    rows = _load_scope_rows(conn, collection=collection)
    if not rows:
        return []

    profile = analyze_query(query)
    scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in rows:
        payload = dict(row)
        label = str(payload.get("label") or "").strip()
        summary_text = str(payload.get("summary_text") or "").strip()
        representative_works = _loads_json_list(payload.get("representative_works_json"))
        source_refs = _loads_json_list(payload.get("source_refs_json"))
        metadata = _loads_json_dict(payload.get("metadata_json"))
        combined_text = " ".join(
            part
            for part in (
                label,
                payload.get("facet_kind"),
                payload.get("facet_key"),
                summary_text,
                " ".join(
                    str(item.get("title") or "")
                    for item in representative_works
                    if isinstance(item, dict)
                ),
                " ".join(str(item) for item in metadata.get("collaborations") or []),
                " ".join(str(item) for item in metadata.get("result_kinds") or []),
                " ".join(str(item) for item in metadata.get("method_families") or []),
            )
            if part
        )
        hits, coverage = query_match_stats(combined_text, profile)
        label_hits, label_coverage = query_match_stats(label, profile)
        if profile.content_tokens and hits == 0 and label_hits == 0:
            continue

        work_count = int(payload.get("work_count") or 0)
        signal_count = int(payload.get("signal_count") or 0)
        score = (
            float(hits) * 1.5
            + float(coverage)
            + float(label_hits) * 0.8
            + float(label_coverage) * 0.4
            + min(work_count, 25) * 0.02
            + min(signal_count, 25) * 0.01
        )
        payload.update(
            {
                "summary_id": str(payload["summary_id"]),
                "title": label,
                "summary": summary_text,
                "representative_works": representative_works,
                "source_refs": source_refs,
                "metadata": metadata,
                "query_group_hits": hits,
                "query_group_coverage": coverage,
                "label_group_hits": label_hits,
                "label_group_coverage": label_coverage,
                "hybrid_score": round(score, 6),
                "search_type": "ontology",
            }
        )
        scored.append(
            (
                (
                    -float(score),
                    -int(hits),
                    -float(coverage),
                    -int(label_hits),
                    -int(work_count),
                    -int(signal_count),
                    label.casefold(),
                ),
                payload,
            )
        )

    ordered = [payload for _, payload in sorted(scored)[: max(1, int(limit))]]
    for rank, payload in enumerate(ordered, start=1):
        payload["rank"] = rank
    return ordered


def ontology_summary_counts(conn: sqlite3.Connection) -> dict[str, int]:
    ensure_ontology_schema(conn)
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS ontology_summaries,
          SUM(CASE WHEN facet_kind = 'collaboration' THEN 1 ELSE 0 END) AS ontology_collaboration_summaries,
          SUM(CASE WHEN facet_kind = 'topic' THEN 1 ELSE 0 END) AS ontology_topic_summaries,
          SUM(CASE WHEN facet_kind = 'result_kind' THEN 1 ELSE 0 END) AS ontology_result_summaries,
          SUM(CASE WHEN facet_kind = 'method_family' THEN 1 ELSE 0 END) AS ontology_method_summaries
        FROM ontology_summaries
        """
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()} if row is not None else {}


def _scope_summary_count(conn: sqlite3.Connection, *, collection: str | None) -> int:
    if collection:
        collection_id = _collection_id(conn, collection)
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM ontology_summaries WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()
        return int(row["n"] or 0) if row is not None else 0
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM ontology_summaries WHERE collection_id IS NULL"
    ).fetchone()
    return int(row["n"] or 0) if row is not None else 0


def _load_scope_rows(conn: sqlite3.Connection, *, collection: str | None) -> list[sqlite3.Row]:
    if collection:
        collection_id = _collection_id(conn, collection)
        return conn.execute(
            """
            SELECT summary_id, facet_kind, facet_key, label, summary_text, status,
                   work_count, signal_count, representative_works_json, source_refs_json, metadata_json
            FROM ontology_summaries
            WHERE collection_id = ?
            ORDER BY facet_kind, work_count DESC, label
            """,
            (collection_id,),
        ).fetchall()
    return conn.execute(
        """
        SELECT summary_id, facet_kind, facet_key, label, summary_text, status,
               work_count, signal_count, representative_works_json, source_refs_json, metadata_json
        FROM ontology_summaries
        WHERE collection_id IS NULL
        ORDER BY facet_kind, work_count DESC, label
        """
    ).fetchall()


def _clear_scope(conn: sqlite3.Connection, *, collection_id: int | None) -> None:
    if collection_id is None:
        conn.execute("DELETE FROM ontology_summaries WHERE collection_id IS NULL")
        return
    conn.execute("DELETE FROM ontology_summaries WHERE collection_id = ?", (collection_id,))


def _build_collaboration_summaries(
    conn: sqlite3.Connection,
    *,
    collection_id: int | None,
    scope_name: str,
    representative_limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = w.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    rows = conn.execute(
        f"""
        SELECT c.name AS label, COUNT(DISTINCT w.work_id) AS work_count
        FROM collaborations c
        JOIN work_collaborations wc ON wc.collaboration_id = c.collaboration_id
        JOIN works w ON w.work_id = wc.work_id
        {collection_join}
        GROUP BY c.collaboration_id, c.name
        HAVING work_count >= 2
        ORDER BY work_count DESC, c.name
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        label = str(row["label"] or "").strip()
        if not label:
            continue
        representative_works = _representative_works_for_collaboration(
            conn,
            label=label,
            collection_id=collection_id,
            limit=representative_limit,
        )
        work_ids = [int(item["work_id"]) for item in representative_works]
        result_kinds = _top_value_counts(conn, table="result_objects", field="result_kind", work_ids=work_ids)
        method_families = _top_value_counts(conn, table="method_objects", field="method_family", work_ids=work_ids)
        out.append(
            _summary_payload(
                scope_name=scope_name,
                collection_id=collection_id,
                facet_kind="collaboration",
                facet_key=label,
                label=label,
                summary_text=_compose_summary_text(
                    label=f"{label} collaboration summary",
                    work_count=int(row["work_count"] or 0),
                    signal_count=int(row["work_count"] or 0),
                    representative_works=representative_works,
                    result_kinds=result_kinds,
                    method_families=method_families,
                ),
                work_count=int(row["work_count"] or 0),
                signal_count=int(row["work_count"] or 0),
                representative_works=representative_works,
                source_refs=[f"collaboration:{label}"],
                metadata={
                    "result_kinds": result_kinds,
                    "method_families": method_families,
                },
            )
        )
    return out


def _build_topic_summaries(
    conn: sqlite3.Connection,
    *,
    collection_id: int | None,
    scope_name: str,
    representative_limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = w.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    rows = conn.execute(
        f"""
        SELECT t.source AS topic_source, t.topic_key AS topic_key, t.label AS label, COUNT(DISTINCT w.work_id) AS work_count
        FROM topics t
        JOIN work_topics wt ON wt.topic_id = t.topic_id
        JOIN works w ON w.work_id = wt.work_id
        {collection_join}
        GROUP BY t.topic_id, t.source, t.topic_key, t.label
        HAVING work_count >= 2
        ORDER BY work_count DESC, t.label
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        topic_source = str(row["topic_source"] or "").strip()
        topic_key = str(row["topic_key"] or "").strip()
        label = str(row["label"] or "").strip()
        if not topic_source or not topic_key or not label:
            continue
        representative_works = _representative_works_for_topic(
            conn,
            topic_source=topic_source,
            topic_key=topic_key,
            collection_id=collection_id,
            limit=representative_limit,
        )
        work_ids = [int(item["work_id"]) for item in representative_works]
        collaborations = _top_collaboration_counts(conn, work_ids=work_ids)
        out.append(
            _summary_payload(
                scope_name=scope_name,
                collection_id=collection_id,
                facet_kind="topic",
                facet_key=f"{topic_source}:{topic_key}",
                label=label,
                summary_text=_compose_summary_text(
                    label=f"{label} topic summary",
                    work_count=int(row["work_count"] or 0),
                    signal_count=int(row["work_count"] or 0),
                    representative_works=representative_works,
                    collaborations=collaborations,
                ),
                work_count=int(row["work_count"] or 0),
                signal_count=int(row["work_count"] or 0),
                representative_works=representative_works,
                source_refs=[f"topic:{topic_source}:{topic_key}"],
                metadata={
                    "topic_source": topic_source,
                    "topic_key": topic_key,
                    "collaborations": collaborations,
                },
            )
        )
    return out


def _build_result_kind_summaries(
    conn: sqlite3.Connection,
    *,
    collection_id: int | None,
    scope_name: str,
    representative_limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = ro.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    rows = conn.execute(
        f"""
        SELECT COALESCE(ro.result_kind, '') AS facet_key,
               COUNT(*) AS signal_count,
               COUNT(DISTINCT ro.work_id) AS work_count
        FROM result_objects ro
        {collection_join}
        WHERE COALESCE(ro.result_kind, '') <> ''
        GROUP BY facet_key
        HAVING work_count >= 2
        ORDER BY signal_count DESC, work_count DESC, facet_key
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        facet_key = str(row["facet_key"] or "").strip()
        if not facet_key:
            continue
        representative_works = _representative_works_for_result_kind(
            conn,
            result_kind=facet_key,
            collection_id=collection_id,
            limit=representative_limit,
        )
        work_ids = [int(item["work_id"]) for item in representative_works]
        method_families = _top_value_counts(conn, table="method_objects", field="method_family", work_ids=work_ids)
        out.append(
            _summary_payload(
                scope_name=scope_name,
                collection_id=collection_id,
                facet_kind="result_kind",
                facet_key=facet_key,
                label=facet_key.replace("_", " "),
                summary_text=_compose_summary_text(
                    label=f"{facet_key.replace('_', ' ')} result summary",
                    work_count=int(row["work_count"] or 0),
                    signal_count=int(row["signal_count"] or 0),
                    representative_works=representative_works,
                    method_families=method_families,
                ),
                work_count=int(row["work_count"] or 0),
                signal_count=int(row["signal_count"] or 0),
                representative_works=representative_works,
                source_refs=[f"result_kind:{facet_key}"],
                metadata={
                    "method_families": method_families,
                },
            )
        )
    return out


def _build_method_family_summaries(
    conn: sqlite3.Connection,
    *,
    collection_id: int | None,
    scope_name: str,
    representative_limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = mo.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    rows = conn.execute(
        f"""
        SELECT COALESCE(mo.method_family, '') AS facet_key,
               COUNT(*) AS signal_count,
               COUNT(DISTINCT mo.work_id) AS work_count
        FROM method_objects mo
        {collection_join}
        WHERE COALESCE(mo.method_family, '') <> ''
        GROUP BY facet_key
        HAVING work_count >= 2
        ORDER BY signal_count DESC, work_count DESC, facet_key
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        facet_key = str(row["facet_key"] or "").strip()
        if not facet_key:
            continue
        representative_works = _representative_works_for_method_family(
            conn,
            method_family=facet_key,
            collection_id=collection_id,
            limit=representative_limit,
        )
        work_ids = [int(item["work_id"]) for item in representative_works]
        result_kinds = _top_value_counts(conn, table="result_objects", field="result_kind", work_ids=work_ids)
        out.append(
            _summary_payload(
                scope_name=scope_name,
                collection_id=collection_id,
                facet_kind="method_family",
                facet_key=facet_key,
                label=facet_key.replace("_", " "),
                summary_text=_compose_summary_text(
                    label=f"{facet_key.replace('_', ' ')} method summary",
                    work_count=int(row["work_count"] or 0),
                    signal_count=int(row["signal_count"] or 0),
                    representative_works=representative_works,
                    result_kinds=result_kinds,
                ),
                work_count=int(row["work_count"] or 0),
                signal_count=int(row["signal_count"] or 0),
                representative_works=representative_works,
                source_refs=[f"method_family:{facet_key}"],
                metadata={
                    "result_kinds": result_kinds,
                },
            )
        )
    return out


def _representative_works_for_collaboration(
    conn: sqlite3.Connection,
    *,
    label: str,
    collection_id: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = w.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    params.extend([label, limit])
    rows = conn.execute(
        f"""
        SELECT DISTINCT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
        FROM works w
        JOIN work_collaborations wc ON wc.work_id = w.work_id
        JOIN collaborations c ON c.collaboration_id = wc.collaboration_id
        {collection_join}
        WHERE c.name = ?
        ORDER BY COALESCE(w.year, 0) DESC, COALESCE(w.citation_count, 0) DESC, w.work_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_work_row_payload(row) for row in rows]


def _representative_works_for_topic(
    conn: sqlite3.Connection,
    *,
    topic_source: str,
    topic_key: str,
    collection_id: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = w.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    params.extend([topic_source, topic_key, limit])
    rows = conn.execute(
        f"""
        SELECT DISTINCT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
        FROM works w
        JOIN work_topics wt ON wt.work_id = w.work_id
        JOIN topics t ON t.topic_id = wt.topic_id
        {collection_join}
        WHERE t.source = ? AND t.topic_key = ?
        ORDER BY COALESCE(w.year, 0) DESC, COALESCE(w.citation_count, 0) DESC, w.work_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_work_row_payload(row) for row in rows]


def _representative_works_for_result_kind(
    conn: sqlite3.Connection,
    *,
    result_kind: str,
    collection_id: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = w.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    params.extend([result_kind, limit])
    rows = conn.execute(
        f"""
        SELECT DISTINCT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
        FROM works w
        JOIN result_objects ro ON ro.work_id = w.work_id
        {collection_join}
        WHERE ro.result_kind = ?
        ORDER BY COALESCE(w.year, 0) DESC, COALESCE(w.citation_count, 0) DESC, w.work_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_work_row_payload(row) for row in rows]


def _representative_works_for_method_family(
    conn: sqlite3.Connection,
    *,
    method_family: str,
    collection_id: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    collection_join = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_join = "JOIN collection_works cw ON cw.work_id = w.work_id AND cw.collection_id = ?"
        params.append(collection_id)
    params.extend([method_family, limit])
    rows = conn.execute(
        f"""
        SELECT DISTINCT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
        FROM works w
        JOIN method_objects mo ON mo.work_id = w.work_id
        {collection_join}
        WHERE mo.method_family = ?
        ORDER BY COALESCE(w.year, 0) DESC, COALESCE(w.citation_count, 0) DESC, w.work_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_work_row_payload(row) for row in rows]


def _top_value_counts(
    conn: sqlite3.Connection,
    *,
    table: str,
    field: str,
    work_ids: list[int],
    limit: int = 3,
) -> list[str]:
    if not work_ids:
        return []
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        (
            f"SELECT COALESCE({field}, '') AS value, COUNT(*) AS n "
            f"FROM {table} "
            f"WHERE work_id IN ({placeholders}) AND COALESCE({field}, '') <> '' "
            f"GROUP BY value ORDER BY n DESC, value LIMIT ?"
        ),
        [*work_ids, limit],
    ).fetchall()
    return [f"{str(row['value']).replace('_', ' ')}={int(row['n'])}" for row in rows if row["value"]]


def _top_collaboration_counts(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int],
    limit: int = 3,
) -> list[str]:
    if not work_ids:
        return []
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        f"""
        SELECT c.name AS value, COUNT(*) AS n
        FROM work_collaborations wc
        JOIN collaborations c ON c.collaboration_id = wc.collaboration_id
        WHERE wc.work_id IN ({placeholders})
        GROUP BY c.name
        ORDER BY n DESC, c.name
        LIMIT ?
        """,
        [*work_ids, limit],
    ).fetchall()
    return [f"{str(row['value'])}={int(row['n'])}" for row in rows if row["value"]]


def _compose_summary_text(
    *,
    label: str,
    work_count: int,
    signal_count: int,
    representative_works: list[dict[str, Any]],
    collaborations: list[str] | None = None,
    result_kinds: list[str] | None = None,
    method_families: list[str] | None = None,
) -> str:
    parts = [label, f"works={work_count}", f"signals={signal_count}"]
    if representative_works:
        parts.append(
            "representative works: "
            + "; ".join(
                f"{item.get('title')} ({item.get('year')})"
                for item in representative_works[:4]
                if item.get("title")
            )
        )
    if collaborations:
        parts.append("collaborations: " + ", ".join(collaborations[:3]))
    if result_kinds:
        parts.append("result kinds: " + ", ".join(result_kinds[:3]))
    if method_families:
        parts.append("method families: " + ", ".join(method_families[:3]))
    return " | ".join(part for part in parts if part)


def _summary_payload(
    *,
    scope_name: str,
    collection_id: int | None,
    facet_kind: str,
    facet_key: str,
    label: str,
    summary_text: str,
    work_count: int,
    signal_count: int,
    representative_works: list[dict[str, Any]],
    source_refs: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "summary_id": _summary_id(scope_name=scope_name, facet_kind=facet_kind, facet_key=facet_key),
        "collection_id": collection_id,
        "facet_kind": facet_kind,
        "facet_key": facet_key,
        "label": label,
        "summary_text": summary_text,
        "status": "ready",
        "work_count": int(work_count),
        "signal_count": int(signal_count),
        "representative_works_json": json.dumps(representative_works, ensure_ascii=False),
        "source_refs_json": json.dumps(source_refs, ensure_ascii=False),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }


def _upsert_ontology_summary(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO ontology_summaries (
          summary_id, collection_id, facet_kind, facet_key, label, summary_text, status,
          work_count, signal_count, representative_works_json, source_refs_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(summary_id) DO UPDATE SET
          collection_id = excluded.collection_id,
          facet_kind = excluded.facet_kind,
          facet_key = excluded.facet_key,
          label = excluded.label,
          summary_text = excluded.summary_text,
          status = excluded.status,
          work_count = excluded.work_count,
          signal_count = excluded.signal_count,
          representative_works_json = excluded.representative_works_json,
          source_refs_json = excluded.source_refs_json,
          metadata_json = excluded.metadata_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            payload["summary_id"],
            payload["collection_id"],
            payload["facet_kind"],
            payload["facet_key"],
            payload["label"],
            payload["summary_text"],
            payload["status"],
            payload["work_count"],
            payload["signal_count"],
            payload["representative_works_json"],
            payload["source_refs_json"],
            payload["metadata_json"],
        ),
    )


def _summary_id(*, scope_name: str, facet_kind: str, facet_key: str) -> str:
    return f"ontology_summary:{_safe_key(scope_name)}:{_safe_key(facet_kind)}:{_safe_key(facet_key)}"


def _safe_key(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip().casefold()).strip("_")
    return text or "unknown"


def _collection_id(conn: sqlite3.Connection, collection: str) -> int:
    row = conn.execute(
        "SELECT collection_id FROM collections WHERE name = ?",
        (collection,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown collection: {collection}")
    return int(row["collection_id"])


def _work_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "work_id": int(row["work_id"]),
        "title": str(row["title"] or ""),
        "year": int(row["year"]) if row["year"] is not None else None,
        "canonical_source": str(row["canonical_source"] or ""),
        "canonical_id": str(row["canonical_id"] or ""),
    }


def _loads_json_list(value: Any) -> list[Any]:
    try:
        payload = json.loads(str(value or "[]"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _loads_json_dict(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
