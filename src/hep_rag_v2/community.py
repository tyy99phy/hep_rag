from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from typing import Any

from hep_rag_v2.methods import ensure_method_schema
from hep_rag_v2.query import analyze_query, query_match_stats
from hep_rag_v2.results import ensure_result_schema
from hep_rag_v2.structure import ensure_structure_schema


ALGORITHM = "weighted_components_hierarchy_v1"
MIN_EDGE_WEIGHT = 0.38
TOPIC_EDGE_WEIGHT = 0.18
RESULT_EDGE_WEIGHT = 0.12
METHOD_EDGE_WEIGHT = 0.10
COLLABORATION_EDGE_WEIGHT = 0.08
OVERVIEW_MIN_EDGE_WEIGHT = 0.26
OVERVIEW_TOPIC_WEIGHT = 0.16
OVERVIEW_RESULT_WEIGHT = 0.12
OVERVIEW_METHOD_WEIGHT = 0.10
OVERVIEW_COLLABORATION_WEIGHT = 0.08
OVERVIEW_CROSS_EDGE_CAP = 0.24

SCHEMA = """
CREATE TABLE IF NOT EXISTS community_summaries (
  summary_id TEXT PRIMARY KEY,
  collection_id INTEGER,
  community_key TEXT NOT NULL,
  algorithm TEXT NOT NULL DEFAULT 'weighted_components_v1',
  community_level TEXT NOT NULL DEFAULT 'fine',
  parent_summary_id TEXT,
  label TEXT NOT NULL,
  summary_text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'ready',
  work_count INTEGER NOT NULL DEFAULT 0,
  edge_count INTEGER NOT NULL DEFAULT 0,
  child_summary_ids_json TEXT NOT NULL DEFAULT '[]',
  lineage_json TEXT NOT NULL DEFAULT '[]',
  representative_works_json TEXT NOT NULL DEFAULT '[]',
  source_refs_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(collection_id, community_key),
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_community_summaries_scope
  ON community_summaries(collection_id, community_level, work_count, edge_count);
"""


def ensure_community_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    existing = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(community_summaries)").fetchall()
    }
    migrations = (
        (
            "community_level",
            "ALTER TABLE community_summaries ADD COLUMN community_level TEXT NOT NULL DEFAULT 'fine'",
        ),
        (
            "parent_summary_id",
            "ALTER TABLE community_summaries ADD COLUMN parent_summary_id TEXT",
        ),
        (
            "child_summary_ids_json",
            "ALTER TABLE community_summaries ADD COLUMN child_summary_ids_json TEXT NOT NULL DEFAULT '[]'",
        ),
        (
            "lineage_json",
            "ALTER TABLE community_summaries ADD COLUMN lineage_json TEXT NOT NULL DEFAULT '[]'",
        ),
    )
    for column_name, sql in migrations:
        if column_name not in existing:
            conn.execute(sql)


def rebuild_community_summaries(
    conn: sqlite3.Connection,
    *,
    collection: str | None = None,
    representative_limit: int = 4,
    min_edge_weight: float = MIN_EDGE_WEIGHT,
) -> dict[str, Any]:
    ensure_community_schema(conn)
    ensure_structure_schema(conn)
    ensure_result_schema(conn)
    ensure_method_schema(conn)

    collection_id = _collection_id(conn, collection) if collection else None
    scope_name = collection or "all"
    _clear_scope(conn, collection_id=collection_id)

    work_ids = _scope_work_ids(conn, collection_id=collection_id)
    if len(work_ids) < 2:
        return {
            "collection": scope_name,
            "algorithm": ALGORITHM,
            "total": 0,
            "edge_count": 0,
        }

    work_map = _work_payload_map(conn, work_ids=work_ids)
    edge_map = _aggregate_edges(conn, work_ids=work_ids, collection_id=collection_id)
    communities = _build_components(work_ids=work_ids, edge_map=edge_map, min_edge_weight=min_edge_weight)
    fine_descriptors: list[dict[str, Any]] = []
    active_edge_total = 0
    for member_ids in communities:
        if len(member_ids) < 2:
            continue
        representative_works = _representative_works(
            work_map=work_map,
            edge_map=edge_map,
            member_ids=member_ids,
            min_edge_weight=min_edge_weight,
            limit=representative_limit,
        )
        edge_stats = _community_edge_stats(
            edge_map=edge_map,
            member_ids=member_ids,
            min_edge_weight=min_edge_weight,
        )
        active_edge_total += int(edge_stats["edge_count"])
        top_collaborations = _top_collaboration_counts(conn, work_ids=member_ids)
        top_topics = _top_topic_counts(conn, work_ids=member_ids)
        top_results = _top_value_counts(conn, table="result_objects", field="result_kind", work_ids=member_ids)
        top_methods = _top_value_counts(conn, table="method_objects", field="method_family", work_ids=member_ids)
        label = _community_label(
            representative_works=representative_works,
            collaborations=top_collaborations,
            topics=top_topics,
            result_kinds=top_results,
            method_families=top_methods,
        )
        fine_descriptors.append(
            {
                "community_key": _community_key(member_ids),
                "member_ids": list(member_ids),
                "label": label,
                "work_count": len(member_ids),
                "edge_count": int(edge_stats["edge_count"]),
                "representative_works": representative_works,
                "signal_mix": list(edge_stats["signal_mix"]),
                "collaborations": top_collaborations,
                "topics": top_topics,
                "result_kinds": top_results,
                "method_families": top_methods,
                "source_refs": [f"work:{work_id}" for work_id in member_ids[: min(8, len(member_ids))]],
            }
        )

    overview_descriptors, parent_map = _build_overview_descriptors(
        conn,
        collection_id=collection_id,
        scope_name=scope_name,
        work_map=work_map,
        edge_map=edge_map,
        descriptors=fine_descriptors,
        representative_limit=representative_limit,
        min_edge_weight=min_edge_weight,
    )
    items = [
        _summary_payload(
            scope_name=scope_name,
            collection_id=collection_id,
            community_key=str(descriptor["community_key"]),
            community_level="overview",
            parent_summary_id=None,
            child_summary_ids=list(descriptor.get("child_summary_ids") or []),
            lineage=[],
            label=str(descriptor["label"]),
            summary_text=_compose_summary_text(
                label=str(descriptor["label"]),
                community_level="overview",
                work_count=int(descriptor["work_count"]),
                edge_count=int(descriptor["edge_count"]),
                representative_works=list(descriptor["representative_works"]),
                signal_mix=list(descriptor["signal_mix"]),
                collaborations=list(descriptor["collaborations"]),
                topics=list(descriptor["topics"]),
                result_kinds=list(descriptor["result_kinds"]),
                method_families=list(descriptor["method_families"]),
                child_labels=list(descriptor.get("child_labels") or []),
            ),
            work_count=int(descriptor["work_count"]),
            edge_count=int(descriptor["edge_count"]),
            representative_works=list(descriptor["representative_works"]),
            source_refs=list(descriptor["source_refs"]),
            metadata={
                "algorithm": ALGORITHM,
                "community_level": "overview",
                "member_work_ids": list(descriptor["member_ids"]),
                "signal_mix": list(descriptor["signal_mix"]),
                "collaborations": list(descriptor["collaborations"]),
                "topics": list(descriptor["topics"]),
                "result_kinds": list(descriptor["result_kinds"]),
                "method_families": list(descriptor["method_families"]),
                "child_summary_ids": list(descriptor.get("child_summary_ids") or []),
                "child_labels": list(descriptor.get("child_labels") or []),
                "hierarchy_version": ALGORITHM,
            },
        )
        for descriptor in overview_descriptors
    ]
    for descriptor in fine_descriptors:
        summary_id = _summary_id(scope_name=scope_name, community_key=str(descriptor["community_key"]))
        parent_summary_id = parent_map.get(summary_id)
        lineage = [parent_summary_id] if parent_summary_id else []
        items.append(
            _summary_payload(
                scope_name=scope_name,
                collection_id=collection_id,
                community_key=str(descriptor["community_key"]),
                community_level="fine",
                parent_summary_id=parent_summary_id,
                child_summary_ids=[],
                lineage=lineage,
                label=str(descriptor["label"]),
                summary_text=_compose_summary_text(
                    label=str(descriptor["label"]),
                    community_level="fine",
                    work_count=int(descriptor["work_count"]),
                    edge_count=int(descriptor["edge_count"]),
                    representative_works=list(descriptor["representative_works"]),
                    signal_mix=list(descriptor["signal_mix"]),
                    collaborations=list(descriptor["collaborations"]),
                    topics=list(descriptor["topics"]),
                    result_kinds=list(descriptor["result_kinds"]),
                    method_families=list(descriptor["method_families"]),
                    parent_label=_parent_label_from_descriptors(
                        overview_descriptors,
                        parent_summary_id=parent_summary_id,
                        scope_name=scope_name,
                    ),
                ),
                work_count=int(descriptor["work_count"]),
                edge_count=int(descriptor["edge_count"]),
                representative_works=list(descriptor["representative_works"]),
                source_refs=list(descriptor["source_refs"]),
                metadata={
                    "algorithm": ALGORITHM,
                    "community_level": "fine",
                    "member_work_ids": list(descriptor["member_ids"]),
                    "signal_mix": list(descriptor["signal_mix"]),
                    "collaborations": list(descriptor["collaborations"]),
                    "topics": list(descriptor["topics"]),
                    "result_kinds": list(descriptor["result_kinds"]),
                    "method_families": list(descriptor["method_families"]),
                    "parent_summary_id": parent_summary_id,
                    "hierarchy_version": ALGORITHM,
                },
            )
        )

    for item in items:
        _upsert_community_summary(conn, item)

    return {
        "collection": scope_name,
        "algorithm": ALGORITHM,
        "total": len(items),
        "fine_total": len(fine_descriptors),
        "overview_total": len(overview_descriptors),
        "edge_count": active_edge_total,
    }


def search_community_summaries(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    ensure_community_schema(conn)
    if _scope_summary_count(conn, collection=collection) == 0 or _scope_has_stale_algorithm(conn, collection=collection):
        rebuild_community_summaries(conn, collection=collection)

    rows = _load_scope_rows(conn, collection=collection)
    if not rows:
        return []

    profile = analyze_query(query)
    overview_scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fine_scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in rows:
        payload = dict(row)
        label = str(payload.get("label") or "").strip()
        summary_text = str(payload.get("summary_text") or "").strip()
        representative_works = _loads_json_list(payload.get("representative_works_json"))
        source_refs = _loads_json_list(payload.get("source_refs_json"))
        metadata = _loads_json_dict(payload.get("metadata_json"))
        child_summary_ids = _loads_json_list(payload.get("child_summary_ids_json"))
        lineage = _loads_json_list(payload.get("lineage_json"))
        community_level = str(payload.get("community_level") or metadata.get("community_level") or "fine")
        combined_text = " ".join(
            part
            for part in (
                label,
                summary_text,
                " ".join(
                    str(item.get("title") or "")
                    for item in representative_works
                    if isinstance(item, dict)
                ),
                " ".join(str(item) for item in metadata.get("signal_mix") or []),
                " ".join(str(item) for item in metadata.get("collaborations") or []),
                " ".join(str(item) for item in metadata.get("topics") or []),
                " ".join(str(item) for item in metadata.get("result_kinds") or []),
                " ".join(str(item) for item in metadata.get("method_families") or []),
                " ".join(str(item) for item in metadata.get("child_labels") or []),
            )
            if part
        )
        hits, coverage = query_match_stats(combined_text, profile)
        label_hits, label_coverage = query_match_stats(label, profile)
        if profile.content_tokens and hits == 0 and label_hits == 0:
            continue

        work_count = int(payload.get("work_count") or 0)
        edge_count = int(payload.get("edge_count") or 0)
        child_count = len(child_summary_ids)
        score = (
            float(hits) * 1.6
            + float(coverage)
            + float(label_hits) * 0.8
            + float(label_coverage) * 0.4
            + min(work_count, 30) * 0.02
            + min(edge_count, 30) * 0.015
            + (min(child_count, 6) * 0.03 if community_level == "overview" else 0.0)
        )
        payload.update(
            {
                "summary_id": str(payload["summary_id"]),
                "title": label,
                "summary": summary_text,
                "community_level": community_level,
                "parent_summary_id": payload.get("parent_summary_id"),
                "child_summary_ids": child_summary_ids,
                "lineage": lineage,
                "representative_works": representative_works,
                "source_refs": source_refs,
                "metadata": metadata,
                "query_group_hits": hits,
                "query_group_coverage": coverage,
                "label_group_hits": label_hits,
                "label_group_coverage": label_coverage,
                "hybrid_score": round(score, 6),
                "search_type": "community",
            }
        )
        record = (
            (
                -float(score),
                0 if community_level == "overview" else 1,
                -int(hits),
                -float(coverage),
                -int(label_hits),
                -int(work_count),
                -int(edge_count),
                label.casefold(),
            ),
            payload,
        )
        if community_level == "overview":
            overview_scored.append(record)
        else:
            fine_scored.append(record)

    if not overview_scored:
        ordered = [payload for _, payload in sorted(fine_scored)[: max(1, int(limit))]]
        for rank, payload in enumerate(ordered, start=1):
            payload["rank"] = rank
        return ordered

    overview_limit = min(max(1, int(limit)), 2)
    selected_overviews = [payload for _, payload in sorted(overview_scored)[:overview_limit]]
    selected_parent_ids = {str(item["summary_id"]) for item in selected_overviews}
    reranked_fine: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for _, payload in fine_scored:
        parent_summary_id = str(payload.get("parent_summary_id") or "")
        lineage = [str(item) for item in list(payload.get("lineage") or [])]
        in_selected_route = parent_summary_id in selected_parent_ids or any(
            item in selected_parent_ids for item in lineage
        )
        if not in_selected_route and int(payload.get("query_group_hits") or 0) == 0 and int(payload.get("label_group_hits") or 0) == 0:
            continue
        route_score = float(payload.get("hybrid_score") or 0.0) + (0.35 if in_selected_route else 0.0)
        payload["route_score"] = round(route_score, 6)
        reranked_fine.append(
            (
                (
                    -route_score,
                    0 if in_selected_route else 1,
                    -int(payload.get("query_group_hits") or 0),
                    -float(payload.get("query_group_coverage") or 0.0),
                    -int(payload.get("work_count") or 0),
                    str(payload.get("title") or "").casefold(),
                ),
                payload,
            )
        )

    remaining_limit = max(0, int(limit) - len(selected_overviews))
    selected_fine = [payload for _, payload in sorted(reranked_fine)[:remaining_limit]]
    ordered = selected_overviews + selected_fine
    for rank, payload in enumerate(ordered, start=1):
        payload["rank"] = rank
    return ordered


def community_summary_counts(conn: sqlite3.Connection) -> dict[str, int]:
    ensure_community_schema(conn)
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS community_summaries,
          SUM(CASE WHEN collection_id IS NULL THEN 1 ELSE 0 END) AS global_community_summaries,
          SUM(CASE WHEN collection_id IS NOT NULL THEN 1 ELSE 0 END) AS collection_community_summaries,
          SUM(CASE WHEN community_level = 'overview' THEN 1 ELSE 0 END) AS overview_community_summaries,
          SUM(CASE WHEN community_level = 'fine' THEN 1 ELSE 0 END) AS fine_community_summaries
        FROM community_summaries
        """
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()} if row is not None else {}


def _build_overview_descriptors(
    conn: sqlite3.Connection,
    *,
    collection_id: int | None,
    scope_name: str,
    work_map: dict[int, dict[str, Any]],
    edge_map: dict[tuple[int, int], dict[str, Any]],
    descriptors: list[dict[str, Any]],
    representative_limit: int,
    min_edge_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if len(descriptors) < 2:
        return ([], {})

    member_index: dict[int, int] = {}
    for index, descriptor in enumerate(descriptors):
        descriptor["summary_id"] = _summary_id(scope_name=scope_name, community_key=str(descriptor["community_key"]))
        for work_id in list(descriptor.get("member_ids") or []):
            member_index[int(work_id)] = index

    cross_edges: dict[tuple[int, int], dict[str, Any]] = {}
    for (left, right), payload in edge_map.items():
        if float(payload.get("weight") or 0.0) < float(min_edge_weight):
            continue
        left_index = member_index.get(int(left))
        right_index = member_index.get(int(right))
        if left_index is None or right_index is None or left_index == right_index:
            continue
        edge_key = (left_index, right_index) if left_index < right_index else (right_index, left_index)
        pair_payload = cross_edges.setdefault(edge_key, {"raw_weight": 0.0, "signals": Counter()})
        pair_payload["raw_weight"] = float(pair_payload["raw_weight"]) + float(payload.get("weight") or 0.0)
        pair_payload["signals"].update(dict(payload.get("signals") or {}))

    overview_edge_map: dict[tuple[int, int], dict[str, Any]] = {}
    descriptor_ids = list(range(len(descriptors)))
    for left_index, left_descriptor in enumerate(descriptors):
        for right_index in range(left_index + 1, len(descriptors)):
            right_descriptor = descriptors[right_index]
            score, signals = _overview_pair_score(
                left_descriptor,
                right_descriptor,
                cross_payload=cross_edges.get((left_index, right_index)),
            )
            if score < OVERVIEW_MIN_EDGE_WEIGHT:
                continue
            overview_edge_map[(left_index, right_index)] = {
                "weight": round(score, 6),
                "signals": Counter(signals),
            }

    overview_components = _build_components(
        work_ids=descriptor_ids,
        edge_map=overview_edge_map,
        min_edge_weight=OVERVIEW_MIN_EDGE_WEIGHT,
    )
    if not overview_components:
        return ([], {})

    overview_descriptors: list[dict[str, Any]] = []
    parent_map: dict[str, str] = {}
    for component in overview_components:
        child_descriptors = [descriptors[index] for index in component]
        member_ids = sorted(
            {
                int(work_id)
                for descriptor in child_descriptors
                for work_id in list(descriptor.get("member_ids") or [])
            }
        )
        if len(member_ids) < 2:
            continue
        child_summary_ids = [str(descriptor["summary_id"]) for descriptor in child_descriptors]
        representative_works = _representative_works(
            work_map=work_map,
            edge_map=edge_map,
            member_ids=member_ids,
            min_edge_weight=min_edge_weight,
            limit=representative_limit,
        )
        edge_stats = _community_edge_stats(
            edge_map=edge_map,
            member_ids=member_ids,
            min_edge_weight=min_edge_weight,
        )
        top_collaborations = _top_collaboration_counts(conn, work_ids=member_ids)
        top_topics = _top_topic_counts(conn, work_ids=member_ids)
        top_results = _top_value_counts(conn, table="result_objects", field="result_kind", work_ids=member_ids)
        top_methods = _top_value_counts(conn, table="method_objects", field="method_family", work_ids=member_ids)
        label = _community_label(
            representative_works=representative_works,
            collaborations=top_collaborations,
            topics=top_topics,
            result_kinds=top_results,
            method_families=top_methods,
        )
        community_key = f"overview_{_group_key(child_summary_ids)}"
        summary_id = _summary_id(scope_name=scope_name, community_key=community_key)
        for child_summary_id in child_summary_ids:
            parent_map[child_summary_id] = summary_id
        overview_descriptors.append(
            {
                "summary_id": summary_id,
                "community_key": community_key,
                "member_ids": member_ids,
                "label": label,
                "work_count": len(member_ids),
                "edge_count": int(edge_stats["edge_count"]),
                "representative_works": representative_works,
                "signal_mix": list(edge_stats["signal_mix"]),
                "collaborations": top_collaborations,
                "topics": top_topics,
                "result_kinds": top_results,
                "method_families": top_methods,
                "child_summary_ids": child_summary_ids,
                "child_labels": [str(descriptor["label"]) for descriptor in child_descriptors[:4]],
                "source_refs": [*child_summary_ids[:4], *[f"work:{work_id}" for work_id in member_ids[:4]]],
            }
        )
    return (overview_descriptors, parent_map)


def _overview_pair_score(
    left_descriptor: dict[str, Any],
    right_descriptor: dict[str, Any],
    *,
    cross_payload: dict[str, Any] | None,
) -> tuple[float, Counter]:
    score = 0.0
    signals = Counter()
    if cross_payload is not None:
        left_size = max(1, len(list(left_descriptor.get("member_ids") or [])))
        right_size = max(1, len(list(right_descriptor.get("member_ids") or [])))
        normalized = float(cross_payload.get("raw_weight") or 0.0) / max(1.0, math.sqrt(left_size * right_size))
        contribution = min(OVERVIEW_CROSS_EDGE_CAP, normalized * 0.22)
        if contribution > 0:
            score += contribution
            signals["cross_edge"] += 1
            signals.update(dict(cross_payload.get("signals") or {}))
    score += _shared_signal_contribution(
        left_descriptor.get("topics"),
        right_descriptor.get("topics"),
        weight=OVERVIEW_TOPIC_WEIGHT,
        signal_name="shared_topic",
        signals=signals,
    )
    score += _shared_signal_contribution(
        left_descriptor.get("result_kinds"),
        right_descriptor.get("result_kinds"),
        weight=OVERVIEW_RESULT_WEIGHT,
        signal_name="shared_result",
        signals=signals,
    )
    score += _shared_signal_contribution(
        left_descriptor.get("method_families"),
        right_descriptor.get("method_families"),
        weight=OVERVIEW_METHOD_WEIGHT,
        signal_name="shared_method",
        signals=signals,
    )
    score += _shared_signal_contribution(
        left_descriptor.get("collaborations"),
        right_descriptor.get("collaborations"),
        weight=OVERVIEW_COLLABORATION_WEIGHT,
        signal_name="shared_collaboration",
        signals=signals,
    )
    return (score, signals)


def _shared_signal_contribution(
    left_values: Any,
    right_values: Any,
    *,
    weight: float,
    signal_name: str,
    signals: Counter,
) -> float:
    shared = _count_label_set(left_values) & _count_label_set(right_values)
    if not shared:
        return 0.0
    signals[signal_name] += len(shared)
    return float(weight)


def _aggregate_edges(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int],
    collection_id: int | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    edge_map: dict[tuple[int, int], dict[str, Any]] = {}
    work_id_set = {int(work_id) for work_id in work_ids}
    _load_similarity_edges(conn, work_id_set=work_id_set, collection_id=collection_id, edge_map=edge_map)
    _load_bibliographic_edges(conn, work_id_set=work_id_set, collection_id=collection_id, edge_map=edge_map)
    _load_co_citation_edges(conn, work_id_set=work_id_set, collection_id=collection_id, edge_map=edge_map)
    _load_group_overlap_edges(
        conn,
        table_sql="""
            SELECT wt.topic_id AS key_id, wt.work_id AS work_id
            FROM work_topics wt
            {scope_join}
            ORDER BY wt.topic_id, wt.work_id
        """,
        scope_work_column="wt.work_id",
        collection_id=collection_id,
        edge_map=edge_map,
        work_id_set=work_id_set,
        weight=TOPIC_EDGE_WEIGHT,
        max_group_size=24,
        signal_name="topic_overlap",
    )
    _load_group_overlap_edges(
        conn,
        table_sql="""
            SELECT ro.result_kind AS key_id, ro.work_id AS work_id
            FROM result_objects ro
            {scope_join}
            WHERE COALESCE(ro.result_kind, '') <> ''
            ORDER BY ro.result_kind, ro.work_id
        """,
        scope_work_column="ro.work_id",
        collection_id=collection_id,
        edge_map=edge_map,
        work_id_set=work_id_set,
        weight=RESULT_EDGE_WEIGHT,
        max_group_size=16,
        signal_name="result_overlap",
    )
    _load_group_overlap_edges(
        conn,
        table_sql="""
            SELECT mo.method_family AS key_id, mo.work_id AS work_id
            FROM method_objects mo
            {scope_join}
            WHERE COALESCE(mo.method_family, '') <> ''
            ORDER BY mo.method_family, mo.work_id
        """,
        scope_work_column="mo.work_id",
        collection_id=collection_id,
        edge_map=edge_map,
        work_id_set=work_id_set,
        weight=METHOD_EDGE_WEIGHT,
        max_group_size=16,
        signal_name="method_overlap",
    )
    _load_group_overlap_edges(
        conn,
        table_sql="""
            SELECT wc.collaboration_id AS key_id, wc.work_id AS work_id
            FROM work_collaborations wc
            {scope_join}
            ORDER BY wc.collaboration_id, wc.work_id
        """,
        scope_work_column="wc.work_id",
        collection_id=collection_id,
        edge_map=edge_map,
        work_id_set=work_id_set,
        weight=COLLABORATION_EDGE_WEIGHT,
        max_group_size=8,
        signal_name="collaboration_overlap",
    )
    return edge_map


def _load_similarity_edges(
    conn: sqlite3.Connection,
    *,
    work_id_set: set[int],
    collection_id: int | None,
    edge_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    collection_sql = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_sql = """
        WHERE src_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
          AND dst_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
        """
        params.extend([collection_id, collection_id])
    rows = conn.execute(
        f"""
        SELECT src_work_id, dst_work_id, MAX(score) AS score
        FROM similarity_edges
        {collection_sql}
        GROUP BY src_work_id, dst_work_id
        ORDER BY src_work_id, dst_work_id
        """,
        params,
    ).fetchall()
    for row in rows:
        left = int(row["src_work_id"])
        right = int(row["dst_work_id"])
        if left not in work_id_set or right not in work_id_set:
            continue
        contribution = max(0.0, min(1.0, float(row["score"] or 0.0)))
        _add_edge_signal(edge_map, left, right, signal="similarity", contribution=contribution)


def _load_bibliographic_edges(
    conn: sqlite3.Connection,
    *,
    work_id_set: set[int],
    collection_id: int | None,
    edge_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    collection_sql = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_sql = """
        WHERE src_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
          AND dst_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
        """
        params.extend([collection_id, collection_id])
    rows = conn.execute(
        f"""
        SELECT src_work_id, dst_work_id, shared_reference_count, score
        FROM bibliographic_coupling_edges
        {collection_sql}
        ORDER BY src_work_id, dst_work_id
        """,
        params,
    ).fetchall()
    for row in rows:
        left = int(row["src_work_id"])
        right = int(row["dst_work_id"])
        if left not in work_id_set or right not in work_id_set:
            continue
        shared = int(row["shared_reference_count"] or 0)
        score = float(row["score"] or 0.0)
        contribution = min(0.6, max(score * 1.6, min(0.35, shared * 0.06)))
        _add_edge_signal(edge_map, left, right, signal="bibliographic", contribution=contribution)


def _load_co_citation_edges(
    conn: sqlite3.Connection,
    *,
    work_id_set: set[int],
    collection_id: int | None,
    edge_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    collection_sql = ""
    params: list[Any] = []
    if collection_id is not None:
        collection_sql = """
        WHERE src_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
          AND dst_work_id IN (SELECT work_id FROM collection_works WHERE collection_id = ?)
        """
        params.extend([collection_id, collection_id])
    rows = conn.execute(
        f"""
        SELECT src_work_id, dst_work_id, shared_citer_count, score
        FROM co_citation_edges
        {collection_sql}
        ORDER BY src_work_id, dst_work_id
        """,
        params,
    ).fetchall()
    for row in rows:
        left = int(row["src_work_id"])
        right = int(row["dst_work_id"])
        if left not in work_id_set or right not in work_id_set:
            continue
        shared = int(row["shared_citer_count"] or 0)
        score = float(row["score"] or 0.0)
        contribution = min(0.6, max(score * 1.6, min(0.35, shared * 0.06)))
        _add_edge_signal(edge_map, left, right, signal="co_citation", contribution=contribution)


def _load_group_overlap_edges(
    conn: sqlite3.Connection,
    *,
    table_sql: str,
    scope_work_column: str,
    collection_id: int | None,
    edge_map: dict[tuple[int, int], dict[str, Any]],
    work_id_set: set[int],
    weight: float,
    max_group_size: int,
    signal_name: str,
) -> None:
    scope_join = ""
    params: list[Any] = []
    if collection_id is not None:
        scope_join = f"JOIN collection_works cw ON cw.work_id = {scope_work_column} AND cw.collection_id = ?"
        params.append(collection_id)
    rows = conn.execute(table_sql.format(scope_join=scope_join), params).fetchall()
    grouped: dict[str, list[int]] = {}
    for row in rows:
        key = str(row["key_id"] or "").strip()
        if not key:
            continue
        work_id = int(row["work_id"])
        if work_id not in work_id_set:
            continue
        grouped.setdefault(key, []).append(work_id)
    for members in grouped.values():
        deduped = sorted({int(work_id) for work_id in members})
        if len(deduped) < 2 or len(deduped) > max_group_size:
            continue
        for index, left in enumerate(deduped):
            for right in deduped[index + 1:]:
                _add_edge_signal(edge_map, left, right, signal=signal_name, contribution=weight)


def _add_edge_signal(
    edge_map: dict[tuple[int, int], dict[str, Any]],
    left: int,
    right: int,
    *,
    signal: str,
    contribution: float,
) -> None:
    if contribution <= 0 or left == right:
        return
    edge_key = (left, right) if left < right else (right, left)
    payload = edge_map.setdefault(edge_key, {"weight": 0.0, "signals": Counter()})
    payload["weight"] = float(payload["weight"]) + float(contribution)
    payload["signals"][signal] += 1


def _build_components(
    *,
    work_ids: list[int],
    edge_map: dict[tuple[int, int], dict[str, Any]],
    min_edge_weight: float,
) -> list[list[int]]:
    parent = {int(work_id): int(work_id) for work_id in work_ids}

    def find(node: int) -> int:
        current = int(node)
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for (left, right), payload in edge_map.items():
        if float(payload.get("weight") or 0.0) >= float(min_edge_weight):
            union(left, right)

    grouped: dict[int, list[int]] = {}
    for work_id in work_ids:
        grouped.setdefault(find(int(work_id)), []).append(int(work_id))
    components = [sorted(member_ids) for member_ids in grouped.values() if len(member_ids) >= 2]
    components.sort(key=lambda member_ids: (-len(member_ids), member_ids[0]))
    return components


def _representative_works(
    *,
    work_map: dict[int, dict[str, Any]],
    edge_map: dict[tuple[int, int], dict[str, Any]],
    member_ids: list[int],
    min_edge_weight: float,
    limit: int,
) -> list[dict[str, Any]]:
    centrality = Counter()
    member_id_set = {int(work_id) for work_id in member_ids}
    for (left, right), payload in edge_map.items():
        weight = float(payload.get("weight") or 0.0)
        if weight < float(min_edge_weight):
            continue
        if left in member_id_set and right in member_id_set:
            centrality[left] += weight
            centrality[right] += weight

    ordered = sorted(
        member_ids,
        key=lambda work_id: (
            -float(centrality.get(int(work_id), 0.0)),
            -int(work_map.get(int(work_id), {}).get("citation_count") or 0),
            -int(work_map.get(int(work_id), {}).get("year") or 0),
            str(work_map.get(int(work_id), {}).get("title") or "").casefold(),
            int(work_id),
        ),
    )[: max(1, int(limit))]
    out: list[dict[str, Any]] = []
    for work_id in ordered:
        payload = dict(work_map.get(int(work_id)) or {})
        if not payload:
            continue
        out.append(
            {
                "work_id": int(work_id),
                "title": str(payload.get("title") or ""),
                "year": payload.get("year"),
                "canonical_source": str(payload.get("canonical_source") or ""),
                "canonical_id": str(payload.get("canonical_id") or ""),
                "centrality": round(float(centrality.get(int(work_id), 0.0)), 4),
            }
        )
    return out


def _community_edge_stats(
    *,
    edge_map: dict[tuple[int, int], dict[str, Any]],
    member_ids: list[int],
    min_edge_weight: float,
) -> dict[str, Any]:
    member_id_set = {int(work_id) for work_id in member_ids}
    edge_count = 0
    signal_mix = Counter()
    for (left, right), payload in edge_map.items():
        if left not in member_id_set or right not in member_id_set:
            continue
        if float(payload.get("weight") or 0.0) < float(min_edge_weight):
            continue
        edge_count += 1
        signal_mix.update(dict(payload.get("signals") or {}))
    return {
        "edge_count": edge_count,
        "signal_mix": [f"{key}={signal_mix[key]}" for key in sorted(signal_mix)],
    }


def _community_label(
    *,
    representative_works: list[dict[str, Any]],
    collaborations: list[str],
    topics: list[str],
    result_kinds: list[str],
    method_families: list[str],
) -> str:
    topic = _count_label(topics[0]) if topics else ""
    collaboration = _count_label(collaborations[0]) if collaborations else ""
    result_kind = _count_label(result_kinds[0]) if result_kinds else ""
    method_family = _count_label(method_families[0]) if method_families else ""
    if topic and collaboration:
        return f"{collaboration} / {topic} community"
    if topic:
        return f"{topic} community"
    if result_kind and collaboration:
        return f"{collaboration} / {result_kind} community"
    if result_kind:
        return f"{result_kind} community"
    if method_family and collaboration:
        return f"{collaboration} / {method_family} community"
    if method_family:
        return f"{method_family} community"
    if collaboration:
        return f"{collaboration} literature community"
    if representative_works:
        return f"Community around {representative_works[0].get('title')}"
    return "Literature community"


def _compose_summary_text(
    *,
    label: str,
    community_level: str,
    work_count: int,
    edge_count: int,
    representative_works: list[dict[str, Any]],
    signal_mix: list[str],
    collaborations: list[str],
    topics: list[str],
    result_kinds: list[str],
    method_families: list[str],
    child_labels: list[str] | None = None,
    parent_label: str | None = None,
) -> str:
    parts = [label, f"level={community_level}", f"works={work_count}", f"edges={edge_count}"]
    if parent_label:
        parts.append(f"parent: {parent_label}")
    if child_labels:
        parts.append("child communities: " + "; ".join(child_labels[:3]))
    if signal_mix:
        parts.append("signals: " + ", ".join(signal_mix[:5]))
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
    if topics:
        parts.append("topics: " + ", ".join(topics[:3]))
    if result_kinds:
        parts.append("result kinds: " + ", ".join(result_kinds[:3]))
    if method_families:
        parts.append("method families: " + ", ".join(method_families[:3]))
    return " | ".join(part for part in parts if part)


def _summary_payload(
    *,
    scope_name: str,
    collection_id: int | None,
    community_key: str,
    community_level: str,
    parent_summary_id: str | None,
    child_summary_ids: list[str],
    lineage: list[str],
    label: str,
    summary_text: str,
    work_count: int,
    edge_count: int,
    representative_works: list[dict[str, Any]],
    source_refs: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "summary_id": _summary_id(scope_name=scope_name, community_key=community_key),
        "collection_id": collection_id,
        "community_key": community_key,
        "algorithm": ALGORITHM,
        "community_level": community_level,
        "parent_summary_id": parent_summary_id,
        "label": label,
        "summary_text": summary_text,
        "status": "ready",
        "work_count": int(work_count),
        "edge_count": int(edge_count),
        "child_summary_ids_json": json.dumps(child_summary_ids, ensure_ascii=False),
        "lineage_json": json.dumps(lineage, ensure_ascii=False),
        "representative_works_json": json.dumps(representative_works, ensure_ascii=False),
        "source_refs_json": json.dumps(source_refs, ensure_ascii=False),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }


def _upsert_community_summary(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO community_summaries (
          summary_id, collection_id, community_key, algorithm, community_level, parent_summary_id,
          label, summary_text, status, work_count, edge_count, child_summary_ids_json, lineage_json,
          representative_works_json, source_refs_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(summary_id) DO UPDATE SET
          collection_id = excluded.collection_id,
          community_key = excluded.community_key,
          algorithm = excluded.algorithm,
          community_level = excluded.community_level,
          parent_summary_id = excluded.parent_summary_id,
          label = excluded.label,
          summary_text = excluded.summary_text,
          status = excluded.status,
          work_count = excluded.work_count,
          edge_count = excluded.edge_count,
          child_summary_ids_json = excluded.child_summary_ids_json,
          lineage_json = excluded.lineage_json,
          representative_works_json = excluded.representative_works_json,
          source_refs_json = excluded.source_refs_json,
          metadata_json = excluded.metadata_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            payload["summary_id"],
            payload["collection_id"],
            payload["community_key"],
            payload["algorithm"],
            payload["community_level"],
            payload["parent_summary_id"],
            payload["label"],
            payload["summary_text"],
            payload["status"],
            payload["work_count"],
            payload["edge_count"],
            payload["child_summary_ids_json"],
            payload["lineage_json"],
            payload["representative_works_json"],
            payload["source_refs_json"],
            payload["metadata_json"],
        ),
    )


def _scope_summary_count(conn: sqlite3.Connection, *, collection: str | None) -> int:
    if collection:
        collection_id = _collection_id(conn, collection)
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM community_summaries WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()
        return int(row["n"] or 0) if row is not None else 0
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM community_summaries WHERE collection_id IS NULL"
    ).fetchone()
    return int(row["n"] or 0) if row is not None else 0


def _scope_has_stale_algorithm(conn: sqlite3.Connection, *, collection: str | None) -> bool:
    if collection:
        collection_id = _collection_id(conn, collection)
        row = conn.execute(
            """
            SELECT COUNT(*) AS stale
            FROM community_summaries
            WHERE collection_id = ? AND algorithm <> ?
            """,
            (collection_id, ALGORITHM),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COUNT(*) AS stale
            FROM community_summaries
            WHERE collection_id IS NULL AND algorithm <> ?
            """,
            (ALGORITHM,),
        ).fetchone()
    return bool(int(row["stale"] or 0)) if row is not None else False


def _load_scope_rows(conn: sqlite3.Connection, *, collection: str | None) -> list[sqlite3.Row]:
    if collection:
        collection_id = _collection_id(conn, collection)
        return conn.execute(
            """
            SELECT summary_id, algorithm, community_level, parent_summary_id, label, summary_text, status,
                   work_count, edge_count, child_summary_ids_json, lineage_json,
                   representative_works_json, source_refs_json, metadata_json
            FROM community_summaries
            WHERE collection_id = ?
            ORDER BY CASE community_level WHEN 'overview' THEN 0 ELSE 1 END, work_count DESC, edge_count DESC, label
            """,
            (collection_id,),
        ).fetchall()
    return conn.execute(
        """
        SELECT summary_id, algorithm, community_level, parent_summary_id, label, summary_text, status,
               work_count, edge_count, child_summary_ids_json, lineage_json,
               representative_works_json, source_refs_json, metadata_json
        FROM community_summaries
        WHERE collection_id IS NULL
        ORDER BY CASE community_level WHEN 'overview' THEN 0 ELSE 1 END, work_count DESC, edge_count DESC, label
        """
    ).fetchall()


def _clear_scope(conn: sqlite3.Connection, *, collection_id: int | None) -> None:
    if collection_id is None:
        conn.execute("DELETE FROM community_summaries WHERE collection_id IS NULL")
        return
    conn.execute("DELETE FROM community_summaries WHERE collection_id = ?", (collection_id,))


def _scope_work_ids(conn: sqlite3.Connection, *, collection_id: int | None) -> list[int]:
    if collection_id is None:
        rows = conn.execute("SELECT work_id FROM works ORDER BY work_id").fetchall()
    else:
        rows = conn.execute(
            """
            SELECT work_id
            FROM collection_works
            WHERE collection_id = ?
            ORDER BY work_id
            """,
            (collection_id,),
        ).fetchall()
    return [int(row["work_id"]) for row in rows]


def _work_payload_map(conn: sqlite3.Connection, *, work_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not work_ids:
        return {}
    rows = conn.execute(
        """
        SELECT work_id, title, year, canonical_source, canonical_id, citation_count
        FROM works
        WHERE work_id IN ({placeholders})
        ORDER BY work_id
        """.format(placeholders=",".join("?" for _ in work_ids)),
        work_ids,
    ).fetchall()
    return {
        int(row["work_id"]): {
            "title": str(row["title"] or ""),
            "year": int(row["year"]) if row["year"] is not None else None,
            "canonical_source": str(row["canonical_source"] or ""),
            "canonical_id": str(row["canonical_id"] or ""),
            "citation_count": int(row["citation_count"] or 0),
        }
        for row in rows
    }


def _top_collaboration_counts(conn: sqlite3.Connection, *, work_ids: list[int], limit: int = 3) -> list[str]:
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


def _top_topic_counts(conn: sqlite3.Connection, *, work_ids: list[int], limit: int = 3) -> list[str]:
    if not work_ids:
        return []
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        f"""
        SELECT t.label AS value, COUNT(*) AS n
        FROM work_topics wt
        JOIN topics t ON t.topic_id = wt.topic_id
        WHERE wt.work_id IN ({placeholders})
        GROUP BY t.label
        ORDER BY n DESC, t.label
        LIMIT ?
        """,
        [*work_ids, limit],
    ).fetchall()
    return [f"{str(row['value'])}={int(row['n'])}" for row in rows if row["value"]]


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


def _community_key(member_ids: list[int]) -> str:
    return _group_key([str(int(work_id)) for work_id in sorted(member_ids)])


def _summary_id(*, scope_name: str, community_key: str) -> str:
    return f"community_summary:{_safe_key(scope_name)}:{community_key}"


def _count_label(value: str) -> str:
    return str(value or "").split("=", 1)[0].strip()


def _count_label_set(values: Any) -> set[str]:
    return {
        _count_label(str(item))
        for item in list(values or [])
        if _count_label(str(item))
    }


def _group_key(values: list[str]) -> str:
    digest = hashlib.sha1(",".join(sorted(str(value) for value in values)).encode("utf-8")).hexdigest()
    return digest[:16]


def _parent_label_from_descriptors(
    descriptors: list[dict[str, Any]],
    *,
    parent_summary_id: str | None,
    scope_name: str,
) -> str | None:
    if not parent_summary_id:
        return None
    for descriptor in descriptors:
        descriptor_summary_id = str(
            descriptor.get("summary_id")
            or _summary_id(scope_name=scope_name, community_key=str(descriptor.get("community_key") or ""))
        )
        if descriptor_summary_id == parent_summary_id:
            label = str(descriptor.get("label") or "").strip()
            return label or None
    return None


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
