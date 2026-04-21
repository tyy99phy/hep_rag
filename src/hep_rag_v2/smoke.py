from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from hep_rag_v2.config import resolve_embedding_settings, runtime_collection_config
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.graph import rebuild_graph_edges
from hep_rag_v2.pipeline import ingest_online, retrieve
from hep_rag_v2.search import rebuild_search_indices, search_index_counts
from hep_rag_v2.service.workspace import workspace_status_payload
from hep_rag_v2.vector import configure_embedding_runtime, rebuild_vector_indices, vector_index_counts


ProgressCallback = Callable[[str], None] | None

DEFAULT_SMOKE_CORPORA: dict[str, dict[str, Any]] = {
    "cms_atlas_2k": {
        "ingest_query": 'collaboration:"CMS" or collaboration:"ATLAS"',
        "validation_queries": [
            {"query": "CMS same-sign WW latest result", "target": "works", "limit": 8},
            {"query": "ATLAS Higgs combination", "target": "works", "limit": 8},
            {"query": "CMS jet tagging graph neural network", "target": "works", "limit": 8},
        ],
    },
    "cms_2k": {
        "ingest_query": 'collaboration:"CMS"',
        "validation_queries": [
            {"query": "CMS same-sign WW latest result", "target": "works", "limit": 8},
            {"query": "CMS jet tagging graph neural network", "target": "works", "limit": 8},
        ],
    },
    "atlas_2k": {
        "ingest_query": 'collaboration:"ATLAS"',
        "validation_queries": [
            {"query": "ATLAS Higgs combination", "target": "works", "limit": 8},
            {"query": "ATLAS vector boson scattering", "target": "works", "limit": 8},
        ],
    },
}


def load_smoke_queries(path: str | Path) -> list[dict[str, Any]]:
    query_path = Path(path).expanduser().resolve()
    suffix = query_path.suffix.lower()
    raw_text = query_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text) if suffix == ".json" else yaml.safe_load(raw_text)
    if isinstance(payload, dict):
        payload = payload.get("queries") or []
    if not isinstance(payload, list):
        raise ValueError(f"Smoke queries must be a list or a mapping with 'queries': {query_path}")
    return [_normalize_query_entry(item) for item in payload]


def run_metadata_smoke(
    config: dict[str, Any],
    *,
    corpus: str = "cms_atlas_2k",
    ingest_query: str | None = None,
    collection_name: str | None = None,
    limit: int = 2000,
    download_limit: int = 0,
    parse_limit: int = 0,
    max_parallelism: int | None = None,
    build_search: bool = True,
    build_vectors: bool = True,
    build_graph: bool = False,
    embedding_profile: str | None = None,
    queries_file: str | Path | None = None,
    validation_queries: list[str] | None = None,
    query_target: str | None = "works",
    query_limit: int = 8,
    export_report: str | Path | None = None,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    effective_config = copy.deepcopy(config)
    if embedding_profile:
        effective_config.setdefault("profiles", {})["embedding"] = embedding_profile
        effective_config.setdefault("embedding", {})["profile"] = embedding_profile

    preset = DEFAULT_SMOKE_CORPORA.get(corpus, {})
    effective_query = str(ingest_query or preset.get("ingest_query") or "").strip()
    if not effective_query:
        raise ValueError("smoke-metadata requires either a known --corpus preset or an explicit --ingest-query.")

    collection_payload = runtime_collection_config(effective_config, name=collection_name or corpus)
    collection = collection_payload["name"]
    resolved_queries = _resolve_validation_queries(
        corpus=corpus,
        queries_file=queries_file,
        validation_queries=validation_queries,
        query_target=query_target,
        query_limit=query_limit,
    )

    report: dict[str, Any] = {
        "mode": "metadata_smoke",
        "corpus": corpus,
        "collection": collection,
        "ingest_query": effective_query,
        "limit_requested": max(1, int(limit)),
        "download_limit": max(0, int(download_limit)),
        "parse_limit": max(0, int(parse_limit)),
        "build_search": bool(build_search),
        "build_vectors": bool(build_vectors),
        "build_graph": bool(build_graph),
        "validation_query_count": len(resolved_queries),
        "validation_queries": resolved_queries,
        "steps": {},
    }

    embedding_settings = resolve_embedding_settings(effective_config)
    report["embedding"] = {
        "profile": str((effective_config.get("profiles") or {}).get("embedding") or ""),
        "model": str(embedding_settings.get("model") or ""),
        "dim": int(embedding_settings.get("dim") or 0),
        "runtime": dict(embedding_settings.get("runtime") or {}),
    }

    overall_started = time.perf_counter()

    report["steps"]["ingest"] = _run_timed(
        lambda: ingest_online(
            effective_config,
            query=effective_query,
            limit=max(1, int(limit)),
            collection_name=collection,
            max_parallelism=max_parallelism,
            download_limit=max(0, int(download_limit)),
            parse_limit=max(0, int(parse_limit)),
            skip_parse=max(0, int(parse_limit)) == 0,
            progress=progress,
        )
    )

    if build_search:
        report["steps"]["sync_search"] = _run_timed(
            lambda: _rebuild_search_snapshot(progress=progress)
        )
    else:
        report["steps"]["sync_search"] = {"enabled": False}

    if build_vectors:
        configure_embedding_runtime(
            model=str(embedding_settings.get("model") or ""),
            settings=embedding_settings,
        )
        report["steps"]["sync_vectors"] = _run_timed(
            lambda: _rebuild_vector_snapshot(
                model=str(embedding_settings.get("model") or ""),
                dim=int(embedding_settings.get("dim") or 0),
                progress=progress,
            )
        )
    else:
        report["steps"]["sync_vectors"] = {"enabled": False}

    if build_graph:
        report["steps"]["sync_graph"] = _run_timed(
            lambda: _rebuild_graph_snapshot(
                collection=collection,
                model=str(embedding_settings.get("model") or ""),
                progress=progress,
            )
        )
    else:
        report["steps"]["sync_graph"] = {"enabled": False}

    query_started = time.perf_counter()
    query_items: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for item in resolved_queries:
        started = time.perf_counter()
        payload = retrieve(
            effective_config,
            query=str(item["query"]),
            limit=int(item.get("limit") or query_limit),
            target=str(item.get("target") or query_target or "works"),
            collection_name=collection,
            model=str(embedding_settings.get("model") or None),
            progress=progress,
        )
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        latencies_ms.append(elapsed_ms)
        query_items.append(
            {
                "query": str(item["query"]),
                "target": str(item.get("target") or query_target or "works"),
                "limit": int(item.get("limit") or query_limit),
                "latency_ms": round(elapsed_ms, 3),
                "works": len(payload.get("works") or []),
                "chunks": len(payload.get("evidence_chunks") or []),
                "community_summaries": len(payload.get("community_summaries") or []),
                "ontology_summaries": len(payload.get("ontology_summaries") or []),
                "top_work_titles": [
                    str(row.get("raw_title") or "")
                    for row in (payload.get("works") or [])[:3]
                    if str(row.get("raw_title") or "").strip()
                ],
            }
        )
    report["steps"]["queries"] = {
        "enabled": bool(resolved_queries),
        "count": len(query_items),
        "seconds": round(max(0.0, time.perf_counter() - query_started), 6),
        "latency_ms": _latency_summary_ms(latencies_ms),
        "items": query_items,
    }

    report["workspace_status"] = workspace_status_payload()
    report["total_seconds"] = round(max(0.0, time.perf_counter() - overall_started), 6)

    if export_report is not None:
        export_path = Path(export_report).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(export_path)
        export_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def _rebuild_search_snapshot(*, progress: ProgressCallback = None) -> dict[str, Any]:
    ensure_db()
    with connect() as conn:
        summary = rebuild_search_indices(conn, target="all")
        conn.commit()
        summary.update(search_index_counts(conn))
    return summary


def _rebuild_vector_snapshot(*, model: str, dim: int, progress: ProgressCallback = None) -> dict[str, Any]:
    ensure_db()
    with connect() as conn:
        summary = rebuild_vector_indices(conn, target="all", model=model, dim=dim, progress=progress)
        conn.commit()
        summary.update(vector_index_counts(conn))
    return summary


def _rebuild_graph_snapshot(*, collection: str, model: str, progress: ProgressCallback = None) -> dict[str, Any]:
    ensure_db()
    with connect() as conn:
        summary = rebuild_graph_edges(
            conn,
            target="all",
            collection=collection,
            min_shared=2,
            similarity_model=model,
            similarity_top_k=10,
            similarity_min_score=0.35,
            progress=progress,
        )
        conn.commit()
    return summary


def _resolve_validation_queries(
    *,
    corpus: str,
    queries_file: str | Path | None,
    validation_queries: list[str] | None,
    query_target: str | None,
    query_limit: int,
) -> list[dict[str, Any]]:
    if queries_file is not None:
        return load_smoke_queries(queries_file)
    if validation_queries:
        return [
            {
                "query": str(item),
                "target": str(query_target or "works"),
                "limit": int(query_limit),
            }
            for item in validation_queries
            if str(item).strip()
        ]
    preset = DEFAULT_SMOKE_CORPORA.get(corpus, {})
    return [_normalize_query_entry(item, target=query_target, limit=query_limit) for item in (preset.get("validation_queries") or [])]


def _normalize_query_entry(
    item: Any,
    *,
    target: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if isinstance(item, str):
        payload: dict[str, Any] = {"query": item}
    elif isinstance(item, dict):
        payload = dict(item)
    else:
        raise ValueError(f"Unsupported smoke query entry: {item!r}")

    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError(f"Smoke query entry is missing a non-empty query: {item!r}")

    normalized = {"query": query}
    resolved_target = payload.get("target", target)
    if resolved_target is not None:
        normalized["target"] = str(resolved_target)
    resolved_limit = payload.get("limit", limit)
    if resolved_limit is not None:
        normalized["limit"] = max(1, int(resolved_limit))
    return normalized


def _latency_summary_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(float(item) for item in samples)
    return {
        "p50": round(_percentile(ordered, 50.0), 3),
        "p95": round(_percentile(ordered, 95.0), 3),
    }


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    position = (len(samples) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(samples[lower])
    fraction = position - lower
    return float(samples[lower] + (samples[upper] - samples[lower]) * fraction)


def _run_timed(fn) -> dict[str, Any]:
    started = time.perf_counter()
    summary = fn()
    return {
        "enabled": True,
        "seconds": round(max(0.0, time.perf_counter() - started), 6),
        "summary": summary,
    }
