from __future__ import annotations

import json
import math
import resource
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hep_rag_v2 import paths
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit
from hep_rag_v2.search import rebuild_search_indices, search_works_bm25
from hep_rag_v2.vector import DEFAULT_VECTOR_DIM, DEFAULT_VECTOR_MODEL, rebuild_vector_indices, search_works_vector


DEFAULT_COLLECTION = "scale-benchmark"


class _TierKey(str):
    _ORDER = {"10k": 0, "100k": 1, "50k": 2}

    def __lt__(self, other: object) -> bool:
        return self._ORDER.get(str(self), 99) < self._ORDER.get(str(other), 99)


@dataclass(frozen=True, slots=True)
class BenchmarkTier:
    name: str
    work_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    workspace_root: Path | None = None
    collection_name: str = DEFAULT_COLLECTION
    query_repeats: int = 3
    query_limit: int = 10
    build_search: bool = True
    build_vectors: bool = True
    dry_run: bool = False
    export_filename: str | None = None
    vector_model: str = DEFAULT_VECTOR_MODEL
    vector_dim: int = DEFAULT_VECTOR_DIM


def default_scale_tiers() -> dict[str, BenchmarkTier]:
    return {
        _TierKey("10k"): BenchmarkTier(name="10k", work_count=10_000),
        _TierKey("50k"): BenchmarkTier(name="50k", work_count=50_000),
        _TierKey("100k"): BenchmarkTier(name="100k", work_count=100_000),
    }


def benchmark_scale(tier: BenchmarkTier | str, *, options: BenchmarkOptions | None = None) -> dict[str, Any]:
    opts = options or BenchmarkOptions()
    tier_obj = default_scale_tiers()[tier] if isinstance(tier, str) else tier
    export_path = _resolve_export_path(tier_obj=tier_obj, options=opts)
    if opts.dry_run:
        return {
            "tier": tier_obj.name,
            "work_count": tier_obj.work_count,
            "dry_run": True,
            "collection_name": opts.collection_name,
            "query_repeats": opts.query_repeats,
            "query_limit": opts.query_limit,
            "queries": list(_benchmark_queries()),
            "export_path": str(export_path),
            "workspace_root": str((opts.workspace_root or paths.workspace_root()).resolve()),
        }

    original_root = paths.workspace_root()
    try:
        if opts.workspace_root is not None:
            paths.set_workspace_root(opts.workspace_root)
        ensure_db()
        started_at = datetime.now(UTC)
        with connect() as conn:
            collection_id = upsert_collection(conn, {"name": opts.collection_name, "label": "Scale Benchmark"})
            ingest_started = time.perf_counter()
            created = 0
            for index in range(tier_obj.work_count):
                summary = upsert_work_from_hit(conn, collection_id=collection_id, hit=_synthetic_hit(index))
                created += int(summary["created"]) + int(summary["updated"])
            ingest_seconds = time.perf_counter() - ingest_started

            search_summary = {
                "enabled": bool(opts.build_search),
                "index_rows": 0,
                "build_seconds": 0.0,
                "latency_ms": {"p50": 0.0, "p95": 0.0},
            }
            if opts.build_search:
                built_at = time.perf_counter()
                search_summary["index_rows"] = int(rebuild_search_indices(conn, target="works")["works"])
                search_summary["build_seconds"] = round(time.perf_counter() - built_at, 6)
                search_summary["latency_ms"] = _latency_summary_ms(
                    _measure_queries(
                        lambda query: search_works_bm25(
                            conn,
                            query=query,
                            collection=opts.collection_name,
                            limit=opts.query_limit,
                        ),
                        repeats=opts.query_repeats,
                    )
                )

            vector_summary = {
                "enabled": bool(opts.build_vectors),
                "model": opts.vector_model,
                "index_rows": 0,
                "build_seconds": 0.0,
                "latency_ms": {"p50": 0.0, "p95": 0.0},
            }
            if opts.build_vectors:
                built_at = time.perf_counter()
                vector_summary["index_rows"] = int(
                    rebuild_vector_indices(
                        conn,
                        target="works",
                        model=opts.vector_model,
                        dim=opts.vector_dim,
                    )["works"]
                )
                vector_summary["build_seconds"] = round(time.perf_counter() - built_at, 6)
                vector_summary["latency_ms"] = _latency_summary_ms(
                    _measure_queries(
                        lambda query: search_works_vector(
                            conn,
                            query=query,
                            collection=opts.collection_name,
                            limit=opts.query_limit,
                            model=opts.vector_model,
                        ),
                        repeats=opts.query_repeats,
                    )
                )

            result = {
                "tier": tier_obj.name,
                "work_count": tier_obj.work_count,
                "dry_run": False,
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "workspace_root": str(paths.workspace_root()),
                "collection_name": opts.collection_name,
                "queries": list(_benchmark_queries()),
                "ingest": {
                    "work_count": tier_obj.work_count,
                    "rows_written": created,
                    "seconds": round(ingest_seconds, 6),
                    "works_per_second": round(tier_obj.work_count / ingest_seconds, 3) if ingest_seconds else float("inf"),
                },
                "search": search_summary,
                "vectors": vector_summary,
                "sqlite_size_mb": round(_sqlite_size_mb(), 3),
                "memory_peak_mb": round(_peak_rss_mb(), 3),
                "export_path": str(export_path),
            }
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        if opts.workspace_root is not None:
            paths.set_workspace_root(original_root)


def _benchmark_queries() -> tuple[str, ...]:
    return (
        "CMS same-sign WW vector boson scattering",
        "ATLAS Higgs diphoton cross section",
        "LHC supersymmetry jets missing transverse momentum",
    )


def _measure_queries(fn: Any, *, repeats: int) -> list[float]:
    latencies: list[float] = []
    for _ in range(max(1, repeats)):
        for query in _benchmark_queries():
            started = time.perf_counter()
            fn(query)
            latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def _latency_summary_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(samples)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(_percentile(ordered, 95), 3),
    }


def _percentile(samples: list[float], percentile: int) -> float:
    if len(samples) == 1:
        return float(samples[0])
    rank = ((percentile / 100.0) * (len(samples) - 1))
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(samples[lower])
    weight = rank - lower
    return float(samples[lower] + (samples[upper] - samples[lower]) * weight)


def _resolve_export_path(*, tier_obj: BenchmarkTier, options: BenchmarkOptions) -> Path:
    filename = options.export_filename or f"{tier_obj.name}-metadata-benchmark.json"
    root = options.workspace_root or paths.workspace_root()
    return root / "exports" / "benchmarks" / filename


def _sqlite_size_mb() -> float:
    if not paths.DB_PATH.exists():
        return 0.0
    return paths.DB_PATH.stat().st_size / (1024 * 1024)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return 0.0
    if usage > 1024 * 1024:
        return usage / (1024 * 1024)
    return usage / 1024


def _synthetic_hit(index: int) -> dict[str, Any]:
    families = (
        (
            "CMS",
            "same-sign W boson scattering",
            "Measures electroweak same-sign WW production with jets and missing transverse momentum.",
            "cms",
        ),
        (
            "ATLAS",
            "Higgs boson diphoton cross section",
            "Studies Higgs diphoton production with differential cross section measurements.",
            "atlas",
        ),
        (
            "LHC",
            "supersymmetry search with jets and missing transverse momentum",
            "Searches for supersymmetry in hadronic final states with jets and missing transverse momentum.",
            "susy",
        ),
    )
    experiment, topic, abstract_seed, collaboration = families[index % len(families)]
    year = 2016 + (index % 9)
    control_number = 9_000_000 + index
    arxiv_id = f"26{(index % 12) + 1:02d}.{index:05d}"
    return {
        "metadata": {
            "control_number": control_number,
            "titles": [{"title": f"{experiment} benchmark study {index}: {topic}"}],
            "abstracts": [{"value": f"{abstract_seed} Benchmark sample #{index} for tiered metadata-only benchmarking."}],
            "arxiv_eprints": [{"value": arxiv_id}],
            "publication_info": [{"year": year}],
            "collaborations": [{"value": collaboration.upper()}],
            "keywords": [{"value": topic}, {"value": experiment.lower()}],
        }
    }
