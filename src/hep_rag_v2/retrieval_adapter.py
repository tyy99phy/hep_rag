from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

HEP_OBJECT_CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class TypedRetrievalMetadata:
    query: str = ""
    collection: str | None = None
    requested_target: str | None = None
    target: str | None = None
    model: str | None = None
    max_parallelism: int | None = None
    graph_expand: int = 0
    reasons: tuple[str, ...] = ()
    lane: str | None = None
    rank: int | None = None
    hybrid_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    search_type: str | None = None
    query_group_hits: int | None = None
    query_group_coverage: float | None = None
    family_id: int | None = None
    family_label: str | None = None
    family_primary_work_id: int | None = None
    related_versions: tuple[dict[str, Any], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "query": self.query,
            "collection": self.collection,
            "requested_target": self.requested_target,
            "target": self.target,
            "model": self.model,
            "max_parallelism": self.max_parallelism,
            "graph_expand": self.graph_expand,
            "reasons": list(self.reasons),
            "lane": self.lane,
            "rank": self.rank,
            "hybrid_score": self.hybrid_score,
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "search_type": self.search_type,
            "query_group_hits": self.query_group_hits,
            "query_group_coverage": self.query_group_coverage,
            "family_id": self.family_id,
            "family_label": self.family_label,
            "family_primary_work_id": self.family_primary_work_id,
            "related_versions": list(self.related_versions),
        }
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != () and value != []
        }


@dataclass(frozen=True)
class TypedRetrievalResult:
    source_type: str | None = None
    object_type: str | None = None
    object_id: str | None = None
    evidence_key: str | None = None
    work_id: int | None = None
    chunk_id: int | None = None
    title: str | None = None
    content: str = ""
    score: float | None = None
    rank: int | None = None
    canonical_source: str | None = None
    canonical_id: str | None = None
    collection: str | None = None
    section_hint: str | None = None
    page_hint: str | None = None
    chunk_role: str | None = None
    metadata: TypedRetrievalMetadata = field(default_factory=TypedRetrievalMetadata)
    typed_objects: tuple["TypedRetrievalResult", ...] = ()
    works: tuple["TypedRetrievalResult", ...] = ()
    chunks: tuple["TypedRetrievalResult", ...] = ()

    def primary_items(self, *, prefer_chunks: bool = True) -> tuple["TypedRetrievalResult", ...]:
        if self.typed_objects:
            return self.typed_objects
        if self.works or self.chunks:
            if prefer_chunks and self.chunks:
                return self.chunks
            if self.works:
                return self.works
            return self.chunks
        return (self,)

    def all_items(self) -> tuple["TypedRetrievalResult", ...]:
        if self.typed_objects or self.works or self.chunks:
            return self.typed_objects + self.works + self.chunks
        return (self,)

    def to_payload(self, *, prefer_chunks: bool = True) -> dict[str, Any]:
        if self.typed_objects or self.works or self.chunks:
            return {
                "metadata": self.metadata.to_payload(),
                "typed_objects": [item.to_payload() for item in self.typed_objects],
                "works": [item.to_payload() for item in self.works],
                "chunks": [item.to_payload() for item in self.chunks],
                "primary": [item.to_payload() for item in self.primary_items(prefer_chunks=prefer_chunks)],
            }
        payload = {
            "source_type": self.source_type,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "evidence_key": self.evidence_key,
            "work_id": self.work_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "canonical_source": self.canonical_source,
            "canonical_id": self.canonical_id,
            "collection": self.collection,
            "section_hint": self.section_hint,
            "page_hint": self.page_hint,
            "chunk_role": self.chunk_role,
            "metadata": self.metadata.to_payload(),
        }
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != "" and value != () and value != []
        }


TypedRetrievalItem = TypedRetrievalResult


def adapt_work_hit(row: Mapping[str, Any], *, collection: str | None = None) -> TypedRetrievalResult:
    work_id = _int_or_none(row.get("work_id"))
    title = _text_or_none(row.get("raw_title") or row.get("title"))
    abstract = str(row.get("abstract") or "").strip()
    metadata = TypedRetrievalMetadata(
        collection=collection,
        lane="works",
        rank=_int_or_none(row.get("rank")),
        hybrid_score=_float_or_none(row.get("hybrid_score", row.get("score"))),
        bm25_rank=_int_or_none(row.get("bm25_rank")),
        vector_rank=_int_or_none(row.get("vector_rank")),
        search_type=_text_or_none(row.get("search_type")),
        query_group_hits=_int_or_none(row.get("query_group_hits")),
        query_group_coverage=_float_or_none(row.get("query_group_coverage")),
        family_id=_int_or_none(row.get("family_id")),
        family_label=_text_or_none(row.get("family_label")),
        family_primary_work_id=_int_or_none(row.get("family_primary_work_id")),
        related_versions=tuple(dict(item) for item in list(row.get("related_versions") or [])),
    )
    return TypedRetrievalResult(
        source_type="work",
        object_type="work",
        object_id=f"work:{work_id}" if work_id is not None else None,
        evidence_key=f"work:{work_id}" if work_id is not None else None,
        work_id=work_id,
        title=title,
        content=abstract,
        score=_float_or_none(row.get("hybrid_score", row.get("score"))),
        rank=_int_or_none(row.get("rank")),
        canonical_source=_text_or_none(row.get("canonical_source")),
        canonical_id=_text_or_none(row.get("canonical_id")),
        collection=collection,
        metadata=metadata,
    )


def adapt_chunk_hit(row: Mapping[str, Any], *, collection: str | None = None) -> TypedRetrievalResult:
    chunk_id = _int_or_none(row.get("chunk_id"))
    work_id = _int_or_none(row.get("work_id"))
    metadata = TypedRetrievalMetadata(
        collection=collection,
        lane="chunks",
        rank=_int_or_none(row.get("rank")),
        hybrid_score=_float_or_none(row.get("hybrid_score", row.get("score"))),
        bm25_rank=_int_or_none(row.get("bm25_rank")),
        vector_rank=_int_or_none(row.get("vector_rank")),
        search_type=_text_or_none(row.get("search_type")),
        query_group_hits=_int_or_none(row.get("query_group_hits")),
        query_group_coverage=_float_or_none(row.get("query_group_coverage")),
        family_id=_int_or_none(row.get("family_id")),
        family_label=_text_or_none(row.get("family_label")),
        family_primary_work_id=_int_or_none(row.get("family_primary_work_id")),
        related_versions=tuple(dict(item) for item in list(row.get("related_versions") or [])),
    )
    return TypedRetrievalResult(
        source_type="chunk",
        object_type="chunk",
        object_id=f"chunk:{chunk_id}" if chunk_id is not None else None,
        evidence_key=f"chunk:{chunk_id}" if chunk_id is not None else None,
        work_id=work_id,
        chunk_id=chunk_id,
        title=_text_or_none(row.get("raw_title") or row.get("title")),
        content=str(row.get("clean_text") or "").strip(),
        score=_float_or_none(row.get("hybrid_score", row.get("score"))),
        rank=_int_or_none(row.get("rank")),
        canonical_source=_text_or_none(row.get("canonical_source")),
        canonical_id=_text_or_none(row.get("canonical_id")),
        collection=collection,
        section_hint=_text_or_none(row.get("section_hint")),
        page_hint=_text_or_none(row.get("page_hint")),
        chunk_role=_text_or_none(row.get("chunk_role")),
        metadata=metadata,
    )


def normalize_retrieval_payload(payload: Mapping[str, Any]) -> TypedRetrievalResult:
    routing = dict(payload.get("routing") or {})
    metadata = TypedRetrievalMetadata(
        query=str(payload.get("query") or ""),
        collection=_text_or_none(payload.get("collection")),
        requested_target=_text_or_none(payload.get("requested_target")),
        target=_text_or_none(routing.get("target")),
        model=_text_or_none(payload.get("model")),
        max_parallelism=_int_or_none(payload.get("max_parallelism")),
        graph_expand=int(routing.get("graph_expand") or 0),
        reasons=tuple(str(item) for item in list(routing.get("reasons") or [])),
    )
    collection = metadata.collection
    typed_objects = tuple(_adapt_typed_collections(payload, collection=collection))
    works = tuple(adapt_work_hit(item, collection=collection) for item in list(payload.get("works") or []))
    chunks = tuple(adapt_chunk_hit(item, collection=collection) for item in list(payload.get("evidence_chunks") or []))
    return TypedRetrievalResult(metadata=metadata, typed_objects=typed_objects, works=works, chunks=chunks)


def build_retrieval_shell(payload: Mapping[str, Any], *, prefer_chunks: bool = True) -> dict[str, Any]:
    from hep_rag_v2.evidence import EvidenceRegistry

    typed = normalize_retrieval_payload(payload)
    registry = EvidenceRegistry()
    results: list[dict[str, Any]] = []
    work_capsules: list[dict[str, Any]] = []
    ordered_items = typed.typed_objects + (
        typed.chunks + typed.works if prefer_chunks else typed.works + typed.chunks
    )
    for item in ordered_items:
        entry = registry.register(item)
        work_capsules.append(
            {
                "contract_version": HEP_OBJECT_CONTRACT_VERSION,
                "object_type": "work_capsule",
                "object_id": item.object_id,
                "source_kind": str(item.source_type or item.object_type or "retrieval"),
                "status": "materialized",
                "source_refs": [entry.citation_id],
                "derivation": {
                    "query": typed.metadata.query,
                    "target": typed.metadata.target,
                    "collection": typed.metadata.collection,
                },
                "capsule_type": item.object_type,
                "work_id": item.work_id,
                "chunk_id": item.chunk_id,
                "title": item.title,
                "content": item.content,
                "evidence_ref": entry.citation_id,
                "evidence_key": entry.evidence_key,
            }
        )
        results.append(
            {
                "source_type": str(item.source_type),
                "object_type": str(item.object_type),
                "object_id": item.object_id,
                "work_id": item.work_id,
                "chunk_id": item.chunk_id,
                "title": item.title,
                "text": item.content,
                "score": item.metadata.hybrid_score,
                "citation": entry["citation"],
            }
        )
    return {
        "query": typed.metadata.query,
        "results": results,
        "typed_retrieval": typed.to_payload(prefer_chunks=prefer_chunks),
        "evidence_registry": registry.export(),
        "object_contracts": {
            "contract_version": HEP_OBJECT_CONTRACT_VERSION,
            "work_capsules": work_capsules,
            "evidence_bundle": registry.export(),
        },
    }


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _adapt_typed_collections(
    payload: Mapping[str, Any],
    *,
    collection: str | None = None,
) -> tuple[TypedRetrievalResult, ...]:
    groups = (
        ("ontology_summaries", "ontology_summary", "summary_id"),
        ("result_objects", "result_object", "result_id"),
        ("method_objects", "method_object", "method_id"),
        ("transfer_candidates", "transfer_candidate", "transfer_id"),
        ("idea_candidates", "idea_candidate", "idea_id"),
    )
    items: list[TypedRetrievalResult] = []
    for field_name, object_type, id_field in groups:
        for row in list(payload.get(field_name) or []):
            items.append(adapt_typed_hit(row, object_type=object_type, id_field=id_field, collection=collection))
    return tuple(items)


def adapt_typed_hit(
    row: Mapping[str, Any],
    *,
    object_type: str,
    id_field: str,
    collection: str | None = None,
) -> TypedRetrievalResult:
    raw_identifier = row.get(id_field) or row.get("object_id") or row.get("id")
    identifier = _text_or_none(raw_identifier)
    object_id = identifier if identifier and ":" in identifier else f"{object_type}:{identifier}" if identifier else None
    content = (
        row.get("summary")
        or row.get("summary_text")
        or row.get("description")
        or row.get("text")
        or row.get("content")
        or row.get("abstract")
        or row.get("clean_text")
        or ""
    )
    metadata = TypedRetrievalMetadata(
        collection=collection,
        lane=object_type,
        rank=_int_or_none(row.get("rank")),
        hybrid_score=_float_or_none(row.get("hybrid_score", row.get("score"))),
        search_type=_text_or_none(row.get("search_type")),
    )
    return TypedRetrievalResult(
        source_type=object_type,
        object_type=object_type,
        object_id=object_id,
        evidence_key=object_id,
        work_id=_int_or_none(row.get("work_id")),
        chunk_id=_int_or_none(row.get("chunk_id")),
        title=_text_or_none(row.get("title") or row.get("name") or row.get("label")),
        content=str(content).strip(),
        score=_float_or_none(row.get("hybrid_score", row.get("score"))),
        rank=_int_or_none(row.get("rank")),
        canonical_source=_text_or_none(row.get("canonical_source")),
        canonical_id=_text_or_none(row.get("canonical_id")),
        collection=collection,
        section_hint=_text_or_none(row.get("section_hint")),
        page_hint=_text_or_none(row.get("page_hint")),
        metadata=metadata,
    )


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
