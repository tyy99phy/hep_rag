from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hep_rag_v2.retrieval_adapter import (
    TypedRetrievalItem,
    TypedRetrievalResult,
    adapt_chunk_hit,
    adapt_work_hit,
    normalize_retrieval_payload,
)


@dataclass
class EvidenceEntry:
    citation_id: str
    evidence_key: str
    result: TypedRetrievalResult
    occurrences: int = 1

    @property
    def ref(self) -> str:
        return self.citation_id

    @property
    def citation(self) -> str:
        return f"[{self.evidence_key}]"

    @property
    def work_id(self) -> int | None:
        return self.result.work_id

    @property
    def chunk_id(self) -> int | None:
        return self.result.chunk_id

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "ref": self.citation_id,
            "citation_id": self.citation_id,
            "citation": self.citation,
            "evidence_key": self.evidence_key,
            "object_type": self.result.object_type,
            "source_type": self.result.source_type,
            "object_id": self.result.object_id,
            "work_id": self.result.work_id,
            "chunk_id": self.result.chunk_id,
            "title": self.result.title,
            "text": self.result.content,
            "section_hint": self.result.section_hint,
            "page_hint": self.result.page_hint,
            "occurrences": self.occurrences,
        }
        return {key: value for key, value in payload.items() if value is not None and value != ""}

    def __getitem__(self, key: str) -> Any:
        return self.to_payload()[key]


class EvidenceRegistry:
    def __init__(self) -> None:
        self._items: list[EvidenceEntry] = []
        self._by_key: dict[str, EvidenceEntry] = {}
        self._by_ref: dict[str, EvidenceEntry] = {}
        self._counters = {"work": 0, "chunk": 0, "generic": 0}

    def __len__(self) -> int:
        return len(self._items)

    def register(self, item: Mapping[str, Any] | TypedRetrievalResult) -> EvidenceEntry:
        typed = _coerce_result(item)
        key = str(typed.evidence_key or typed.object_id or "")
        if not key:
            raise ValueError("Cannot register evidence without evidence_key/object_id.")
        existing = self._by_key.get(key)
        if existing is not None:
            existing.occurrences += 1
            return existing
        self._counters["generic"] += 1
        entry = EvidenceEntry(citation_id=f"E{self._counters['generic']}", evidence_key=key, result=typed)
        self._items.append(entry)
        self._by_key[key] = entry
        self._by_ref[entry.citation_id] = entry
        return entry

    def register_many(self, items: list[Mapping[str, Any] | TypedRetrievalResult]) -> list[EvidenceEntry]:
        return [self.register(item) for item in items]

    def register_work(self, item: Mapping[str, Any] | TypedRetrievalResult) -> EvidenceEntry:
        typed = _coerce_result(item)
        key = str(typed.evidence_key or typed.object_id or "")
        existing = self._by_key.get(key)
        if existing is not None:
            existing.occurrences += 1
            return existing
        self._counters["work"] += 1
        entry = EvidenceEntry(citation_id=f"W{self._counters['work']}", evidence_key=key, result=typed)
        self._items.append(entry)
        self._by_key[key] = entry
        self._by_ref[entry.citation_id] = entry
        return entry

    def register_chunk(self, item: Mapping[str, Any] | TypedRetrievalResult) -> EvidenceEntry:
        typed = _coerce_result(item)
        key = str(typed.evidence_key or typed.object_id or "")
        existing = self._by_key.get(key)
        if existing is not None:
            existing.occurrences += 1
            return existing
        self._counters["chunk"] += 1
        entry = EvidenceEntry(citation_id=f"C{self._counters['chunk']}", evidence_key=key, result=typed)
        self._items.append(entry)
        self._by_key[key] = entry
        self._by_ref[entry.citation_id] = entry
        return entry

    def register_retrieval(self, payload: Mapping[str, Any] | TypedRetrievalResult) -> list[EvidenceEntry]:
        typed = payload if isinstance(payload, TypedRetrievalResult) else normalize_retrieval_payload(payload)
        out: list[EvidenceEntry] = []
        for item in typed.works:
            out.append(self.register_work(item))
        for item in typed.chunks:
            out.append(self.register_chunk(item))
        return out

    def records(self) -> list[EvidenceEntry]:
        return list(self._items)

    def items(self) -> list[EvidenceEntry]:
        return list(self._items)

    def resolve(self, ref: str) -> EvidenceEntry:
        return self._by_ref[ref]

    def to_payload(self) -> list[dict[str, Any]]:
        return [item.to_payload() for item in self._items]

    def export(self) -> dict[str, list[dict[str, Any]]]:
        return {"items": self.to_payload()}


def build_evidence_registry(payload: Mapping[str, Any] | TypedRetrievalResult) -> dict[str, Any]:
    registry = EvidenceRegistry()
    registry.register_retrieval(payload)
    return registry.export()


def _coerce_result(item: Mapping[str, Any] | TypedRetrievalResult) -> TypedRetrievalResult:
    if isinstance(item, TypedRetrievalResult):
        return item
    if item.get("object_type") == "chunk" or item.get("source_type") == "chunk":
        return adapt_chunk_hit(item)
    if "chunk_id" in item or "clean_text" in item or "section_hint" in item:
        return adapt_chunk_hit(item)
    if "work_id" in item or "abstract" in item:
        return adapt_work_hit(item)
    if item.get("object_type") == "work" or item.get("source_type") == "work":
        return adapt_work_hit(item)
    return TypedRetrievalItem(
        source_type="work",
        object_type="work",
        object_id=str(item.get("object_id") or item.get("evidence_key") or item.get("source_id") or ""),
        evidence_key=str(item.get("evidence_key") or item.get("object_id") or item.get("source_id") or ""),
        work_id=item.get("work_id") or item.get("source_id"),
        title=item.get("title"),
        content=str(item.get("text") or item.get("content") or ""),
    )
