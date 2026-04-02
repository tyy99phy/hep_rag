from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hep_rag_v2.evidence import EvidenceRegistry
from hep_rag_v2.pipeline import ask as pipeline_ask
from hep_rag_v2.pipeline import fetch_online_candidates as pipeline_fetch_online_candidates
from hep_rag_v2.pipeline import ingest_online as pipeline_ingest_online
from hep_rag_v2.pipeline import reparse_cached_pdfs as pipeline_reparse_cached_pdfs
from hep_rag_v2.pipeline import retrieve as pipeline_retrieve
from hep_rag_v2.retrieval_adapter import build_retrieval_shell, normalize_retrieval_payload
from hep_rag_v2.service.inspect import audit_document_payload, show_document_payload, show_graph_payload
from hep_rag_v2.service.workspace import workspace_status_payload

ProgressCallback = Callable[[str], None] | None


@dataclass
class HepRagServiceFacade:
    config: dict[str, Any]
    progress: ProgressCallback = None

    def retrieve(
        self,
        *,
        query: str,
        limit: int | None = None,
        target: str | None = None,
        collection_name: str | None = None,
        max_parallelism: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload = pipeline_retrieve(
            self.config,
            query=query,
            limit=limit,
            target=target,
            collection_name=collection_name,
            max_parallelism=max_parallelism,
            model=model,
            progress=self.progress,
        )
        return self._enrich_retrieval_payload(payload)

    def ask(
        self,
        *,
        query: str,
        mode: str = "answer",
        limit: int | None = None,
        target: str | None = None,
        collection_name: str | None = None,
        max_parallelism: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload = pipeline_ask(
            self.config,
            query=query,
            mode=mode,
            limit=limit,
            target=target,
            collection_name=collection_name,
            max_parallelism=max_parallelism,
            model=model,
            progress=self.progress,
        )
        registry = EvidenceRegistry()
        evidence = payload.get("evidence") or {}
        registry.register_retrieval(
            {
                "query": query,
                "collection": collection_name,
                "requested_target": target,
                "routing": {"target": target or "answer", "graph_expand": 0, "reasons": ["answer_evidence"]},
                "model": model,
                "works": list(evidence.get("works") or []),
                "evidence_chunks": list(evidence.get("chunks") or []),
            }
        )
        enriched = dict(payload)
        enriched["evidence_registry"] = registry.to_payload()
        return enriched

    def fetch_papers(self, *, query: str, limit: int, max_parallelism: int | None = None) -> dict[str, Any]:
        return pipeline_fetch_online_candidates(
            self.config,
            query=query,
            limit=limit,
            max_parallelism=max_parallelism,
            progress=self.progress,
        )

    def ingest_online(self, **kwargs: Any) -> dict[str, Any]:
        return pipeline_ingest_online(self.config, progress=self.progress, **kwargs)

    def reparse_pdfs(self, **kwargs: Any) -> dict[str, Any]:
        return pipeline_reparse_cached_pdfs(self.config, progress=self.progress, **kwargs)

    def workspace_status(self) -> dict[str, Any]:
        return workspace_status_payload()

    def graph_neighbors(self, **kwargs: Any) -> dict[str, Any]:
        return show_graph_payload(**kwargs)

    def show_document(self, **kwargs: Any) -> dict[str, Any]:
        return show_document_payload(**kwargs)

    def audit_document(self, **kwargs: Any) -> dict[str, Any]:
        return audit_document_payload(**kwargs)

    def _enrich_retrieval_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        typed = normalize_retrieval_payload(payload)
        registry = EvidenceRegistry()
        registry.register_retrieval(typed)
        shell = build_retrieval_shell(payload)
        enriched = dict(payload)
        enriched["typed_retrieval"] = typed.to_payload()
        enriched["evidence_registry"] = registry.to_payload()
        enriched["results"] = shell["results"]
        return enriched
