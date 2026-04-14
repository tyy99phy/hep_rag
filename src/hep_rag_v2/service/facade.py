from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hep_rag_v2.evidence import EvidenceRegistry
from hep_rag_v2.pipeline import ask as pipeline_ask
from hep_rag_v2.pipeline import fetch_online_candidates as pipeline_fetch_online_candidates
from hep_rag_v2.pipeline import ingest_online as pipeline_ingest_online
from hep_rag_v2.pipeline import reparse_cached_pdfs as pipeline_reparse_cached_pdfs
from hep_rag_v2.pipeline import retrieve as pipeline_retrieve
from hep_rag_v2.retrieval_adapter import HEP_OBJECT_CONTRACT_VERSION, build_retrieval_shell
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

    def generate_ideas(
        self,
        *,
        query: str,
        limit: int | None = None,
        target: str | None = None,
        collection_name: str | None = None,
        max_parallelism: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        retrieval = self.retrieve(
            query=query,
            limit=limit,
            target=target,
            collection_name=collection_name,
            max_parallelism=max_parallelism,
            model=model,
        )
        trace_steps: list[dict[str, Any]] = [
            {
                "contract_version": HEP_OBJECT_CONTRACT_VERSION,
                "object_type": "trace_step",
                "object_id": "trace_step:retrieve",
                "source_kind": "service_facade",
                "status": "materialized",
                "source_refs": list((retrieval.get("evidence_registry") or {}).get("source_refs") or []),
                "derivation": {
                    "query": query,
                    "phase": "retrieve",
                    "target": (retrieval.get("typed_retrieval") or {}).get("metadata", {}).get("target"),
                },
                "step_type": "retrieve",
                "summary": f"retrieved {len(retrieval.get('results') or [])} evidence items",
                "target": (retrieval.get("typed_retrieval") or {}).get("metadata", {}).get("target"),
            }
        ]
        if self.progress is not None:
            self.progress("reasoning step: retrieve")

        idea_limit = max(1, int((self.config.get("ideas") or {}).get("top_k") or limit or 3))
        ideas = _rank_idea_candidates(retrieval, limit=idea_limit)
        trace_steps.append(
            {
                "contract_version": HEP_OBJECT_CONTRACT_VERSION,
                "object_type": "trace_step",
                "object_id": "trace_step:generate_idea",
                "source_kind": "service_facade",
                "status": "materialized",
                "source_refs": [item["object_id"] for item in ideas],
                "derivation": {"query": query, "phase": "generate_idea", "idea_count": len(ideas)},
                "step_type": "generate_idea",
                "summary": f"ranked {len(ideas)} idea candidates",
                "idea_ids": [item["object_id"] for item in ideas],
            }
        )
        if self.progress is not None:
            self.progress("reasoning step: generate_idea")

        registry = EvidenceRegistry()
        registry.register_many(ideas)
        payload = {
            "query": query,
            "ideas": ideas,
            "trace": {"steps": trace_steps},
            "retrieval": {
                "query": retrieval.get("query"),
                "results": retrieval.get("results"),
                "typed_retrieval": retrieval.get("typed_retrieval"),
            },
            "evidence_registry": registry.export(),
            "object_contracts": {
                "contract_version": HEP_OBJECT_CONTRACT_VERSION,
                "evidence_bundle": registry.export(),
                "trace_steps": trace_steps,
                "retrieval": retrieval.get("object_contracts"),
            },
        }
        return payload

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
        shell = build_retrieval_shell(payload)
        enriched = dict(payload)
        enriched["typed_retrieval"] = shell["typed_retrieval"]
        enriched["evidence_registry"] = shell["evidence_registry"]
        enriched["object_contracts"] = shell["object_contracts"]
        enriched["results"] = shell["results"]
        return enriched


def _rank_idea_candidates(retrieval: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    results = list(retrieval.get("results") or [])
    ideas: list[dict[str, Any]] = []
    for index, item in enumerate(results[: max(1, limit)], start=1):
        title = str(item.get("title") or item.get("object_id") or f"Idea {index}").strip()
        evidence_key = str(item.get("citation") or item.get("object_id") or "")
        score = float(item.get("score") or 0.0)
        ideas.append(
            {
                "source_type": "idea_candidate",
                "object_type": "idea_candidate",
                "object_id": f"idea_candidate:idea-{index}",
                "evidence_key": f"idea_candidate:idea-{index}",
                "title": f"Hypothesis from {title}",
                "content": (
                    f"Investigate whether {title.lower()} can transfer to a new analysis lane "
                    f"using evidence anchored in {evidence_key or 'retrieved evidence'}."
                ),
                "score": round(min(1.0, 0.45 + (score * 0.5)), 3),
                "metadata": {
                    "source_object_id": item.get("object_id"),
                    "source_object_type": item.get("object_type"),
                    "evidence_refs": [evidence_key] if evidence_key else [],
                },
            }
        )
    return ideas
