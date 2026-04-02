from __future__ import annotations

from typing import Any, Callable

from hep_rag_v2.service.facade import HepRagServiceFacade, ProgressCallback
from hep_rag_v2.tools import ToolRegistry


def create_facade(config: dict[str, Any], *, progress: ProgressCallback = None) -> HepRagServiceFacade:
    return HepRagServiceFacade(config=config, progress=progress)


def create_tool_registry(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    max_parallelism: int | None = None,
    model: str | None = None,
    mode: str = "answer",
    progress: ProgressCallback = None,
) -> ToolRegistry:
    facade = create_facade(config, progress=progress)
    registry = ToolRegistry()
    registry.register(
        name="retrieve",
        description="Run the typed hep-rag retrieval pipeline.",
        handler=lambda *, query: facade.retrieve(
            query=query,
            collection_name=collection_name,
            target=target,
            limit=limit,
            max_parallelism=max_parallelism,
            model=model,
        ),
    )
    registry.register(
        name="ask",
        description="Run retrieval plus answer synthesis with evidence citations.",
        handler=lambda *, query: facade.ask(
            query=query,
            mode=mode,
            collection_name=collection_name,
            target=target,
            limit=limit,
            max_parallelism=max_parallelism,
            model=model,
        ),
    )
    registry.register(
        name="workspace_status",
        description="Inspect workspace status and index counts.",
        handler=lambda: facade.workspace_status(),
    )
    registry.register(
        name="show_graph",
        description="Inspect graph neighbors for a known work.",
        handler=lambda **kwargs: facade.graph_neighbors(**kwargs),
    )
    registry.register(
        name="show_document",
        description="Inspect a parsed document and chunk samples.",
        handler=lambda **kwargs: facade.show_document(**kwargs),
    )
    registry.register(
        name="audit_document",
        description="Audit parser noise and retrieval readiness.",
        handler=lambda **kwargs: facade.audit_document(**kwargs),
    )
    return registry
