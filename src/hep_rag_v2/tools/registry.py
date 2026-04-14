from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, *, name: str, description: str, handler: Callable[..., Any]) -> ToolSpec:
        key = str(name).strip()
        if not key:
            raise ValueError("Tool name cannot be empty.")
        if key in self._tools:
            raise ValueError(f"Tool already registered: {key}")
        spec = ToolSpec(name=key, description=str(description).strip(), handler=handler)
        self._tools[key] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def invoke(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        return self.get(name).handler(*args, **kwargs)


def build_default_tool_registry(facade: Any) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="retrieve",
        description="Run the typed hep-rag retrieval pipeline.",
        handler=lambda *, query: facade.retrieve(query=query),
    )
    registry.register(
        name="ask",
        description="Run retrieval plus answer synthesis with evidence citations.",
        handler=lambda *, query: facade.ask(query=query),
    )
    registry.register(
        name="generate_ideas",
        description="Generate ranked idea candidates with a structured reasoning trace.",
        handler=lambda *, query: facade.generate_ideas(query=query),
    )
    registry.register(
        name="workspace_status",
        description="Inspect workspace status and index counts.",
        handler=lambda: facade.workspace_status(),
    )
    return registry
