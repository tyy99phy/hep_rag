from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.tools import StructuredTool
except Exception as exc:  # pragma: no cover - import error is validated by packaging
    raise RuntimeError(
        "LangChain integration requires langchain-core. Install it with `pip install -e .[langchain]`."
    ) from exc

from pydantic import ConfigDict, Field, PrivateAttr

from hep_rag_v2.config import apply_runtime_config
from hep_rag_v2.pipeline import _build_answer_messages, _build_llm_client, ask, retrieve
from hep_rag_v2.service.inspect import audit_document_payload, show_document_payload, show_graph_payload

ProgressCallback = Callable[[str], None] | None


def load_langchain_runtime(
    *,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    return apply_runtime_config(config_path=config_path, workspace_root=workspace_root)


def retrieval_to_documents(
    payload: Mapping[str, Any],
    *,
    prefer_chunks: bool = True,
) -> list[Document]:
    if prefer_chunks:
        chunk_docs = [_chunk_to_document(item) for item in list(payload.get("evidence_chunks") or [])]
        if chunk_docs:
            return chunk_docs
    return [_work_to_document(item) for item in list(payload.get("works") or [])]


class HepRagRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: dict[str, Any]
    collection_name: str | None = None
    target: str | None = None
    limit: int | None = None
    model: str | None = None
    prefer_chunks: bool = True
    progress: ProgressCallback = Field(default=None, exclude=True, repr=False)

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        payload = retrieve(
            self.config,
            query=query,
            limit=self.limit,
            target=self.target,
            collection_name=self.collection_name,
            model=self.model,
            progress=self.progress,
        )
        return retrieval_to_documents(payload, prefer_chunks=self.prefer_chunks)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "target": self.target,
            "limit": self.limit,
            "model": self.model,
            "prefer_chunks": self.prefer_chunks,
        }


class HepRagChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: dict[str, Any]
    model_override: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    _client: Any = PrivateAttr(default=None)
    _llm_cfg: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        llm_cfg = dict(self.config.get("llm") or {})
        if not bool(llm_cfg.get("enabled")):
            raise ValueError(
                "LLM is disabled in config. Set llm.enabled=true before using the LangChain chat adapter."
            )
        if self.model_override:
            llm_cfg["model"] = self.model_override
        self._llm_cfg = llm_cfg
        self._client = _build_llm_client(llm_cfg)

    @property
    def _llm_type(self) -> str:
        return str(self._llm_cfg.get("backend") or "openai_compatible")

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "backend": self._llm_type,
            "model": str(self._llm_cfg.get("model") or ""),
            "temperature": self._resolved_temperature(),
            "max_tokens": self._resolved_max_tokens(),
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self._client.chat(
            messages=[_message_to_payload(item) for item in messages],
            temperature=float(kwargs.get("temperature") or self._resolved_temperature()),
            max_tokens=int(kwargs.get("max_tokens") or self._resolved_max_tokens()),
        )
        content = _apply_stop_tokens(str(response.get("content") or ""), stop)
        generation = ChatGeneration(
            message=AIMessage(content=content),
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "backend": self._llm_type,
                "model": str(response.get("model") or self._llm_cfg.get("model") or ""),
                "raw": response.get("raw"),
            },
        )

    def _resolved_temperature(self) -> float:
        return float(self.temperature if self.temperature is not None else self._llm_cfg.get("temperature") or 0.2)

    def _resolved_max_tokens(self) -> int:
        return int(self.max_tokens if self.max_tokens is not None else self._llm_cfg.get("max_tokens") or 1200)


def build_langchain_retrieval_tool(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    progress: ProgressCallback = None,
    name: str = "hep_rag_retrieve",
    description: str = "Retrieve structured HEP evidence from the local hep-rag workspace.",
) -> StructuredTool:
    def _tool(query: str) -> dict[str, Any]:
        return retrieve(
            config,
            query=query,
            limit=limit,
            target=target,
            collection_name=collection_name,
            model=model,
            progress=progress,
        )

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )


def build_langchain_answer_tool(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    mode: str = "answer",
    progress: ProgressCallback = None,
    name: str = "hep_rag_answer",
    description: str = "Answer a HEP literature question using the local hep-rag evidence pipeline.",
) -> StructuredTool:
    def _tool(query: str) -> dict[str, Any]:
        return ask(
            config,
            query=query,
            mode=mode,
            limit=limit,
            target=target,
            collection_name=collection_name,
            model=model,
            progress=progress,
        )

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )


def build_langchain_graph_tool(
    *,
    collection_name: str | None = None,
    similarity_model: str | None = None,
    limit: int = 20,
    name: str = "hep_rag_graph_neighbors",
    description: str = "Inspect graph neighbors for a paper in the local hep-rag workspace.",
) -> StructuredTool:
    def _tool(
        work_id: int | None = None,
        id_type: str | None = None,
        id_value: str | None = None,
        edge_kind: str = "all",
        collection: str | None = None,
        result_limit: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        return show_graph_payload(
            work_id=work_id,
            id_type=id_type,
            id_value=id_value,
            edge_kind=edge_kind,
            collection=collection or collection_name,
            limit=result_limit if result_limit is not None else limit,
            similarity_model=model or similarity_model,
        )

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )


def build_langchain_document_tool(
    *,
    limit: int = 20,
    name: str = "hep_rag_show_document",
    description: str = "Inspect a parsed MinerU document, including sections, blocks, and chunks.",
) -> StructuredTool:
    def _tool(
        work_id: int | None = None,
        id_type: str | None = None,
        id_value: str | None = None,
        result_limit: int | None = None,
    ) -> dict[str, Any]:
        return show_document_payload(
            work_id=work_id,
            id_type=id_type,
            id_value=id_value,
            limit=result_limit if result_limit is not None else limit,
        )

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )


def build_langchain_document_audit_tool(
    *,
    limit: int = 20,
    name: str = "hep_rag_audit_document",
    description: str = "Audit a parsed MinerU document for readiness and parser noise.",
) -> StructuredTool:
    def _tool(
        work_id: int | None = None,
        id_type: str | None = None,
        id_value: str | None = None,
        result_limit: int | None = None,
    ) -> dict[str, Any]:
        return audit_document_payload(
            work_id=work_id,
            id_type=id_type,
            id_value=id_value,
            limit=result_limit if result_limit is not None else limit,
        )

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )


def build_langchain_toolkit(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    mode: str = "answer",
    progress: ProgressCallback = None,
    include_debug_tools: bool = True,
) -> list[StructuredTool]:
    tools = [
        build_langchain_retrieval_tool(
            config,
            collection_name=collection_name,
            target=target,
            limit=limit,
            model=model,
            progress=progress,
        ),
        build_langchain_answer_tool(
            config,
            collection_name=collection_name,
            target=target,
            limit=limit,
            model=model,
            mode=mode,
            progress=progress,
        ),
        build_langchain_graph_tool(
            collection_name=collection_name,
            similarity_model=model,
        ),
        build_langchain_document_tool(),
    ]
    if include_debug_tools:
        tools.append(build_langchain_document_audit_tool())
    return tools


def build_langchain_answer_runnable(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    mode: str = "answer",
    progress: ProgressCallback = None,
) -> Any:
    chat_model = HepRagChatModel(config=config, model_override=model)

    def _normalize_input(value: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(value, str):
            return {"query": value}
        if isinstance(value, Mapping):
            query = str(value.get("query") or "").strip()
            if not query:
                raise ValueError("LangChain answer runnable expects a non-empty `query`.")
            return dict(value)
        raise TypeError("LangChain answer runnable expects either a string query or a mapping with `query`.")

    def _attach_retrieval(data: Mapping[str, Any]) -> dict[str, Any]:
        runtime_query = str(data.get("query") or "").strip()
        runtime_collection = _mapping_text(data, "collection_name") or _mapping_text(data, "collection") or collection_name
        runtime_target = _mapping_text(data, "target") or target
        runtime_model = _mapping_text(data, "model") or model
        runtime_mode = _mapping_text(data, "mode") or mode
        runtime_limit = _mapping_int(data, "limit", default=limit)
        retrieval_payload = retrieve(
            config,
            query=runtime_query,
            limit=runtime_limit,
            target=runtime_target,
            collection_name=runtime_collection,
            model=runtime_model,
            progress=progress,
        )
        return {
            "query": runtime_query,
            "mode": runtime_mode,
            "collection_name": runtime_collection,
            "target": runtime_target,
            "limit": runtime_limit,
            "model": runtime_model,
            "retrieval": retrieval_payload,
        }

    def _build_prompt_messages(data: Mapping[str, Any]) -> list[BaseMessage]:
        retrieval_payload = dict(data["retrieval"])
        messages = _build_answer_messages(
            query=str(data["query"]),
            mode=str(data["mode"]),
            works=list(retrieval_payload.get("works") or []),
            chunks=list(retrieval_payload.get("evidence_chunks") or []),
        )
        return [_payload_to_message(item) for item in messages]

    def _finalize_output(data: Mapping[str, Any]) -> dict[str, Any]:
        retrieval_payload = dict(data["retrieval"])
        return {
            "query": str(data["query"]),
            "mode": str(data["mode"]),
            "answer": str(data["answer"]),
            "documents": retrieval_to_documents(retrieval_payload),
            "retrieval": retrieval_payload,
        }

    return (
        RunnableLambda(_normalize_input)
        | RunnableLambda(_attach_retrieval)
        | RunnablePassthrough.assign(
            answer=RunnableLambda(_build_prompt_messages) | chat_model | StrOutputParser(),
        )
        | RunnableLambda(_finalize_output)
    )


def build_langchain_answer_chain(
    config: dict[str, Any],
    *,
    collection_name: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    mode: str = "answer",
    progress: ProgressCallback = None,
) -> Any:
    return build_langchain_answer_runnable(
        config,
        collection_name=collection_name,
        target=target,
        limit=limit,
        model=model,
        mode=mode,
        progress=progress,
    ) | RunnableLambda(lambda item: item["answer"])


def _chunk_to_document(item: Mapping[str, Any]) -> Document:
    metadata = {
        "source_type": "chunk",
        "chunk_id": item.get("chunk_id"),
        "work_id": item.get("work_id"),
        "section_hint": item.get("section_hint"),
        "chunk_role": item.get("chunk_role"),
        "canonical_source": item.get("canonical_source"),
        "canonical_id": item.get("canonical_id"),
        "title": item.get("raw_title"),
        "score": item.get("hybrid_score", item.get("score")),
        "rank": item.get("rank"),
    }
    content = str(item.get("clean_text") or "").strip()
    if not content:
        content = str(item.get("raw_title") or "").strip()
    return Document(page_content=content, metadata=_drop_none(metadata))


def _work_to_document(item: Mapping[str, Any]) -> Document:
    content_parts = [
        str(item.get("raw_title") or "").strip(),
        str(item.get("abstract") or "").strip(),
    ]
    metadata = {
        "source_type": "work",
        "work_id": item.get("work_id"),
        "canonical_source": item.get("canonical_source"),
        "canonical_id": item.get("canonical_id"),
        "title": item.get("raw_title"),
        "year": item.get("year"),
        "score": item.get("hybrid_score", item.get("score")),
        "rank": item.get("rank"),
    }
    return Document(
        page_content="\n\n".join(part for part in content_parts if part),
        metadata=_drop_none(metadata),
    )


def _drop_none(data: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in data.items() if value is not None}


def _mapping_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping_int(data: Mapping[str, Any], key: str, *, default: int | None) -> int | None:
    value = data.get(key)
    if value in {None, ""}:
        return default
    return int(value)


def _payload_to_message(payload: Mapping[str, Any]) -> BaseMessage:
    role = str(payload.get("role") or "user").strip().casefold()
    content = str(payload.get("content") or "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    if role == "tool":
        return ToolMessage(content=content, tool_call_id="hep_rag_tool")
    if role == "user":
        return HumanMessage(content=content)
    return ChatMessage(role=role, content=content)


def _message_to_payload(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ChatMessage):
        role = str(message.role or "user").strip() or "user"
    else:
        role = "user"
    return {
        "role": role,
        "content": _message_content_text(message),
    }


def _message_content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, Mapping):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _apply_stop_tokens(text: str, stop: list[str] | None) -> str:
    value = str(text or "")
    if not stop:
        return value
    cut = len(value)
    for token in stop:
        if not token:
            continue
        idx = value.find(token)
        if idx >= 0:
            cut = min(cut, idx)
    return value[:cut]
