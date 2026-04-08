from __future__ import annotations

from typing import Any, Callable

from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.metadata import expand_work_ids_with_family, family_payload_map
from hep_rag_v2.providers.local_transformers import LocalTransformersClient
from hep_rag_v2.providers.openai_compatible import OpenAICompatibleClient
from hep_rag_v2.vector import (
    DEFAULT_VECTOR_MODEL,
    route_query,
    search_chunks_hybrid,
    search_works_hybrid,
)

ProgressCallback = Callable[[str], None] | None


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is not None:
        progress(message)


def _resolve_parallelism(
    *,
    requested: int | None,
    configured: Any,
    fallback: int,
    ceiling: int | None = None,
) -> int:
    value = requested
    if value is None:
        try:
            value = int(configured) if configured not in {None, ""} else None
        except (TypeError, ValueError):
            value = None
    if value is None:
        value = int(fallback)
    value = max(1, int(value))
    if ceiling is not None:
        value = min(value, max(1, int(ceiling)))
    return value


def retrieve(
    config: dict[str, Any],
    *,
    query: str,
    limit: int | None = None,
    target: str | None = None,
    collection_name: str | None = None,
    max_parallelism: int | None = None,
    model: str | None = None,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_db()
    retrieval_cfg = config.get("retrieval") or {}
    collection = collection_name or str((config.get("collection") or {}).get("name") or "default")
    requested_target = str(target or retrieval_cfg.get("target") or "auto")
    embedding_model = str(model or (config.get("embedding") or {}).get("model") or DEFAULT_VECTOR_MODEL)
    limit_value = max(1, int(limit or retrieval_cfg.get("limit") or 8))
    chunk_limit = max(limit_value, int(retrieval_cfg.get("chunk_limit") or max(limit_value, 12)))
    retrieval_workers = _resolve_parallelism(
        requested=max_parallelism,
        configured=retrieval_cfg.get("max_parallelism"),
        fallback=2,
        ceiling=2,
    )

    routing = route_query(query) if requested_target == "auto" else {
        "target": requested_target,
        "graph_expand": 0,
        "reasons": ["manual_target"],
    }
    actual_target = str(routing["target"])
    _emit_progress(progress, f"routing query to {actual_target} retrieval...")
    graph_expand = retrieval_cfg.get("graph_expand")
    if graph_expand is None:
        graph_expand = routing.get("graph_expand") or 0

    with connect() as conn:
        if actual_target == "works":
            _emit_progress(progress, "searching works...")
            works = search_works_hybrid(
                conn,
                query=query,
                collection=collection,
                limit=limit_value,
                model=embedding_model,
                graph_expand=int(graph_expand),
                seed_limit=int(retrieval_cfg.get("seed_limit") or 5),
                max_parallelism=retrieval_workers,
            )
            _emit_progress(progress, "collecting supporting chunks...")
            chunks = _supporting_chunks(
                conn,
                query=query,
                collection=collection,
                model=embedding_model,
                work_ids=[int(row["work_id"]) for row in works],
                limit=chunk_limit,
                max_parallelism=retrieval_workers,
            )
        else:
            _emit_progress(progress, "searching chunks...")
            chunks = search_chunks_hybrid(
                conn,
                query=query,
                collection=collection,
                limit=chunk_limit,
                model=embedding_model,
                max_parallelism=retrieval_workers,
            )
            _emit_progress(progress, "hydrating work-level evidence...")
            works = _hydrate_works_from_chunks(conn, chunks=chunks, limit=limit_value)

    return {
        "query": query,
        "collection": collection,
        "max_parallelism": retrieval_workers,
        "requested_target": requested_target,
        "routing": {
            "target": actual_target,
            "graph_expand": int(graph_expand),
            "reasons": list(routing.get("reasons") or []),
        },
        "model": embedding_model,
        "works": works,
        "evidence_chunks": chunks,
    }


def ask(
    config: dict[str, Any],
    *,
    query: str,
    mode: str = "answer",
    limit: int | None = None,
    target: str | None = None,
    collection_name: str | None = None,
    max_parallelism: int | None = None,
    model: str | None = None,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    llm_cfg = config.get("llm") or {}
    if not bool(llm_cfg.get("enabled")):
        raise ValueError("LLM is disabled in config. Set llm.enabled=true and configure either openai_compatible or local_transformers.")

    _emit_progress(progress, "retrieving evidence...")
    retrieval = retrieve(
        config,
        query=query,
        limit=limit,
        target=target,
        collection_name=collection_name,
        max_parallelism=max_parallelism,
        model=model,
        progress=progress,
    )
    evidence_limit = max(1, int((config.get("retrieval") or {}).get("answer_evidence_limit") or 6))
    evidence_chunks = retrieval["evidence_chunks"][:evidence_limit]
    evidence_works = retrieval["works"][: max(3, min(len(retrieval["works"]), evidence_limit))]

    client = _build_llm_client(llm_cfg)
    messages = _build_answer_messages(
        query=query,
        mode=mode,
        works=evidence_works,
        chunks=evidence_chunks,
    )
    _emit_progress(progress, "generating answer with LLM...")
    answer = client.chat(
        messages=messages,
        temperature=float(llm_cfg.get("temperature") or 0.2),
        max_tokens=int(llm_cfg.get("max_tokens") or 1200),
    )
    return {
        "query": query,
        "mode": mode,
        "llm_backend": str(llm_cfg.get("backend") or "openai_compatible"),
        "llm_model": answer["model"],
        "answer": answer["content"],
        "evidence": {
            "works": evidence_works,
            "chunks": evidence_chunks,
        },
    }


def _build_llm_client(llm_cfg: dict[str, Any]) -> OpenAICompatibleClient | LocalTransformersClient:
    backend = str(llm_cfg.get("backend") or "openai_compatible").strip() or "openai_compatible"
    if backend == "openai_compatible":
        return OpenAICompatibleClient(
            api_base=str(llm_cfg.get("api_base") or ""),
            api_key=str(llm_cfg.get("api_key") or ""),
            model=str(llm_cfg.get("model") or ""),
            chat_path=str(llm_cfg.get("chat_path") or "/chat/completions"),
            timeout_sec=int(llm_cfg.get("timeout_sec") or 120),
            extra_headers=dict(llm_cfg.get("extra_headers") or {}),
        )
    if backend == "local_transformers":
        return LocalTransformersClient(
            model_name_or_path=str(llm_cfg.get("local_model_path") or llm_cfg.get("model") or ""),
            device=str(llm_cfg.get("device") or "cpu"),
            torch_dtype=str(llm_cfg.get("torch_dtype") or "auto"),
            trust_remote_code=bool(llm_cfg.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unsupported llm.backend: {backend}")


def _build_mineru_client(config: dict[str, Any]) -> Any:
    from hep_rag_v2.providers.mineru_api import MinerUClient

    mineru = config.get("mineru") or {}
    if not bool(mineru.get("enabled")):
        return None
    return MinerUClient(
        api_base=str(mineru.get("api_base") or ""),
        api_token=str(mineru.get("api_token") or ""),
        model_version=str(mineru.get("model_version") or "pipeline"),
        is_ocr=bool(mineru.get("is_ocr", False)),
        enable_formula=bool(mineru.get("enable_formula", True)),
        enable_table=bool(mineru.get("enable_table", True)),
        language=str(mineru.get("language") or "en"),
        poll_interval_sec=int(mineru.get("poll_interval_sec") or 10),
        max_wait_sec=int(mineru.get("max_wait_sec") or 1800),
        timeout_sec=int(mineru.get("timeout_sec") or 120),
    )


def _supporting_chunks(
    conn: Any,
    *,
    query: str,
    collection: str,
    model: str,
    work_ids: list[int],
    limit: int,
    max_parallelism: int = 1,
) -> list[dict[str, Any]]:
    raw = search_chunks_hybrid(
        conn,
        query=query,
        collection=collection,
        limit=max(limit * 3, 30),
        model=model,
        max_parallelism=max_parallelism,
    )
    if not work_ids:
        return raw[:limit]
    expanded_work_ids = expand_work_ids_with_family(conn, work_ids=work_ids)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    work_id_set = {int(item) for item in (expanded_work_ids or work_ids)}
    for row in raw:
        chunk_id = int(row["chunk_id"])
        if int(row["work_id"]) not in work_id_set or chunk_id in seen:
            continue
        seen.add(chunk_id)
        selected.append(row)
        if len(selected) >= limit:
            return selected
    for row in raw:
        chunk_id = int(row["chunk_id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _hydrate_works_from_chunks(
    conn: Any,
    *,
    chunks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ids: list[int] = []
    seen_families: set[int] = set()
    family_map = family_payload_map(
        conn,
        work_ids=[int(row["work_id"]) for row in chunks if row.get("work_id") is not None],
    )
    for row in chunks:
        work_id = int(row["work_id"])
        family_payload = family_map.get(work_id) or {}
        family_id = family_payload.get("family_id")
        chosen_work_id = int(family_payload.get("family_primary_work_id") or work_id)
        if family_id is not None:
            family_key = int(family_id)
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
        if chosen_work_id not in ids:
            ids.append(chosen_work_id)
        if len(ids) >= limit:
            break
    if not ids:
        return []
    placeholder_scores = {int(row["work_id"]): float(row.get("hybrid_score") or 0.0) for row in chunks}
    rows = conn.execute(
        """
        SELECT work_id, year, canonical_source, canonical_id, title AS raw_title
        FROM works
        WHERE work_id IN ({placeholders})
        ORDER BY work_id
        """.format(placeholders=",".join("?" for _ in ids)),
        ids,
    ).fetchall()
    row_map = {int(row["work_id"]): dict(row) for row in rows}
    out: list[dict[str, Any]] = []
    for rank, work_id in enumerate(ids, start=1):
        row = row_map.get(work_id)
        if row is None:
            continue
        row["rank"] = rank
        row["search_type"] = "chunk_support"
        row["hybrid_score"] = placeholder_scores.get(work_id, 0.0)
        family_payload = family_map.get(work_id)
        if family_payload is not None:
            row.update(family_payload)
        out.append(row)
    return out


def _build_answer_messages(
    *,
    query: str,
    mode: str,
    works: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    mode_text = {
        "answer": "直接回答问题，并明确区分结论、证据、不确定性。",
        "survey": "写成短综述，按研究方向或方法分组，并点出代表论文与差异。",
        "idea": "从证据中提炼潜在研究空白与可操作想法，并说明依据与风险。",
    }.get(mode, "基于证据回答问题。")

    work_lines = []
    for idx, item in enumerate(works, start=1):
        related_versions = list(item.get("related_versions") or [])[:3]
        related_text = ""
        if related_versions:
            related_text = " | related versions: " + "; ".join(
                (
                    f"{version.get('member_role')}: {version.get('title')} "
                    f"({version.get('year')}) {version.get('canonical_source')}:{version.get('canonical_id')}"
                ).strip()
                for version in related_versions
            )
        work_lines.append(
            f"[W{idx}] {item.get('raw_title')} ({item.get('year')}) "
            f"{item.get('canonical_source')}:{item.get('canonical_id')}{related_text}"
        )

    chunk_lines = []
    for idx, item in enumerate(chunks, start=1):
        snippet = " ".join(str(item.get("clean_text") or "").split())
        if len(snippet) > 700:
            snippet = snippet[:700].rstrip() + " ..."
        chunk_lines.append(
            f"[C{idx}] work_id={item.get('work_id')} title={item.get('raw_title')} "
            f"section={item.get('section_hint')}\n{snippet}"
        )

    user_prompt = (
        f"用户问题:\n{query}\n\n"
        f"写作模式:\n{mode_text}\n\n"
        "可用论文列表:\n"
        + ("\n".join(work_lines) if work_lines else "(none)")
        + "\n\n可用证据片段:\n"
        + ("\n\n".join(chunk_lines) if chunk_lines else "(none)")
        + "\n\n要求:\n"
        "1. 只能基于上面证据作答，不要补造未给出的论文结论。\n"
        "2. 如果证据不足，要明确说不足在哪里。\n"
        "3. 回答时尽量引用 [W1] / [C2] 这种编号，便于回溯。\n"
        "4. 使用与用户问题相同的语言。"
    )
    return [
        {
            "role": "system",
            "content": "你是一个科研文献分析助手。你的任务是基于给定证据做严格、可回溯的回答。",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
