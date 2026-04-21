from __future__ import annotations

import re
from typing import Any, Callable

from hep_rag_v2.community import search_community_summaries
from hep_rag_v2.config import resolve_embedding_settings
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.metadata import expand_work_ids_with_family, family_payload_map
from hep_rag_v2.ontology import search_ontology_summaries
from hep_rag_v2.providers.local_transformers import LocalTransformersClient
from hep_rag_v2.providers.openai_compatible import OpenAICompatibleClient
from hep_rag_v2.query import is_relation_query, is_result_query
from hep_rag_v2.search_scope import normalize_search_scope
from hep_rag_v2.vector import (
    DEFAULT_VECTOR_MODEL,
    configure_embedding_runtime,
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
    search_scope = normalize_search_scope(collection_name)
    collection = search_scope.collection_name
    requested_target = str(target or retrieval_cfg.get("target") or "auto")
    embedding_settings = resolve_embedding_settings(config, model=model)
    embedding_model = str(embedding_settings.get("model") or DEFAULT_VECTOR_MODEL)
    configure_embedding_runtime(model=embedding_model, settings=embedding_settings)
    limit_value = max(1, int(limit or retrieval_cfg.get("limit") or 8))
    chunk_limit = max(limit_value, int(retrieval_cfg.get("chunk_limit") or max(limit_value, 12)))
    community_limit = max(1, int(retrieval_cfg.get("community_limit") or min(limit_value, 4)))
    ontology_limit = max(1, int(retrieval_cfg.get("ontology_limit") or min(limit_value, 4)))
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
        community_summaries: list[dict[str, Any]] = []
        ontology_summaries: list[dict[str, Any]] = []
        if actual_target == "community":
            _emit_progress(progress, "searching community summaries...")
            community_summaries = search_community_summaries(
                conn,
                query=query,
                collection=collection,
                limit=community_limit,
            )
            representative_work_ids = _summary_representative_work_ids(
                community_summaries,
                limit=max(limit_value * 2, community_limit * 4),
            )
            works = _hydrate_works_by_ids(conn, work_ids=representative_work_ids, limit=limit_value)
            if works:
                _emit_progress(progress, "collecting supporting chunks from community summaries...")
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
                chunks = []
        elif actual_target == "ontology":
            _emit_progress(progress, "searching ontology summaries...")
            ontology_summaries = search_ontology_summaries(
                conn,
                query=query,
                collection=collection,
                limit=ontology_limit,
            )
            representative_work_ids = _summary_representative_work_ids(
                ontology_summaries,
                limit=max(limit_value * 2, ontology_limit * 4),
            )
            works = _hydrate_works_by_ids(conn, work_ids=representative_work_ids, limit=limit_value)
            if works:
                _emit_progress(progress, "collecting supporting chunks from ontology summaries...")
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
                chunks = []
        elif actual_target == "works":
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
            if _should_attach_community_summaries(query=query, actual_target=actual_target):
                _emit_progress(progress, "attaching community summaries...")
                community_summaries = search_community_summaries(
                    conn,
                    query=query,
                    collection=collection,
                    limit=community_limit,
                )
            if _should_attach_ontology_summaries(query=query, actual_target=actual_target):
                _emit_progress(progress, "attaching ontology summaries...")
                ontology_summaries = search_ontology_summaries(
                    conn,
                    query=query,
                    collection=collection,
                    limit=ontology_limit,
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
        "collection": collection or "all",
        "search_scope": search_scope.to_payload(),
        "max_parallelism": retrieval_workers,
        "requested_target": requested_target,
        "routing": {
            "target": actual_target,
            "graph_expand": int(graph_expand),
            "reasons": list(routing.get("reasons") or []),
        },
        "model": embedding_model,
        "community_summaries": community_summaries,
        "ontology_summaries": ontology_summaries,
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
    community_summaries = list(retrieval.get("community_summaries") or [])
    evidence_community = community_summaries[: max(2, min(len(community_summaries), 4))]
    ontology_summaries = list(retrieval.get("ontology_summaries") or [])
    evidence_ontology = ontology_summaries[: max(2, min(len(ontology_summaries), 4))]

    if not any((evidence_works, evidence_chunks, evidence_community, evidence_ontology)):
        return {
            "status": "insufficient_evidence",
            "query": query,
            "mode": mode,
            "collection": retrieval.get("collection"),
            "search_scope": retrieval.get("search_scope"),
            "requested_target": retrieval.get("requested_target"),
            "routing": retrieval.get("routing"),
            "retrieval_model": retrieval.get("model"),
            "llm_backend": str(llm_cfg.get("backend") or "openai_compatible"),
            "llm_model": None,
            "answer_strategy": "insufficient_evidence",
            "answer": "Knowledge-base evidence is insufficient to answer this query.",
            "community_map_notes": [],
            "evidence": {
                "works": evidence_works,
                "chunks": evidence_chunks,
                "community_summaries": evidence_community,
                "ontology_summaries": evidence_ontology,
            },
        }

    client = _build_llm_client(llm_cfg)
    answer_strategy = _resolve_answer_strategy(
        llm_cfg=llm_cfg,
        query=query,
        mode=mode,
        community_summaries=evidence_community,
    )
    community_map_notes: list[dict[str, Any]] = []
    if answer_strategy == "community_map_reduce":
        _emit_progress(progress, "generating community map notes...")
        answer, community_map_notes = _run_community_map_reduce(
            client=client,
            llm_cfg=llm_cfg,
            query=query,
            mode=mode,
            works=evidence_works,
            chunks=evidence_chunks,
            community_summaries=evidence_community,
            ontology_summaries=evidence_ontology,
            progress=progress,
        )
    else:
        messages = _build_answer_messages(
            query=query,
            mode=mode,
            works=evidence_works,
            chunks=evidence_chunks,
            community_summaries=evidence_community,
            ontology_summaries=evidence_ontology,
        )
        _emit_progress(progress, "generating answer with LLM...")
        answer = client.chat(
            messages=messages,
            temperature=float(llm_cfg.get("temperature") or 0.2),
            max_tokens=int(llm_cfg.get("max_tokens") or 1200),
        )
    return {
        "status": "ok",
        "query": query,
        "mode": mode,
        "collection": retrieval.get("collection"),
        "search_scope": retrieval.get("search_scope"),
        "requested_target": retrieval.get("requested_target"),
        "routing": retrieval.get("routing"),
        "retrieval_model": retrieval.get("model"),
        "llm_backend": str(llm_cfg.get("backend") or "openai_compatible"),
        "llm_model": answer["model"],
        "answer_strategy": answer_strategy,
        "answer": answer["content"],
        "community_map_notes": community_map_notes,
        "evidence": {
            "works": evidence_works,
            "chunks": evidence_chunks,
            "community_summaries": evidence_community,
            "ontology_summaries": evidence_ontology,
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


def _resolve_answer_strategy(
    *,
    llm_cfg: dict[str, Any],
    query: str,
    mode: str,
    community_summaries: list[dict[str, Any]],
) -> str:
    configured = str(llm_cfg.get("answer_strategy") or "auto").strip().casefold() or "auto"
    if configured not in {"auto", "single_pass", "community_map_reduce"}:
        configured = "auto"
    if configured == "single_pass":
        return "single_pass"
    can_map_reduce = _can_use_community_map_reduce(
        llm_cfg=llm_cfg,
        query=query,
        mode=mode,
        community_summaries=community_summaries,
    )
    if configured == "community_map_reduce":
        return "community_map_reduce" if can_map_reduce else "single_pass"
    return "community_map_reduce" if can_map_reduce else "single_pass"


def _can_use_community_map_reduce(
    *,
    llm_cfg: dict[str, Any],
    query: str,
    mode: str,
    community_summaries: list[dict[str, Any]],
) -> bool:
    if not bool(llm_cfg.get("map_reduce_enabled", True)):
        return False
    overview_count = sum(
        1
        for item in community_summaries
        if str(item.get("community_level") or "fine").strip().casefold() == "overview"
    )
    if overview_count == 0:
        return False
    if mode == "survey":
        return True
    return is_relation_query(query) or is_result_query(query)


def _run_community_map_reduce(
    *,
    client: OpenAICompatibleClient | LocalTransformersClient,
    llm_cfg: dict[str, Any],
    query: str,
    mode: str,
    works: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    community_summaries: list[dict[str, Any]],
    ontology_summaries: list[dict[str, Any]],
    progress: ProgressCallback = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    community_units = _prepare_map_reduce_communities(
        llm_cfg=llm_cfg,
        community_summaries=community_summaries,
        ontology_summaries=ontology_summaries,
        works=works,
        chunks=chunks,
    )
    if not community_units:
        messages = _build_answer_messages(
            query=query,
            mode=mode,
            works=works,
            chunks=chunks,
            community_summaries=community_summaries,
            ontology_summaries=ontology_summaries,
        )
        _emit_progress(progress, "generating answer with LLM...")
        answer = client.chat(
            messages=messages,
            temperature=float(llm_cfg.get("temperature") or 0.2),
            max_tokens=int(llm_cfg.get("max_tokens") or 1200),
        )
        return (answer, [])

    map_notes: list[dict[str, Any]] = []
    map_max_tokens = int(llm_cfg.get("map_max_tokens") or 500)
    temperature = float(llm_cfg.get("temperature") or 0.2)
    for index, unit in enumerate(community_units, start=1):
        _emit_progress(progress, f"mapping community {index}/{len(community_units)}...")
        map_response = client.chat(
            messages=_build_community_map_messages(query=query, mode=mode, unit=unit),
            temperature=min(temperature, 0.2),
            max_tokens=map_max_tokens,
        )
        map_notes.append(
            {
                "summary_id": unit["overview"].get("summary_id"),
                "label": unit["overview"].get("label") or unit["overview"].get("title"),
                "community_level": unit["overview"].get("community_level"),
                "source_refs": list(unit.get("source_refs") or []),
                "content": map_response["content"],
                "llm_model": map_response["model"],
            }
        )

    _emit_progress(progress, "reducing community map notes...")
    answer = client.chat(
        messages=_build_community_reduce_messages(
            query=query,
            mode=mode,
            map_notes=map_notes,
            works=works,
            chunks=chunks,
            community_summaries=community_summaries,
            ontology_summaries=ontology_summaries,
        ),
        temperature=temperature,
        max_tokens=int(llm_cfg.get("max_tokens") or 1200),
    )
    return (answer, map_notes)


def _prepare_map_reduce_communities(
    *,
    llm_cfg: dict[str, Any],
    community_summaries: list[dict[str, Any]],
    ontology_summaries: list[dict[str, Any]],
    works: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    community_limit = max(1, int(llm_cfg.get("map_reduce_max_communities") or 3))
    child_limit = max(1, int(llm_cfg.get("map_reduce_child_limit") or 3))
    overview_items = [
        dict(item)
        for item in community_summaries
        if str(item.get("community_level") or "fine").strip().casefold() == "overview"
    ][:community_limit]
    if not overview_items:
        return []
    fine_by_parent: dict[str, list[dict[str, Any]]] = {}
    for item in community_summaries:
        if str(item.get("community_level") or "fine").strip().casefold() != "fine":
            continue
        parent_summary_id = str(item.get("parent_summary_id") or "").strip()
        if not parent_summary_id:
            continue
        fine_by_parent.setdefault(parent_summary_id, []).append(dict(item))
    community_ref_map = {
        str(item.get("summary_id")): f"G{index}"
        for index, item in enumerate(community_summaries, start=1)
        if item.get("summary_id")
    }
    ontology_ref_map = {
        str(item.get("summary_id")): f"O{index}"
        for index, item in enumerate(ontology_summaries, start=1)
        if item.get("summary_id")
    }
    work_ref_map = {
        int(item["work_id"]): f"W{index}"
        for index, item in enumerate(works, start=1)
        if item.get("work_id") is not None
    }
    chunk_ref_map = {
        int(item["chunk_id"]): f"C{index}"
        for index, item in enumerate(chunks, start=1)
        if item.get("chunk_id") is not None
    }
    units: list[dict[str, Any]] = []
    for overview in overview_items:
        overview_id = str(overview.get("summary_id") or "")
        fine_items = fine_by_parent.get(overview_id, [])[:child_limit]
        member_work_ids = {
            int(work_id)
            for work_id in list((overview.get("metadata") or {}).get("member_work_ids") or [])
            if work_id not in {None, ""}
        }
        scoped_works = [
            dict(item)
            for item in works
            if item.get("work_id") is not None and int(item["work_id"]) in member_work_ids
        ]
        if not scoped_works:
            scoped_works = [dict(item) for item in works[:2]]
        scoped_chunks = [
            dict(item)
            for item in chunks
            if item.get("work_id") is not None and int(item["work_id"]) in member_work_ids
        ]
        if not scoped_chunks:
            scoped_chunks = [dict(item) for item in chunks[:2]]
        scoped_ontology = _select_ontology_for_community(
            overview=overview,
            fine_items=fine_items,
            ontology_summaries=ontology_summaries,
        )
        source_refs = [community_ref_map.get(overview_id, overview_id)]
        source_refs.extend(
            community_ref_map.get(str(item.get("summary_id") or ""), str(item.get("summary_id") or ""))
            for item in fine_items
            if item.get("summary_id")
        )
        source_refs.extend(
            ontology_ref_map.get(str(item.get("summary_id") or ""), str(item.get("summary_id") or ""))
            for item in scoped_ontology
            if item.get("summary_id")
        )
        source_refs.extend(
            work_ref_map.get(int(item["work_id"]), f"work:{item['work_id']}")
            for item in scoped_works
            if item.get("work_id") is not None
        )
        source_refs.extend(
            chunk_ref_map.get(int(item["chunk_id"]), f"chunk:{item['chunk_id']}")
            for item in scoped_chunks
            if item.get("chunk_id") is not None
        )
        units.append(
            {
                "overview": overview,
                "overview_ref": community_ref_map.get(overview_id, overview_id),
                "fine_items": fine_items,
                "community_ref_map": community_ref_map,
                "ontology_items": scoped_ontology,
                "ontology_ref_map": ontology_ref_map,
                "works": scoped_works[:3],
                "work_ref_map": work_ref_map,
                "chunks": scoped_chunks[:3],
                "chunk_ref_map": chunk_ref_map,
                "source_refs": [str(item) for item in source_refs if item],
            }
        )
    return units


def _select_ontology_for_community(
    *,
    overview: dict[str, Any],
    fine_items: list[dict[str, Any]],
    ontology_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    community_terms = _community_match_terms(overview)
    for item in fine_items:
        community_terms.update(_community_match_terms(item))
    scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for item in ontology_summaries:
        ontology_terms = _ontology_match_terms(item)
        overlap = len(community_terms & ontology_terms)
        score = overlap * 2
        if str(item.get("label") or "").casefold() in community_terms:
            score += 1
        if score == 0 and int(item.get("rank") or 99) > 2:
            continue
        scored.append(
            (
                (-score, int(item.get("rank") or 99), str(item.get("label") or "").casefold()),
                dict(item),
            )
        )
    if not scored:
        return [dict(item) for item in ontology_summaries[:1]]
    return [payload for _, payload in sorted(scored)[:2]]


def _community_match_terms(item: dict[str, Any]) -> set[str]:
    metadata = dict(item.get("metadata") or {})
    terms = _label_terms(
        str(item.get("label") or ""),
        metadata.get("collaborations"),
        metadata.get("topics"),
        metadata.get("result_kinds"),
        metadata.get("method_families"),
        metadata.get("child_labels"),
    )
    return terms


def _ontology_match_terms(item: dict[str, Any]) -> set[str]:
    metadata = dict(item.get("metadata") or {})
    return _label_terms(
        str(item.get("label") or ""),
        str(item.get("facet_kind") or ""),
        metadata.get("collaborations"),
        metadata.get("topics"),
        metadata.get("result_kinds"),
        metadata.get("method_families"),
    )


def _label_terms(*values: Any) -> set[str]:
    terms: set[str] = set()
    for value in values:
        if isinstance(value, list):
            for item in value:
                terms.update(_label_terms(item))
            continue
        text = str(value or "").strip().casefold()
        if not text:
            continue
        text = text.split("=", 1)[0].strip()
        if not text:
            continue
        terms.add(text)
        for token in re.split(r"[^a-z0-9]+", text):
            token = token.strip()
            if len(token) >= 3:
                terms.add(token)
    return terms


def _build_answer_messages(
    *,
    query: str,
    mode: str,
    works: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    community_summaries: list[dict[str, Any]],
    ontology_summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    mode_text = _mode_prompt(mode)
    work_lines = _format_work_lines(works)
    chunk_lines = _format_chunk_lines(chunks)
    community_lines = _format_community_lines(community_summaries)
    ontology_lines = _format_ontology_lines(ontology_summaries)
    user_prompt = (
        f"用户问题:\n{query}\n\n"
        f"写作模式:\n{mode_text}\n\n"
        "可用 community summaries:\n"
        + ("\n\n".join(community_lines) if community_lines else "(none)")
        + "\n\n"
        "可用 ontology summaries:\n"
        + ("\n\n".join(ontology_lines) if ontology_lines else "(none)")
        + "\n\n"
        "可用论文列表:\n"
        + ("\n".join(work_lines) if work_lines else "(none)")
        + "\n\n可用证据片段:\n"
        + ("\n\n".join(chunk_lines) if chunk_lines else "(none)")
        + "\n\n要求:\n"
        "1. 只能基于上面证据作答，不要补造未给出的论文结论。\n"
        "2. 如果证据不足，要明确说不足在哪里。\n"
        "3. community summaries 里如果出现 level=overview，可以把它当作较高层主题簇；level=fine 则是更具体的子社区。\n"
        "4. 回答时尽量引用 [G1] / [O1] / [W1] / [C2] 这种编号，便于回溯。\n"
        "5. 使用与用户问题相同的语言。"
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


def _build_community_map_messages(
    *,
    query: str,
    mode: str,
    unit: dict[str, Any],
) -> list[dict[str, str]]:
    overview = dict(unit.get("overview") or {})
    fine_items = list(unit.get("fine_items") or [])
    ontology_items = list(unit.get("ontology_items") or [])
    work_lines = _format_work_lines(list(unit.get("works") or []), ref_map=unit.get("work_ref_map"))
    chunk_lines = _format_chunk_lines(list(unit.get("chunks") or []), ref_map=unit.get("chunk_ref_map"))
    community_lines = _format_community_lines(
        [overview, *fine_items],
        ref_map=unit.get("community_ref_map"),
    )
    ontology_lines = _format_ontology_lines(
        ontology_items,
        ref_map=unit.get("ontology_ref_map"),
    )
    user_prompt = (
        f"用户问题:\n{query}\n\n"
        f"写作模式:\n{_mode_prompt(mode)}\n\n"
        "当前要分析的社区簇:\n"
        + ("\n\n".join(community_lines) if community_lines else "(none)")
        + "\n\n"
        "与该社区最相关的 ontology summaries:\n"
        + ("\n\n".join(ontology_lines) if ontology_lines else "(none)")
        + "\n\n"
        "与该社区最相关的论文:\n"
        + ("\n".join(work_lines) if work_lines else "(none)")
        + "\n\n与该社区最相关的证据片段:\n"
        + ("\n\n".join(chunk_lines) if chunk_lines else "(none)")
        + "\n\n要求:\n"
        "1. 你现在只分析这个社区，不要试图回答整个问题。\n"
        "2. 输出三段：Community Focus、Evidence-backed Points、Open Gaps。\n"
        "3. 每个段落都尽量引用 [G] / [O] / [W] / [C] 编号。\n"
        "4. 只能根据给定证据写，不要引入外部知识。\n"
        "5. 使用与用户问题相同的语言。"
    )
    return [
        {
            "role": "system",
            "content": "你是一个社区级文献分析助手。你的任务是对单个图谱社区做局部证据归纳，而不是回答全局问题。",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _build_community_reduce_messages(
    *,
    query: str,
    mode: str,
    map_notes: list[dict[str, Any]],
    works: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    community_summaries: list[dict[str, Any]],
    ontology_summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    community_lines = _format_community_lines(community_summaries)
    ontology_lines = _format_ontology_lines(ontology_summaries)
    work_lines = _format_work_lines(works)
    chunk_lines = _format_chunk_lines(chunks)
    map_lines = _format_map_note_lines(map_notes)
    user_prompt = (
        f"用户问题:\n{query}\n\n"
        f"写作模式:\n{_mode_prompt(mode)}\n\n"
        "community map notes:\n"
        + ("\n\n".join(map_lines) if map_lines else "(none)")
        + "\n\n"
        "原始 community summaries:\n"
        + ("\n\n".join(community_lines) if community_lines else "(none)")
        + "\n\n"
        "原始 ontology summaries:\n"
        + ("\n\n".join(ontology_lines) if ontology_lines else "(none)")
        + "\n\n"
        "原始论文列表:\n"
        + ("\n".join(work_lines) if work_lines else "(none)")
        + "\n\n原始证据片段:\n"
        + ("\n\n".join(chunk_lines) if chunk_lines else "(none)")
        + "\n\n要求:\n"
        "1. community map notes 只是中间分析；最终回答必须以原始 evidence 为准。\n"
        "2. 只能基于原始 evidence 作答，不要补造未给出的论文结论。\n"
        "3. 如果证据不足，要明确指出不足。\n"
        "4. 回答时尽量引用 [G1] / [O1] / [W1] / [C1] 这种编号。\n"
        "5. 使用与用户问题相同的语言。"
    )
    return [
        {
            "role": "system",
            "content": "你是一个科研文献综合助手。你的任务是把多个社区级分析整合成严格、可回溯的全局回答。",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _mode_prompt(mode: str) -> str:
    mode_text = {
        "answer": "直接回答问题，并明确区分结论、证据、不确定性。",
        "survey": "写成短综述，按研究方向或方法分组，并点出代表论文与差异。",
        "idea": "从证据中提炼潜在研究空白与可操作想法，并说明依据与风险。",
    }.get(mode, "基于证据回答问题。")
    return mode_text


def _format_work_lines(
    works: list[dict[str, Any]],
    *,
    ref_map: dict[int, str] | None = None,
) -> list[str]:
    work_lines = []
    for idx, item in enumerate(works, start=1):
        ref = None
        if ref_map is not None and item.get("work_id") is not None:
            ref = ref_map.get(int(item["work_id"]))
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
            f"[{ref or f'W{idx}'}] {item.get('raw_title')} ({item.get('year')}) "
            f"{item.get('canonical_source')}:{item.get('canonical_id')}{related_text}"
        )
    return work_lines


def _format_chunk_lines(
    chunks: list[dict[str, Any]],
    *,
    ref_map: dict[int, str] | None = None,
) -> list[str]:
    chunk_lines = []
    for idx, item in enumerate(chunks, start=1):
        ref = None
        if ref_map is not None and item.get("chunk_id") is not None:
            ref = ref_map.get(int(item["chunk_id"]))
        snippet = " ".join(str(item.get("clean_text") or "").split())
        if len(snippet) > 700:
            snippet = snippet[:700].rstrip() + " ..."
        chunk_lines.append(
            f"[{ref or f'C{idx}'}] work_id={item.get('work_id')} title={item.get('raw_title')} "
            f"section={item.get('section_hint')}\n{snippet}"
        )
    return chunk_lines


def _format_community_lines(
    community_summaries: list[dict[str, Any]],
    *,
    ref_map: dict[str, str] | None = None,
) -> list[str]:
    community_lines = []
    for idx, item in enumerate(community_summaries, start=1):
        ref = None
        if ref_map is not None and item.get("summary_id"):
            ref = ref_map.get(str(item["summary_id"]))
        summary_text = " ".join(str(item.get("summary_text") or item.get("summary") or "").split())
        if len(summary_text) > 500:
            summary_text = summary_text[:500].rstrip() + " ..."
        representative = list(item.get("representative_works") or [])[:3]
        representative_text = "; ".join(
            f"{work.get('title')} ({work.get('year')})"
            for work in representative
            if isinstance(work, dict) and work.get("title")
        )
        header = (
            f"[{ref or f'G{idx}'}] {item.get('label') or item.get('title')} "
            f"| level={item.get('community_level') or 'fine'} "
            f"| works={item.get('work_count') or 0} "
            f"| edges={item.get('edge_count') or 0}"
        )
        if item.get("parent_summary_id"):
            header += f" | parent={item.get('parent_summary_id')}"
        if representative_text:
            header += f"\nrepresentative works: {representative_text}"
        community_lines.append(f"{header}\n{summary_text}")
    return community_lines


def _format_ontology_lines(
    ontology_summaries: list[dict[str, Any]],
    *,
    ref_map: dict[str, str] | None = None,
) -> list[str]:
    ontology_lines = []
    for idx, item in enumerate(ontology_summaries, start=1):
        ref = None
        if ref_map is not None and item.get("summary_id"):
            ref = ref_map.get(str(item["summary_id"]))
        summary_text = " ".join(str(item.get("summary_text") or item.get("summary") or "").split())
        if len(summary_text) > 500:
            summary_text = summary_text[:500].rstrip() + " ..."
        representative = list(item.get("representative_works") or [])[:3]
        representative_text = "; ".join(
            f"{work.get('title')} ({work.get('year')})"
            for work in representative
            if isinstance(work, dict) and work.get("title")
        )
        header = (
            f"[{ref or f'O{idx}'}] {item.get('label') or item.get('title')} "
            f"| facet={item.get('facet_kind')} "
            f"| works={item.get('work_count') or 0} "
            f"| signals={item.get('signal_count') or 0}"
        )
        if representative_text:
            header += f"\nrepresentative works: {representative_text}"
        ontology_lines.append(f"{header}\n{summary_text}")
    return ontology_lines


def _format_map_note_lines(map_notes: list[dict[str, Any]]) -> list[str]:
    lines = []
    for index, item in enumerate(map_notes, start=1):
        source_refs = ", ".join(str(ref) for ref in list(item.get("source_refs") or [])[:6])
        header = f"[M{index}] {item.get('label') or 'Community map note'}"
        if source_refs:
            header += f" | refs={source_refs}"
        lines.append(f"{header}\n{str(item.get('content') or '').strip()}")
    return lines


def _should_attach_ontology_summaries(*, query: str, actual_target: str) -> bool:
    if actual_target == "ontology":
        return True
    if actual_target != "works":
        return False
    return is_relation_query(query) or is_result_query(query)


def _should_attach_community_summaries(*, query: str, actual_target: str) -> bool:
    if actual_target == "community":
        return True
    if actual_target != "works":
        return False
    return is_relation_query(query) or is_result_query(query)


def _summary_representative_work_ids(
    summaries: list[dict[str, Any]],
    *,
    limit: int,
) -> list[int]:
    work_ids: list[int] = []
    seen: set[int] = set()
    for summary in summaries:
        for item in list(summary.get("representative_works") or []):
            if not isinstance(item, dict) or item.get("work_id") is None:
                continue
            work_id = int(item["work_id"])
            if work_id in seen:
                continue
            seen.add(work_id)
            work_ids.append(work_id)
            if len(work_ids) >= limit:
                return work_ids
    return work_ids


def _hydrate_works_by_ids(
    conn: Any,
    *,
    work_ids: list[int],
    limit: int,
) -> list[dict[str, Any]]:
    if not work_ids:
        return []
    selected_ids = []
    seen: set[int] = set()
    for work_id in work_ids:
        key = int(work_id)
        if key in seen:
            continue
        seen.add(key)
        selected_ids.append(key)
        if len(selected_ids) >= limit:
            break
    if not selected_ids:
        return []
    family_map = family_payload_map(conn, work_ids=selected_ids)
    rows = conn.execute(
        """
        SELECT work_id, year, canonical_source, canonical_id, title AS raw_title
        FROM works
        WHERE work_id IN ({placeholders})
        ORDER BY work_id
        """.format(placeholders=",".join("?" for _ in selected_ids)),
        selected_ids,
    ).fetchall()
    row_map = {int(row["work_id"]): dict(row) for row in rows}
    out: list[dict[str, Any]] = []
    for rank, work_id in enumerate(selected_ids, start=1):
        row = row_map.get(work_id)
        if row is None:
            continue
        row["rank"] = rank
        row["search_type"] = "ontology_support"
        row["hybrid_score"] = 0.0
        family_payload = family_map.get(work_id)
        if family_payload is not None:
            row.update(family_payload)
        out.append(row)
    return out
