from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hep_rag_v2 import paths
from hep_rag_v2.config import runtime_collection_config
from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2.fulltext import import_mineru_source, materialize_mineru_document
from hep_rag_v2.graph import rebuild_graph_edges
from hep_rag_v2.metadata import canonical_identity, upsert_collection, upsert_work_from_hit
from hep_rag_v2.providers.inspire import (
    content_addressed_name,
    download_pdf_candidates,
    list_pdf_candidates,
    search_literature,
    summarize_hit,
)
from hep_rag_v2.providers.local_transformers import LocalTransformersClient
from hep_rag_v2.providers.mineru_api import MinerUClient
from hep_rag_v2.providers.openai_compatible import OpenAICompatibleClient
from hep_rag_v2.records import paper_storage_stem, parsed_doc_dir
from hep_rag_v2.search import rebuild_search_indices
from hep_rag_v2.vector import (
    DEFAULT_VECTOR_MODEL,
    rebuild_vector_indices,
    route_query,
    search_chunks_hybrid,
    search_works_hybrid,
)


def initialize_workspace(config: dict[str, Any], *, collection_name: str | None = None) -> dict[str, Any]:
    ensure_db()
    collection_payload = runtime_collection_config(config, name=collection_name)
    with connect() as conn:
        collection_id = upsert_collection(conn, collection_payload)
        snapshot = _snapshot(conn)
    return {
        "workspace_root": str(paths.workspace_root()),
        "db_path": str(paths.DB_PATH),
        "collection": {
            "collection_id": collection_id,
            "name": collection_payload["name"],
            "label": collection_payload["label"],
        },
        "snapshot": snapshot,
    }


def fetch_online_candidates(
    config: dict[str, Any],
    *,
    query: str,
    limit: int,
) -> dict[str, Any]:
    online = config.get("online") or {}
    hits = search_literature(
        query,
        limit=limit,
        page_size=int(online.get("page_size") or 25),
        fields=list(online.get("fields") or []),
        published_only=bool(online.get("published_only", True)),
        query_suffix=str(online.get("query_suffix") or ""),
        timeout=int(online.get("timeout_sec") or 60),
        retries=int(online.get("retries") or 3),
        sleep_sec=float(online.get("sleep_sec") or 0.2),
    )
    return {
        "query": query,
        "effective_count": len(hits),
        "results": [summarize_hit(hit) for hit in hits],
    }


def ingest_online(
    config: dict[str, Any],
    *,
    query: str,
    limit: int,
    collection_name: str | None = None,
    download_limit: int | None = None,
    parse_limit: int | None = None,
    replace_existing: bool = False,
    skip_parse: bool = False,
    skip_index: bool = False,
    skip_graph: bool = False,
) -> dict[str, Any]:
    ensure_db()
    collection_payload = runtime_collection_config(config, name=collection_name)
    online = config.get("online") or {}
    download_cfg = config.get("download") or {}
    embedding_cfg = config.get("embedding") or {}

    hits = search_literature(
        query,
        limit=limit,
        page_size=int(online.get("page_size") or 25),
        fields=list(online.get("fields") or []),
        published_only=bool(online.get("published_only", True)),
        query_suffix=str(online.get("query_suffix") or ""),
        timeout=int(online.get("timeout_sec") or 60),
        retries=int(online.get("retries") or 3),
        sleep_sec=float(online.get("sleep_sec") or 0.2),
    )

    download_cap = max(0, download_limit if download_limit is not None else len(hits))
    parse_cap = 0 if skip_parse else max(0, parse_limit if parse_limit is not None else download_cap)

    summary: dict[str, Any] = {
        "query": query,
        "workspace_root": str(paths.workspace_root()),
        "collection": collection_payload["name"],
        "metadata": {
            "hits_seen": len(hits),
            "created": 0,
            "updated": 0,
            "citations_written": 0,
        },
        "downloads": {
            "attempted": 0,
            "ok": 0,
            "failed": 0,
            "items": [],
        },
        "mineru": {
            "enabled": bool((config.get("mineru") or {}).get("enabled")),
            "attempted": 0,
            "materialized": 0,
            "failed": 0,
            "items": [],
        },
        "search": None,
        "vectors": None,
        "graph": None,
    }

    mineru_client = _build_mineru_client(config)
    max_workers = max(1, int(download_cfg.get("max_download_workers") or 4))

    with connect() as conn:
        collection_id = upsert_collection(conn, collection_payload)
        run_id = _start_ingest_run(
            conn,
            collection_id=collection_id,
            source="online_search",
            query_json=json.dumps({"query": query, "limit": limit}, ensure_ascii=False),
            limit_requested=limit,
        )

        try:
            # Phase 1: Metadata upsert (sequential, in DB transaction)
            download_tasks: list[dict[str, Any]] = []
            for hit in hits:
                stats = upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                summary["metadata"]["created"] += int(stats["created"])
                summary["metadata"]["updated"] += int(stats["updated"])
                summary["metadata"]["citations_written"] += int(stats["citations_written"])

                work_id = _find_work_id_for_hit(conn, hit)
                if work_id is None:
                    continue

                if len(download_tasks) >= download_cap:
                    continue

                download_tasks.append(
                    _prepare_download(
                        conn,
                        hit=hit,
                        work_id=work_id,
                        collection_name=collection_payload["name"],
                    )
                )

            # Phase 2: PDF download (parallel with ThreadPoolExecutor, no DB)
            dl_timeout = int(download_cfg.get("timeout_sec") or 120)
            dl_retries = int(download_cfg.get("retries") or 3)
            dl_verify_ssl = bool(download_cfg.get("verify_ssl", True))

            download_results: list[dict[str, Any]] = []
            if download_tasks:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(download_tasks))) as pool:
                    futures = {
                        pool.submit(
                            _execute_download,
                            task=task,
                            timeout=dl_timeout,
                            retries=dl_retries,
                            verify_ssl=dl_verify_ssl,
                        ): task
                        for task in download_tasks
                    }
                    for future in as_completed(futures):
                        download_results.append(future.result())

            summary["downloads"]["attempted"] = len(download_results)
            for pdf_info in download_results:
                summary["downloads"]["items"].append(pdf_info)
                if pdf_info["ok"]:
                    summary["downloads"]["ok"] += 1
                else:
                    summary["downloads"]["failed"] += 1

            # Phase 3: MinerU parse (sequential, in DB transaction)
            parse_count = 0
            for pdf_info in download_results:
                if not pdf_info["ok"] or parse_count >= parse_cap or mineru_client is None:
                    continue

                parse_count += 1
                summary["mineru"]["attempted"] += 1
                try:
                    parsed = _parse_with_mineru(
                        conn,
                        config=config,
                        client=mineru_client,
                        work_id=pdf_info["work_id"],
                        pdf_path=Path(pdf_info["path"]),
                        collection_name=collection_payload["name"],
                        replace_existing=replace_existing,
                    )
                    summary["mineru"]["materialized"] += 1
                    summary["mineru"]["items"].append(parsed)
                except Exception as exc:
                    summary["mineru"]["failed"] += 1
                    summary["mineru"]["items"].append(
                        {
                            "ok": False,
                            "work_id": pdf_info["work_id"],
                            "title": _work_title(conn, pdf_info["work_id"]),
                            "error": str(exc),
                        }
                    )

            if not skip_index:
                summary["search"] = rebuild_search_indices(conn, target="all")
                if bool(embedding_cfg.get("build_after_ingest", True)):
                    try:
                        summary["vectors"] = rebuild_vector_indices(
                            conn,
                            target="all",
                            model=str(embedding_cfg.get("model") or DEFAULT_VECTOR_MODEL),
                            dim=int(embedding_cfg.get("dim") or 768),
                        )
                    except Exception as exc:
                        summary["vectors"] = {"ok": False, "error": str(exc)}
            if not skip_graph:
                try:
                    summary["graph"] = rebuild_graph_edges(
                        conn,
                        target="all",
                        collection=collection_payload["name"],
                        similarity_model=str(embedding_cfg.get("model") or DEFAULT_VECTOR_MODEL),
                    )
                except Exception as exc:
                    summary["graph"] = {"ok": False, "error": str(exc)}

            _finish_ingest_run(
                conn,
                run_id=run_id,
                status="finished",
                processed_hits=len(hits),
                works_created=int(summary["metadata"]["created"]),
                works_updated=int(summary["metadata"]["updated"]),
                citations_written=int(summary["metadata"]["citations_written"]),
            )
        except Exception:
            _finish_ingest_run(
                conn,
                run_id=run_id,
                status="failed",
                processed_hits=len(hits),
                works_created=int(summary["metadata"]["created"]),
                works_updated=int(summary["metadata"]["updated"]),
                citations_written=int(summary["metadata"]["citations_written"]),
            )
            raise

        summary["snapshot"] = _snapshot(conn)
    return summary


def retrieve(
    config: dict[str, Any],
    *,
    query: str,
    limit: int | None = None,
    target: str | None = None,
    collection_name: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    ensure_db()
    retrieval_cfg = config.get("retrieval") or {}
    collection = collection_name or str((config.get("collection") or {}).get("name") or "default")
    requested_target = str(target or retrieval_cfg.get("target") or "auto")
    embedding_model = str(model or (config.get("embedding") or {}).get("model") or DEFAULT_VECTOR_MODEL)
    limit_value = max(1, int(limit or retrieval_cfg.get("limit") or 8))
    chunk_limit = max(limit_value, int(retrieval_cfg.get("chunk_limit") or max(limit_value, 12)))

    routing = route_query(query) if requested_target == "auto" else {
        "target": requested_target,
        "graph_expand": 0,
        "reasons": ["manual_target"],
    }
    actual_target = str(routing["target"])
    graph_expand = retrieval_cfg.get("graph_expand")
    if graph_expand is None:
        graph_expand = routing.get("graph_expand") or 0

    with connect() as conn:
        if actual_target == "works":
            works = search_works_hybrid(
                conn,
                query=query,
                collection=collection,
                limit=limit_value,
                model=embedding_model,
                graph_expand=int(graph_expand),
                seed_limit=int(retrieval_cfg.get("seed_limit") or 5),
            )
            chunks = _supporting_chunks(
                conn,
                query=query,
                collection=collection,
                model=embedding_model,
                work_ids=[int(row["work_id"]) for row in works],
                limit=chunk_limit,
            )
        else:
            chunks = search_chunks_hybrid(
                conn,
                query=query,
                collection=collection,
                limit=chunk_limit,
                model=embedding_model,
            )
            works = _hydrate_works_from_chunks(conn, chunks=chunks, limit=limit_value)

    return {
        "query": query,
        "collection": collection,
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
    model: str | None = None,
) -> dict[str, Any]:
    llm_cfg = config.get("llm") or {}
    if not bool(llm_cfg.get("enabled")):
        raise ValueError("LLM is disabled in config. Set llm.enabled=true and configure either openai_compatible or local_transformers.")

    retrieval = retrieve(
        config,
        query=query,
        limit=limit,
        target=target,
        collection_name=collection_name,
        model=model,
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


def _build_mineru_client(config: dict[str, Any]) -> MinerUClient | None:
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


def _prepare_download(
    conn: sqlite3.Connection,
    *,
    hit: dict[str, Any],
    work_id: int,
    collection_name: str,
) -> dict[str, Any]:
    """Phase 1 helper: runs inside DB transaction, pre-computes download info."""
    stem = paper_storage_stem(conn, work_id)
    output_path = paths.PDF_DIR / collection_name / f"{stem}.pdf"
    candidates = list_pdf_candidates(
        hit,
        resolve_arxiv_from_doi=True,
        timeout=30,
        retries=3,
    )
    return {
        "work_id": work_id,
        "title": _work_title(conn, work_id),
        "output_path": str(output_path),
        "candidates": candidates,
        "content_addressed_name": content_addressed_name(hit),
    }


def _execute_download(
    *,
    task: dict[str, Any],
    timeout: int,
    retries: int,
    verify_ssl: bool,
) -> dict[str, Any]:
    """Phase 2 helper: pure I/O, no DB access. Thread-safe."""
    output_path = Path(task["output_path"])
    if output_path.exists():
        return {
            "ok": True,
            "work_id": task["work_id"],
            "title": task["title"],
            "path": str(output_path),
            "source": "cached",
        }

    result = download_pdf_candidates(
        task["candidates"],
        output_path=output_path,
        timeout=timeout,
        retries=retries,
        verify_ssl=verify_ssl,
    )
    result.update(
        {
            "work_id": task["work_id"],
            "title": task["title"],
            "candidate_name": task["content_addressed_name"],
        }
    )
    return result


def _download_hit_pdf(
    conn: sqlite3.Connection,
    *,
    hit: dict[str, Any],
    work_id: int,
    collection_name: str,
    timeout: int,
    retries: int,
    verify_ssl: bool,
) -> dict[str, Any]:
    stem = paper_storage_stem(conn, work_id)
    output_path = paths.PDF_DIR / collection_name / f"{stem}.pdf"
    if output_path.exists():
        return {
            "ok": True,
            "work_id": work_id,
            "title": _work_title(conn, work_id),
            "path": str(output_path),
            "source": "cached",
        }

    candidates = list_pdf_candidates(
        hit,
        resolve_arxiv_from_doi=True,
        timeout=max(10, min(timeout, 30)),
        retries=max(1, min(retries, 3)),
    )
    result = download_pdf_candidates(
        candidates,
        output_path=output_path,
        timeout=timeout,
        retries=retries,
        verify_ssl=verify_ssl,
    )
    result.update(
        {
            "work_id": work_id,
            "title": _work_title(conn, work_id),
            "candidate_name": content_addressed_name(hit),
        }
    )
    return result


def _parse_with_mineru(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    client: MinerUClient,
    work_id: int,
    pdf_path: Path,
    collection_name: str,
    replace_existing: bool,
) -> dict[str, Any]:
    stem = paper_storage_stem(conn, work_id)
    dest_dir = parsed_doc_dir(collection_name, stem)
    if (dest_dir / "manifest.json").exists() and not replace_existing:
        return {
            "ok": True,
            "work_id": work_id,
            "title": _work_title(conn, work_id),
            "state": "cached",
            "parsed_dir": str(dest_dir),
        }

    raw_zip_dir = paths.RAW_DIR / "mineru" / collection_name
    raw_zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_zip_dir / f"{stem}.zip"
    task = client.submit_local_pdf(pdf_path, data_id=stem)
    client.download_result_zip(task, output_path=zip_path)
    import_mineru_source(source_path=zip_path, dest_dir=dest_dir, replace=replace_existing)

    ingest_cfg = config.get("ingest") or {}
    summary = materialize_mineru_document(
        conn,
        work_id=work_id,
        manifest_path=dest_dir / "manifest.json",
        replace=replace_existing,
        chunk_size=int(ingest_cfg.get("chunk_size") or 2400),
        overlap_blocks=int(ingest_cfg.get("overlap_blocks") or 1),
        section_parent_char_limit=int(ingest_cfg.get("section_parent_char_limit") or 12000),
    )
    return {
        "ok": True,
        "work_id": work_id,
        "title": _work_title(conn, work_id),
        "state": task.state,
        "parsed_dir": str(dest_dir),
        "zip_path": str(zip_path),
        "document": summary,
    }


def _supporting_chunks(
    conn: sqlite3.Connection,
    *,
    query: str,
    collection: str,
    model: str,
    work_ids: list[int],
    limit: int,
) -> list[dict[str, Any]]:
    raw = search_chunks_hybrid(
        conn,
        query=query,
        collection=collection,
        limit=max(limit * 3, 30),
        model=model,
    )
    if not work_ids:
        return raw[:limit]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    work_id_set = {int(item) for item in work_ids}
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
    conn: sqlite3.Connection,
    *,
    chunks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ids: list[int] = []
    for row in chunks:
        work_id = int(row["work_id"])
        if work_id not in ids:
            ids.append(work_id)
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
        work_lines.append(
            f"[W{idx}] {item.get('raw_title')} ({item.get('year')}) "
            f"{item.get('canonical_source')}:{item.get('canonical_id')}"
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


def _find_work_id_for_hit(conn: sqlite3.Connection, hit: dict[str, Any]) -> int | None:
    metadata = hit.get("metadata") or {}
    source, external_id = canonical_identity(metadata)
    row = conn.execute(
        """
        SELECT work_id
        FROM works
        WHERE canonical_source = ? AND canonical_id = ?
        """,
        (source, external_id),
    ).fetchone()
    if row is None:
        return None
    return int(row["work_id"])


def _work_title(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute("SELECT title FROM works WHERE work_id = ?", (work_id,)).fetchone()
    return str(row["title"]) if row is not None else None


def _snapshot(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM collections) AS collections,
          (SELECT COUNT(*) FROM works) AS works,
          (SELECT COUNT(*) FROM documents) AS documents,
          (SELECT COUNT(*) FROM chunks) AS chunks,
          (SELECT COUNT(*) FROM citations) AS citations,
          (SELECT COUNT(*) FROM similarity_edges) AS similarity_edges
        """
    ).fetchone()
    return {key: int(row[key]) for key in row.keys()} if row is not None else {}


def _start_ingest_run(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    source: str,
    query_json: str,
    limit_requested: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingest_runs (
          collection_id, source, status, query_json, page_size, limit_requested, raw_dir
        ) VALUES (?, ?, 'running', ?, NULL, ?, ?)
        """,
        (collection_id, source, query_json, limit_requested, str(paths.RAW_DIR)),
    )
    return int(cur.lastrowid)


def _finish_ingest_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    status: str,
    processed_hits: int,
    works_created: int,
    works_updated: int,
    citations_written: int,
) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET status = ?,
            processed_hits = ?,
            works_created = ?,
            works_updated = ?,
            citations_written = ?,
            finished_at = CURRENT_TIMESTAMP
        WHERE run_id = ?
        """,
        (status, processed_hits, works_created, works_updated, citations_written, run_id),
    )
