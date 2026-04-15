from __future__ import annotations

import copy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from hep_rag_v2.metadata import (
    first_arxiv_id,
    first_doi,
    first_title,
    year_from_metadata,
    _document_stage_rank,
    _has_publication_signal,
    _has_distinct_publication_identities,
)
from hep_rag_v2.providers.inspire import list_pdf_candidates, search_literature, summarize_hit

ProgressCallback = Callable[[str], None] | None

ONLINE_REQUIRED_FIELDS = (
    "control_number",
    "titles",
    "abstracts",
    "collaborations",
    "document_type",
    "earliest_date",
    "publication_info",
    "references",
)
INSPIRE_QUERY_FIELD_PATTERN = re.compile(
    r"\b(?:collaboration|collection|collections|title|abstract|keyword|keywords|recid|doi|arxiv|author|document_type|date|earliest_date)\s*:",
    re.IGNORECASE,
)
QUERY_REWRITE_SYSTEM_PROMPT = (
    "You rewrite literature-search queries for academic databases such as INSPIRE-HEP. "
    "Return ONLY valid JSON."
)
HEP_COLLABORATIONS = (
    "CMS",
    "ATLAS",
    "ALICE",
    "LHCB",
    "TOTEM",
    "CDF",
    "D0",
    "DZERO",
    "BELLE",
    "BABAR",
)
HEP_ENERGY_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*TEV\b", re.IGNORECASE)
TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TITLE_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
}


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is not None:
        progress(message)


def _truncate_progress_text(value: Any, *, limit: int = 88) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


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


def _search_online_hits(
    config: dict[str, Any],
    *,
    query: str,
    limit: int,
    max_parallelism: int | None = None,
    progress: ProgressCallback = None,
    build_llm_client: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    online = config.get("online") or {}
    page_size = int(online.get("page_size") or 25)
    fields = _ensure_online_fields(list(online.get("fields") or []))
    published_only = bool(online.get("published_only", False))
    query_suffix = str(online.get("query_suffix") or "")
    timeout = int(online.get("timeout_sec") or 60)
    retries = int(online.get("retries") or 3)
    sleep_sec = float(online.get("sleep_sec") or 0.2)

    search_plan = _plan_online_queries(
        config,
        query=query,
        progress=progress,
        build_llm_client=build_llm_client,
    )
    query_cfg = config.get("query_rewrite") or {}
    per_query_limit = max(
        max(1, limit),
        int(query_cfg.get("per_query_limit") or max(limit * 3, 10)),
    )
    search_workers = _resolve_parallelism(
        requested=max_parallelism,
        configured=online.get("max_parallelism"),
        fallback=4,
        ceiling=max(1, len(search_plan["queries"])),
    )
    search_plan["max_parallelism"] = search_workers
    ranked_batches: list[tuple[str, list[dict[str, Any]]]] = [("", []) for _ in search_plan["queries"]]

    if len(search_plan["queries"]) == 1 or search_workers <= 1:
        for idx, search_query in enumerate(search_plan["queries"], start=1):
            query_label = f"INSPIRE query {idx}/{len(search_plan['queries'])}"
            query_preview = _truncate_progress_text(search_query)
            _emit_progress(
                progress,
                f'searching {query_label} limit={per_query_limit} page_size={page_size} q="{query_preview}"...',
            )
            hits = search_literature(
                search_query,
                limit=per_query_limit,
                page_size=page_size,
                fields=fields,
                published_only=published_only,
                query_suffix=query_suffix,
                timeout=timeout,
                retries=retries,
                sleep_sec=sleep_sec,
                progress=progress,
                progress_label=query_label,
            )
            ranked_batches[idx - 1] = (search_query, hits)
            _emit_progress(
                progress,
                f"finished {query_label} results={len(hits)}",
            )
    else:
        _emit_progress(
            progress,
            f"searching INSPIRE with up to {search_workers} parallel requests across {len(search_plan['queries'])} queries...",
        )
        with ThreadPoolExecutor(max_workers=search_workers) as pool:
            futures: dict[Any, tuple[int, str]] = {}
            for idx, search_query in enumerate(search_plan["queries"], start=1):
                query_label = f"INSPIRE query {idx}/{len(search_plan['queries'])}"
                query_preview = _truncate_progress_text(search_query)
                _emit_progress(
                    progress,
                    f'searching {query_label} limit={per_query_limit} page_size={page_size} q="{query_preview}"...',
                )
                future = pool.submit(
                    search_literature,
                    search_query,
                    limit=per_query_limit,
                    page_size=page_size,
                    fields=fields,
                    published_only=published_only,
                    query_suffix=query_suffix,
                    timeout=timeout,
                    retries=retries,
                    sleep_sec=sleep_sec,
                    progress=progress,
                    progress_label=query_label,
                )
                futures[future] = (idx, search_query)
            for future in as_completed(futures):
                idx, search_query = futures[future]
                hits = future.result()
                ranked_batches[idx - 1] = (search_query, hits)
                _emit_progress(
                    progress,
                    f"finished INSPIRE query {idx}/{len(search_plan['queries'])} results={len(hits)}",
                )

    merged_hits, merge_stats = _merge_ranked_hits(ranked_batches, limit=max(1, limit))
    search_plan["published_only"] = published_only
    search_plan.update(merge_stats)
    return (merged_hits, search_plan)


def _plan_online_queries(
    config: dict[str, Any],
    *,
    query: str,
    progress: ProgressCallback = None,
    build_llm_client: Any = None,
) -> dict[str, Any]:
    plan = {
        "original_query": query,
        "queries": [query],
        "seed_queries": [],
        "seed_used": False,
        "rewrite_used": False,
        "rewrite_backend": None,
        "rewrite_model": None,
    }
    rewrite_cfg = config.get("query_rewrite") or {}
    max_queries = max(2, int(rewrite_cfg.get("max_queries") or 4))

    seen = {query.casefold()}
    for item in _seed_online_queries(query):
        value = str(item or "").strip()
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        plan["queries"].append(value)
        plan["seed_queries"].append(value)
        if len(plan["queries"]) >= max_queries:
            plan["seed_used"] = True
            return plan
    if plan["seed_queries"]:
        plan["seed_used"] = True

    if not _should_rewrite_query(config, query=query):
        return plan

    llm_cfg = config.get("llm") or {}
    remaining = max_queries - len(plan["queries"])
    if remaining <= 0:
        return plan
    _emit_progress(progress, "rewriting search query...")
    try:
        if build_llm_client is None:
            from hep_rag_v2.rag import _build_llm_client
            build_llm_client = _build_llm_client
        client = build_llm_client(llm_cfg)
        response = client.chat(
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _query_rewrite_prompt(query, max_queries=remaining),
                },
            ],
            temperature=float(rewrite_cfg.get("temperature") or 0.0),
            max_tokens=int(rewrite_cfg.get("max_tokens") or 320),
        )
        variants = _parse_query_rewrite_output(response["content"])
    except Exception:
        _emit_progress(progress, "query rewrite unavailable; using the original query.")
        return plan

    for item in variants:
        value = str(item or "").strip()
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        plan["queries"].append(value)
        if len(plan["queries"]) >= max_queries:
            break
    if len(plan["queries"]) == 1 + len(plan["seed_queries"]):
        if plan["seed_queries"]:
            _emit_progress(progress, "query rewrite produced no useful alternatives; continuing with deterministic search variants.")
        else:
            _emit_progress(progress, "query rewrite produced no useful alternatives; using the original query.")
        return plan
    plan["rewrite_used"] = True
    plan["rewrite_backend"] = str(llm_cfg.get("backend") or "openai_compatible")
    plan["rewrite_model"] = response["model"]
    return plan


def _should_rewrite_query(config: dict[str, Any], *, query: str) -> bool:
    value = str(query or "").strip()
    if not value:
        return False
    rewrite_cfg = config.get("query_rewrite") or {}
    if not bool(rewrite_cfg.get("enabled", True)):
        return False
    if int(rewrite_cfg.get("max_queries") or 4) <= 1:
        return False
    llm_cfg = config.get("llm") or {}
    if not bool(llm_cfg.get("enabled")):
        return False
    if INSPIRE_QUERY_FIELD_PATTERN.search(value):
        return False
    return True


def _seed_online_queries(query: str) -> list[str]:
    value = str(query or "").strip()
    if not value or INSPIRE_QUERY_FIELD_PATTERN.search(value):
        return []

    collaboration = _detect_hep_collaboration(value)
    channel_terms = _detect_hep_channel_terms(value)
    topology_terms = _detect_hep_topology_terms(value)
    energy_terms = _detect_hep_energy_terms(value)

    clauses: list[str] = []
    if collaboration:
        clauses.append(f"collaboration:{collaboration}")
    if channel_terms:
        clauses.append(_inspire_field_clause(channel_terms))
    if topology_terms:
        clauses.append(_inspire_field_clause(topology_terms))
    if energy_terms and (channel_terms or topology_terms):
        clauses.append(_inspire_field_clause(energy_terms))

    if len(clauses) < 2:
        return []
    return [" and ".join(clauses)]


def _detect_hep_collaboration(query: str) -> str | None:
    upper = str(query or "").upper()
    for name in HEP_COLLABORATIONS:
        if re.search(rf"\b{re.escape(name)}\b", upper):
            return "D0" if name in {"D0", "DZERO"} else name
    return None


def _detect_hep_channel_terms(query: str) -> list[str]:
    upper = str(query or "").upper()
    terms: list[str] = []

    if re.search(r"\bSS\s*WW\b|\bSSWW\b", upper):
        terms.extend(["same-sign WW", "same-sign W boson", "same-sign W boson pairs"])
    if re.search(r"\bOS\s*WW\b|\bOSWW\b", upper):
        terms.extend(["opposite-sign WW", "W+W-", "WW"])
    if re.search(r"\bWZ(?:JJ)?\b", upper):
        terms.extend(["WZ", "WZ boson pairs"])
    if re.search(r"\bWW(?:JJ)?\b", upper) and not re.search(r"\bSS\s*WW\b|\bSSWW\b|\bOS\s*WW\b|\bOSWW\b", upper):
        terms.extend(["WW", "W boson pairs"])
    if re.search(r"\bZZ(?:JJ)?\b", upper):
        terms.extend(["ZZ", "Z boson pairs"])
    if re.search(r"\bVVJJ\b|\bVV\s*JJ\b", upper):
        terms.extend(["diboson", "vector boson pairs"])

    return _dedupe_preserve_order(terms)


def _detect_hep_topology_terms(query: str) -> list[str]:
    upper = str(query or "").upper()
    has_two_jets = bool(
        re.search(r"\b(?:2J|2JETS?|TWO JETS?|DIJET|JETS?)\b", upper)
        or re.search(r"\b[A-Z0-9+/.-]*JJ\b", upper)
    )
    terms: list[str] = []

    if re.search(r"\bVBS\b", upper):
        terms.extend(["vector boson scattering", "electroweak production"])
        has_two_jets = True
    if re.search(r"\bVBF\b", upper):
        terms.append("vector boson fusion")
        has_two_jets = True
    if re.search(r"\b(?:EWK|EW)\b", upper):
        terms.extend(["electroweak production", "electroweak"])
    if has_two_jets:
        terms.extend(["in association with two jets", "two jets", "dijet"])

    return _dedupe_preserve_order(terms)


def _detect_hep_energy_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in HEP_ENERGY_PATTERN.finditer(str(query or "")):
        value = f"{match.group(1)} TeV"
        terms.append(value)
        terms.append(f"sqrt(s)={value}")
    return _dedupe_preserve_order(terms)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _inspire_field_clause(terms: list[str]) -> str:
    atoms: list[str] = []
    for term in _dedupe_preserve_order(terms):
        quoted = _quote_inspire_term(term)
        atoms.append(f'title:{quoted}')
        atoms.append(f'abstract:{quoted}')
    return "(" + " or ".join(atoms) + ")"


def _quote_inspire_term(term: str) -> str:
    return '"' + str(term).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _query_rewrite_prompt(query: str, *, max_queries: int) -> str:
    return (
        "Rewrite the following literature-search query into concise alternatives for INSPIRE-HEP.\n"
        "Guidelines:\n"
        "- Preserve concrete entities such as experiment names, collaborations, energies, and channels.\n"
        "- Expand abbreviations when helpful.\n"
        "- Produce alternatives that a paper title or abstract is likely to contain.\n"
        "- Make the alternatives diverse:\n"
        "  1. one long-form natural-language expansion,\n"
        "  2. one paper-title-style phrasing,\n"
        "  3. one INSPIRE-friendly fielded query using collaboration/title/abstract when obvious,\n"
        "  4. one alternate notation query using particle names, charges, symbols, or final-state/topology wording when relevant.\n"
        "- For collider-physics queries, prefer title-like phrases such as 'same-sign W boson pairs', "
        "'vector boson scattering', and 'in association with two jets' when they are implied by the input.\n"
        "- If a vector-boson-scattering query is underspecified, include standard companion channel wording that often appears in paper titles, "
        "such as WW/WZ when appropriate.\n"
        "- Do not add explanations.\n"
        f"- Return a JSON array with up to {max_queries} strings.\n\n"
        f"Query: {query}"
    )


def _parse_query_rewrite_output(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in (fenced, _extract_json_array(fenced)):
        items = _coerce_query_list(candidate)
        if items:
            return items

    queries: list[str] = []
    for line in fenced.splitlines():
        value = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"').strip("'")
        if value:
            queries.append(value)
    return queries


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_query_list(text: str | None) -> list[str]:
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("queries")
    if not isinstance(payload, list):
        return []
    out: list[str] = []
    for item in payload:
        value = str(item or "").strip()
        if value:
            out.append(value)
    return out


def _merge_ranked_hits(
    ranked_batches: list[tuple[str, list[dict[str, Any]]]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not ranked_batches:
        return [], {
            "merged_hit_count": 0,
            "deduped_hit_count": 0,
            "dedupe_removed": 0,
        }

    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    hit_map: dict[str, dict[str, Any]] = {}
    for batch_idx, (_, hits) in enumerate(ranked_batches):
        for rank, hit in enumerate(hits, start=1):
            key = _online_hit_key(hit)
            hit_map[key] = hit
            scores[key] = scores.get(key, 0.0) + 1.0 / float(60 + rank) + _document_type_score_bonus(hit)
            if key not in first_seen:
                first_seen[key] = (batch_idx, rank)

    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], first_seen[key][0], first_seen[key][1]),
    )
    ordered_hits = [hit_map[key] for key in ordered]
    family_hits = _collapse_online_hit_families(ordered_hits)
    return family_hits[:limit], {
        "merged_hit_count": len(ordered_hits),
        "deduped_hit_count": len(family_hits),
        "dedupe_removed": max(0, len(ordered_hits) - len(family_hits)),
    }


def _online_hit_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    control_number = metadata.get("control_number")
    if control_number is not None:
        return f"cn:{control_number}"
    self_link = str((hit.get("links") or {}).get("self") or "").strip()
    if self_link:
        return f"self:{self_link}"
    title = str(((metadata.get("titles") or [{}])[0].get("title") or "")).strip()
    year = str(((metadata.get("publication_info") or [{}])[0].get("year") or metadata.get("earliest_date") or "")).strip()
    digest = hashlib.sha1(f"{title}|{year}".encode("utf-8")).hexdigest()
    return f"fallback:{digest}"


def _ensure_online_fields(fields: list[str]) -> list[str]:
    seen = {str(item).strip() for item in fields if str(item).strip()}
    out = [str(item).strip() for item in fields if str(item).strip()]
    for field in ONLINE_REQUIRED_FIELDS:
        if field not in seen:
            seen.add(field)
            out.append(field)
    return out


def _document_type_score_bonus(hit: dict[str, Any]) -> float:
    metadata = hit.get("metadata") or {}
    doc_types = {str(item).casefold() for item in (metadata.get("document_type") or [])}
    if "article" in doc_types:
        return 0.003
    if "review" in doc_types:
        return 0.0015
    if "note" in doc_types:
        return -0.0015
    if "conference paper" in doc_types or "proceedings article" in doc_types:
        return -0.001
    return 0.0


def _collapse_online_hit_families(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: list[list[dict[str, Any]]] = []
    for hit in hits:
        matched_family: list[dict[str, Any]] | None = None
        for family in families:
            if any(_hits_are_semantic_duplicates(existing, hit) for existing in family):
                matched_family = family
                break
        if matched_family is None:
            families.append([hit])
            continue
        matched_family.append(hit)

    collapsed: list[dict[str, Any]] = []
    for family in families:
        primary = max(family, key=_hit_richness_key)
        payload = copy.deepcopy(primary)
        payload["_family_members"] = [
            copy.deepcopy(item)
            for item in family
            if _online_hit_key(item) != _online_hit_key(primary)
        ]
        payload["_family_size"] = len(family)
        collapsed.append(payload)
    return collapsed


def _hits_are_semantic_duplicates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_md = left.get("metadata") or {}
    right_md = right.get("metadata") or {}

    left_arxiv = str(first_arxiv_id(left_md) or "").strip().casefold()
    right_arxiv = str(first_arxiv_id(right_md) or "").strip().casefold()
    if left_arxiv and right_arxiv and left_arxiv == right_arxiv:
        return True
    if left_arxiv and right_arxiv and left_arxiv != right_arxiv:
        return False

    left_doi = str(first_doi(left_md) or "").strip().casefold()
    right_doi = str(first_doi(right_md) or "").strip().casefold()
    if left_doi and right_doi and left_doi == right_doi:
        return True
    if left_doi and right_doi and left_doi != right_doi:
        return False

    left_year = year_from_metadata(left_md)
    right_year = year_from_metadata(right_md)
    if left_year is not None and right_year is not None and abs(left_year - right_year) > 1:
        return False

    if not _titles_look_equivalent(first_title(left_md), first_title(right_md)):
        return False
    if _shared_report_roots(left_md, right_md):
        return True
    if not (_hit_collaborations(left_md) & _hit_collaborations(right_md)):
        return False
    return _looks_like_stage_variant_duplicate(left_md, right_md)


def _hit_richness_key(hit: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int]:
    metadata = hit.get("metadata") or {}
    return (
        _document_type_preference(metadata),
        1 if list_pdf_candidates(hit) else 0,
        1 if metadata.get("publication_info") else 0,
        1 if first_arxiv_id(metadata) else 0,
        1 if first_doi(metadata) else 0,
        len(metadata.get("documents") or []),
        _coerce_int_like(metadata.get("citation_count")) or 0,
        len(first_title(metadata) or ""),
    )


def _document_type_preference(metadata: dict[str, Any]) -> int:
    doc_types = {str(item).casefold() for item in (metadata.get("document_type") or [])}
    if "article" in doc_types:
        return 5
    if "review" in doc_types:
        return 4
    if "note" in doc_types:
        return 3
    if "conference paper" in doc_types or "proceedings article" in doc_types:
        return 2
    return 1


def _hit_collaborations(metadata: dict[str, Any]) -> set[str]:
    collabs: set[str] = set()
    for item in metadata.get("collaborations") or []:
        if isinstance(item, dict) and item.get("value"):
            collabs.add(str(item["value"]).strip().casefold())
        elif item:
            collabs.add(str(item).strip().casefold())
    return collabs


def _shared_report_roots(left_md: dict[str, Any], right_md: dict[str, Any]) -> set[str]:
    return _hit_report_roots(left_md) & _hit_report_roots(right_md)


def _hit_report_roots(metadata: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    for item in metadata.get("report_numbers") or []:
        value = item.get("value") if isinstance(item, dict) else item
        normalized = _normalize_report_root(value)
        if normalized:
            roots.add(normalized)
    return roots


def _normalize_report_root(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    tokens = [token for token in re.findall(r"[a-z0-9]+", text) if token]
    if not tokens:
        return None
    tokens = [token for token in tokens if token not in {"pas", "prelim", "preliminary", "public", "internal", "note", "notes"}] or tokens
    return "-".join(tokens)


def _looks_like_stage_variant_duplicate(left_md: dict[str, Any], right_md: dict[str, Any]) -> bool:
    left_stage = _document_stage_rank(left_md)
    right_stage = _document_stage_rank(right_md)

    if left_stage == right_stage:
        return False
    if max(left_stage, right_stage) < 3:
        return False

    higher = left_md if left_stage > right_stage else right_md
    lower = right_md if higher is left_md else left_md

    if _has_distinct_publication_identities(left_md, right_md):
        return False
    if not _has_publication_signal(higher):
        return False
    if _has_publication_signal(lower) and _document_stage_rank(lower) >= 4:
        return False
    return True


def _titles_look_equivalent(left: str | None, right: str | None) -> bool:
    left_norm = _normalize_title_text(left)
    right_norm = _normalize_title_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = _title_token_set(left_norm)
    right_tokens = _title_token_set(right_norm)
    if not left_tokens or not right_tokens:
        return False

    shared = len(left_tokens & right_tokens)
    if shared < 6:
        return False
    if left_tokens == right_tokens:
        return True
    return shared / float(len(left_tokens | right_tokens)) >= 0.92


def _normalize_title_text(title: str | None) -> str:
    value = str(title or "").casefold()
    if not value:
        return ""
    value = value.replace("same-sign", "same sign")
    value = value.replace("proton-proton", "proton proton")
    value = value.replace("w±w±", "ww")
    value = re.sub(r"\\[a-z]+", " ", value)
    value = value.replace("±", " ")
    value = re.sub(r"\$+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _title_token_set(normalized_title: str) -> set[str]:
    return {
        token
        for token in TITLE_TOKEN_PATTERN.findall(str(normalized_title or ""))
        if token and token not in TITLE_TOKEN_STOPWORDS
    }


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, str, tuple, set)):
        return bool(value)
    return True


def _coerce_int_like(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
