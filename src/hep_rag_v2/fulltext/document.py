from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hep_rag_v2.textnorm import normalize_display_text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INLINE_CITATION_PATTERNS = [
    re.compile(r"\(\s*(?:see\s+)?(?:e\.g\.\s*,?\s*)?(?:Refs?\.?|references?)\s*\[[0-9,\s;–-]+\]\s*\)", re.IGNORECASE),
    re.compile(r"(?:Refs?\.?|references?)\s*\[[0-9,\s;–-]+\]", re.IGNORECASE),
    re.compile(r"\[[0-9,\s;–-]+\]"),
]
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_PATTERN = re.compile(
    r"\b(?:arxiv\s*:?\s*)?((?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)\b",
    re.IGNORECASE,
)
BRACKETED_REFERENCE_PATTERN = re.compile(r"(?=(?:^|\s)\[\d+\]\s*)")
NUMBERED_REFERENCE_PATTERN = re.compile(r"(?=(?:^|\s)\d+\.\s+)")

ABSTRACT_HEADINGS = {"abstract"}
BIBLIOGRAPHY_HEADINGS = {"references", "reference", "bibliography"}
BACK_MATTER_HEADINGS = {"acknowledgments", "acknowledgements"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParsedBlock:
    index: int
    block_type: str
    raw_text: str
    page: int | None
    order: int
    text_level: int | None
    latex: str | None
    caption: str | None
    asset_path: str | None
    raw: dict[str, Any]


@dataclass
class AnnotatedBlock:
    parsed: ParsedBlock
    block_role: str
    is_heading: bool
    heading_level: int | None
    heading_ordinal: str | None
    heading_clean_title: str | None
    section_kind: str
    clean_text: str
    is_retrievable: bool
    exclusion_reason: str | None
    flags: dict[str, Any]


@dataclass
class SectionFrame:
    section_id: int
    level: int
    title: str
    path_text: str
    section_kind: str


# ---------------------------------------------------------------------------
# materialize_mineru_document
# ---------------------------------------------------------------------------


def materialize_mineru_document(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    manifest_path: Path,
    replace: bool = False,
    chunk_size: int = 2400,
    overlap_blocks: int = 1,
    section_parent_char_limit: int = 12000,
) -> dict[str, Any]:
    from .chunks import build_chunks
    from .parser import load_content_list, load_manifest, parsed_blocks_from_content_list

    work_row = conn.execute(
        "SELECT title FROM works WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if work_row is None:
        raise ValueError(f"Unknown work_id: {work_id}")

    manifest = load_manifest(manifest_path)
    content_list_path = Path(str(manifest["content_list_path"]))
    items = load_content_list(content_list_path)
    parsed_blocks = parsed_blocks_from_content_list(items)
    annotated_blocks = annotate_blocks(parsed_blocks, work_title=str(work_row["title"]))

    existing = conn.execute(
        "SELECT document_id FROM documents WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if existing is not None:
        if not replace:
            raise FileExistsError(f"Document already exists for work_id={work_id}. Pass replace=True to rebuild.")
        conn.execute("DELETE FROM documents WHERE work_id = ?", (work_id,))

    cur = conn.execute(
        """
        INSERT INTO documents (
          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            work_id,
            "mineru",
            "v2-contract",
            "materialized",
            str(manifest_path.parent),
            str(manifest_path),
        ),
    )
    document_id = int(cur.lastrowid)

    title = normalize_display_text(str(work_row["title"]).strip()) or str(work_row["title"]).strip()
    root_section_id = _insert_section(
        conn,
        document_id=document_id,
        parent_section_id=None,
        ordinal=None,
        title=title,
        clean_title=title,
        path_text=title,
        section_kind="document",
        level=0,
        order_index=0,
    )
    section_stack = [
        SectionFrame(
            section_id=root_section_id,
            level=0,
            title=title,
            path_text=title,
            section_kind="document",
        )
    ]
    section_page_ranges: dict[int, list[int | None]] = {root_section_id: [None, None]}
    inserted_blocks: list[dict[str, Any]] = []
    section_order = 1

    for annotated in annotated_blocks:
        if annotated.is_heading:
            heading_level = min(max(annotated.heading_level or 1, 1), 6)
            while len(section_stack) > 1 and section_stack[-1].level >= heading_level:
                section_stack.pop()
            parent = section_stack[-1]
            clean_title = normalize_display_text(
                annotated.heading_clean_title or _single_line(annotated.clean_text or annotated.parsed.raw_text)
            )
            path_text = f"{parent.path_text} / {clean_title}"
            section_id = _insert_section(
                conn,
                document_id=document_id,
                parent_section_id=parent.section_id,
                ordinal=annotated.heading_ordinal,
                title=normalize_display_text(_single_line(annotated.parsed.raw_text)),
                clean_title=clean_title,
                path_text=path_text,
                section_kind=annotated.section_kind,
                level=heading_level,
                order_index=section_order,
            )
            section_order += 1
            section_stack.append(
                SectionFrame(
                    section_id=section_id,
                    level=heading_level,
                    title=clean_title,
                    path_text=path_text,
                    section_kind=annotated.section_kind,
                )
            )
            section_page_ranges[section_id] = [None, None]

        current_section = section_stack[-1]
        current_page = annotated.parsed.page
        if current_page is not None:
            for frame in section_stack:
                _bump_page_range(section_page_ranges, frame.section_id, current_page)

        raw_text = annotated.parsed.raw_text.strip()
        clean_text = annotated.clean_text.strip()
        block_cur = conn.execute(
            """
            INSERT INTO blocks (
              document_id, section_id, block_type, page, order_index, text, raw_text, clean_text,
              text_level, block_role, is_heading, is_retrievable, exclusion_reason,
              latex, caption, asset_path, flags_json, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                current_section.section_id,
                annotated.parsed.block_type,
                current_page,
                annotated.parsed.order,
                clean_text or raw_text,
                raw_text or None,
                clean_text or None,
                annotated.parsed.text_level,
                annotated.block_role,
                int(annotated.is_heading),
                int(annotated.is_retrievable),
                annotated.exclusion_reason,
                annotated.parsed.latex,
                annotated.parsed.caption,
                annotated.parsed.asset_path,
                json.dumps(annotated.flags, ensure_ascii=False),
                json.dumps(annotated.parsed.raw, ensure_ascii=False),
            ),
        )
        block_id = int(block_cur.lastrowid)
        inserted_block = {
            "block_id": block_id,
            "section_id": current_section.section_id,
            "section_kind": current_section.section_kind,
            "section_hint": current_section.title,
            "section_path": current_section.path_text,
            "block_type": annotated.parsed.block_type,
            "page": current_page,
            "order_index": annotated.parsed.order,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "block_role": annotated.block_role,
            "is_heading": annotated.is_heading,
            "is_retrievable": annotated.is_retrievable,
            "flags": annotated.flags,
        }
        inserted_blocks.append(inserted_block)

        if annotated.parsed.block_type == "equation" and annotated.parsed.latex:
            conn.execute(
                """
                INSERT INTO formulas (
                  document_id, section_id, block_id, page, order_index, latex, normalized_latex
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    current_section.section_id,
                    block_id,
                    current_page,
                    annotated.parsed.order,
                    annotated.parsed.latex,
                    _normalize_latex(annotated.parsed.latex),
                ),
            )
        if annotated.parsed.block_type in {"table", "image"}:
            conn.execute(
                """
                INSERT INTO assets (
                  document_id, section_id, block_id, asset_type, page, caption, asset_path, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    current_section.section_id,
                    block_id,
                    annotated.parsed.block_type,
                    current_page,
                    annotated.parsed.caption,
                    annotated.parsed.asset_path,
                    json.dumps(annotated.parsed.raw, ensure_ascii=False),
                ),
            )

    for section_id, page_range in section_page_ranges.items():
        conn.execute(
            """
            UPDATE document_sections
            SET page_start = ?, page_end = ?
            WHERE section_id = ?
            """,
            (page_range[0], page_range[1], section_id),
        )

    chunk_rows = build_chunks(
        inserted_blocks,
        chunk_size=chunk_size,
        overlap_blocks=overlap_blocks,
        section_parent_char_limit=section_parent_char_limit,
    )
    for row in chunk_rows:
        clean_text = str(row["clean_text"]).strip()
        raw_text = str(row["raw_text"]).strip()
        payload = (
            work_id,
            document_id,
            row["section_id"],
            row["block_start_id"],
            row["block_end_id"],
            row["chunk_role"],
            row["page_hint"],
            row["section_hint"],
            clean_text or raw_text,
            raw_text or None,
            clean_text or None,
            _hash_text(clean_text or raw_text),
            int(row["is_retrievable"]),
            row["exclusion_reason"],
            json.dumps(row["source_block_ids"], ensure_ascii=False),
            json.dumps(row["flags"], ensure_ascii=False),
        )
        conn.execute(
            """
            INSERT INTO chunks (
              work_id, document_id, section_id, block_start_id, block_end_id, chunk_role,
              page_hint, section_hint, text, raw_text, clean_text, text_hash, is_retrievable,
              exclusion_reason, source_block_ids_json, flags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

    role_counts = dict(
        conn.execute(
            """
            SELECT block_role, COUNT(*) AS n
            FROM blocks
            WHERE document_id = ?
            GROUP BY block_role
            ORDER BY block_role
            """,
            (document_id,),
        ).fetchall()
    )
    chunk_counts = dict(
        conn.execute(
            """
            SELECT chunk_role, COUNT(*) AS n
            FROM chunks
            WHERE document_id = ?
            GROUP BY chunk_role
            ORDER BY chunk_role
            """,
            (document_id,),
        ).fetchall()
    )
    retrievable_blocks = sum(1 for block in inserted_blocks if block["is_retrievable"])
    citation_summary = _rewrite_mineru_bibliography_citations(
        conn,
        work_id=work_id,
        document_id=document_id,
        blocks=inserted_blocks,
    )
    return {
        "document_id": document_id,
        "work_id": work_id,
        "parsed_dir": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "blocks": len(inserted_blocks),
        "retrievable_blocks": retrievable_blocks,
        "chunks": len(chunk_rows),
        "block_roles": role_counts,
        "chunk_roles": chunk_counts,
        "citations_written": int(citation_summary["citations_written"]),
        "citations_resolved": int(citation_summary["citations_resolved"]),
        "bibliography_entries": int(citation_summary["bibliography_entries"]),
    }


# ---------------------------------------------------------------------------
# annotate_blocks
# ---------------------------------------------------------------------------


def annotate_blocks(blocks: list[ParsedBlock], *, work_title: str) -> list[AnnotatedBlock]:
    annotated: list[AnnotatedBlock] = []
    state = "front_matter"
    title_taken = False

    for block in blocks:
        single_line = _single_line(block.raw_text)
        is_heading_candidate = _is_heading_candidate(block, single_line)
        ordinal, heading_title = _split_heading(single_line) if is_heading_candidate else (None, None)
        heading_clean = normalize_display_text(heading_title or single_line or "") or None
        heading_kind = _heading_kind(heading_clean) if heading_clean else None

        is_heading = False
        block_role = state
        section_kind = state if state != "front_matter" else "front_matter"
        heading_level = block.text_level

        if not title_taken and block.block_type == "text" and single_line:
            if _looks_like_title(block, work_title):
                title_taken = True
                block_role = "title"
                section_kind = "front_matter"
        if block_role != "title":
            if state == "front_matter":
                if is_heading_candidate and heading_kind == "abstract":
                    state = "abstract"
                    block_role = "abstract"
                    section_kind = "abstract"
                    is_heading = True
                    heading_level = heading_level or 1
                elif is_heading_candidate and heading_kind in {"body", "appendix", "bibliography", "back_matter"}:
                    if _can_start_main_text(heading_clean, ordinal):
                        state = heading_kind
                        block_role = state
                        section_kind = state
                        is_heading = True
                        heading_level = heading_level or 1
                    else:
                        block_role = "front_matter"
                        section_kind = "front_matter"
                else:
                    block_role = "front_matter"
                    section_kind = "front_matter"
            elif state == "abstract":
                if is_heading_candidate and heading_kind != "abstract":
                    state = heading_kind or "body"
                    if state not in {"body", "appendix", "bibliography", "back_matter"}:
                        state = "body"
                    block_role = state
                    section_kind = state
                    is_heading = True
                    heading_level = heading_level or 1
                else:
                    block_role = "abstract"
                    section_kind = "abstract"
            elif state == "bibliography":
                if is_heading_candidate and heading_kind == "appendix":
                    state = "appendix"
                    block_role = "appendix"
                    section_kind = "appendix"
                    is_heading = True
                    heading_level = heading_level or 1
                else:
                    block_role = "bibliography"
                    section_kind = "bibliography"
            else:
                if is_heading_candidate:
                    if heading_kind == "bibliography":
                        state = "bibliography"
                    elif heading_kind == "appendix":
                        state = "appendix"
                    elif heading_kind == "back_matter":
                        state = "back_matter"
                    elif heading_kind == "abstract":
                        state = "abstract"
                    else:
                        state = "appendix" if state == "appendix" else "body"
                    block_role = state
                    section_kind = state
                    is_heading = True
                    heading_level = heading_level or 1
                else:
                    block_role = state
                    section_kind = state

        clean_text, flags = _clean_block_text(
            block.raw_text,
            block_role=block_role,
            block_type=block.block_type,
        )
        is_retrievable = bool(clean_text)
        exclusion_reason: str | None = None
        if block_role in {"title", "front_matter", "bibliography", "back_matter"}:
            is_retrievable = False
            exclusion_reason = block_role
        elif not clean_text:
            is_retrievable = False
            exclusion_reason = "empty_after_clean"

        annotated.append(
            AnnotatedBlock(
                parsed=block,
                block_role=block_role,
                is_heading=is_heading,
                heading_level=heading_level,
                heading_ordinal=ordinal if is_heading else None,
                heading_clean_title=heading_clean if is_heading else None,
                section_kind=section_kind,
                clean_text=clean_text,
                is_retrievable=is_retrievable,
                exclusion_reason=exclusion_reason,
                flags=flags,
            )
        )
    return annotated


# ---------------------------------------------------------------------------
# Section / heading helpers
# ---------------------------------------------------------------------------


def _insert_section(
    conn: sqlite3.Connection,
    *,
    document_id: int,
    parent_section_id: int | None,
    ordinal: str | None,
    title: str,
    clean_title: str,
    path_text: str,
    section_kind: str,
    level: int,
    order_index: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO document_sections (
          document_id, parent_section_id, ordinal, title, clean_title,
          path_text, section_kind, level, order_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            parent_section_id,
            ordinal,
            title,
            clean_title,
            path_text,
            section_kind,
            level,
            order_index,
        ),
    )
    return int(cur.lastrowid)


def _bump_page_range(ranges: dict[int, list[int | None]], section_id: int, page: int) -> None:
    low, high = ranges.setdefault(section_id, [None, None])
    if low is None or page < low:
        low = page
    if high is None or page > high:
        high = page
    ranges[section_id] = [low, high]


def _is_heading_candidate(block: ParsedBlock, single_line: str) -> bool:
    if block.block_type != "text":
        return False
    if not single_line or len(single_line) > 200:
        return False
    if block.text_level is not None and block.text_level <= 3:
        return True
    return bool(re.match(r"^(?:\d+(?:\.\d+)*|Appendix\s+[A-Z0-9]+)\s+\S", single_line, re.IGNORECASE))


def _split_heading(text: str) -> tuple[str | None, str | None]:
    single = _single_line(text)
    if not single:
        return None, None
    appendix_match = re.match(r"^(Appendix\s+[A-Z0-9]+)\s*[:.-]?\s*(.*)$", single, re.IGNORECASE)
    if appendix_match:
        ordinal = appendix_match.group(1).strip()
        title = appendix_match.group(2).strip() or ordinal
        return ordinal, title
    numbered_match = re.match(r"^((?:\d+(?:\.\d+)*))[\s.)-]+(.+)$", single)
    if numbered_match:
        return numbered_match.group(1).strip(), numbered_match.group(2).strip()
    return None, single


def _heading_kind(title: str | None) -> str | None:
    if not title:
        return None
    normalized = _normalize_heading(title)
    if normalized in ABSTRACT_HEADINGS:
        return "abstract"
    if normalized in BIBLIOGRAPHY_HEADINGS:
        return "bibliography"
    if normalized in BACK_MATTER_HEADINGS:
        return "back_matter"
    if normalized.startswith("appendix"):
        return "appendix"
    return "body"


def _can_start_main_text(heading_title: str | None, ordinal: str | None) -> bool:
    if ordinal:
        return True
    normalized = _normalize_heading(heading_title or "")
    return normalized in ABSTRACT_HEADINGS | BIBLIOGRAPHY_HEADINGS | BACK_MATTER_HEADINGS | {"introduction"} or normalized.startswith(
        "appendix"
    )


def _looks_like_title(block: ParsedBlock, work_title: str) -> bool:
    if block.block_type != "text":
        return False
    text = _single_line(block.raw_text)
    if not text:
        return False
    if block.text_level is not None and block.text_level > 1:
        return False
    title_norm = _normalize_heading(work_title)
    text_norm = _normalize_heading(text)
    if text_norm and text_norm == title_norm:
        return True
    return block.index == 0 and len(text) >= 20


def _normalize_heading(text: str) -> str:
    single = _single_line(text)
    _, title = _split_heading(single)
    return re.sub(r"\s+", " ", (title or single).strip().lower())


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------


def _clean_block_text(raw_text: str, *, block_role: str, block_type: str) -> tuple[str, dict[str, Any]]:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").strip()
    if not text:
        return "", {"citation_markers_removed": 0}

    removed = 0
    if block_role in {"abstract", "body", "appendix"} and block_type in {"text", "table", "image"}:
        for pattern in INLINE_CITATION_PATTERNS:
            text, n = pattern.subn(" ", text)
            removed += n

    text = normalize_display_text(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(), {"citation_markers_removed": removed}


def _rewrite_mineru_bibliography_citations(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    document_id: int,
    blocks: list[dict[str, Any]],
) -> dict[str, int]:
    from hep_rag_v2.metadata import find_work_id

    existing_target_keys: set[tuple[str, str]] = set()
    stale_citation_ids: list[int] = []
    for row in conn.execute(
        """
        SELECT citation_id, dst_source, dst_external_id, raw_json
        FROM citations
        WHERE src_work_id = ?
        """,
        (work_id,),
    ).fetchall():
        raw_payload = _load_raw_citation_payload(row["raw_json"])
        if raw_payload.get("source") == "mineru_bibliography":
            stale_citation_ids.append(int(row["citation_id"]))
            continue
        dst_source = str(row["dst_source"] or "").strip()
        dst_external_id = str(row["dst_external_id"] or "").strip()
        if dst_source and dst_external_id:
            existing_target_keys.add((dst_source, dst_external_id))

    if stale_citation_ids:
        conn.executemany(
            "DELETE FROM citations WHERE citation_id = ?",
            [(citation_id,) for citation_id in stale_citation_ids],
        )

    written = 0
    resolved = 0
    seen_raw_hashes: set[str] = set()
    bibliography_entries = _bibliography_entries_from_blocks(blocks)
    for entry_index, entry_text in enumerate(bibliography_entries, start=1):
        target = _bibliography_target(entry_text)
        raw_hash = _hash_text(entry_text)
        if target is not None:
            if target in existing_target_keys:
                continue
            existing_target_keys.add(target)
        elif raw_hash and raw_hash in seen_raw_hashes:
            continue
        if raw_hash:
            seen_raw_hashes.add(raw_hash)

        dst_work_id = None
        dst_source = None
        dst_external_id = None
        if target is not None:
            dst_source, dst_external_id = target
            dst_work_id = find_work_id(conn, id_type=dst_source, id_value=dst_external_id)

        raw_payload = {
            "source": "mineru_bibliography",
            "document_id": document_id,
            "entry_index": entry_index,
            "entry_text": entry_text,
        }
        if dst_source and dst_external_id:
            raw_payload["target"] = {
                "dst_source": dst_source,
                "dst_external_id": dst_external_id,
            }

        conn.execute(
            """
            INSERT INTO citations (
              src_work_id, dst_work_id, dst_source, dst_external_id, raw_json, resolution_status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                work_id,
                dst_work_id,
                dst_source,
                dst_external_id,
                json.dumps(raw_payload, ensure_ascii=False),
                "resolved" if dst_work_id is not None else "unresolved",
            ),
        )
        written += 1
        if dst_work_id is not None:
            resolved += 1

    return {
        "citations_written": written,
        "citations_resolved": resolved,
        "bibliography_entries": len(bibliography_entries),
    }


def _bibliography_entries_from_blocks(blocks: list[dict[str, Any]]) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        if str(block.get("block_role") or "") != "bibliography":
            continue
        if bool(block.get("is_heading")):
            continue
        text = str(block.get("raw_text") or block.get("clean_text") or "").strip()
        if not text:
            continue
        for entry in _split_bibliography_entries(text):
            normalized = _single_line(entry)
            if not normalized or normalized.casefold() in seen:
                continue
            seen.add(normalized.casefold())
            entries.append(normalized)
    return entries


def _split_bibliography_entries(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return []
    for pattern in (BRACKETED_REFERENCE_PATTERN, NUMBERED_REFERENCE_PATTERN):
        parts: list[str] = []
        for part in pattern.split(compact):
            cleaned = _clean_bibliography_entry(part)
            if cleaned:
                parts.append(cleaned)
        if len(parts) >= 2:
            return parts
    cleaned = _clean_bibliography_entry(compact)
    return [cleaned] if cleaned else []


def _clean_bibliography_entry(text: str) -> str:
    value = re.sub(r"^(?:\[\d+\]|\d+\.)\s*", "", str(text or "").strip())
    value = re.sub(r"\s+", " ", value).strip(" ;,")
    if len(value) < 12:
        return ""
    if value.casefold() in BIBLIOGRAPHY_HEADINGS:
        return ""
    return value


def _bibliography_target(entry_text: str) -> tuple[str, str] | None:
    doi = _extract_doi(entry_text)
    if doi:
        return ("doi", doi)
    arxiv_id = _extract_arxiv_id(entry_text)
    if arxiv_id:
        return ("arxiv", arxiv_id)
    return None


def _extract_doi(text: str) -> str | None:
    match = DOI_PATTERN.search(str(text or ""))
    if not match:
        return None
    value = match.group(0).rstrip(").,;]")
    return value.lower()


def _extract_arxiv_id(text: str) -> str | None:
    match = ARXIV_PATTERN.search(str(text or ""))
    if not match:
        return None
    value = match.group(1).strip().rstrip(").,;]")
    lower = value.casefold()
    if lower.startswith("10.") or len(lower) < 7:
        return None
    return value


def _load_raw_citation_payload(raw_json: str | None) -> dict[str, Any]:
    if not raw_json:
        return {}
    try:
        payload = json.loads(raw_json)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_latex(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _hash_text(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _single_line(text: str) -> str:
    return " ".join(text.split()).strip()


def _page_hint(pages: list[int]) -> str | None:
    if not pages:
        return None
    if len(set(pages)) == 1:
        return str(pages[0])
    return f"{min(pages)}-{max(pages)}"
