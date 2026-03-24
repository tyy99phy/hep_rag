from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hep_rag_v2.textnorm import normalize_display_text


INLINE_CITATION_PATTERNS = [
    re.compile(r"\(\s*(?:see\s+)?(?:e\.g\.\s*,?\s*)?(?:Refs?\.?|references?)\s*\[[0-9,\s;–-]+\]\s*\)", re.IGNORECASE),
    re.compile(r"(?:Refs?\.?|references?)\s*\[[0-9,\s;–-]+\]", re.IGNORECASE),
    re.compile(r"\[[0-9,\s;–-]+\]"),
]

ABSTRACT_HEADINGS = {"abstract"}
BIBLIOGRAPHY_HEADINGS = {"references", "reference", "bibliography"}
BACK_MATTER_HEADINGS = {"acknowledgments", "acknowledgements"}
CONTINUATION_START_RE = re.compile(
    r"^(?:[a-zα-ω]|where\b|which\b|that\b|and\b|or\b|but\b|with\b|without\b|for\b|to\b|of\b|in\b|on\b|from\b|by\b|as\b|at\b|via\b|using\b|including\b|excluding\b|respectively\b|trimuon\b|dimuon\b|signal\b|background\b|events\b|the\b|a\b|an\b|[({\[])",  # noqa: E501
)
STRONG_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")
WEAK_SENTENCE_END_RE = re.compile(r"[:,;][\"')\]]*$")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=(?:["“(]*[A-Z0-9ΔΣΩ]))')
LATEX_FRACTION_COMMANDS = ("\\frac", "\\dfrac", "\\tfrac")
UNSAFE_TRUNCATION_TOKENS = {"-", "<", ">", "/", "=", "^", "+", "*", "(", "[", "{", ","}


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


def import_mineru_source(*, source_path: Path, dest_dir: Path, replace: bool = False) -> dict[str, Any]:
    source = source_path.resolve()
    if source.is_file() and source.name == "manifest.json":
        manifest = load_manifest(source)
        raw_source = Path(str(manifest.get("raw_dir") or source.parent / "raw"))
        return import_mineru_bundle(bundle_path=raw_source, dest_dir=dest_dir, replace=replace)
    if source.is_dir() and (source / "manifest.json").exists():
        manifest = load_manifest(source / "manifest.json")
        raw_source = Path(str(manifest.get("raw_dir") or source / "raw"))
        return import_mineru_bundle(bundle_path=raw_source, dest_dir=dest_dir, replace=replace)
    return import_mineru_bundle(bundle_path=source, dest_dir=dest_dir, replace=replace)


def import_mineru_bundle(*, bundle_path: Path, dest_dir: Path, replace: bool = False) -> dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"MinerU bundle not found: {bundle_path}")

    raw_dir = dest_dir / "raw"
    if raw_dir.exists():
        if not replace:
            raise FileExistsError(f"Target parse directory already exists: {raw_dir}")
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if bundle_path.is_file() and bundle_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(raw_dir)
    elif bundle_path.is_dir():
        for child in bundle_path.iterdir():
            target = raw_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
    else:
        raise ValueError(f"Unsupported MinerU bundle type: {bundle_path}")

    full_md_path = _first_match(raw_dir, "*full.md")
    content_list_path = _first_match(raw_dir, "*content_list.json")
    model_json_path = _first_match(raw_dir, "*model.json")
    if not full_md_path or not content_list_path:
        raise ValueError(
            "MinerU bundle is missing expected outputs. Need at least '*full.md' and '*content_list.json'."
        )

    manifest = {
        "engine": "mineru",
        "bundle_source": str(bundle_path),
        "raw_dir": str(raw_dir),
        "full_md_path": str(full_md_path),
        "content_list_path": str(content_list_path),
        "model_json_path": str(model_json_path) if model_json_path else None,
    }
    (dest_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_content_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("content_list", "items", "data", "contents", "blocks"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported MinerU content_list schema: {path}")


def parsed_blocks_from_content_list(items: list[dict[str, Any]]) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    for idx, item in enumerate(items):
        block_type = _normalize_type(item.get("type"))
        if block_type == "discarded":
            continue
        raw_text = _render_block_text(item, block_type)
        latex = _extract_latex(item, block_type)
        caption = _extract_caption(item, block_type)
        asset_path = _first_str(item, "image_path", "img_path", "path")
        if not raw_text and not latex and not caption:
            continue
        blocks.append(
            ParsedBlock(
                index=idx,
                block_type=block_type,
                raw_text=raw_text,
                page=_first_int(item, "page_idx", "page", "page_no"),
                order=idx,
                text_level=_first_int(item, "text_level", "level"),
                latex=latex,
                caption=caption,
                asset_path=asset_path,
                raw=item,
            )
        )
    return blocks


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
    }


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


def build_chunks(
    blocks: list[dict[str, Any]],
    *,
    chunk_size: int,
    overlap_blocks: int,
    section_parent_char_limit: int,
) -> list[dict[str, Any]]:
    token_budget = _token_budget_from_char_budget(chunk_size)
    by_section: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        by_section.setdefault(int(block["section_id"]), []).append(block)

    chunk_rows: list[dict[str, Any]] = []
    for section_blocks in by_section.values():
        ordered = sorted(section_blocks, key=lambda item: int(item["order_index"]))
        retrievable = [block for block in ordered if bool(block["is_retrievable"]) and str(block["clean_text"]).strip()]
        linear_blocks = _linearize_blocks_for_chunking(retrievable)
        linear_units = _split_linear_units_for_budget(
            _build_linear_units(linear_blocks),
            token_budget=token_budget,
            char_budget=chunk_size,
        )
        if not retrievable:
            continue

        section_kind = str(retrievable[0]["section_kind"])
        if section_kind == "abstract":
            if linear_units:
                chunk_rows.append(_compose_chunk(linear_units, chunk_role="abstract_chunk"))
            continue

        if section_kind in {"body", "appendix"}:
            if linear_units:
                total_chars = sum(len(str(unit["clean_text"])) for unit in linear_units)
                if total_chars <= section_parent_char_limit:
                    chunk_rows.append(_compose_chunk(linear_units, chunk_role="section_parent"))
                chunk_rows.extend(
                    _chunk_linear_units(
                        linear_units,
                        chunk_role="section_child",
                        token_budget=token_budget,
                        char_budget=chunk_size,
                        chunk_size=chunk_size,
                        overlap_blocks=overlap_blocks,
                    )
                )
            chunk_rows.extend(
                _context_windows(
                    ordered,
                    target_type="equation",
                    chunk_role="formula_window",
                    include_target=False,
                )
            )
            chunk_rows.extend(
                _context_windows(
                    ordered,
                    target_type="image",
                    chunk_role="asset_window",
                    include_target=True,
                )
            )
            chunk_rows.extend(
                _context_windows(
                    ordered,
                    target_type="table",
                    chunk_role="asset_window",
                    include_target=True,
                )
            )
    return chunk_rows


def _chunk_linear_units(
    units: list[dict[str, Any]],
    *,
    chunk_role: str,
    token_budget: int,
    char_budget: int,
    chunk_size: int,
    overlap_blocks: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    window: list[dict[str, Any]] = []
    window_chars = 0
    window_tokens = 0

    def flush() -> None:
        nonlocal window, window_chars, window_tokens
        if not window:
            return
        out.append(_compose_chunk(window, chunk_role=chunk_role))
        carry = window[-overlap_blocks:] if overlap_blocks > 0 else []
        window = list(carry)
        window_chars = sum(len(str(item["clean_text"])) for item in window)
        window_tokens = sum(_approx_token_count(str(item["clean_text"])) for item in window)

    for unit in units:
        unit_chars = len(str(unit["clean_text"]))
        unit_tokens = _approx_token_count(str(unit["clean_text"]))
        if window and (window_tokens + unit_tokens > token_budget or window_chars + unit_chars > char_budget):
            flush()
        if not window and (unit_tokens > token_budget or unit_chars > char_budget):
            out.append(_compose_chunk([unit], chunk_role=chunk_role))
            continue
        window.append(unit)
        window_chars += unit_chars
        window_tokens += unit_tokens
    flush()
    return out


def _context_windows(
    blocks: list[dict[str, Any]],
    *,
    target_type: str,
    chunk_role: str,
    include_target: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        if str(block["block_type"]) != target_type:
            continue
        window: list[dict[str, Any]] = []
        prev_block = _nearest_retrievable(blocks, start=idx - 1, step=-1, include_equations=False)
        next_block = _nearest_retrievable(blocks, start=idx + 1, step=1, include_equations=False)
        if prev_block is not None:
            window.append(prev_block)
        if include_target and bool(block["is_retrievable"]) and str(block["clean_text"]).strip():
            window.append(block)
        if next_block is not None and (not window or next_block["block_id"] != window[-1]["block_id"]):
            window.append(next_block)
        if not window:
            continue
        out.append(_compose_chunk(window, chunk_role=chunk_role))
    return out


def _nearest_retrievable(
    blocks: list[dict[str, Any]],
    *,
    start: int,
    step: int,
    include_equations: bool,
) -> dict[str, Any] | None:
    idx = start
    while 0 <= idx < len(blocks):
        candidate = blocks[idx]
        if (
            bool(candidate["is_retrievable"])
            and str(candidate["clean_text"]).strip()
            and (include_equations or str(candidate["block_type"]) != "equation")
        ):
            return candidate
        idx += step
    return None


def _include_in_linear_chunks(block: dict[str, Any]) -> bool:
    return str(block.get("block_type") or "") != "equation"


def _compose_chunk(blocks: list[dict[str, Any]], *, chunk_role: str) -> dict[str, Any]:
    unique_blocks: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    raw_segments: list[str] = []
    clean_segments: list[str] = []
    for block in blocks:
        raw_segment = str(block["raw_text"]).strip()
        clean_segment = str(block["clean_text"]).strip()
        if raw_segment:
            raw_segments.append(raw_segment)
        if clean_segment:
            clean_segments.append(clean_segment)
        for source_block in _source_blocks(block):
            block_id = int(source_block["block_id"])
            if block_id in seen_ids:
                continue
            seen_ids.add(block_id)
            unique_blocks.append(source_block)

    raw_text = "\n\n".join(raw_segments)
    clean_text = "\n\n".join(clean_segments)
    flags = {
        "block_count": len(unique_blocks),
        "citation_markers_removed": sum(
            int((block.get("flags") or {}).get("citation_markers_removed") or 0)
            for block in unique_blocks
        ),
    }
    pages = [int(block["page"]) for block in unique_blocks if block.get("page") is not None]
    return {
        "section_id": unique_blocks[0]["section_id"],
        "block_start_id": unique_blocks[0]["block_id"],
        "block_end_id": unique_blocks[-1]["block_id"],
        "chunk_role": chunk_role,
        "page_hint": _page_hint(pages),
        "section_hint": unique_blocks[0]["section_path"],
        "raw_text": raw_text,
        "clean_text": clean_text,
        "is_retrievable": bool(clean_text.strip()),
        "exclusion_reason": None if clean_text.strip() else "empty_after_clean",
        "source_block_ids": [int(block["block_id"]) for block in unique_blocks],
        "flags": flags,
    }


def _build_linear_units(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []

    units: list[list[dict[str, Any]]] = [[blocks[0]]]
    for block in blocks[1:]:
        current = units[-1]
        if _should_merge_linear_blocks(current[-1], block):
            current.append(block)
        else:
            units.append([block])
    return [_compose_linear_unit(group) for group in units]


def _linearize_blocks_for_chunking(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in blocks:
        block_type = str(block.get("block_type") or "")
        if block_type != "equation":
            out.append(block)
            continue
        placeholder = _equation_bridge_block(block)
        if placeholder is not None:
            out.append(placeholder)
    return out


def _split_linear_units_for_budget(
    units: list[dict[str, Any]],
    *,
    token_budget: int,
    char_budget: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for unit in units:
        if _approx_token_count(str(unit.get("clean_text") or "")) <= token_budget and len(str(unit.get("clean_text") or "")) <= char_budget:
            out.append(unit)
            continue
        out.extend(_split_unit_into_sentence_fragments(unit, token_budget=token_budget, char_budget=char_budget))
    return out


def _compose_linear_unit(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(blocks) == 1:
        block = dict(blocks[0])
        block["source_blocks"] = [blocks[0]]
        return block

    first = blocks[0]
    raw_text = str(first["raw_text"]).strip()
    clean_text = str(first["clean_text"]).strip()
    for previous, current in zip(blocks, blocks[1:]):
        raw_text = _merge_inline_text(raw_text, str(current["raw_text"]).strip())
        clean_text = _merge_inline_text(clean_text, str(current["clean_text"]).strip())

    unit = dict(first)
    unit["raw_text"] = raw_text
    unit["clean_text"] = clean_text
    unit["source_blocks"] = list(blocks)
    flags = dict(first.get("flags") or {})
    flags["citation_markers_removed"] = sum(
        int((block.get("flags") or {}).get("citation_markers_removed") or 0)
        for block in blocks
    )
    unit["flags"] = flags
    return unit


def _split_unit_into_sentence_fragments(
    unit: dict[str, Any],
    *,
    token_budget: int,
    char_budget: int,
) -> list[dict[str, Any]]:
    clean_parts = _sentence_like_parts(str(unit.get("clean_text") or ""))
    if len(clean_parts) <= 1:
        return [unit]

    raw_parts = _sentence_like_parts(str(unit.get("raw_text") or ""))
    if len(raw_parts) != len(clean_parts):
        raw_parts = clean_parts

    out: list[dict[str, Any]] = []
    current_clean: list[str] = []
    current_raw: list[str] = []
    current_tokens = 0
    current_chars = 0

    def flush() -> None:
        nonlocal current_clean, current_raw, current_tokens, current_chars
        if not current_clean:
            return
        fragment = dict(unit)
        fragment["clean_text"] = _join_sentence_parts(current_clean)
        fragment["raw_text"] = _join_sentence_parts(current_raw)
        out.append(fragment)
        current_clean = []
        current_raw = []
        current_tokens = 0
        current_chars = 0

    for clean_part, raw_part in zip(clean_parts, raw_parts):
        part_tokens = _approx_token_count(clean_part)
        part_chars = len(clean_part)
        if current_clean and (current_tokens + part_tokens > token_budget or current_chars + part_chars > char_budget):
            flush()
        if not current_clean and (part_tokens > token_budget or part_chars > char_budget):
            clean_slices = _hard_split_text_by_budget(clean_part, token_budget=token_budget, char_budget=char_budget)
            raw_slices = _hard_split_text_by_budget(raw_part, token_budget=token_budget, char_budget=char_budget)
            if len(raw_slices) != len(clean_slices):
                raw_slices = clean_slices
            for clean_slice, raw_slice in zip(clean_slices, raw_slices):
                fragment = dict(unit)
                fragment["clean_text"] = clean_slice
                fragment["raw_text"] = raw_slice
                out.append(fragment)
            continue
        current_clean.append(clean_part)
        current_raw.append(raw_part)
        current_tokens += part_tokens
        current_chars += part_chars
    flush()
    return out or [unit]


def _should_merge_linear_blocks(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("block_type") or "") != "text" or str(right.get("block_type") or "") != "text":
        return False
    if bool(left.get("is_heading")) or bool(right.get("is_heading")):
        return False

    left_text = str(left.get("clean_text") or "").strip()
    right_text = str(right.get("clean_text") or "").strip()
    if not left_text or not right_text:
        return False
    if abs(int((left.get("page") or 0) or 0) - int((right.get("page") or 0) or 0)) > 1:
        return False

    if _ends_with_strong_sentence_boundary(left_text) and _starts_like_new_sentence(right_text):
        return False
    if _ends_with_weak_sentence_boundary(left_text):
        return True
    if _starts_like_continuation(right_text):
        return True
    if not _ends_with_strong_sentence_boundary(left_text):
        return True
    return False


def _ends_with_strong_sentence_boundary(text: str) -> bool:
    return bool(STRONG_SENTENCE_END_RE.search(text.strip()))


def _ends_with_weak_sentence_boundary(text: str) -> bool:
    return bool(WEAK_SENTENCE_END_RE.search(text.strip()))


def _starts_like_new_sentence(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if _starts_like_continuation(text):
        return False
    return bool(re.match(r'^(?:["“(]*[A-Z0-9ΔΣΩ])', text))


def _starts_like_continuation(text: str) -> bool:
    return bool(CONTINUATION_START_RE.match(text.strip()))


def _merge_inline_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith(("-", "/", "(")) or re.match(r"^[,.;:!?%)\]}]", right):
        return f"{left}{right}"
    return f"{left} {right}"


def _source_blocks(block: dict[str, Any]) -> list[dict[str, Any]]:
    source_blocks = block.get("source_blocks")
    if isinstance(source_blocks, list) and source_blocks:
        return [item for item in source_blocks if isinstance(item, dict)]
    return [block]


def _equation_bridge_block(block: dict[str, Any]) -> dict[str, Any] | None:
    summary = _equation_placeholder_text(
        str(block.get("clean_text") or ""),
        raw_text=str(block.get("raw_text") or ""),
    )
    if not summary:
        return None
    bridge = dict(block)
    bridge["block_type"] = "equation_bridge"
    bridge["raw_text"] = summary
    bridge["clean_text"] = summary
    bridge["source_blocks"] = _source_blocks(block)
    return bridge


def _equation_placeholder_text(text: str, *, raw_text: str | None = None, token_limit: int = 32) -> str:
    compact = ""
    if raw_text and _looks_like_latex_equation(raw_text):
        compact = _linearize_equation_placeholder(raw_text)
    if not compact:
        compact = _linearize_equation_placeholder(text)
    if not compact:
        return ""
    if _looks_like_non_equation_display_list(raw_text or "", compact):
        return ""
    compact = _truncate_text_by_tokens(compact, token_limit)
    return f"Equation: {compact}"


def _linearize_equation_placeholder(text: str) -> str:
    candidate = _strip_math_fences(text).strip()
    if not candidate:
        return ""
    if any(command in candidate for command in LATEX_FRACTION_COMMANDS) or "\\" in candidate or "{" in candidate or "}" in candidate:
        candidate = _expand_latex_fractions(candidate)
        candidate = normalize_display_text(candidate)
    candidate = _compact_equation_placeholder(candidate)
    return candidate


def _looks_like_latex_equation(text: str) -> bool:
    return bool(text and any(marker in text for marker in ("\\", "{", "}", "$")))


def _looks_like_non_equation_display_list(raw_text: str, compact_text: str) -> bool:
    raw_lower = raw_text.lower()
    compact_lower = compact_text.lower()
    return "\\bullet" in raw_lower or "bullet" in compact_lower


def _compact_equation_placeholder(text: str) -> str:
    compact = re.sub(r"\s*=\s*", " = ", text)
    compact = re.sub(r"\s*/\s*", " / ", compact)
    compact = re.sub(r"\)\s*(?=\()", ") x ", compact)
    compact = re.sub(r"\)\s*(?=(?!x\b)[A-Za-zΑ-Ωα-ωϵσχτμπνλρϕωΔ])", ") x ", compact)
    compact = re.sub(r"\]\s*(?=\()", "] x ", compact)
    compact = re.sub(r"\]\s*(?=(?!x\b)[A-Za-zΑ-Ωα-ωϵσχτμπνλρϕωΔ])", "] x ", compact)
    compact = re.sub(r"\bx\s+x\b", "x", compact)
    compact = re.sub(r"\s+", " ", compact)
    return compact.strip().strip(",;")


def _expand_latex_fractions(text: str) -> str:
    out: list[str] = []
    idx = 0
    while idx < len(text):
        command = next((item for item in LATEX_FRACTION_COMMANDS if text.startswith(item, idx)), None)
        if command is None:
            out.append(text[idx])
            idx += 1
            continue

        idx += len(command)
        numerator, idx = _consume_latex_group(text, idx)
        denominator, idx = _consume_latex_group(text, idx)
        if numerator and denominator:
            out.append(f" ( {_expand_latex_fractions(numerator)} ) / ( {_expand_latex_fractions(denominator)} ) ")
        else:
            out.append(command)
    return "".join(out)


def _consume_latex_group(text: str, idx: int) -> tuple[str, int]:
    idx = _skip_latex_whitespace(text, idx)
    if idx >= len(text):
        return "", idx
    if text[idx] != "{":
        return text[idx], idx + 1

    start = idx + 1
    depth = 0
    idx += 1
    while idx < len(text):
        token = text[idx]
        if token == "{":
            depth += 1
        elif token == "}":
            if depth == 0:
                return text[start:idx], idx + 1
            depth -= 1
        idx += 1
    return text[start:], idx


def _skip_latex_whitespace(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _token_budget_from_char_budget(char_budget: int) -> int:
    return max(20, char_budget // 4)


def _approx_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def _sentence_like_parts(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text or "") if part.strip()]
    parts: list[str] = []
    for paragraph in paragraphs:
        sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(paragraph) if item.strip()]
        if sentences:
            parts.extend(sentences)
        else:
            parts.append(paragraph)
    return parts


def _join_sentence_parts(parts: list[str]) -> str:
    out = ""
    for part in parts:
        out = _merge_inline_text(out, part)
    return out.strip()


def _hard_split_text_by_budget(text: str, *, token_budget: int, char_budget: int) -> list[str]:
    tokens = TOKEN_RE.findall(text or "")
    if not tokens:
        return []

    out: list[str] = []
    current: list[str] = []
    for token in tokens:
        candidate = _join_tokens(current + [token])
        if current and (_approx_token_count(candidate) > token_budget or len(candidate) > char_budget):
            out.append(_join_tokens(current))
            current = [token]
        else:
            current.append(token)
    if current:
        out.append(_join_tokens(current))
    return [item for item in out if item.strip()]


def _truncate_text_by_tokens(text: str, limit: int) -> str:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= limit:
        return text
    cutoff = min(len(tokens), limit)
    while cutoff > 1 and tokens[cutoff - 1] in UNSAFE_TRUNCATION_TOKENS:
        cutoff -= 1
    if cutoff <= 0:
        cutoff = min(len(tokens), limit)
    return _join_tokens(tokens[:cutoff]) + " ..."


def _join_tokens(tokens: list[str]) -> str:
    out = ""
    for token in tokens:
        if not out:
            out = token
            continue
        if token == "-" and out.endswith("<"):
            out += token
        elif token == ">" and out.endswith(("-", "<-")):
            out += token
        elif re.match(r"^[,.;:!?%)\]}]$", token):
            out += token
        elif re.match(r"^[({\[]$", token) or out.endswith(("(", "[", "{", "/")):
            out += token
        else:
            out += f" {token}"
    return out


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


def _normalize_type(value: Any) -> str:
    if not isinstance(value, str):
        return "text"
    normalized = value.strip().lower()
    aliases = {
        "title": "text",
        "heading": "text",
        "header": "text",
        "paragraph": "text",
        "discarded": "discarded",
        "equation": "equation",
        "formula": "equation",
        "inline_formula": "equation",
        "table": "table",
        "image": "image",
        "figure": "image",
    }
    return aliases.get(normalized, normalized)


def _render_block_text(item: dict[str, Any], block_type: str) -> str:
    if block_type == "equation":
        latex = _first_str(item, "latex", "text", "content")
        if not latex:
            return ""
        latex = latex.strip()
        if latex.startswith("$"):
            return latex
        return f"$$\n{latex}\n$$"

    if block_type == "table":
        caption = _extract_caption(item, block_type)
        latex = _first_str(item, "latex")
        if latex:
            return _prepend_caption(latex.strip(), caption)
        html = _first_str(item, "html", "table_body")
        if html:
            return _prepend_caption(html.strip(), caption)
        text = _first_str(item, "text", "content")
        return _prepend_caption(text.strip(), caption) if text else caption or ""

    if block_type == "image":
        caption = _extract_caption(item, block_type)
        if caption:
            return caption
        text = _first_str(item, "caption", "text", "content")
        return text.strip() if text else ""

    text = _first_str(item, "text", "content", "md")
    return text.strip() if text else ""


def _extract_latex(item: dict[str, Any], block_type: str) -> str | None:
    if block_type not in {"equation", "table"}:
        return None
    latex = _first_str(item, "latex")
    if latex:
        return _strip_math_fences(latex)
    if block_type == "equation":
        fallback = _first_str(item, "text", "content")
        if fallback:
            return _strip_math_fences(fallback)
    return None


def _extract_caption(item: dict[str, Any], block_type: str) -> str | None:
    if block_type == "table":
        return _join_text(item.get("table_caption")) or _join_text(item.get("table_footnote"))
    if block_type == "image":
        return _join_text(item.get("image_caption")) or _join_text(item.get("image_footnote")) or _first_str(
            item,
            "caption",
        )
    return None


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


def _normalize_latex(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _strip_math_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("$$") and value.endswith("$$"):
        value = value[2:-2].strip()
    elif value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value


def _page_hint(pages: list[int]) -> str | None:
    if not pages:
        return None
    if len(set(pages)) == 1:
        return str(pages[0])
    return f"{min(pages)}-{max(pages)}"


def _hash_text(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _single_line(text: str) -> str:
    return " ".join(text.split()).strip()


def _join_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        if parts:
            return " ".join(parts).strip()
    return None


def _prepend_caption(text: str, caption: str | None) -> str:
    if caption:
        return f"{caption.strip()}\n\n{text}"
    return text


def _first_match(root: Path, pattern: str) -> Path | None:
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            return path
    return None


def _first_str(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_int(item: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None
