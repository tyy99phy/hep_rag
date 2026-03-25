from __future__ import annotations

import re
from typing import Any

from hep_rag_v2.textnorm import normalize_display_text

from .document import _page_hint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTINUATION_START_RE = re.compile(
    r"^(?:[a-zα-ω]|where\b|which\b|that\b|and\b|or\b|but\b|with\b|without\b|for\b|to\b|of\b|in\b|on\b|from\b|by\b|as\b|at\b|via\b|using\b|including\b|excluding\b|respectively\b|trimuon\b|dimuon\b|signal\b|background\b|events\b|the\b|a\b|an\b|[({\[])",  # noqa: E501
)
STRONG_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")
WEAK_SENTENCE_END_RE = re.compile(r"[:,;][\"')\]]*$")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=(?:[""(]*[A-Z0-9ΔΣΩ]))')
LATEX_FRACTION_COMMANDS = ("\\frac", "\\dfrac", "\\tfrac")
UNSAFE_TRUNCATION_TOKENS = {"-", "<", ">", "/", "=", "^", "+", "*", "(", "[", "{", ","}


# ---------------------------------------------------------------------------
# build_chunks  (public API)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Chunk assembly helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Linear-unit helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Block merge heuristics
# ---------------------------------------------------------------------------


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
    return bool(re.match(r'^(?:[""(]*[A-Z0-9ΔΣΩ])', text))


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


# ---------------------------------------------------------------------------
# Equation bridge / placeholder helpers
# ---------------------------------------------------------------------------


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


def _strip_math_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("$$") and value.endswith("$$"):
        value = value[2:-2].strip()
    elif value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value


# ---------------------------------------------------------------------------
# Token / budget helpers
# ---------------------------------------------------------------------------


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
