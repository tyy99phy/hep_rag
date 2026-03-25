from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .document import ParsedBlock


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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


def _strip_math_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("$$") and value.endswith("$$"):
        value = value[2:-2].strip()
    elif value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value


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
