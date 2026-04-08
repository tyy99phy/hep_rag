from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from hep_rag_v2 import paths
from hep_rag_v2.fulltext import annotate_blocks, import_mineru_source, load_content_list, load_manifest, parsed_blocks_from_content_list
from hep_rag_v2.textnorm import normalize_display_text

PDG_SCHEMA = """
CREATE TABLE IF NOT EXISTS pdg_sources (
  source_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  manifest_path TEXT,
  parsed_dir TEXT,
  block_count INTEGER NOT NULL DEFAULT 0,
  capsule_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pdg_sections (
  pdg_section_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  parent_title TEXT,
  title TEXT NOT NULL,
  clean_title TEXT,
  path_text TEXT,
  section_kind TEXT NOT NULL DEFAULT 'body',
  level INTEGER NOT NULL DEFAULT 1,
  order_index INTEGER NOT NULL DEFAULT 0,
  page_start INTEGER,
  page_end INTEGER,
  raw_text TEXT,
  capsule_text TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (source_id) REFERENCES pdg_sources(source_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pdg_sections_source ON pdg_sections(source_id, order_index);
"""


def ensure_pdg_schema(conn: sqlite3.Connection) -> None:
    for statement in PDG_SCHEMA.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)


def import_pdg_source(
    conn: sqlite3.Connection,
    *,
    source_path: str | Path,
    source_id: str,
    title: str,
    replace: bool = False,
    max_capsule_chars: int = 1200,
) -> dict[str, Any]:
    ensure_pdg_schema(conn)
    source_path = Path(source_path).expanduser().resolve()
    dest_dir = paths.PARSED_DIR / "pdg" / source_id
    manifest = import_mineru_source(source_path=source_path, dest_dir=dest_dir, replace=replace)
    manifest_path = dest_dir / "manifest.json"
    loaded_manifest = load_manifest(manifest_path)
    content_list_path = Path(str(loaded_manifest["content_list_path"]))
    items = load_content_list(content_list_path)
    parsed_blocks = parsed_blocks_from_content_list(items)
    annotated_blocks = annotate_blocks(parsed_blocks, work_title=title)
    sections = _collect_pdg_sections(annotated_blocks, title=title, max_capsule_chars=max_capsule_chars)

    with conn:
        conn.execute("DELETE FROM pdg_sections WHERE source_id = ?", (source_id,))
        conn.execute(
            """
            INSERT INTO pdg_sources (source_id, title, manifest_path, parsed_dir, block_count, capsule_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
              title = excluded.title,
              manifest_path = excluded.manifest_path,
              parsed_dir = excluded.parsed_dir,
              block_count = excluded.block_count,
              capsule_count = excluded.capsule_count,
              updated_at = CURRENT_TIMESTAMP
            """,
            (
                source_id,
                title,
                str(manifest_path),
                str(dest_dir),
                len(parsed_blocks),
                len(sections),
            ),
        )
        for idx, section in enumerate(sections, start=1):
            conn.execute(
                """
                INSERT INTO pdg_sections (
                  source_id, parent_title, title, clean_title, path_text, section_kind, level,
                  order_index, page_start, page_end, raw_text, capsule_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    section["parent_title"],
                    section["title"],
                    section["clean_title"],
                    section["path_text"],
                    section["section_kind"],
                    section["level"],
                    idx,
                    section["page_start"],
                    section["page_end"],
                    section["raw_text"],
                    section["capsule_text"],
                ),
            )
    return {
        "source_id": source_id,
        "title": title,
        "manifest_path": str(manifest_path),
        "parsed_dir": str(dest_dir),
        "block_count": len(parsed_blocks),
        "capsule_count": len(sections),
        "import_manifest": manifest,
    }


def _collect_pdg_sections(annotated_blocks: list[Any], *, title: str, max_capsule_chars: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    root_title = normalize_display_text(title) or title
    for annotated in annotated_blocks:
        if annotated.is_heading and annotated.section_kind in {"body", "appendix", "abstract"}:
            if current is not None and current["raw_text"].strip():
                current["capsule_text"] = _capsule_text(current, max_capsule_chars=max_capsule_chars)
                sections.append(current)
            heading_title = normalize_display_text(annotated.heading_clean_title or annotated.clean_text or annotated.parsed.raw_text) or root_title
            current = {
                "parent_title": root_title,
                "title": heading_title,
                "clean_title": heading_title,
                "path_text": f"{root_title} / {heading_title}",
                "section_kind": annotated.section_kind,
                "level": int(annotated.heading_level or 1),
                "page_start": annotated.parsed.page,
                "page_end": annotated.parsed.page,
                "raw_text": "",
            }
            continue
        if not annotated.is_retrievable:
            continue
        if current is None:
            continue
        snippet = str(annotated.clean_text or annotated.parsed.raw_text or "").strip()
        if not snippet:
            continue
        current["raw_text"] = (current["raw_text"] + "\n\n" + snippet).strip()
        if annotated.parsed.page is not None:
            if current["page_start"] is None:
                current["page_start"] = annotated.parsed.page
            current["page_end"] = annotated.parsed.page
    if current is not None and current["raw_text"].strip():
        current["capsule_text"] = _capsule_text(current, max_capsule_chars=max_capsule_chars)
        sections.append(current)
    return sections


def _capsule_text(section: dict[str, Any], *, max_capsule_chars: int) -> str:
    body = " ".join(str(section.get("raw_text") or "").split())
    if len(body) > max_capsule_chars:
        body = body[: max(0, max_capsule_chars - 1)].rstrip() + "…"
    return f"{section['title']}\n{body}".strip()
