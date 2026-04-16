from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from hep_rag_v2 import paths
from hep_rag_v2.fulltext import annotate_blocks, import_mineru_source, load_content_list, load_manifest, parsed_blocks_from_content_list
from hep_rag_v2.physics import build_physics_substrate
from hep_rag_v2.textnorm import normalize_display_text

_CONTEXT_HEADING_RE = re.compile(r"\b(review|meson|boson|quark|lepton|neutrino|higgs|matrix|mixing|decay|physics)\b", re.IGNORECASE)
_GENERIC_CHILD_RE = re.compile(r"\b(properties|property|decays?|masses?|lifetimes?|parameters?|branching|couplings?|matrix|form factors?|widths?)\b", re.IGNORECASE)

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
        physics_summary = build_physics_substrate(conn, collection=None, work_ids=None)
    return {
        "source_id": source_id,
        "title": title,
        "manifest_path": str(manifest_path),
        "parsed_dir": str(dest_dir),
        "block_count": len(parsed_blocks),
        "capsule_count": len(sections),
        "physics": physics_summary,
        "import_manifest": manifest,
    }


def _collect_pdg_sections(annotated_blocks: list[Any], *, title: str, max_capsule_chars: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    root_title = normalize_display_text(title) or title
    heading_stack: list[tuple[int, str]] = []
    latent_context_title: str | None = None
    for annotated in annotated_blocks:
        heading_title = normalize_display_text(annotated.heading_clean_title or annotated.clean_text or annotated.parsed.raw_text) or root_title
        if annotated.is_heading and annotated.section_kind not in {"body", "appendix", "abstract"}:
            if _CONTEXT_HEADING_RE.search(heading_title):
                latent_context_title = heading_title
            continue
        if annotated.is_heading and annotated.section_kind in {"body", "appendix", "abstract"}:
            promote_as_child = False
            if current is not None and current["raw_text"].strip():
                current["capsule_text"] = _capsule_text(current, max_capsule_chars=max_capsule_chars)
                sections.append(current)
            heading_level = max(1, int(annotated.heading_level or 1))
            if current is not None and not current["raw_text"].strip():
                current_title = str(current.get("title") or "")
                promote_as_child = (
                    bool(_CONTEXT_HEADING_RE.search(current_title))
                    and bool(_GENERIC_CHILD_RE.search(heading_title))
                )
            if latent_context_title and heading_title != latent_context_title:
                if not heading_stack:
                    heading_stack = [(1, latent_context_title)]
                    heading_level = max(2, heading_level)
                elif heading_stack[0][1] != latent_context_title:
                    heading_stack = [(1, latent_context_title), *heading_stack]
                    heading_level = max(2, heading_level)
            if promote_as_child and heading_stack:
                heading_level = max(heading_level + 1, int(heading_stack[-1][0]) + 1)
            else:
                heading_stack = [item for item in heading_stack if item[0] < heading_level]
                if latent_context_title and heading_title != latent_context_title and not heading_stack:
                    heading_stack = [(1, latent_context_title)]
                    heading_level = max(2, heading_level)
            heading_stack.append((heading_level, heading_title))
            path_titles = [root_title, *[item[1] for item in heading_stack]]
            current = {
                "parent_title": heading_stack[-2][1] if len(heading_stack) >= 2 else root_title,
                "title": heading_title,
                "clean_title": heading_title,
                "path_text": " / ".join(path_titles),
                "section_kind": annotated.section_kind,
                "level": heading_level,
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
