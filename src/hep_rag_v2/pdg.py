from __future__ import annotations

import json
import re
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup

from hep_rag_v2 import paths
from hep_rag_v2.fulltext import annotate_blocks, import_mineru_source, load_content_list, load_manifest, parsed_blocks_from_content_list
from hep_rag_v2.physics import build_physics_substrate
from hep_rag_v2.textnorm import normalize_display_text

_CONTEXT_HEADING_RE = re.compile(r"\b(review|meson|boson|quark|lepton|neutrino|higgs|matrix|mixing|decay|physics)\b", re.IGNORECASE)
_GENERIC_CHILD_RE = re.compile(r"\b(properties|property|decays?|masses?|lifetimes?|parameters?|branching|couplings?|matrix|form factors?|widths?)\b", re.IGNORECASE)
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_TAGS = {"p", "li", "td", "th", "dd", "dt", "pre"}
_DIR_LABELS = {
    "reviews": "Reviews",
    "tables": "Summary Tables",
    "listings": "Particle Listings",
    "pdgid": "PDG Identifiers",
    "html": "PDG Meta",
}
_INCLUDED_HTML_ROOTS = {"reviews", "tables", "listings", "pdgid", "html"}
_SKIP_HTML_FILES = {
    "navigationinclude.html",
    "copyright.html",
    "copyright-1.html",
    "news.html",
}
_INCLUDED_HTML_FILES = {
    "index.html",
    "html/errata.html",
}
_INCLUDED_HTML_GLOBS = (
    "html/authors_*.html",
    "html/booklet.html",
)
_GENERIC_PAGE_TITLES = {
    "particle data group",
    "pdg",
}

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
CREATE TABLE IF NOT EXISTS pdg_artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  artifact_kind TEXT NOT NULL,
  edition TEXT,
  title TEXT,
  local_path TEXT,
  source_url TEXT,
  file_name TEXT,
  byte_size INTEGER,
  metadata_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(source_id, artifact_kind)
);
CREATE INDEX IF NOT EXISTS idx_pdg_sections_source ON pdg_sections(source_id, order_index);
CREATE INDEX IF NOT EXISTS idx_pdg_artifacts_source ON pdg_artifacts(source_id, artifact_kind);
"""

ProgressCallback = Callable[[str], None] | None


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is None:
        return
    text = str(message or "").strip()
    if text:
        progress(text)


def ensure_pdg_schema(conn: sqlite3.Connection) -> None:
    for statement in PDG_SCHEMA.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)


def register_pdg_artifact(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    artifact_kind: str,
    edition: str | None,
    title: str | None,
    local_path: str | Path | None,
    source_url: str | None,
    file_name: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_pdg_schema(conn)
    resolved_path = Path(local_path).expanduser().resolve() if local_path is not None else None
    byte_size = resolved_path.stat().st_size if resolved_path is not None and resolved_path.exists() else None
    conn.execute(
        """
        INSERT INTO pdg_artifacts (
          source_id, artifact_kind, edition, title, local_path, source_url, file_name, byte_size, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, artifact_kind) DO UPDATE SET
          edition = excluded.edition,
          title = excluded.title,
          local_path = excluded.local_path,
          source_url = excluded.source_url,
          file_name = excluded.file_name,
          byte_size = excluded.byte_size,
          metadata_json = excluded.metadata_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            str(source_id).strip(),
            str(artifact_kind).strip(),
            str(edition).strip() if edition is not None else None,
            str(title).strip() if title is not None else None,
            str(resolved_path) if resolved_path is not None else None,
            str(source_url).strip() if source_url else None,
            str(file_name).strip() if file_name else None,
            int(byte_size) if byte_size is not None else None,
            json.dumps(metadata or {}, ensure_ascii=False) if metadata is not None else None,
        ),
    )
    return {
        "source_id": str(source_id).strip(),
        "artifact_kind": str(artifact_kind).strip(),
        "edition": str(edition).strip() if edition is not None else None,
        "title": str(title).strip() if title is not None else None,
        "local_path": str(resolved_path) if resolved_path is not None else None,
        "source_url": str(source_url).strip() if source_url else None,
        "file_name": str(file_name).strip() if file_name else None,
        "byte_size": int(byte_size) if byte_size is not None else None,
        "metadata": dict(metadata or {}),
    }


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
        _replace_pdg_sections(
            conn,
            source_id=source_id,
            title=title,
            manifest_path=manifest_path,
            parsed_dir=dest_dir,
            sections=sections,
            block_count=len(parsed_blocks),
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


def import_pdg_website_source(
    conn: sqlite3.Connection,
    *,
    source_path: str | Path,
    source_id: str,
    title: str,
    replace: bool = False,
    max_capsule_chars: int = 1200,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    ensure_pdg_schema(conn)
    source_path = Path(source_path).expanduser().resolve()
    dest_dir = paths.PARSED_DIR / "pdg" / source_id
    manifest = import_pdg_website_bundle(source_path=source_path, dest_dir=dest_dir, replace=replace, progress=progress)
    site_root = Path(str(manifest["site_root"]))
    _emit_progress(progress, f"extracting PDG HTML sections from {manifest['html_count']} HTML files...")
    sections, block_count = _collect_pdg_website_sections(
        site_root=site_root,
        title=title,
        max_capsule_chars=max_capsule_chars,
    )
    manifest_path = dest_dir / "manifest.json"

    with conn:
        _replace_pdg_sections(
            conn,
            source_id=source_id,
            title=title,
            manifest_path=manifest_path,
            parsed_dir=dest_dir,
            sections=sections,
            block_count=block_count,
        )
        physics_summary = build_physics_substrate(conn, collection=None, work_ids=None)
    return {
        "source_id": source_id,
        "title": title,
        "manifest_path": str(manifest_path),
        "parsed_dir": str(dest_dir),
        "block_count": block_count,
        "capsule_count": len(sections),
        "physics": physics_summary,
        "import_manifest": manifest,
    }


def import_pdg_website_bundle(*, source_path: Path, dest_dir: Path, replace: bool = False, progress: ProgressCallback = None) -> dict[str, Any]:
    source = source_path.resolve()
    raw_dir = dest_dir / "raw"
    if raw_dir.exists():
        if not replace:
            raise FileExistsError(f"Target PDG website directory already exists: {raw_dir}")
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = [member for member in zf.infolist() if not member.is_dir()]
            total = len(members)
            _emit_progress(progress, f"extracting PDG website bundle: {total} files...")
            for index, member in enumerate(members, start=1):
                zf.extract(member, raw_dir)
                if index % 250 == 0 or index == total:
                    _emit_progress(progress, f"extracted website bundle members: {index}/{total}")
    elif source.is_dir():
        copied = 0
        for child in source.iterdir():
            target = raw_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
            copied += 1
        _emit_progress(progress, f"copied PDG website source directory entries: {copied}")
    else:
        raise ValueError(f"Unsupported PDG website source: {source}")

    site_root = _detect_pdg_website_root(raw_dir)
    html_files = [str(path.relative_to(site_root)) for path in _iter_pdg_website_html_files(site_root)]
    _emit_progress(progress, f"detected PDG website root: {site_root.name} ({len(html_files)} HTML files selected)")
    manifest = {
        "engine": "pdg_website",
        "bundle_source": str(source),
        "raw_dir": str(raw_dir),
        "site_root": str(site_root),
        "html_count": len(html_files),
        "html_files": html_files,
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _replace_pdg_sections(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    title: str,
    manifest_path: Path,
    parsed_dir: Path,
    sections: list[dict[str, Any]],
    block_count: int,
) -> None:
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
            str(parsed_dir),
            int(block_count),
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
                section.get("page_start"),
                section.get("page_end"),
                section["raw_text"],
                section["capsule_text"],
            ),
        )


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


def _collect_pdg_website_sections(
    *,
    site_root: Path,
    title: str,
    max_capsule_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    root_title = normalize_display_text(title) or title
    sections: list[dict[str, Any]] = []
    block_count = 0
    for html_path in _iter_pdg_website_html_files(site_root):
        page_sections, page_blocks = _extract_sections_from_pdg_html(
            html_path=html_path,
            site_root=site_root,
            root_title=root_title,
            max_capsule_chars=max_capsule_chars,
        )
        sections.extend(page_sections)
        block_count += page_blocks
    return sections, block_count


def _extract_sections_from_pdg_html(
    *,
    html_path: Path,
    site_root: Path,
    root_title: str,
    max_capsule_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    content_root = soup.find(id="details") or soup.find("main") or soup.body or soup
    rel_path = html_path.relative_to(site_root)
    page_title = _page_title_from_html(soup, rel_path=rel_path)
    page_segments = [root_title, *_path_segments_from_rel_path(rel_path, page_title=page_title)]

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    heading_stack: list[tuple[int, str]] = []
    block_count = 0

    for tag in content_root.find_all(list(_HEADING_TAGS | _TEXT_TAGS)):
        if tag.name in _HEADING_TAGS:
            heading_text = _clean_text(tag.get_text(" ", strip=True))
            if not heading_text or heading_text.casefold() == page_title.casefold():
                continue
            if current is not None and current["raw_text"].strip():
                current["capsule_text"] = _capsule_text(current, max_capsule_chars=max_capsule_chars)
                sections.append(current)
            heading_level = int(tag.name[1]) + 1
            heading_stack = [item for item in heading_stack if item[0] < heading_level]
            heading_stack.append((heading_level, heading_text))
            current = {
                "parent_title": heading_stack[-2][1] if len(heading_stack) >= 2 else page_title,
                "title": heading_text,
                "clean_title": heading_text,
                "path_text": " / ".join([*page_segments, *[item[1] for item in heading_stack]]),
                "section_kind": "body",
                "level": heading_level,
                "page_start": None,
                "page_end": None,
                "raw_text": "",
            }
            continue

        snippet = _clean_text(tag.get_text(" ", strip=True))
        if not snippet:
            continue
        if current is None:
            current = {
                "parent_title": page_segments[-2] if len(page_segments) >= 2 else root_title,
                "title": page_title,
                "clean_title": page_title,
                "path_text": " / ".join(page_segments),
                "section_kind": "body",
                "level": 1,
                "page_start": None,
                "page_end": None,
                "raw_text": "",
            }
        current["raw_text"] = (current["raw_text"] + "\n\n" + snippet).strip()
        block_count += 1

    if current is not None and current["raw_text"].strip():
        current["capsule_text"] = _capsule_text(current, max_capsule_chars=max_capsule_chars)
        sections.append(current)
    return sections, block_count


def _detect_pdg_website_root(raw_dir: Path) -> Path:
    candidates = []
    for index_path in raw_dir.rglob("index.html"):
        parent = index_path.parent
        if (parent / "reviews").exists() and (parent / "listings").exists() and (parent / "tables").exists():
            candidates.append(parent)
    if not candidates:
        raise ValueError(f"Could not locate PDG website root inside {raw_dir}")
    return sorted(candidates)[0]


def _iter_pdg_website_html_files(site_root: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(site_root.rglob("*.html")):
        rel_path = path.relative_to(site_root)
        rel_posix = rel_path.as_posix()
        if path.name in _SKIP_HTML_FILES:
            continue
        if rel_posix in _INCLUDED_HTML_FILES:
            selected.append(path)
            continue
        if any(rel_path.match(pattern) for pattern in _INCLUDED_HTML_GLOBS):
            selected.append(path)
            continue
        if not rel_path.parts:
            continue
        root = rel_path.parts[0]
        if root not in _INCLUDED_HTML_ROOTS:
            continue
        selected.append(path)
    return selected


def _page_title_from_html(soup: BeautifulSoup, *, rel_path: Path) -> str:
    candidates: list[str] = []
    title_tag = soup.find("title")
    if title_tag is not None:
        candidates.append(str(title_tag.get_text(" ", strip=True)))
    for selector in ("h1", "h2", "div.title", "p.title", "div.download-title"):
        node = soup.select_one(selector)
        if node is not None:
            candidates.append(str(node.get_text(" ", strip=True)))
    for candidate in candidates:
        text = _clean_text(candidate)
        if not text:
            continue
        lowered = text.casefold()
        if lowered in _GENERIC_PAGE_TITLES:
            continue
        if lowered.startswith("particle data group"):
            continue
        return text
    return _humanize_file_stem(rel_path.stem)


def _path_segments_from_rel_path(rel_path: Path, *, page_title: str) -> list[str]:
    segments: list[str] = []
    if rel_path.parts:
        root = rel_path.parts[0]
        if root in _DIR_LABELS:
            segments.append(_DIR_LABELS[root])
    if page_title:
        segments.append(page_title)
    return segments


def _humanize_file_stem(stem: str) -> str:
    text = str(stem or "").replace("_", " ").replace("-", " ")
    return normalize_display_text(text) or str(stem or "").strip() or "PDG Page"


def _clean_text(value: str) -> str:
    text = normalize_display_text(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _capsule_text(section: dict[str, Any], *, max_capsule_chars: int) -> str:
    body = " ".join(str(section.get("raw_text") or "").split())
    if len(body) > max_capsule_chars:
        body = body[: max(0, max_capsule_chars - 1)].rstrip() + "…"
    return f"{section['title']}\n{body}".strip()
