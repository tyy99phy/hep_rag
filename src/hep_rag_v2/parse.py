from __future__ import annotations

import copy
import json
import sqlite3
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hep_rag_v2 import paths
from hep_rag_v2.fulltext import import_mineru_source, load_content_list, materialize_mineru_document
from hep_rag_v2.records import paper_storage_stem, parsed_doc_dir

ProgressCallback = Callable[[str], None] | None
_PAGE_NUMBER_KEYS = ("page_idx", "page", "page_no")
_ASSET_PATH_KEYS = ("image_path", "img_path", "path")


@dataclass(frozen=True)
class _PdfSplitPart:
    index: int
    start_page: int
    end_page: int
    pdf_path: Path


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is not None:
        progress(message)


def _parse_with_mineru(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    client: Any,
    work_id: int,
    pdf_path: Path,
    collection_name: str,
    replace_existing: bool,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    stem = paper_storage_stem(conn, work_id)
    dest_dir = parsed_doc_dir(collection_name, stem)
    manifest_path = dest_dir / "manifest.json"
    document_row = _document_row(conn, work_id=work_id)
    if manifest_path.exists() and not replace_existing:
        if document_row is not None and str(document_row["parse_status"] or "") == "materialized":
            _emit_progress(progress, "using cached parsed document; skipping MinerU submission.")
            return {
                "ok": True,
                "work_id": work_id,
                "title": _work_title(conn, work_id),
                "state": "cached",
                "parsed_dir": str(dest_dir),
            }
        _emit_progress(progress, "found cached manifest; materializing into the database...")
        return _materialize_existing_manifest(
            conn,
            config=config,
            work_id=work_id,
            manifest_path=manifest_path,
            replace_existing=document_row is not None,
        )

    raw_zip_dir = paths.RAW_DIR / "mineru" / collection_name
    raw_zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_zip_dir / f"{stem}.zip"
    _upsert_document_parse_record(
        conn,
        work_id=work_id,
        parse_status="submitted",
        parsed_dir=dest_dir,
        manifest_path=manifest_path,
        parse_error=None,
    )
    mineru_cfg = config.get("mineru") or {}
    strategy = str(mineru_cfg.get("oversize_strategy") or "split").strip().lower()
    page_limit = max(0, int(mineru_cfg.get("max_pages_per_pdf") or 0))
    page_count = _count_pdf_pages(pdf_path) if page_limit > 0 else 0
    if strategy == "split" and page_limit > 0 and page_count > page_limit:
        _emit_progress(
            progress,
            f"{pdf_path.name} has {page_count} pages; splitting into <= {page_limit}-page MinerU jobs...",
        )
        source_path = _parse_with_mineru_split_pdf(
            client,
            pdf_path=pdf_path,
            stem=stem,
            raw_zip_dir=raw_zip_dir,
            max_pages_per_part=page_limit,
            progress=progress,
        )
    else:
        _emit_progress(progress, f"submitting {pdf_path.name} to MinerU...")
        task = client.submit_local_pdf(pdf_path, data_id=stem, progress=progress)
        client.download_result_zip(task, output_path=zip_path, progress=progress)
        source_path = zip_path
    _emit_progress(progress, f"importing parsed bundle into {dest_dir.name}...")
    import_mineru_source(source_path=source_path, dest_dir=dest_dir, replace=replace_existing)

    _emit_progress(progress, "materializing parsed manifest into SQLite...")
    result = _materialize_existing_manifest(
        conn,
        config=config,
        work_id=work_id,
        manifest_path=manifest_path,
        replace_existing=True,
    )
    result["state"] = "materialized"
    result["bundle_path"] = str(source_path)
    if source_path == zip_path:
        result["zip_path"] = str(zip_path)
    if strategy == "split" and page_limit > 0 and page_count > page_limit:
        result["split_pdf_pages"] = page_count
        result["split_page_limit"] = page_limit
    _emit_progress(progress, "materialized parsed document.")
    return result


def _materialize_existing_manifest(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    work_id: int,
    manifest_path: Path,
    replace_existing: bool,
) -> dict[str, Any]:
    ingest_cfg = config.get("ingest") or {}
    summary = materialize_mineru_document(
        conn,
        work_id=work_id,
        manifest_path=manifest_path,
        replace=replace_existing,
        chunk_size=int(ingest_cfg.get("chunk_size") or 2400),
        overlap_blocks=int(ingest_cfg.get("overlap_blocks") or 1),
        section_parent_char_limit=int(ingest_cfg.get("section_parent_char_limit") or 12000),
    )
    _upsert_document_parse_record(
        conn,
        work_id=work_id,
        parse_status="materialized",
        parsed_dir=manifest_path.parent,
        manifest_path=manifest_path,
        parse_error=None,
    )
    return {
        "ok": True,
        "work_id": work_id,
        "title": _work_title(conn, work_id),
        "state": "materialized",
        "parsed_dir": str(manifest_path.parent),
        "document": summary,
    }


def _document_row(conn: sqlite3.Connection, *, work_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT document_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path, parse_error
        FROM documents
        WHERE work_id = ?
        ORDER BY document_id DESC
        LIMIT 1
        """,
        (work_id,),
    ).fetchone()


def _upsert_document_parse_record(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    parse_status: str,
    parsed_dir: Path | None,
    manifest_path: Path | None,
    parse_error: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO documents (
          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path, parse_error, last_parse_attempt_at, updated_at
        ) VALUES (?, 'mineru', 'v2-contract', ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(work_id) DO UPDATE SET
          parser_name = excluded.parser_name,
          parser_version = excluded.parser_version,
          parse_status = excluded.parse_status,
          parsed_dir = excluded.parsed_dir,
          manifest_path = excluded.manifest_path,
          parse_error = excluded.parse_error,
          last_parse_attempt_at = CURRENT_TIMESTAMP,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            work_id,
            parse_status,
            str(parsed_dir) if parsed_dir is not None else None,
            str(manifest_path) if manifest_path is not None else None,
            str(parse_error) if parse_error else None,
        ),
    )


def _mark_document_parse_failed(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    collection_name: str,
    error: str,
) -> None:
    dest_dir = parsed_doc_dir(collection_name, paper_storage_stem(conn, work_id))
    manifest_path = dest_dir / "manifest.json"
    _upsert_document_parse_record(
        conn,
        work_id=work_id,
        parse_status="failed",
        parsed_dir=dest_dir,
        manifest_path=manifest_path if manifest_path.exists() else None,
        parse_error=error,
    )


def _work_title(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute("SELECT title FROM works WHERE work_id = ?", (work_id,)).fetchone()
    return str(row["title"]) if row is not None else None


def _prefixed_progress(progress: ProgressCallback, prefix: str) -> ProgressCallback:
    if progress is None:
        return None
    label = str(prefix or "").strip()
    if not label:
        return progress

    def _inner(message: str) -> None:
        text = str(message or "").strip()
        if text:
            progress(f"{label}: {text}")

    return _inner


def _count_pdf_pages(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised in environment, not unit tests
        raise RuntimeError(
            "pypdf is required for MinerU oversized-PDF handling. Reinstall hep-rag after syncing dependencies."
        ) from exc
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def _split_pdf_for_mineru(*, pdf_path: Path, output_dir: Path, max_pages_per_part: int) -> list[_PdfSplitPart]:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - exercised in environment, not unit tests
        raise RuntimeError(
            "pypdf is required for MinerU oversized-PDF handling. Reinstall hep-rag after syncing dependencies."
        ) from exc
    if max_pages_per_part <= 0:
        raise ValueError("max_pages_per_part must be positive")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    parts: list[_PdfSplitPart] = []
    for idx, start in enumerate(range(0, total_pages, max_pages_per_part), start=1):
        end = min(start + max_pages_per_part, total_pages)
        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])
        part_path = output_dir / f"{pdf_path.stem}.part{idx:03d}.pdf"
        with part_path.open("wb") as handle:
            writer.write(handle)
        parts.append(
            _PdfSplitPart(
                index=idx,
                start_page=start + 1,
                end_page=end,
                pdf_path=part_path,
            )
        )
    return parts


def _parse_with_mineru_split_pdf(
    client: Any,
    *,
    pdf_path: Path,
    stem: str,
    raw_zip_dir: Path,
    max_pages_per_part: int,
    progress: ProgressCallback = None,
) -> Path:
    split_pdf_dir = raw_zip_dir / f"{stem}__split_pdf"
    combined_bundle_dir = raw_zip_dir / f"{stem}__combined_bundle"
    split_parts = _split_pdf_for_mineru(
        pdf_path=pdf_path,
        output_dir=split_pdf_dir,
        max_pages_per_part=max_pages_per_part,
    )
    _emit_progress(
        progress,
        f"created {len(split_parts)} PDF parts for MinerU ({max_pages_per_part} pages per part).",
    )

    part_zip_paths: list[Path] = []
    for part in split_parts:
        label = f"part {part.index}/{len(split_parts)} pages {part.start_page}-{part.end_page}"
        _emit_progress(progress, f"submitting {label} to MinerU...")
        task = client.submit_local_pdf(
            part.pdf_path,
            data_id=f"{stem}-part-{part.index:03d}",
            progress=_prefixed_progress(progress, label),
        )
        part_zip_path = raw_zip_dir / f"{stem}.part{part.index:03d}.zip"
        client.download_result_zip(
            task,
            output_path=part_zip_path,
            progress=_prefixed_progress(progress, label),
        )
        part_zip_paths.append(part_zip_path)

    _emit_progress(progress, f"merging {len(part_zip_paths)} MinerU part bundles...")
    return _merge_mineru_part_bundles(
        bundle_paths=part_zip_paths,
        split_parts=split_parts,
        output_dir=combined_bundle_dir,
    )


def _merge_mineru_part_bundles(
    *,
    bundle_paths: list[Path],
    split_parts: list[_PdfSplitPart],
    output_dir: Path,
) -> Path:
    if len(bundle_paths) != len(split_parts):
        raise ValueError("bundle_paths and split_parts must have the same length")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_items: list[dict[str, Any]] = []
    combined_markdown: list[str] = []
    for part, bundle_path in zip(split_parts, bundle_paths):
        part_dir = output_dir / f"part_{part.index:03d}"
        _extract_bundle(bundle_path=bundle_path, output_dir=part_dir)
        content_list_path = _first_bundle_match(part_dir, "*content_list.json")
        if content_list_path is None:
            raise ValueError(f"MinerU bundle is missing '*content_list.json': {bundle_path}")
        full_md_path = _first_bundle_match(part_dir, "*full.md")
        items = load_content_list(content_list_path)
        page_offset = _page_offset_for_items(items, start_page=part.start_page)
        combined_items.extend(
            _rewrite_bundle_items(
                items,
                part_dir=part_dir,
                page_offset=page_offset,
            )
        )
        combined_markdown.append(
            f"<!-- part {part.index} pages {part.start_page}-{part.end_page} -->\n"
        )
        if full_md_path is not None:
            combined_markdown.append(full_md_path.read_text(encoding="utf-8"))
        combined_markdown.append("\n\n")

    (output_dir / "000_combined_content_list.json").write_text(
        json.dumps(combined_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "000_combined_full.md").write_text(
        "".join(combined_markdown).strip() + "\n",
        encoding="utf-8",
    )
    return output_dir


def _extract_bundle(*, bundle_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle_path.is_file() and bundle_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(output_dir)
        return
    if bundle_path.is_dir():
        for child in bundle_path.iterdir():
            target = output_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
        return
    raise ValueError(f"Unsupported MinerU bundle type: {bundle_path}")


def _first_bundle_match(root: Path, pattern: str) -> Path | None:
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            return path
    return None


def _page_offset_for_items(items: list[dict[str, Any]], *, start_page: int) -> int:
    page_values: list[int] = []
    for item in items:
        for key in _PAGE_NUMBER_KEYS:
            value = item.get(key)
            try:
                page_values.append(int(value))
                break
            except (TypeError, ValueError):
                continue
    if not page_values:
        return max(0, start_page - 1)
    return int(start_page) - min(page_values)


def _rewrite_bundle_items(
    items: list[dict[str, Any]],
    *,
    part_dir: Path,
    page_offset: int,
) -> list[dict[str, Any]]:
    part_prefix = part_dir.name
    rewritten: list[dict[str, Any]] = []
    for item in items:
        payload = copy.deepcopy(item)
        for key in _PAGE_NUMBER_KEYS:
            value = payload.get(key)
            try:
                payload[key] = int(value) + page_offset
            except (TypeError, ValueError):
                continue
        for key in _ASSET_PATH_KEYS:
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            relative = _normalize_relative_asset_path(value)
            if relative is None:
                continue
            candidate = (part_dir / relative).resolve()
            try:
                candidate.relative_to(part_dir.resolve())
            except ValueError:
                continue
            if candidate.exists():
                payload[key] = f"{part_prefix}/{relative.as_posix()}"
        rewritten.append(payload)
    return rewritten


def _normalize_relative_asset_path(value: str) -> Path | None:
    text = str(value or "").strip()
    if not text or "://" in text:
        return None
    path = Path(text)
    if path.is_absolute():
        return None
    parts = [segment for segment in path.parts if segment not in {"", "."}]
    if any(segment == ".." for segment in parts):
        return None
    if not parts:
        return None
    return Path(*parts)
