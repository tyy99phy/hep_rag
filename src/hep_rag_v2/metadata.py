from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_collection_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_collection(conn: sqlite3.Connection, config: dict[str, Any]) -> int:
    name = str(config.get("name") or "").strip()
    if not name:
        raise ValueError("Collection config is missing 'name'.")
    conn.execute(
        """
        INSERT INTO collections (name, label, notes, source_priority_json, raw_config_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
          label = excluded.label,
          notes = excluded.notes,
          source_priority_json = excluded.source_priority_json,
          raw_config_json = excluded.raw_config_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            name,
            str(config.get("label") or "").strip() or None,
            str(config.get("notes") or "").strip() or None,
            json.dumps(config.get("source_priority") or [], ensure_ascii=False),
            json.dumps(config, ensure_ascii=False),
        ),
    )
    row = conn.execute("SELECT collection_id FROM collections WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert collection: {name}")
    return int(row["collection_id"])


def find_work_id(conn: sqlite3.Connection, *, id_type: str, id_value: str) -> int | None:
    row = conn.execute(
        "SELECT work_id FROM work_ids WHERE id_type = ? AND id_value = ?",
        (id_type, id_value),
    ).fetchone()
    if row is None:
        return None
    return int(row["work_id"])


def upsert_work_from_hit(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    hit: dict[str, Any],
) -> dict[str, int]:
    metadata = hit.get("metadata") or {}
    title = first_title(metadata)
    if not title:
        return {"created": 0, "updated": 0, "citations_written": 0, "skipped": 1}

    canonical_source, canonical_id = canonical_identity(metadata)
    work_id = _find_existing_work_id(conn, metadata, canonical_source=canonical_source, canonical_id=canonical_id)
    payload = {
        "canonical_source": canonical_source,
        "canonical_id": canonical_id,
        "title": title,
        "abstract": first_abstract(metadata),
        "year": year_from_metadata(metadata),
        "citation_count": _coerce_int(metadata.get("citation_count")),
        "primary_source_url": source_url_from_hit(hit, metadata),
        "primary_pdf_url": pdf_url_from_metadata(metadata),
        "raw_metadata_json": json.dumps(metadata, ensure_ascii=False),
    }

    created = 0
    updated = 0
    if work_id is None:
        cur = conn.execute(
            """
            INSERT INTO works (
              canonical_source, canonical_id, title, abstract, year, citation_count,
              primary_source_url, primary_pdf_url, raw_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["canonical_source"],
                payload["canonical_id"],
                payload["title"],
                payload["abstract"],
                payload["year"],
                payload["citation_count"],
                payload["primary_source_url"],
                payload["primary_pdf_url"],
                payload["raw_metadata_json"],
            ),
        )
        work_id = int(cur.lastrowid)
        created = 1
    else:
        conn.execute(
            """
            UPDATE works
            SET canonical_source = ?, canonical_id = ?, title = ?, abstract = ?, year = ?,
                citation_count = ?, primary_source_url = ?, primary_pdf_url = ?,
                raw_metadata_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE work_id = ?
            """,
            (
                payload["canonical_source"],
                payload["canonical_id"],
                payload["title"],
                payload["abstract"],
                payload["year"],
                payload["citation_count"],
                payload["primary_source_url"],
                payload["primary_pdf_url"],
                payload["raw_metadata_json"],
                work_id,
            ),
        )
        updated = 1

    _upsert_work_ids(conn, work_id=work_id, metadata=metadata, canonical_source=canonical_source, canonical_id=canonical_id)
    conn.execute(
        "INSERT OR IGNORE INTO collection_works (collection_id, work_id) VALUES (?, ?)",
        (collection_id, work_id),
    )
    _rewrite_work_authors(conn, work_id=work_id, metadata=metadata)
    _rewrite_work_collaborations(conn, work_id=work_id, metadata=metadata)
    _rewrite_work_venues(conn, work_id=work_id, metadata=metadata)
    _rewrite_work_topics(conn, work_id=work_id, metadata=metadata)
    citations_written = _rewrite_citations(conn, work_id=work_id, metadata=metadata)
    return {
        "created": created,
        "updated": updated,
        "citations_written": citations_written,
        "skipped": 0,
    }


def backfill_unresolved_citations(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT citation_id, dst_source, dst_external_id
        FROM citations
        WHERE dst_work_id IS NULL
          AND dst_source IS NOT NULL
          AND dst_external_id IS NOT NULL
        """
    ).fetchall()
    resolved = 0
    for row in rows:
        work_id = find_work_id(
            conn,
            id_type=str(row["dst_source"]).strip(),
            id_value=str(row["dst_external_id"]).strip(),
        )
        if work_id is None:
            continue
        conn.execute(
            """
            UPDATE citations
            SET dst_work_id = ?, resolution_status = 'resolved'
            WHERE citation_id = ?
            """,
            (work_id, int(row["citation_id"])),
        )
        resolved += 1
    return resolved


def canonical_identity(metadata: dict[str, Any]) -> tuple[str, str]:
    inspire_id = _control_number(metadata)
    if inspire_id:
        return ("inspire", inspire_id)
    arxiv_id = first_arxiv_id(metadata)
    if arxiv_id:
        return ("arxiv", arxiv_id)
    doi = first_doi(metadata)
    if doi:
        return ("doi", doi)
    fingerprint = hashlib.sha1(
        f"{first_title(metadata) or ''}|{year_from_metadata(metadata) or ''}".encode("utf-8")
    ).hexdigest()
    return ("local", fingerprint)


def first_title(metadata: dict[str, Any]) -> str | None:
    for item in metadata.get("titles") or []:
        if isinstance(item, dict) and item.get("title"):
            return str(item["title"]).strip()
    return None


def first_abstract(metadata: dict[str, Any]) -> str | None:
    for item in metadata.get("abstracts") or []:
        if isinstance(item, dict) and item.get("value"):
            return str(item["value"]).strip()
    return None


def first_arxiv_id(metadata: dict[str, Any]) -> str | None:
    for item in metadata.get("arxiv_eprints") or []:
        if isinstance(item, dict) and item.get("value"):
            return str(item["value"]).strip()
    return None


def first_doi(metadata: dict[str, Any]) -> str | None:
    for item in metadata.get("dois") or []:
        if isinstance(item, dict) and item.get("value"):
            return str(item["value"]).strip().lower()
    return None


def year_from_metadata(metadata: dict[str, Any]) -> int | None:
    for item in metadata.get("publication_info") or []:
        if isinstance(item, dict):
            year = _coerce_int(item.get("year"))
            if year is not None:
                return year
    for key in ("preprint_date", "earliest_date"):
        value = str(metadata.get(key) or "").strip()
        if len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def source_url_from_hit(hit: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    links = hit.get("links") or {}
    if isinstance(links, dict) and links.get("self"):
        return str(links["self"]).strip()
    control_number = _control_number(metadata)
    if control_number:
        return f"https://inspirehep.net/literature/{control_number}"
    return None


def pdf_url_from_metadata(metadata: dict[str, Any]) -> str | None:
    arxiv_id = first_arxiv_id(metadata)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    for item in metadata.get("documents") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if url.lower().endswith(".pdf") or "/files/" in url:
            return url
    for item in metadata.get("files") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("file") or item.get("path") or "").strip()
        if url.lower().endswith(".pdf") or ".pdf?" in url.lower() or "/pdf/" in url.lower():
            return url
    return None


def _find_existing_work_id(
    conn: sqlite3.Connection,
    metadata: dict[str, Any],
    *,
    canonical_source: str,
    canonical_id: str,
) -> int | None:
    row = conn.execute(
        "SELECT work_id FROM works WHERE canonical_source = ? AND canonical_id = ?",
        (canonical_source, canonical_id),
    ).fetchone()
    if row is not None:
        return int(row["work_id"])
    candidates = [
        ("inspire", _control_number(metadata)),
        ("arxiv", first_arxiv_id(metadata)),
        ("doi", first_doi(metadata)),
    ]
    for id_type, id_value in candidates:
        if not id_value:
            continue
        work_id = find_work_id(conn, id_type=id_type, id_value=id_value)
        if work_id is not None:
            return work_id
    return None


def _upsert_work_ids(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    metadata: dict[str, Any],
    canonical_source: str,
    canonical_id: str,
) -> None:
    ids = [
        (canonical_source, canonical_id, 1),
        ("inspire", _control_number(metadata), 1 if canonical_source == "inspire" else 0),
        ("arxiv", first_arxiv_id(metadata), 1 if canonical_source == "arxiv" else 0),
        ("doi", first_doi(metadata), 1 if canonical_source == "doi" else 0),
    ]
    for id_type, id_value, is_primary in ids:
        if not id_value:
            continue
        conn.execute(
            """
            INSERT INTO work_ids (id_type, id_value, work_id, is_primary)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id_type, id_value) DO UPDATE SET
              work_id = excluded.work_id,
              is_primary = excluded.is_primary
            """,
            (id_type, id_value, work_id, is_primary),
        )


def _rewrite_work_authors(conn: sqlite3.Connection, *, work_id: int, metadata: dict[str, Any]) -> None:
    conn.execute("DELETE FROM work_authors WHERE work_id = ?", (work_id,))
    for idx, item in enumerate(metadata.get("authors") or [], start=1):
        if not isinstance(item, dict):
            continue
        display_name = str(item.get("full_name") or "").strip()
        if not display_name:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO authors (display_name, raw_json) VALUES (?, ?)",
            (display_name, json.dumps(item, ensure_ascii=False)),
        )
        row = conn.execute(
            "SELECT author_id FROM authors WHERE display_name = ?",
            (display_name,),
        ).fetchone()
        if row is None:
            continue
        affiliations = []
        for aff in item.get("affiliations") or []:
            if isinstance(aff, dict) and aff.get("value"):
                affiliations.append(str(aff["value"]).strip())
        conn.execute(
            """
            INSERT INTO work_authors (work_id, author_position, author_id, affiliations_json, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                work_id,
                idx,
                int(row["author_id"]),
                json.dumps(affiliations, ensure_ascii=False),
                json.dumps(item, ensure_ascii=False),
            ),
        )


def _rewrite_work_collaborations(conn: sqlite3.Connection, *, work_id: int, metadata: dict[str, Any]) -> None:
    conn.execute("DELETE FROM work_collaborations WHERE work_id = ?", (work_id,))
    for item in metadata.get("collaborations") or []:
        value = None
        if isinstance(item, dict):
            value = item.get("value") or item.get("name")
        elif isinstance(item, str):
            value = item
        name = str(value or "").strip()
        if not name:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO collaborations (name) VALUES (?)",
            (name,),
        )
        row = conn.execute(
            "SELECT collaboration_id FROM collaborations WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO work_collaborations (work_id, collaboration_id) VALUES (?, ?)",
            (work_id, int(row["collaboration_id"])),
        )


def _rewrite_work_venues(conn: sqlite3.Connection, *, work_id: int, metadata: dict[str, Any]) -> None:
    conn.execute("DELETE FROM work_venues WHERE work_id = ?", (work_id,))
    seen: set[str] = set()
    for item in metadata.get("publication_info") or []:
        if not isinstance(item, dict):
            continue
        for key, venue_type in (("journal_title", "journal"), ("conference_title", "conference"), ("pubinfo_freetext", "other")):
            value = str(item.get(key) or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            conn.execute(
                "INSERT OR IGNORE INTO venues (name, venue_type, raw_json) VALUES (?, ?, ?)",
                (value, venue_type, json.dumps(item, ensure_ascii=False)),
            )
            row = conn.execute(
                "SELECT venue_id FROM venues WHERE name = ?",
                (value,),
            ).fetchone()
            if row is None:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO work_venues (work_id, venue_id, is_primary, raw_json)
                VALUES (?, ?, ?, ?)
                """,
                (work_id, int(row["venue_id"]), 1 if len(seen) == 1 else 0, json.dumps(item, ensure_ascii=False)),
            )


def _rewrite_work_topics(conn: sqlite3.Connection, *, work_id: int, metadata: dict[str, Any]) -> None:
    conn.execute("DELETE FROM work_topics WHERE work_id = ?", (work_id,))
    seen: set[tuple[str, str]] = set()
    for source, topic_key, label, raw_json in _topic_rows(metadata):
        key = (source, topic_key)
        if key in seen:
            continue
        seen.add(key)
        conn.execute(
            """
            INSERT INTO topics (source, topic_key, label)
            VALUES (?, ?, ?)
            ON CONFLICT(source, topic_key) DO UPDATE SET
              label = excluded.label
            """,
            (source, topic_key, label),
        )
        row = conn.execute(
            "SELECT topic_id FROM topics WHERE source = ? AND topic_key = ?",
            (source, topic_key),
        ).fetchone()
        if row is None:
            continue
        conn.execute(
            """
            INSERT INTO work_topics (work_id, topic_id, score, raw_json)
            VALUES (?, ?, ?, ?)
            """,
            (work_id, int(row["topic_id"]), None, raw_json),
        )


def _rewrite_citations(conn: sqlite3.Connection, *, work_id: int, metadata: dict[str, Any]) -> int:
    conn.execute("DELETE FROM citations WHERE src_work_id = ?", (work_id,))
    written = 0
    seen: set[tuple[str, str]] = set()
    for item in metadata.get("references") or []:
        if not isinstance(item, dict):
            continue
        target = _reference_target(item)
        if target is None:
            continue
        dst_source, dst_external_id = target
        key = (dst_source, dst_external_id)
        if key in seen:
            continue
        seen.add(key)
        dst_work_id = find_work_id(conn, id_type=dst_source, id_value=dst_external_id)
        resolution_status = "resolved" if dst_work_id is not None else "unresolved"
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
                json.dumps(item, ensure_ascii=False),
                resolution_status,
            ),
        )
        written += 1
    return written


def _reference_target(item: dict[str, Any]) -> tuple[str, str] | None:
    control_number = item.get("control_number")
    if control_number is not None:
        return ("inspire", str(control_number).strip())
    record = item.get("record") or {}
    if isinstance(record, dict):
        ref_value = str(record.get("$ref") or "").strip()
        if ref_value:
            return ("inspire", ref_value.rstrip("/").split("/")[-1])
    for doi in item.get("dois") or []:
        if isinstance(doi, dict) and doi.get("value"):
            return ("doi", str(doi["value"]).strip().lower())
    arxiv_id = str(item.get("arxiv_eprint") or item.get("arxiv_id") or "").strip()
    if arxiv_id:
        return ("arxiv", arxiv_id)
    return None


def _topic_rows(metadata: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for item in metadata.get("keywords") or []:
        value = None
        if isinstance(item, dict):
            value = item.get("value") or item.get("term")
        elif isinstance(item, str):
            value = item
        label = str(value or "").strip()
        if label:
            rows.append(("keyword", label.casefold(), label, json.dumps(item, ensure_ascii=False)))
    for item in metadata.get("inspire_categories") or []:
        label = None
        if isinstance(item, dict):
            label = item.get("term")
        elif isinstance(item, str):
            label = item
        label_text = str(label or "").strip()
        if label_text:
            rows.append(("category", label_text.casefold(), label_text, json.dumps(item, ensure_ascii=False)))
    for item in metadata.get("accelerator_experiments") or []:
        label = None
        if isinstance(item, dict):
            label = item.get("experiment")
        elif isinstance(item, str):
            label = item
        label_text = str(label or "").strip()
        if label_text:
            rows.append(("experiment", label_text.casefold(), label_text, json.dumps(item, ensure_ascii=False)))
    return rows


def _control_number(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("control_number")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
