from __future__ import annotations

import hashlib
import json
import re
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
    _assign_work_family(conn, work_id=work_id, metadata=metadata)
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
    fingerprint = hashlib.sha256(
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


FAMILY_TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FAMILY_TITLE_STOPWORDS = {
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
FAMILY_STAGE_NEUTRAL_TOKENS = {
    "pas",
    "prelim",
    "preliminary",
    "public",
    "internal",
    "analysis",
    "note",
    "notes",
}
FAMILY_MEMBER_ROLE_ORDER = {
    "article": 6,
    "review": 5,
    "note": 4,
    "proceedings": 3,
    "preprint": 2,
    "standalone": 1,
}


def expand_work_ids_with_family(conn: sqlite3.Connection, *, work_ids: list[int]) -> list[int]:
    ordered_work_ids = [int(item) for item in work_ids]
    if not ordered_work_ids:
        return []

    placeholders = ",".join("?" for _ in ordered_work_ids)
    rows = conn.execute(
        f"""
        SELECT
          seed.work_id AS seed_work_id,
          member.work_id AS related_work_id,
          family.primary_work_id
        FROM work_family_members seed
        JOIN work_families family ON family.family_id = seed.family_id
        JOIN work_family_members member ON member.family_id = family.family_id
        WHERE seed.work_id IN ({placeholders})
        ORDER BY seed.work_id, CASE WHEN member.work_id = family.primary_work_id THEN 0 ELSE 1 END, member.work_id
        """,
        ordered_work_ids,
    ).fetchall()

    by_seed: dict[int, list[int]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed_work_id"]), []).append(int(row["related_work_id"]))

    expanded: list[int] = []
    seen: set[int] = set()
    for work_id in ordered_work_ids:
        candidates = by_seed.get(work_id) or [work_id]
        for candidate_id in candidates:
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            expanded.append(candidate_id)
    return expanded


def family_payload_map(conn: sqlite3.Connection, *, work_ids: list[int]) -> dict[int, dict[str, Any]]:
    ordered_work_ids = [int(item) for item in work_ids]
    if not ordered_work_ids:
        return {}

    placeholders = ",".join("?" for _ in ordered_work_ids)
    membership_rows = conn.execute(
        f"""
        SELECT
          m.work_id,
          m.family_id,
          m.member_role,
          m.confidence,
          m.reason_json,
          f.family_key,
          f.label,
          f.primary_work_id,
          f.relation_kind
        FROM work_family_members m
        JOIN work_families f ON f.family_id = m.family_id
        WHERE m.work_id IN ({placeholders})
        """,
        ordered_work_ids,
    ).fetchall()
    if not membership_rows:
        return {
            work_id: {
                "family_id": None,
                "family_key": None,
                "family_label": None,
                "family_primary_work_id": work_id,
                "family_relation_kind": "standalone",
                "family_member_role": "standalone",
                "family_confidence": 1.0,
                "family_size": 1,
                "related_versions": [],
            }
            for work_id in ordered_work_ids
        }

    membership_map = {int(row["work_id"]): dict(row) for row in membership_rows}
    family_ids = sorted({int(row["family_id"]) for row in membership_rows})
    family_placeholders = ",".join("?" for _ in family_ids)
    family_member_rows = conn.execute(
        f"""
        SELECT
          m.family_id,
          m.work_id,
          m.member_role,
          w.title,
          w.year,
          w.canonical_source,
          w.canonical_id,
          w.primary_source_url,
          w.primary_pdf_url
        FROM work_family_members m
        JOIN works w ON w.work_id = m.work_id
        WHERE m.family_id IN ({family_placeholders})
        ORDER BY m.family_id, w.work_id
        """,
        family_ids,
    ).fetchall()

    members_by_family: dict[int, list[dict[str, Any]]] = {}
    for row in family_member_rows:
        members_by_family.setdefault(int(row["family_id"]), []).append(
            {
                "work_id": int(row["work_id"]),
                "member_role": str(row["member_role"] or "standalone"),
                "title": row["title"],
                "year": row["year"],
                "canonical_source": row["canonical_source"],
                "canonical_id": row["canonical_id"],
                "primary_source_url": row["primary_source_url"],
                "primary_pdf_url": row["primary_pdf_url"],
            }
        )

    out: dict[int, dict[str, Any]] = {}
    for work_id in ordered_work_ids:
        membership = membership_map.get(work_id)
        if membership is None:
            out[work_id] = {
                "family_id": None,
                "family_key": None,
                "family_label": None,
                "family_primary_work_id": work_id,
                "family_relation_kind": "standalone",
                "family_member_role": "standalone",
                "family_confidence": 1.0,
                "family_size": 1,
                "related_versions": [],
            }
            continue

        family_id = int(membership["family_id"])
        primary_work_id = int(membership["primary_work_id"] or work_id)
        members = list(members_by_family.get(family_id) or [])
        members.sort(
            key=lambda item: (
                0 if int(item["work_id"]) == primary_work_id else 1,
                -FAMILY_MEMBER_ROLE_ORDER.get(str(item["member_role"] or "standalone"), 0),
                int(item["work_id"]),
            )
        )
        related_versions = [item for item in members if int(item["work_id"]) != work_id]
        out[work_id] = {
            "family_id": family_id,
            "family_key": membership["family_key"],
            "family_label": membership["label"],
            "family_primary_work_id": primary_work_id,
            "family_relation_kind": str(membership["relation_kind"] or "standalone"),
            "family_member_role": str(membership["member_role"] or "standalone"),
            "family_confidence": float(membership["confidence"] or 1.0),
            "family_size": len(members),
            "related_versions": related_versions,
        }
    return out


def _assign_work_family(conn: sqlite3.Connection, *, work_id: int, metadata: dict[str, Any]) -> int:
    existing_family_id = _family_id_for_work(conn, work_id=work_id)
    match = _best_family_match(conn, work_id=work_id, metadata=metadata)
    if match is None:
        family_id = existing_family_id or _create_work_family(
            conn,
            family_key=f"work:{work_id}",
            label=first_title(metadata),
            primary_work_id=work_id,
            relation_kind="standalone",
            confidence=1.0,
            reason={"matched_by": "self"},
        )
    else:
        target_family_id = _family_id_for_work(conn, work_id=int(match["work_id"]))
        if target_family_id is None:
            target_family_id = _create_work_family(
                conn,
                family_key=f"work:{int(match['work_id'])}",
                label=str(match.get("title") or first_title(metadata) or "").strip() or None,
                primary_work_id=int(match["work_id"]),
                relation_kind=str(match.get("relation_kind") or "version_family"),
                confidence=float(match.get("confidence") or 0.8),
                reason=match,
            )
            _upsert_work_family_member(
                conn,
                family_id=target_family_id,
                work_id=int(match["work_id"]),
                member_role=str(match.get("matched_member_role") or "standalone"),
                confidence=float(match.get("confidence") or 0.8),
                reason=match,
            )
        family_id = target_family_id
        if existing_family_id is not None and existing_family_id != family_id:
            family_id = _merge_work_families(conn, target_family_id=family_id, source_family_id=existing_family_id)

    membership_reason = match or {"matched_by": "self"}
    member_role = _family_member_role(metadata)
    member_confidence = float((match or {}).get("confidence") or 1.0)
    _upsert_work_family_member(
        conn,
        family_id=family_id,
        work_id=work_id,
        member_role=member_role,
        confidence=member_confidence,
        reason=membership_reason,
    )
    _refresh_work_family_summary(conn, family_id=family_id)
    return family_id


def _family_id_for_work(conn: sqlite3.Connection, *, work_id: int) -> int | None:
    row = conn.execute(
        "SELECT family_id FROM work_family_members WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row["family_id"])


def _create_work_family(
    conn: sqlite3.Connection,
    *,
    family_key: str,
    label: str | None,
    primary_work_id: int | None,
    relation_kind: str,
    confidence: float,
    reason: dict[str, Any],
) -> int:
    conn.execute(
        """
        INSERT INTO work_families (family_key, label, primary_work_id, relation_kind, confidence, reason_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(family_key) DO UPDATE SET
          label = excluded.label,
          primary_work_id = COALESCE(work_families.primary_work_id, excluded.primary_work_id),
          relation_kind = excluded.relation_kind,
          confidence = excluded.confidence,
          reason_json = excluded.reason_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            family_key,
            str(label or "").strip() or None,
            primary_work_id,
            relation_kind,
            confidence,
            json.dumps(reason, ensure_ascii=False),
        ),
    )
    row = conn.execute(
        "SELECT family_id FROM work_families WHERE family_key = ?",
        (family_key,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert work family: {family_key}")
    return int(row["family_id"])


def _merge_work_families(conn: sqlite3.Connection, *, target_family_id: int, source_family_id: int) -> int:
    if target_family_id == source_family_id:
        return target_family_id
    conn.execute(
        """
        UPDATE work_family_members
        SET family_id = ?, updated_at = CURRENT_TIMESTAMP
        WHERE family_id = ?
        """,
        (target_family_id, source_family_id),
    )
    conn.execute("DELETE FROM work_families WHERE family_id = ?", (source_family_id,))
    _refresh_work_family_summary(conn, family_id=target_family_id)
    return target_family_id


def _upsert_work_family_member(
    conn: sqlite3.Connection,
    *,
    family_id: int,
    work_id: int,
    member_role: str,
    confidence: float,
    reason: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO work_family_members (family_id, work_id, member_role, confidence, reason_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(work_id) DO UPDATE SET
          family_id = excluded.family_id,
          member_role = excluded.member_role,
          confidence = excluded.confidence,
          reason_json = excluded.reason_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            family_id,
            work_id,
            member_role,
            confidence,
            json.dumps(reason, ensure_ascii=False),
        ),
    )


def _refresh_work_family_summary(conn: sqlite3.Connection, *, family_id: int) -> None:
    rows = conn.execute(
        """
        SELECT
          m.work_id,
          m.member_role,
          m.confidence,
          w.title,
          w.year,
          w.raw_metadata_json
        FROM work_family_members m
        JOIN works w ON w.work_id = m.work_id
        WHERE m.family_id = ?
        ORDER BY w.work_id
        """,
        (family_id,),
    ).fetchall()
    if not rows:
        return

    member_rows = [dict(row) for row in rows]
    primary = max(member_rows, key=_family_primary_sort_key)
    relation_kind = "version_family" if len(member_rows) > 1 else "standalone"
    confidence = max(float(row["confidence"] or 0.0) for row in member_rows) or 1.0
    label = str(primary.get("title") or "").strip() or None
    conn.execute(
        """
        UPDATE work_families
        SET primary_work_id = ?, label = ?, relation_kind = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP
        WHERE family_id = ?
        """,
        (
            int(primary["work_id"]),
            label,
            relation_kind,
            confidence,
            family_id,
        ),
    )


def _family_primary_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int]:
    metadata = _loads_json_object(row.get("raw_metadata_json"))
    return (
        FAMILY_MEMBER_ROLE_ORDER.get(str(row.get("member_role") or "standalone"), 0),
        1 if pdf_url_from_metadata(metadata) else 0,
        1 if metadata.get("publication_info") else 0,
        1 if first_arxiv_id(metadata) else 0,
        1 if first_doi(metadata) else 0,
        _coerce_int(metadata.get("citation_count")) or 0,
        _coerce_int(row.get("year")) or 0,
        len(str(row.get("title") or "")),
    )


def _best_family_match(
    conn: sqlite3.Connection,
    *,
    work_id: int,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    year = year_from_metadata(metadata)
    query = """
        SELECT w.work_id, w.title, w.year, w.raw_metadata_json
        FROM works w
        WHERE w.work_id != ?
    """
    params: list[Any] = [work_id]
    if year is not None:
        query += " AND (w.year BETWEEN ? AND ? OR w.year IS NULL)"
        params.extend([year - 2, year + 2])
    query += " ORDER BY w.work_id DESC LIMIT 250"

    best: dict[str, Any] | None = None
    for row in conn.execute(query, params).fetchall():
        candidate_metadata = _loads_json_object(row["raw_metadata_json"])
        match = _family_match_metadata(metadata, candidate_metadata)
        if match is None:
            continue
        match["work_id"] = int(row["work_id"])
        match["title"] = row["title"]
        match["matched_member_role"] = _family_member_role(candidate_metadata)
        if best is None or float(match["confidence"]) > float(best["confidence"]):
            best = match
    return best


def _family_match_metadata(left_md: dict[str, Any], right_md: dict[str, Any]) -> dict[str, Any] | None:
    if _has_distinct_publication_identities(left_md, right_md):
        return None

    title_equivalent = _family_titles_look_equivalent(first_title(left_md), first_title(right_md))
    year_close = _family_years_are_close(left_md, right_md)
    shared_reports = sorted(_normalized_report_roots(left_md) & _normalized_report_roots(right_md))
    shared_collaborations = sorted(_normalized_collaborations(left_md) & _normalized_collaborations(right_md))

    if title_equivalent and year_close and shared_reports:
        return {
            "matched_by": "report_number+title",
            "relation_kind": "version_family",
            "confidence": 0.99,
            "shared_report_roots": shared_reports,
        }
    if title_equivalent and shared_collaborations and _looks_like_stage_variant_family(left_md, right_md):
        confidence = 0.9 + (0.03 if shared_collaborations else 0.0)
        payload = {
            "matched_by": "title+stage_variant",
            "relation_kind": "version_family",
            "confidence": confidence,
        }
        if shared_collaborations:
            payload["shared_collaborations"] = shared_collaborations[:6]
        return payload
    if shared_reports and year_close and shared_collaborations:
        return {
            "matched_by": "report_number+collaboration",
            "relation_kind": "version_family",
            "confidence": 0.82,
            "shared_report_roots": shared_reports,
            "shared_collaborations": shared_collaborations[:6],
        }
    return None


def _family_years_are_close(left_md: dict[str, Any], right_md: dict[str, Any]) -> bool:
    left_year = year_from_metadata(left_md)
    right_year = year_from_metadata(right_md)
    if left_year is None or right_year is None:
        return True
    return abs(left_year - right_year) <= 2


def _family_titles_look_equivalent(left: str | None, right: str | None) -> bool:
    left_norm = _normalize_family_title(left)
    right_norm = _normalize_family_title(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = _family_title_tokens(left_norm)
    right_tokens = _family_title_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return False
    shared = len(left_tokens & right_tokens)
    if shared < 6:
        return False
    if left_tokens == right_tokens:
        return True
    return shared / float(len(left_tokens | right_tokens)) >= 0.92


def _normalize_family_title(title: str | None) -> str:
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


def _family_title_tokens(normalized_title: str) -> set[str]:
    return {
        token
        for token in FAMILY_TITLE_TOKEN_PATTERN.findall(str(normalized_title or ""))
        if token and token not in FAMILY_TITLE_STOPWORDS
    }


def _looks_like_stage_variant_family(left_md: dict[str, Any], right_md: dict[str, Any]) -> bool:
    left_stage = _document_stage_rank(left_md)
    right_stage = _document_stage_rank(right_md)
    if left_stage == right_stage:
        return False
    if max(left_stage, right_stage) < 3:
        return False

    higher = left_md if left_stage > right_stage else right_md
    lower = right_md if higher is left_md else left_md
    if not _has_publication_signal(higher):
        return False
    if _has_publication_signal(lower) and _document_stage_rank(lower) >= 4:
        return False
    return True


def _document_stage_rank(metadata: dict[str, Any]) -> int:
    doc_types = {str(item).casefold() for item in (metadata.get("document_type") or [])}
    if "article" in doc_types or "review" in doc_types:
        return 5
    if "note" in doc_types:
        return 4
    if "conference paper" in doc_types or "proceedings article" in doc_types:
        return 3
    if metadata.get("publication_info"):
        return 2
    return 1


def _has_publication_signal(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("publication_info") or first_arxiv_id(metadata) or first_doi(metadata) or metadata.get("documents"))


def _has_distinct_publication_identities(left_md: dict[str, Any], right_md: dict[str, Any]) -> bool:
    for extractor in (first_arxiv_id, first_doi):
        left_value = str(extractor(left_md) or "").strip().casefold()
        right_value = str(extractor(right_md) or "").strip().casefold()
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _normalized_report_roots(metadata: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    for item in metadata.get("report_numbers") or []:
        value = None
        if isinstance(item, dict):
            value = item.get("value")
        elif item:
            value = item
        normalized = _normalize_report_number(value)
        if normalized:
            roots.add(normalized)
    return roots


def _normalize_report_number(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    tokens = [token for token in re.findall(r"[a-z0-9]+", text) if token]
    if not tokens:
        return None
    normalized_tokens = [token for token in tokens if token not in FAMILY_STAGE_NEUTRAL_TOKENS]
    if normalized_tokens:
        tokens = normalized_tokens
    return "-".join(tokens)


def _normalized_collaborations(metadata: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in metadata.get("collaborations") or []:
        if isinstance(item, dict):
            value = item.get("value") or item.get("name")
        else:
            value = item
        text = str(value or "").strip().casefold()
        if text:
            out.add(text)
    return out


def _family_member_role(metadata: dict[str, Any]) -> str:
    doc_types = {str(item).casefold() for item in (metadata.get("document_type") or [])}
    if "article" in doc_types:
        return "article"
    if "review" in doc_types:
        return "review"
    if "note" in doc_types:
        return "note"
    if "conference paper" in doc_types or "proceedings article" in doc_types:
        return "proceedings"
    if first_arxiv_id(metadata):
        return "preprint"
    return "standalone"


def _loads_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
