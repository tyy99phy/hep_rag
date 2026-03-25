from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from hep_rag_v2 import paths
from hep_rag_v2.metadata import find_work_id, load_collection_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSPIRE_API = "https://inspirehep.net/api/literature"
AUDIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("latex_frac_literal", re.compile(r"\bfrac\b", re.IGNORECASE)),
    ("style_command_literal", re.compile(r"\b(?:mathnormal|boldsymbol|textcircled|operatorname)\b", re.IGNORECASE)),
    ("array_artifact", re.compile(r"\b(?:begin|end)\s+array\b", re.IGNORECASE)),
    ("arrow_split", re.compile(r"(?<=[A-Za-zΑ-Ωα-ω0-9])\s+-\s+>(?=\s*[A-Za-zΑ-Ωα-ω0-9(])")),
    ("double_punct", re.compile(r"(?:,\s*,|,,|(?<!\.)\.\s*\.(?!\.))")),
    (
        "orphan_citation_phrase",
        re.compile(r"\bdetailed\s+in(?:\s+(?:Ref(?:s)?\.?|reference(?:s)?))?(?=\s*(?:[.,;:]|$))", re.IGNORECASE),
    ),
]
READINESS_THRESHOLDS = {
    "max_retrievable_chunk_noise": 0,
    "max_retrievable_block_noise": 1,
    "min_structured_equation_ratio": 0.6,
}

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def load_collection(name: str) -> dict[str, Any]:
    path = paths.COLLECTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Collection config not found: {path}")
    return load_collection_config(path)


def http_get_json(url: str, *, timeout: int = 60, retries: int = 3) -> dict[str, Any]:
    headers = {
        "User-Agent": "hep-rag-v2/0.1 (+local CLI)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Failed to fetch JSON: {last_error}")


def emit_cli_status(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    print(f"[hep-rag] {text}", file=sys.stderr, flush=True)


def save_raw_payload(*, run_id: int, collection: str, shard_slug: str, page: int, payload: dict[str, Any]) -> Path:
    out_dir = paths.RAW_INSPIRE_DIR / collection / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{shard_slug}_page_{page:04d}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_year_buckets(text: str | None) -> list[tuple[int, int]]:
    default = [(2010, 2014), (2015, 2018), (2019, 2021), (2022, 2026)]
    if not text:
        return default
    buckets: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise SystemExit(f"Invalid year bucket format: {part}. Use e.g. 2010-2014,2015-2018")
        a, b = part.split("-", 1)
        buckets.append((int(a), int(b)))
    return buckets or default


# ---------------------------------------------------------------------------
# Ingest-run DB helpers (used by ingest and inspect)
# ---------------------------------------------------------------------------


def _start_ingest_run(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    source: str,
    status: str,
    query_json: str,
    page_size: int,
    limit_requested: int,
    raw_dir: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingest_runs (
          collection_id, source, status, query_json, page_size, limit_requested, raw_dir
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (collection_id, source, status, query_json, page_size, limit_requested, raw_dir),
    )
    return int(cur.lastrowid)


def _update_ingest_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    processed_hits: int,
    works_created: int,
    works_updated: int,
    citations_written: int,
    notes: str | None,
) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET processed_hits = ?, works_created = ?, works_updated = ?, citations_written = ?, notes = ?
        WHERE run_id = ?
        """,
        (processed_hits, works_created, works_updated, citations_written, notes, run_id),
    )


def _finish_ingest_run(conn: sqlite3.Connection, *, run_id: int, status: str, notes: str | None) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET status = ?, notes = ?, finished_at = CURRENT_TIMESTAMP
        WHERE run_id = ?
        """,
        (status, notes, run_id),
    )


# ---------------------------------------------------------------------------
# Work-resolution helpers (used by ingest, inspect, search)
# ---------------------------------------------------------------------------


def _resolve_work_row(
    conn: sqlite3.Connection,
    *,
    work_id: int | None,
    id_type: str | None,
    id_value: str | None,
) -> sqlite3.Row:
    if work_id is not None:
        row = conn.execute(
            """
            SELECT work_id, title, year, canonical_source, canonical_id
            FROM works
            WHERE work_id = ?
            """,
            (work_id,),
        ).fetchone()
        if row is None:
            raise SystemExit(f"Unknown work_id: {work_id}")
        return row

    if id_type and id_value:
        row = conn.execute(
            """
            SELECT w.work_id, w.title, w.year, w.canonical_source, w.canonical_id
            FROM work_ids wi
            JOIN works w ON w.work_id = wi.work_id
            WHERE wi.id_type = ? AND wi.id_value = ?
            """,
            (id_type, id_value),
        ).fetchone()
        if row is None:
            raise SystemExit(f"Unknown work identity: {id_type}:{id_value}")
        return row

    raise SystemExit("Specify either --work-id or both --id-type and --id-value.")


def _collection_work_count(conn: sqlite3.Connection, collection_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM collection_works WHERE collection_id = ?",
        (collection_id,),
    ).fetchone()
    return int(row["n"] if row is not None else 0)


def _infer_collection_name(conn: sqlite3.Connection, work_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT c.name
        FROM collection_works cw
        JOIN collections c ON c.collection_id = cw.collection_id
        WHERE cw.work_id = ?
        ORDER BY c.name
        LIMIT 1
        """,
        (work_id,),
    ).fetchone()
    return str(row["name"]) if row is not None else None


def _paper_storage_stem(conn: sqlite3.Connection, work_id: int) -> str:
    id_rows = conn.execute(
        """
        SELECT id_type, id_value, is_primary
        FROM work_ids
        WHERE work_id = ?
        ORDER BY is_primary DESC, CASE id_type WHEN 'arxiv' THEN 0 WHEN 'inspire' THEN 1 ELSE 2 END, id_type
        """,
        (work_id,),
    ).fetchall()
    for row in id_rows:
        value = str(row["id_value"]).strip()
        if value:
            return _safe_stem(value)
    row = conn.execute(
        "SELECT canonical_id FROM works WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    if row is not None and str(row["canonical_id"]).strip():
        return _safe_stem(str(row["canonical_id"]).strip())
    return str(work_id)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "paper"


def _parsed_doc_dir(collection: str, stem: str) -> Path:
    return paths.PARSED_DIR / collection / stem


# ---------------------------------------------------------------------------
# INSPIRE helpers
# ---------------------------------------------------------------------------


def _required_inspire_fields(fields: list[str]) -> list[str]:
    required = [
        "control_number",
        "titles",
        "abstracts",
        "authors",
        "collaborations",
        "publication_info",
        "arxiv_eprints",
        "dois",
        "citation_count",
        "references",
        "keywords",
        "inspire_categories",
        "documents",
        "accelerator_experiments",
        "preprint_date",
        "earliest_date",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for field in [*fields, *required]:
        name = str(field or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _inspire_literature_url(inspire_id: str, *, fields: list[str]) -> str:
    query = urllib.parse.urlencode({"fields": ",".join(fields)})
    return f"{INSPIRE_API}/{urllib.parse.quote(str(inspire_id).strip())}?{query}"


def _select_inspire_enrichment_targets(
    conn: sqlite3.Connection,
    *,
    collection: str,
    limit: int | None,
    force: bool,
) -> list[sqlite3.Row]:
    sql = """
        SELECT
          w.work_id,
          w.canonical_source,
          w.canonical_id,
          COALESCE(
            (SELECT wi.id_value FROM work_ids wi WHERE wi.work_id = w.work_id AND wi.id_type = 'inspire' LIMIT 1),
            CASE WHEN w.canonical_source = 'inspire' THEN w.canonical_id ELSE NULL END
          ) AS inspire_id,
          EXISTS(SELECT 1 FROM citations c WHERE c.src_work_id = w.work_id) AS has_citations
        FROM works w
        JOIN collection_works cw ON cw.work_id = w.work_id
        JOIN collections c ON c.collection_id = cw.collection_id
        WHERE c.name = ?
    """
    params: list[Any] = [collection]
    if not force:
        sql += " AND NOT EXISTS(SELECT 1 FROM citations c WHERE c.src_work_id = w.work_id)"
    sql += " ORDER BY w.year DESC, w.work_id DESC"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Legacy-corpus helpers
# ---------------------------------------------------------------------------


def _load_legacy_papers(
    conn: sqlite3.Connection,
    *,
    collection: str,
    limit: int | None,
) -> list[sqlite3.Row]:
    sql = """
        SELECT *
        FROM papers
        WHERE collection = ?
        ORDER BY paper_id
    """
    params: list[Any] = [collection]
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _legacy_hit_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata: dict[str, Any]
    raw_metadata = str(row["raw_metadata_json"] or "").strip()
    if raw_metadata:
        metadata = json.loads(raw_metadata)
    else:
        metadata = {}

    if not metadata.get("control_number") and row["inspire_id"]:
        metadata["control_number"] = int(row["inspire_id"]) if str(row["inspire_id"]).isdigit() else str(row["inspire_id"])
    if row["title"] and not metadata.get("titles"):
        metadata["titles"] = [{"title": str(row["title"])}]
    if row["abstract"] and not metadata.get("abstracts"):
        metadata["abstracts"] = [{"value": str(row["abstract"])}]
    if row["year"] and not metadata.get("publication_info"):
        metadata["publication_info"] = [{"year": int(row["year"])}]
    if row["arxiv_id"] and not metadata.get("arxiv_eprints"):
        metadata["arxiv_eprints"] = [{"value": str(row["arxiv_id"])}]
    if row["doi"] and not metadata.get("dois"):
        metadata["dois"] = [{"value": str(row["doi"])}]
    if row["citation_count"] and metadata.get("citation_count") is None:
        metadata["citation_count"] = int(row["citation_count"])

    links: dict[str, Any] = {}
    if row["source_url"]:
        links["self"] = str(row["source_url"])
    return {"metadata": metadata, "links": links}


def _legacy_row_identities(row: sqlite3.Row) -> set[tuple[str, str]]:
    ids: set[tuple[str, str]] = set()
    if row["inspire_id"]:
        ids.add(("inspire", str(row["inspire_id"]).strip()))
    if row["arxiv_id"]:
        ids.add(("arxiv", str(row["arxiv_id"]).strip()))
    if row["doi"]:
        ids.add(("doi", str(row["doi"]).strip().lower()))
    return ids


def _discover_manifests(parsed_root: Path) -> list[Path]:
    return sorted(path.resolve() for path in parsed_root.rglob("manifest.json"))


def _manifest_identity_candidates(stem: str) -> list[tuple[str, str]]:
    value = stem.strip()
    candidates: list[tuple[str, str]] = []
    if not value:
        return candidates
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$", value):
        candidates.append(("arxiv", value))
    elif re.match(r"^[A-Za-z-]+_[0-9]{7}$", value):
        candidates.append(("arxiv", value.replace("_", "/", 1)))
    elif re.match(r"^[0-9]+$", value):
        candidates.append(("inspire", value))
    return candidates


def _resolve_manifest_work_id(conn: sqlite3.Connection, manifest_path: Path) -> int | None:
    for id_type, id_value in _manifest_identity_candidates(manifest_path.parent.name):
        work_id = find_work_id(conn, id_type=id_type, id_value=id_value)
        if work_id is not None:
            return work_id
    return None
