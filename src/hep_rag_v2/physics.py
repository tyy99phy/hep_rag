from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from hep_rag_v2.results import ensure_result_schema
from hep_rag_v2.textnorm import normalize_display_text, normalize_search_text


SCHEMA = """
CREATE TABLE IF NOT EXISTS physics_concepts (
  physics_concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_key TEXT NOT NULL UNIQUE,
  label TEXT NOT NULL,
  normalized_label TEXT NOT NULL,
  concept_kind TEXT NOT NULL DEFAULT 'section',
  source_kind TEXT NOT NULL DEFAULT 'pdg_seed',
  source_ref TEXT,
  summary_text TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS physics_aliases (
  physics_alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
  physics_concept_id INTEGER NOT NULL,
  alias_text TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  alias_kind TEXT NOT NULL DEFAULT 'label',
  confidence REAL NOT NULL DEFAULT 1.0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(physics_concept_id, normalized_alias),
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS physics_relations (
  physics_relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_physics_concept_id INTEGER NOT NULL,
  dst_physics_concept_id INTEGER NOT NULL,
  relation_kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  source_ref TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(src_physics_concept_id, dst_physics_concept_id, relation_kind),
  FOREIGN KEY (src_physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE,
  FOREIGN KEY (dst_physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS work_physics_groundings (
  work_physics_grounding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  physics_concept_id INTEGER NOT NULL,
  match_kind TEXT NOT NULL DEFAULT 'alias',
  confidence REAL NOT NULL DEFAULT 0.0,
  matched_alias TEXT,
  evidence_text TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(work_id, physics_concept_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS result_physics_groundings (
  result_physics_grounding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  result_object_id INTEGER NOT NULL,
  physics_concept_id INTEGER NOT NULL,
  match_kind TEXT NOT NULL DEFAULT 'alias',
  confidence REAL NOT NULL DEFAULT 0.0,
  matched_alias TEXT,
  evidence_text TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(result_object_id, physics_concept_id),
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE CASCADE,
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chunk_physics_groundings (
  chunk_physics_grounding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id INTEGER NOT NULL,
  physics_concept_id INTEGER NOT NULL,
  match_kind TEXT NOT NULL DEFAULT 'alias',
  confidence REAL NOT NULL DEFAULT 0.0,
  matched_alias TEXT,
  evidence_text TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(chunk_id, physics_concept_id),
  FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_physics_concepts_kind ON physics_concepts(concept_kind, label);
CREATE INDEX IF NOT EXISTS idx_physics_aliases_norm ON physics_aliases(normalized_alias);
CREATE INDEX IF NOT EXISTS idx_physics_relations_src ON physics_relations(src_physics_concept_id, relation_kind);
CREATE INDEX IF NOT EXISTS idx_work_physics_groundings_work ON work_physics_groundings(work_id, confidence);
CREATE INDEX IF NOT EXISTS idx_work_physics_groundings_concept ON work_physics_groundings(physics_concept_id, confidence);
CREATE INDEX IF NOT EXISTS idx_result_physics_groundings_result ON result_physics_groundings(result_object_id, confidence);
CREATE INDEX IF NOT EXISTS idx_result_physics_groundings_concept ON result_physics_groundings(physics_concept_id, confidence);
CREATE INDEX IF NOT EXISTS idx_chunk_physics_groundings_chunk ON chunk_physics_groundings(chunk_id, confidence);
CREATE INDEX IF NOT EXISTS idx_chunk_physics_groundings_concept ON chunk_physics_groundings(physics_concept_id, confidence);
"""

_HEADING_NUMBER_RE = re.compile(r"^\s*(?:[A-Za-z]?\d+(?:\.\d+)*|[IVXLCDM]+(?=[.)]))\s*[\].:)\-]?\s*", re.IGNORECASE)
_PARENS_RE = re.compile(r"\(([^()]{2,24})\)")
_SUBJECT_PATTERNS = (
    re.compile(r"\b(CKM matrix)\b", re.IGNORECASE),
    re.compile(r"\b(PMNS matrix)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9+-]+\s+mesons?)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9+-]+\s+bosons?)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9+-]+\s+quarks?)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9+-]+\s+leptons?)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9+-]+\s+neutrinos?)\b", re.IGNORECASE),
    re.compile(r"\b(Higgs boson)\b", re.IGNORECASE),
    re.compile(r"\b(vector boson scattering)\b", re.IGNORECASE),
)
_GENERIC_SECTION_NAMES = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "background",
    "bibliography",
    "conclusion",
    "conclusions",
    "contents",
    "discussion",
    "figure",
    "figures",
    "index",
    "introduction",
    "overview",
    "references",
    "review",
    "summary",
    "supplement",
    "table",
    "tables",
}
_CONTEXTUAL_SECTION_NAMES = {
    "branching fractions",
    "branching ratio",
    "branching ratios",
    "coupling",
    "couplings",
    "cross section",
    "cross sections",
    "decay",
    "decays",
    "form factor",
    "form factors",
    "interactions",
    "lifetime",
    "lifetimes",
    "mass",
    "masses",
    "mixing",
    "parameter",
    "parameters",
    "production",
    "properties",
    "property",
    "scattering",
    "spectroscopy",
    "width",
    "widths",
}
_PARTICLE_HINTS = (
    "boson",
    "baryon",
    "hadron",
    "meson",
    "quark",
    "gluon",
    "photon",
    "electron",
    "muon",
    "tau",
    "neutrino",
    "higgs",
    "proton",
    "neutron",
    "kaon",
    "pion",
    "lambda",
    "sigma",
    "omega",
    "charm",
    "bottom",
    "top",
    "strange",
)
_PROCESS_HINTS = (
    "annihilation",
    "collision",
    "fusion",
    "interaction",
    "jet tagging",
    "production",
    "scattering",
    "vector boson scattering",
)
_DECAY_HINTS = ("decay", "branching", "lifetime", "width")
_PARAMETER_HINTS = ("ckm", "pmns", "coupling", "constant", "matrix element", "parameter")
_OBSERVABLE_HINTS = ("cross section", "mass", "width", "asymmetry", "lifetime", "form factor", "mixing")


def ensure_physics_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def build_physics_substrate(
    conn: sqlite3.Connection,
    *,
    collection: str | None = None,
    work_ids: list[int] | None = None,
    max_work_groundings: int = 6,
    max_result_groundings: int = 4,
    max_chunk_groundings: int = 3,
) -> dict[str, Any]:
    ensure_physics_schema(conn)
    ensure_result_schema(conn)

    concept_summary = _rebuild_concepts(conn)
    selected_work_ids = _select_scope_work_ids(conn, collection=collection, work_ids=work_ids)
    if concept_summary["total"] == 0:
        return {
            "collection": collection or "all",
            "scope_work_ids": len(selected_work_ids),
            "concepts": concept_summary,
            "works": {"processed": 0, "grounded": 0},
            "results": {"processed": 0, "grounded": 0},
            "chunks": {"processed": 0, "grounded": 0},
        }

    aliases = _load_alias_records(conn)
    work_summary = _rewrite_work_groundings(
        conn,
        work_ids=selected_work_ids,
        aliases=aliases,
        limit=max_work_groundings,
    )
    result_summary = _rewrite_result_groundings(
        conn,
        work_ids=selected_work_ids,
        aliases=aliases,
        limit=max_result_groundings,
    )
    chunk_summary = _rewrite_chunk_groundings(
        conn,
        work_ids=selected_work_ids,
        aliases=aliases,
        limit=max_chunk_groundings,
    )
    return {
        "collection": collection or "all",
        "scope_work_ids": len(selected_work_ids),
        "concepts": concept_summary,
        "works": work_summary,
        "results": result_summary,
        "chunks": chunk_summary,
    }


def physics_summary_counts(conn: sqlite3.Connection) -> dict[str, int]:
    ensure_physics_schema(conn)
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM physics_concepts) AS physics_concepts,
          (SELECT COUNT(*) FROM physics_aliases) AS physics_aliases,
          (SELECT COUNT(*) FROM physics_relations) AS physics_relations,
          (SELECT COUNT(*) FROM work_physics_groundings) AS work_physics_groundings,
          (SELECT COUNT(*) FROM result_physics_groundings) AS result_physics_groundings,
          (SELECT COUNT(*) FROM chunk_physics_groundings) AS chunk_physics_groundings
        """
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()} if row is not None else {}


def work_physics_text_map(conn: sqlite3.Connection, *, limit: int = 6) -> dict[int, str]:
    ensure_physics_schema(conn)
    rows = conn.execute(
        """
        SELECT wpg.work_id, pc.label, wpg.confidence
        FROM work_physics_groundings wpg
        JOIN physics_concepts pc ON pc.physics_concept_id = wpg.physics_concept_id
        ORDER BY wpg.work_id, wpg.confidence DESC, pc.label
        """
    ).fetchall()
    grouped: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        work_id = int(row["work_id"])
        labels = grouped[work_id]
        label = str(row["label"] or "").strip()
        if label and label not in labels and len(labels) < limit:
            labels.append(label)
    return {work_id: " ".join(labels) for work_id, labels in grouped.items() if labels}


def chunk_physics_text_map(conn: sqlite3.Connection, *, limit: int = 5) -> dict[int, str]:
    ensure_physics_schema(conn)
    rows = conn.execute(
        """
        SELECT cpg.chunk_id, pc.label, cpg.confidence
        FROM chunk_physics_groundings cpg
        JOIN physics_concepts pc ON pc.physics_concept_id = cpg.physics_concept_id
        ORDER BY cpg.chunk_id, cpg.confidence DESC, pc.label
        """
    ).fetchall()
    grouped: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        chunk_id = int(row["chunk_id"])
        labels = grouped[chunk_id]
        label = str(row["label"] or "").strip()
        if label and label not in labels and len(labels) < limit:
            labels.append(label)
    return {chunk_id: " ".join(labels) for chunk_id, labels in grouped.items() if labels}


def top_physics_concept_counts(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int],
    limit: int = 3,
) -> list[str]:
    ensure_physics_schema(conn)
    if not work_ids:
        return []
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        f"""
        SELECT pc.label AS label, COUNT(DISTINCT wpg.work_id) AS n
        FROM work_physics_groundings wpg
        JOIN physics_concepts pc ON pc.physics_concept_id = wpg.physics_concept_id
        WHERE wpg.work_id IN ({placeholders})
        GROUP BY pc.physics_concept_id, pc.label
        ORDER BY n DESC, pc.label
        LIMIT ?
        """,
        [*work_ids, limit],
    ).fetchall()
    return [f"{str(row['label'])}={int(row['n'])}" for row in rows if row["label"]]


def _rebuild_concepts(conn: sqlite3.Connection) -> dict[str, Any]:
    records = _seed_records(conn)
    if not records:
        total = int(conn.execute("SELECT COUNT(*) FROM physics_concepts").fetchone()[0])
        return {"total": total, "aliases": 0, "relations": 0, "seed_records": 0}

    concepts: dict[str, dict[str, Any]] = {}
    relations: set[tuple[str, str]] = set()
    for record in records:
        path_segments = list(record.get("path_segments") or [])
        if not path_segments:
            continue
        prefix_keys: list[str] = []
        prefix_labels: list[str] = []
        for index, raw_segment in enumerate(path_segments):
            segment = _canonical_segment_text(str(raw_segment or ""))
            parent_segment = prefix_labels[-1] if prefix_labels else None
            if not _accept_segment(segment, parent_segment=parent_segment):
                continue
            label = _compose_concept_label(segment=segment, parent_segment=parent_segment)
            concept_key = _concept_key(path_segments[: index + 1])
            payload = concepts.setdefault(
                concept_key,
                {
                    "label": label,
                    "concept_kind": _classify_concept_kind(label),
                    "summary_snippets": [],
                    "aliases": set(),
                    "source_refs": set(),
                    "path_samples": set(),
                },
            )
            payload["label"] = _prefer_label(existing=str(payload["label"]), candidate=label)
            payload["concept_kind"] = _prefer_kind(
                existing=str(payload["concept_kind"]),
                candidate=_classify_concept_kind(label),
            )
            payload["aliases"].update(_candidate_aliases(label=label, raw_segment=segment, parent_segment=parent_segment))
            source_ref = str(record.get("source_ref") or "").strip()
            if source_ref:
                payload["source_refs"].add(source_ref)
            path_text = str(record.get("path_text") or "").strip()
            if path_text:
                payload["path_samples"].add(path_text)
            if index == len(path_segments) - 1:
                summary_text = str(record.get("summary_text") or "").strip()
                if summary_text:
                    payload["summary_snippets"].append(summary_text)
            if prefix_keys:
                relations.add((prefix_keys[-1], concept_key))
            prefix_keys.append(concept_key)
            prefix_labels.append(label)
        for subject_label in _extract_subject_concepts(str(record.get("summary_text") or "")):
            concept_key = f"pdg_entity:{_safe_key(subject_label)}"
            payload = concepts.setdefault(
                concept_key,
                {
                    "label": subject_label,
                    "concept_kind": _classify_concept_kind(subject_label),
                    "summary_snippets": [],
                    "aliases": set(),
                    "source_refs": set(),
                    "path_samples": set(),
                },
            )
            payload["aliases"].update(_candidate_aliases(label=subject_label, raw_segment=subject_label, parent_segment=None))
            source_ref = str(record.get("source_ref") or "").strip()
            if source_ref:
                payload["source_refs"].add(source_ref)
            path_text = str(record.get("path_text") or "").strip()
            if path_text:
                payload["path_samples"].add(path_text)
            summary_text = str(record.get("summary_text") or "").strip()
            if summary_text:
                payload["summary_snippets"].append(summary_text)

    if not concepts:
        total = int(conn.execute("SELECT COUNT(*) FROM physics_concepts").fetchone()[0])
        return {"total": total, "aliases": 0, "relations": 0, "seed_records": len(records)}

    concept_id_by_key: dict[str, int] = {}
    with conn:
        for concept_key, payload in concepts.items():
            summary_text = _summary_from_snippets(payload["summary_snippets"])
            metadata = {
                "path_samples": sorted(payload["path_samples"])[:6],
                "source_refs": sorted(payload["source_refs"])[:12],
            }
            conn.execute(
                """
                INSERT INTO physics_concepts (
                  concept_key, label, normalized_label, concept_kind, source_kind, source_ref, summary_text, metadata_json
                ) VALUES (?, ?, ?, ?, 'pdg_seed', ?, ?, ?)
                ON CONFLICT(concept_key) DO UPDATE SET
                  label = excluded.label,
                  normalized_label = excluded.normalized_label,
                  concept_kind = excluded.concept_kind,
                  source_kind = excluded.source_kind,
                  source_ref = excluded.source_ref,
                  summary_text = excluded.summary_text,
                  metadata_json = excluded.metadata_json,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (
                    concept_key,
                    str(payload["label"]),
                    normalize_search_text(str(payload["label"])),
                    str(payload["concept_kind"]),
                    next(iter(sorted(payload["source_refs"])), None),
                    summary_text,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        rows = conn.execute(
            "SELECT physics_concept_id, concept_key FROM physics_concepts WHERE source_kind = 'pdg_seed'"
        ).fetchall()
        seen_keys = set(concepts)
        stale_ids = [int(row["physics_concept_id"]) for row in rows if str(row["concept_key"]) not in seen_keys]
        if stale_ids:
            placeholders = ",".join("?" for _ in stale_ids)
            conn.execute(f"DELETE FROM physics_concepts WHERE physics_concept_id IN ({placeholders})", stale_ids)
        concept_rows = conn.execute(
            "SELECT physics_concept_id, concept_key FROM physics_concepts WHERE concept_key IN ({})".format(
                ",".join("?" for _ in concepts)
            ),
            list(concepts),
        ).fetchall()
        concept_id_by_key = {str(row["concept_key"]): int(row["physics_concept_id"]) for row in concept_rows}
        if concept_id_by_key:
            placeholders = ",".join("?" for _ in concept_id_by_key)
            conn.execute(
                f"DELETE FROM physics_aliases WHERE physics_concept_id IN ({placeholders})",
                list(concept_id_by_key.values()),
            )
        alias_count = 0
        for concept_key, payload in concepts.items():
            concept_id = concept_id_by_key.get(concept_key)
            if concept_id is None:
                continue
            alias_rows = []
            for alias_text, alias_kind, confidence in _ordered_alias_payloads(
                label=str(payload["label"]),
                aliases=payload["aliases"],
            ):
                alias_rows.append(
                    (
                        concept_id,
                        alias_text,
                        normalize_search_text(alias_text),
                        alias_kind,
                        confidence,
                    )
                )
            if alias_rows:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO physics_aliases (
                      physics_concept_id, alias_text, normalized_alias, alias_kind, confidence
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    alias_rows,
                )
                alias_count += len(alias_rows)
        conn.execute("DELETE FROM physics_relations")
        relation_rows = []
        for parent_key, child_key in sorted(relations):
            parent_id = concept_id_by_key.get(parent_key)
            child_id = concept_id_by_key.get(child_key)
            if parent_id is None or child_id is None or parent_id == child_id:
                continue
            relation_rows.append(
                (
                    parent_id,
                    child_id,
                    "contains",
                    1.0,
                    child_key,
                    json.dumps({"source": "pdg_seed"}, ensure_ascii=False),
                )
            )
        if relation_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO physics_relations (
                  src_physics_concept_id, dst_physics_concept_id, relation_kind, weight, source_ref, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                relation_rows,
            )
    return {
        "total": len(concept_id_by_key),
        "aliases": alias_count,
        "relations": len(relations),
        "seed_records": len(records),
    }


def _seed_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows = conn.execute(
        """
        SELECT source_id, title, parent_title, path_text, capsule_text
        FROM pdg_sections
        ORDER BY source_id, order_index
        """
    ).fetchall()
    for row in rows:
        root_title = _canonical_segment_text(str(row["source_id"] or ""))
        path_segments = _semantic_path_segments(
            path_text=str(row["path_text"] or ""),
            root_candidates=[str(row["source_id"] or "")],
        )
        if not path_segments:
            leaf = _canonical_segment_text(str(row["title"] or ""))
            if leaf:
                path_segments = [leaf]
        if path_segments:
            records.append(
                {
                    "path_segments": path_segments,
                    "summary_text": str(row["capsule_text"] or ""),
                    "source_ref": f"pdg_section:{row['source_id']}:{_safe_key('/'.join(path_segments))}",
                    "path_text": str(row["path_text"] or root_title or ""),
                }
            )

    doc_rows = conn.execute(
        """
        SELECT
          w.work_id,
          COALESCE(w.title, '') AS work_title,
          COALESCE(ds.path_text, COALESCE(ds.clean_title, ds.title, '')) AS path_text,
          COALESCE(
            (
              SELECT COALESCE(c.clean_text, c.text, '')
              FROM chunks c
              WHERE c.section_id = ds.section_id
                AND c.is_retrievable = 1
                AND COALESCE(c.clean_text, c.text, '') <> ''
              ORDER BY c.chunk_id
              LIMIT 1
            ),
            ''
          ) AS snippet
        FROM works w
        JOIN documents d ON d.work_id = w.work_id
        JOIN document_sections ds ON ds.document_id = d.document_id
        WHERE (
          w.canonical_source = 'pdg'
          OR EXISTS (
            SELECT 1
            FROM work_ids wi
            WHERE wi.work_id = w.work_id
              AND wi.id_type = 'pdg'
          )
        )
        ORDER BY w.work_id, ds.order_index
        """
    ).fetchall()
    for row in doc_rows:
        path_segments = _semantic_path_segments(
            path_text=str(row["path_text"] or ""),
            root_candidates=[str(row["work_title"] or "")],
        )
        if not path_segments:
            continue
        records.append(
            {
                "path_segments": path_segments,
                "summary_text": str(row["snippet"] or ""),
                "source_ref": f"pdg_document:{int(row['work_id'])}:{_safe_key('/'.join(path_segments))}",
                "path_text": str(row["path_text"] or ""),
            }
        )
    return records


def _select_scope_work_ids(
    conn: sqlite3.Connection,
    *,
    collection: str | None,
    work_ids: list[int] | None,
) -> list[int]:
    explicit = sorted({int(work_id) for work_id in (work_ids or [])})
    if explicit:
        return _exclude_pdg_work_ids(conn, explicit)
    if collection:
        rows = conn.execute(
            """
            SELECT cw.work_id
            FROM collection_works cw
            JOIN collections c ON c.collection_id = cw.collection_id
            WHERE c.name = ?
            ORDER BY cw.work_id
            """,
            (collection,),
        ).fetchall()
        return _exclude_pdg_work_ids(conn, [int(row["work_id"]) for row in rows])
    rows = conn.execute("SELECT work_id FROM works ORDER BY work_id").fetchall()
    return _exclude_pdg_work_ids(conn, [int(row["work_id"]) for row in rows])


def _exclude_pdg_work_ids(conn: sqlite3.Connection, work_ids: list[int]) -> list[int]:
    if not work_ids:
        return []
    placeholders = ",".join("?" for _ in work_ids)
    pdg_rows = conn.execute(
        f"""
        SELECT w.work_id
        FROM works w
        WHERE w.work_id IN ({placeholders})
          AND (
            w.canonical_source = 'pdg'
            OR EXISTS (
              SELECT 1
              FROM work_ids wi
              WHERE wi.work_id = w.work_id
                AND wi.id_type = 'pdg'
            )
          )
        """,
        work_ids,
    ).fetchall()
    excluded = {int(row["work_id"]) for row in pdg_rows}
    return [int(work_id) for work_id in work_ids if int(work_id) not in excluded]


def _load_alias_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          pa.physics_concept_id,
          pa.alias_text,
          pa.normalized_alias,
          pa.alias_kind,
          pa.confidence,
          pc.label,
          pc.normalized_label,
          pc.concept_kind
        FROM physics_aliases pa
        JOIN physics_concepts pc ON pc.physics_concept_id = pa.physics_concept_id
        WHERE COALESCE(pa.normalized_alias, '') <> ''
        ORDER BY LENGTH(pa.normalized_alias) DESC, pa.confidence DESC, pa.alias_text
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _rewrite_work_groundings(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int],
    aliases: list[dict[str, Any]],
    limit: int,
) -> dict[str, int]:
    if not work_ids:
        return {"processed": 0, "grounded": 0}
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        f"""
        SELECT work_id, COALESCE(title, '') AS title, COALESCE(abstract, '') AS abstract
        FROM works
        WHERE work_id IN ({placeholders})
        ORDER BY work_id
        """,
        work_ids,
    ).fetchall()
    with conn:
        conn.execute(f"DELETE FROM work_physics_groundings WHERE work_id IN ({placeholders})", work_ids)
        insert_rows = []
        grounded = 0
        for row in rows:
            work_id = int(row["work_id"])
            text = "\n".join(part for part in (str(row["title"] or ""), str(row["abstract"] or "")) if part.strip())
            matches = _match_aliases(text=text, aliases=aliases, limit=limit)
            if matches:
                grounded += 1
            for match in matches:
                insert_rows.append(
                    (
                        work_id,
                        int(match["physics_concept_id"]),
                        "alias",
                        float(match["confidence"]),
                        str(match["matched_alias"]),
                        _truncate_evidence(text),
                        json.dumps(
                            {
                                "concept_kind": match["concept_kind"],
                                "label": match["label"],
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
        if insert_rows:
            conn.executemany(
                """
                INSERT INTO work_physics_groundings (
                  work_id, physics_concept_id, match_kind, confidence, matched_alias, evidence_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
    return {"processed": len(rows), "grounded": grounded}


def _rewrite_result_groundings(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int],
    aliases: list[dict[str, Any]],
    limit: int,
) -> dict[str, int]:
    if not work_ids:
        return {"processed": 0, "grounded": 0}
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        f"""
        SELECT result_object_id, work_id, COALESCE(label, '') AS label, COALESCE(summary_text, '') AS summary_text
        FROM result_objects
        WHERE work_id IN ({placeholders})
        ORDER BY result_object_id
        """,
        work_ids,
    ).fetchall()
    with conn:
        if rows:
            result_ids = [int(row["result_object_id"]) for row in rows]
            result_placeholders = ",".join("?" for _ in result_ids)
            conn.execute(
                f"DELETE FROM result_physics_groundings WHERE result_object_id IN ({result_placeholders})",
                result_ids,
            )
        insert_rows = []
        grounded = 0
        for row in rows:
            text = "\n".join(part for part in (str(row["label"] or ""), str(row["summary_text"] or "")) if part.strip())
            matches = _match_aliases(text=text, aliases=aliases, limit=limit)
            if matches:
                grounded += 1
            for match in matches:
                insert_rows.append(
                    (
                        int(row["result_object_id"]),
                        int(match["physics_concept_id"]),
                        "alias",
                        float(match["confidence"]),
                        str(match["matched_alias"]),
                        _truncate_evidence(text),
                        json.dumps(
                            {
                                "work_id": int(row["work_id"]),
                                "concept_kind": match["concept_kind"],
                                "label": match["label"],
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
        if insert_rows:
            conn.executemany(
                """
                INSERT INTO result_physics_groundings (
                  result_object_id, physics_concept_id, match_kind, confidence, matched_alias, evidence_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
    return {"processed": len(rows), "grounded": grounded}


def _rewrite_chunk_groundings(
    conn: sqlite3.Connection,
    *,
    work_ids: list[int],
    aliases: list[dict[str, Any]],
    limit: int,
) -> dict[str, int]:
    if not work_ids:
        return {"processed": 0, "grounded": 0}
    placeholders = ",".join("?" for _ in work_ids)
    rows = conn.execute(
        f"""
        SELECT
          chunk_id,
          work_id,
          COALESCE(section_hint, '') AS section_hint,
          COALESCE(clean_text, text, '') AS body_text
        FROM chunks
        WHERE work_id IN ({placeholders})
          AND is_retrievable = 1
          AND COALESCE(clean_text, text, '') <> ''
        ORDER BY chunk_id
        """,
        work_ids,
    ).fetchall()
    with conn:
        if rows:
            chunk_ids = [int(row["chunk_id"]) for row in rows]
            chunk_placeholders = ",".join("?" for _ in chunk_ids)
            conn.execute(
                f"DELETE FROM chunk_physics_groundings WHERE chunk_id IN ({chunk_placeholders})",
                chunk_ids,
            )
        insert_rows = []
        grounded = 0
        for row in rows:
            text = "\n".join(part for part in (str(row["section_hint"] or ""), str(row["body_text"] or "")) if part.strip())
            matches = _match_aliases(text=text, aliases=aliases, limit=limit)
            if matches:
                grounded += 1
            for match in matches:
                insert_rows.append(
                    (
                        int(row["chunk_id"]),
                        int(match["physics_concept_id"]),
                        "alias",
                        float(match["confidence"]),
                        str(match["matched_alias"]),
                        _truncate_evidence(text),
                        json.dumps(
                            {
                                "work_id": int(row["work_id"]),
                                "concept_kind": match["concept_kind"],
                                "label": match["label"],
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
        if insert_rows:
            conn.executemany(
                """
                INSERT INTO chunk_physics_groundings (
                  chunk_id, physics_concept_id, match_kind, confidence, matched_alias, evidence_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
    return {"processed": len(rows), "grounded": grounded}


def _match_aliases(
    *,
    text: str,
    aliases: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    normalized_text = normalize_search_text(text).casefold().strip()
    if not normalized_text:
        return []
    haystack = f" {normalized_text} "
    by_concept: dict[int, dict[str, Any]] = {}
    for alias in aliases:
        normalized_alias = str(alias.get("normalized_alias") or "").casefold().strip()
        if not normalized_alias or len(normalized_alias) < 3:
            continue
        if f" {normalized_alias} " not in haystack:
            continue
        concept_id = int(alias["physics_concept_id"])
        token_count = len(normalized_alias.split())
        score = 0.46 + min(token_count, 5) * 0.08
        score += min(float(alias.get("confidence") or 0.0), 0.2)
        if str(alias.get("alias_kind") or "") == "label":
            score += 0.06
        if str(alias.get("normalized_label") or "").casefold() == normalized_alias:
            score += 0.05
        if str(alias.get("concept_kind") or "") in {"particle", "process", "decay"}:
            score += 0.04
        payload = {
            "physics_concept_id": concept_id,
            "confidence": min(score, 0.99),
            "matched_alias": str(alias.get("alias_text") or ""),
            "concept_kind": str(alias.get("concept_kind") or ""),
            "label": str(alias.get("label") or ""),
        }
        current = by_concept.get(concept_id)
        if current is None or float(payload["confidence"]) > float(current["confidence"]):
            by_concept[concept_id] = payload
    ordered = sorted(
        by_concept.values(),
        key=lambda item: (
            -float(item["confidence"]),
            -len(str(item["matched_alias"] or "")),
            str(item["label"] or "").casefold(),
        ),
    )
    return ordered[: max(1, int(limit))]


def _semantic_path_segments(path_text: str, *, root_candidates: list[str]) -> list[str]:
    raw_segments = [normalize_display_text(part) for part in str(path_text or "").split("/") if normalize_display_text(part)]
    if not raw_segments:
        return []
    normalized_roots = {
        normalize_search_text(_canonical_segment_text(candidate)).casefold()
        for candidate in root_candidates
        if _canonical_segment_text(candidate)
    }
    while raw_segments:
        head = normalize_search_text(_canonical_segment_text(raw_segments[0])).casefold()
        if head in normalized_roots or (head.startswith("pdg") and len(raw_segments) > 1):
            raw_segments.pop(0)
            continue
        break
    return [_canonical_segment_text(item) for item in raw_segments if _canonical_segment_text(item)]


def _accept_segment(segment: str, *, parent_segment: str | None) -> bool:
    normalized = normalize_search_text(segment).casefold()
    if not normalized or normalized in _GENERIC_SECTION_NAMES:
        return False
    if normalized in _CONTEXTUAL_SECTION_NAMES:
        return bool(parent_segment and _looks_physicsy(parent_segment))
    return _looks_physicsy(segment) or bool(parent_segment and _looks_physicsy(parent_segment))


def _compose_concept_label(*, segment: str, parent_segment: str | None) -> str:
    normalized = normalize_search_text(segment).casefold()
    if parent_segment and normalized in _CONTEXTUAL_SECTION_NAMES:
        parent = _canonical_segment_text(parent_segment)
        if normalized in {"property", "properties"}:
            return f"{parent} properties"
        if normalized in {"parameter", "parameters"}:
            return f"{parent} parameters"
        return f"{parent} {normalized}"
    return segment


def _classify_concept_kind(label: str) -> str:
    text = normalize_search_text(label).casefold()
    if any(hint in text for hint in _DECAY_HINTS):
        return "decay"
    if any(hint in text for hint in _PROCESS_HINTS):
        return "process"
    if any(hint in text for hint in _PARAMETER_HINTS):
        return "parameter"
    if any(hint in text for hint in _OBSERVABLE_HINTS):
        return "observable"
    if any(hint in text for hint in _PARTICLE_HINTS):
        return "particle"
    return "section"


def _candidate_aliases(
    *,
    label: str,
    raw_segment: str,
    parent_segment: str | None,
) -> set[str]:
    aliases = {label}
    normalized_raw = normalize_search_text(raw_segment).casefold()
    if normalized_raw not in _CONTEXTUAL_SECTION_NAMES and normalized_raw not in _GENERIC_SECTION_NAMES:
        aliases.add(raw_segment)
    if parent_segment and normalized_raw in _CONTEXTUAL_SECTION_NAMES:
        aliases.add(f"{raw_segment} of {parent_segment}")
    for source in (label, raw_segment):
        for match in _PARENS_RE.findall(source):
            candidate = normalize_display_text(match)
            if candidate:
                aliases.add(candidate)
        stripped = re.sub(r"\([^()]*\)", " ", source)
        stripped = normalize_display_text(stripped)
        if stripped:
            aliases.add(stripped)
    return {alias for alias in aliases if _allow_alias(alias)}


def _ordered_alias_payloads(*, label: str, aliases: set[str]) -> list[tuple[str, str, float]]:
    ordered = []
    seen: set[str] = set()
    preferred = [label] + sorted(aliases, key=lambda item: (-len(normalize_search_text(item)), item.casefold()))
    for alias_text in preferred:
        normalized_alias = normalize_search_text(alias_text)
        if not normalized_alias or normalized_alias in seen:
            continue
        seen.add(normalized_alias)
        alias_kind = "label" if normalized_alias == normalize_search_text(label) else "alias"
        confidence = 1.0 if alias_kind == "label" else min(0.92, 0.55 + len(normalized_alias.split()) * 0.08)
        ordered.append((alias_text, alias_kind, confidence))
    return ordered


def _allow_alias(alias: str) -> bool:
    normalized = normalize_search_text(alias)
    if not normalized:
        return False
    if normalized.casefold() in _GENERIC_SECTION_NAMES:
        return False
    if len(normalized) < 3:
        return False
    if len(normalized.split()) == 1 and len(normalized) < 4:
        return False
    return True


def _prefer_label(*, existing: str, candidate: str) -> str:
    existing_key = normalize_search_text(existing)
    candidate_key = normalize_search_text(candidate)
    if len(candidate_key) > len(existing_key):
        return candidate
    return existing


def _prefer_kind(*, existing: str, candidate: str) -> str:
    rank = {
        "particle": 5,
        "process": 4,
        "decay": 4,
        "parameter": 3,
        "observable": 3,
        "section": 1,
    }
    return candidate if rank.get(candidate, 0) > rank.get(existing, 0) else existing


def _summary_from_snippets(snippets: list[str]) -> str:
    unique = []
    seen: set[str] = set()
    for snippet in snippets:
        text = " ".join(str(snippet or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
        if len(unique) >= 2:
            break
    return " | ".join(unique)


def _canonical_segment_text(text: str) -> str:
    cleaned = normalize_display_text(text or "")
    cleaned = _HEADING_NUMBER_RE.sub("", cleaned)
    cleaned = re.sub(r"^\s*PDG\s+review\s+of\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*review\s+of\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
    return cleaned


def _looks_physicsy(text: str) -> bool:
    normalized = normalize_search_text(text).casefold()
    if not normalized:
        return False
    if any(hint in normalized for hint in _PARTICLE_HINTS + _PROCESS_HINTS + _DECAY_HINTS + _PARAMETER_HINTS + _OBSERVABLE_HINTS):
        return True
    if re.search(r"\b(?:qcd|ckm|pmns|lhc|hl lhc|bsm|eft|vbs|ssww|cms|atlas)\b", normalized):
        return True
    if re.search(r"\b[a-z]+\d+\b", normalized):
        return True
    if re.search(r"\b(?:b|d|k|w|z)\s+mesons?\b", normalized):
        return True
    return False


def _concept_key(path_segments: list[str]) -> str:
    parts = [_safe_key(_canonical_segment_text(part)) for part in path_segments if _canonical_segment_text(part)]
    return "pdg:" + "/".join(parts)


def _safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip().casefold()).strip("_") or "unknown"


def _truncate_evidence(text: str, *, limit: int = 360) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _extract_subject_concepts(text: str) -> list[str]:
    cleaned = normalize_display_text(text)
    out: list[str] = []
    seen: set[str] = set()
    for pattern in _SUBJECT_PATTERNS:
        for match in pattern.findall(cleaned):
            candidate = _subject_label(str(match))
            if not candidate or candidate in seen or not _looks_physicsy(candidate):
                continue
            seen.add(candidate)
            out.append(candidate)
    return out[:4]


def _subject_label(text: str) -> str:
    candidate = normalize_display_text(text)
    candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE).strip()
    parts = candidate.split()
    if not parts:
        return ""
    if parts[0].islower() and len(parts[0]) == 1:
        parts[0] = parts[0].upper()
    if len(parts) >= 2 and parts[1].casefold() == "matrix":
        return f"{parts[0].upper()} Matrix"
    return " ".join(part.capitalize() if index == 0 and len(part) == 1 else part for index, part in enumerate(parts))
