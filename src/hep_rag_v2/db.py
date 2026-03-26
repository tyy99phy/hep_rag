from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager

from hep_rag_v2 import paths


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(paths.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_directories() -> None:
    paths.DB_DIR.mkdir(parents=True, exist_ok=True)
    paths.COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths.RAW_INSPIRE_DIR.mkdir(parents=True, exist_ok=True)
    paths.PDF_DIR.mkdir(parents=True, exist_ok=True)
    paths.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    paths.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_db() -> None:
    ensure_directories()
    schema = paths.SCHEMA_PATH.read_text(encoding="utf-8")
    with closing(sqlite3.connect(paths.DB_PATH)) as conn:
        conn.executescript(schema)
        _ensure_schema_upgrades(conn)
        conn.commit()


def _ensure_schema_upgrades(conn: sqlite3.Connection) -> None:
    _ensure_columns(
        conn,
        "document_sections",
        {
            "clean_title": "TEXT",
            "section_kind": "TEXT NOT NULL DEFAULT 'body'",
        },
    )
    _ensure_columns(
        conn,
        "blocks",
        {
            "raw_text": "TEXT",
            "clean_text": "TEXT",
            "text_level": "INTEGER",
            "block_role": "TEXT NOT NULL DEFAULT 'body'",
            "is_heading": "INTEGER NOT NULL DEFAULT 0",
            "is_retrievable": "INTEGER NOT NULL DEFAULT 1",
            "exclusion_reason": "TEXT",
            "flags_json": "TEXT",
        },
    )
    _ensure_columns(
        conn,
        "chunks",
        {
            "raw_text": "TEXT",
            "clean_text": "TEXT",
            "is_retrievable": "INTEGER NOT NULL DEFAULT 1",
            "exclusion_reason": "TEXT",
            "source_block_ids_json": "TEXT",
            "flags_json": "TEXT",
        },
    )
    _ensure_columns(
        conn,
        "documents",
        {
            "parse_error": "TEXT",
            "last_parse_attempt_at": "TEXT",
        },
    )


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        existing.add(name)
