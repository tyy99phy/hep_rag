from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PACKAGE_ROOT / "db" / "schema.sql"

_workspace_root = Path(os.environ.get("HEP_RAG_WORKSPACE") or PACKAGE_ROOT).expanduser().resolve()

WORKDIR: Path
DB_DIR: Path
DB_PATH: Path
COLLECTIONS_DIR: Path
DATA_DIR: Path
RAW_DIR: Path
RAW_INSPIRE_DIR: Path
PDF_DIR: Path
PARSED_DIR: Path
INDEX_DIR: Path
EXPORTS_DIR: Path


def set_workspace_root(root: str | Path) -> Path:
    global _workspace_root
    _workspace_root = Path(root).expanduser().resolve()
    _refresh()
    return _workspace_root


def workspace_root() -> Path:
    return _workspace_root


def _refresh() -> None:
    global WORKDIR
    global DB_DIR
    global DB_PATH
    global COLLECTIONS_DIR
    global DATA_DIR
    global RAW_DIR
    global RAW_INSPIRE_DIR
    global PDF_DIR
    global PARSED_DIR
    global INDEX_DIR
    global EXPORTS_DIR

    WORKDIR = _workspace_root
    DB_DIR = WORKDIR / "db"
    DB_PATH = DB_DIR / "hep_rag_v2.db"
    COLLECTIONS_DIR = WORKDIR / "collections"
    DATA_DIR = WORKDIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    RAW_INSPIRE_DIR = RAW_DIR / "inspire"
    PDF_DIR = DATA_DIR / "pdfs"
    PARSED_DIR = DATA_DIR / "parsed"
    INDEX_DIR = WORKDIR / "indexes"
    EXPORTS_DIR = WORKDIR / "exports"


_refresh()
