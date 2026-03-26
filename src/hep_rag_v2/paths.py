from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_workspace_root = Path(os.environ.get("HEP_RAG_WORKSPACE") or _DEFAULT_ROOT).expanduser().resolve()


@dataclass(frozen=True)
class WorkspacePaths:
    workdir: Path
    db_dir: Path
    db_path: Path
    api_db_path: Path
    collections_dir: Path
    data_dir: Path
    raw_dir: Path
    raw_inspire_dir: Path
    pdf_dir: Path
    parsed_dir: Path
    index_dir: Path
    exports_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> WorkspacePaths:
        data_dir = root / "data"
        raw_dir = data_dir / "raw"
        return cls(
            workdir=root,
            db_dir=root / "db",
            db_path=root / "db" / "hep_rag_v2.db",
            api_db_path=root / "db" / "hep_rag_api.db",
            collections_dir=root / "collections",
            data_dir=data_dir,
            raw_dir=raw_dir,
            raw_inspire_dir=raw_dir / "inspire",
            pdf_dir=data_dir / "pdfs",
            parsed_dir=data_dir / "parsed",
            index_dir=root / "indexes",
            exports_dir=root / "exports",
        )


WORKDIR: Path
DB_DIR: Path
DB_PATH: Path
API_DB_PATH: Path
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
    global API_DB_PATH
    global COLLECTIONS_DIR
    global DATA_DIR
    global RAW_DIR
    global RAW_INSPIRE_DIR
    global PDF_DIR
    global PARSED_DIR
    global INDEX_DIR
    global EXPORTS_DIR

    wp = WorkspacePaths.from_root(_workspace_root)
    WORKDIR = wp.workdir
    DB_DIR = wp.db_dir
    DB_PATH = wp.db_path
    API_DB_PATH = wp.api_db_path
    COLLECTIONS_DIR = wp.collections_dir
    DATA_DIR = wp.data_dir
    RAW_DIR = wp.raw_dir
    RAW_INSPIRE_DIR = wp.raw_inspire_dir
    PDF_DIR = wp.pdf_dir
    PARSED_DIR = wp.parsed_dir
    INDEX_DIR = wp.index_dir
    EXPORTS_DIR = wp.exports_dir


_refresh()
