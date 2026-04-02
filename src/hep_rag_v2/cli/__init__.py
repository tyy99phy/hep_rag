"""CLI sub-package for hep-rag-v2.

Re-exports every public name so that ``from hep_rag_v2.cli import main``
and ``from hep_rag_v2 import cli; cli.cmd_search_hybrid(...)`` keep working.
"""
from __future__ import annotations

# -- constants & shared utilities ------------------------------------------
from ._common import (
    AUDIT_PATTERNS,
    INSPIRE_API,
    READINESS_THRESHOLDS,
    http_get_json,
    load_collection,
    parse_year_buckets,
    save_raw_payload,
)

# -- workspace commands ----------------------------------------------------
from .workspace import (
    cmd_collections,
    cmd_init,
    cmd_init_config,
    cmd_status,
)

# -- ingest commands -------------------------------------------------------
from .ingest import (
    cmd_ask,
    cmd_bootstrap_legacy_corpus,
    cmd_enrich_inspire_metadata,
    cmd_fetch_papers,
    cmd_import_mineru,
    cmd_import_pdg,
    cmd_ingest_metadata,
    cmd_ingest_online,
    cmd_query,
    cmd_reparse_pdfs,
    cmd_resolve_citations,
    cmd_search_works,
)

# -- search / index commands -----------------------------------------------
from .search import (
    cmd_build_graph,
    cmd_build_search_index,
    cmd_build_vector_index,
    cmd_search_bm25,
    cmd_search_hybrid,
    cmd_search_vector,
    cmd_show_graph,
    cmd_sync_chroma_index,
)

# -- inspect / audit commands ----------------------------------------------
from .inspect import (
    cmd_audit_document,
    cmd_show_document,
)

# -- parser & entry-point --------------------------------------------------
from ._parser import build_parser, main

__all__ = [
    # constants
    "INSPIRE_API",
    "AUDIT_PATTERNS",
    "READINESS_THRESHOLDS",
    # shared utilities
    "http_get_json",
    "load_collection",
    "parse_year_buckets",
    "save_raw_payload",
    # workspace
    "cmd_init",
    "cmd_collections",
    "cmd_status",
    "cmd_init_config",
    # ingest
    "cmd_fetch_papers",
    "cmd_ingest_online",
    "cmd_query",
    "cmd_ask",
    "cmd_ingest_metadata",
    "cmd_resolve_citations",
    "cmd_search_works",
    "cmd_enrich_inspire_metadata",
    "cmd_import_mineru",
    "cmd_import_pdg",
    "cmd_reparse_pdfs",
    "cmd_bootstrap_legacy_corpus",
    # search / index
    "cmd_build_search_index",
    "cmd_search_bm25",
    "cmd_build_vector_index",
    "cmd_search_vector",
    "cmd_sync_chroma_index",
    "cmd_search_hybrid",
    "cmd_build_graph",
    "cmd_show_graph",
    # inspect / audit
    "cmd_show_document",
    "cmd_audit_document",
    # entry-point
    "build_parser",
    "main",
]
