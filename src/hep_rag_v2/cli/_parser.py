from __future__ import annotations

import argparse

from hep_rag_v2.vector import DEFAULT_VECTOR_MODEL

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
    cmd_benchmark_manifest,
)
from .inspect import (
    cmd_audit_document,
    cmd_show_document,
)
from .search import (
    cmd_build_graph,
    cmd_build_search_index,
    cmd_build_vector_index,
    cmd_search_bm25,
    cmd_search_hybrid,
    cmd_search_vector,
    cmd_show_graph,
    cmd_sync_chroma_index,
    cmd_sync_graph,
    cmd_sync_search,
    cmd_sync_vectors,
)
from .workspace import (
    cmd_collections,
    cmd_init,
    cmd_init_config,
    cmd_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hep-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-config", help="Write a user config file and initialize an empty workspace")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Workspace root for DB/data/indexes")
    s.add_argument("--collection", default=None, help="Initial collection name")
    s.add_argument("--force", action="store_true", help="Overwrite existing config file")
    s.set_defaults(func=cmd_init_config)

    s = sub.add_parser("fetch-papers", help="Search INSPIRE online and show candidate papers before ingest")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--max-parallelism", type=int, default=None, help="Override online query/download parallelism")
    s.set_defaults(func=cmd_fetch_papers)

    s = sub.add_parser("ingest-online", help="Search online, download PDFs, optionally parse with MinerU, then build indices")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--max-parallelism", type=int, default=None, help="Override online query/download parallelism")
    s.add_argument("--download-limit", type=int, default=None, help="Maximum number of PDFs to download")
    s.add_argument("--parse-limit", type=int, default=None, help="Maximum number of PDFs to send to MinerU")
    s.add_argument("--replace-existing", action="store_true", help="Rebuild already materialized documents")
    s.add_argument("--skip-parse", action="store_true", help="Only ingest metadata and PDFs, skip MinerU parsing")
    s.add_argument("--skip-index", action="store_true", help="Skip rebuilding search/vector indices after ingest")
    s.add_argument("--skip-graph", action="store_true", help="Skip rebuilding graph edges after ingest")
    s.set_defaults(func=cmd_ingest_online)

    s = sub.add_parser("reparse-pdfs", help="Submit cached local PDFs that still need MinerU materialization")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--limit", type=int, default=None, help="Maximum number of cached PDFs to reparse")
    s.add_argument("--work-id", type=int, action="append", default=None, help="Restrict to specific work_id values")
    s.add_argument("--replace-existing", action="store_true", help="Rebuild already materialized documents from local PDFs")
    s.add_argument("--skip-index", action="store_true", help="Skip rebuilding search/vector indices after reparse")
    s.add_argument("--skip-graph", action="store_true", help="Skip rebuilding graph edges after reparse")
    s.set_defaults(func=cmd_reparse_pdfs)

    s = sub.add_parser("query", help="Run config-driven retrieval and return structured evidence")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--target", choices=["auto", "works", "chunks", "community", "ontology"], default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--model", default=None, help="Override embedding model")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("ask", help="Run retrieval and synthesize an answer with either an OpenAI-compatible API or a local Transformers model")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default=None, help="Override collection name")
    s.add_argument("--target", choices=["auto", "works", "chunks", "community", "ontology"], default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--model", default=None, help="Override embedding model")
    s.add_argument("--mode", choices=["answer", "survey", "idea"], default="answer")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("benchmark-manifest", help="Write the RAG effect benchmark manifest, including the thinking-engine trace scenario")
    s.add_argument("--output", default=None, help="Manifest output path; defaults to <workspace>/.omx/benchmarks/<model>-rag-effect-manifest.json")
    s.add_argument("--model-label", default="weak-model", help="Model label used in the manifest")
    s.set_defaults(func=cmd_benchmark_manifest)

    s = sub.add_parser("init", help="Initialize local database and directories")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("collections", help="List available collection configs")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.set_defaults(func=cmd_collections)

    s = sub.add_parser("status", help="Show metadata graph snapshot")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("ingest-metadata", help="Ingest INSPIRE metadata into the graph schema")
    s.add_argument("--collection", default="cms_rare_decay")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--page-size", type=int, default=25)
    s.add_argument("--timeout", type=int, default=60)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--sleep", type=float, default=0.2)
    s.add_argument("--year-buckets", default=None)
    s.set_defaults(func=cmd_ingest_metadata)

    s = sub.add_parser("resolve-citations", help="Backfill unresolved citation targets")
    s.set_defaults(func=cmd_resolve_citations)

    s = sub.add_parser("search-works", help="Search titles and abstracts in the metadata graph")
    s.add_argument("query")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_search_works)

    s = sub.add_parser("build-search-index", help="Rebuild SQLite FTS5 BM25 search indices")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "works", "chunks", "formulas", "assets", "structure"], default="all")
    s.set_defaults(func=cmd_build_search_index)

    s = sub.add_parser("sync-search", help="Incrementally sync SQLite FTS5 BM25 search indices")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "works", "chunks", "formulas", "assets", "structure"], default="all")
    s.add_argument("--scope", choices=["all", "dirty"], default="dirty")
    s.add_argument("--collection", default=None)
    s.add_argument("--updated-since", default=None)
    s.set_defaults(func=cmd_sync_search)

    s = sub.add_parser("search-bm25", help="Search works, chunks, formulas, or assets with SQLite FTS5 BM25")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["works", "chunks", "formulas", "assets"], default="works")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_search_bm25)

    s = sub.add_parser("build-vector-index", help="Build local vector indices for works and/or chunks")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "works", "chunks"], default="all")
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--dim", type=int, default=768)
    s.set_defaults(func=cmd_build_vector_index)

    s = sub.add_parser("sync-vectors", help="Incrementally sync local vector indices")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "works", "chunks"], default="all")
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--dim", type=int, default=768)
    s.add_argument("--scope", choices=["all", "dirty"], default="dirty")
    s.add_argument("--collection", default=None)
    s.add_argument("--updated-since", default=None)
    s.set_defaults(func=cmd_sync_vectors)

    s = sub.add_parser("sync-chroma-index", help="Mirror local vector indices into an optional Chroma vector store")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "works", "chunks"], default="all")
    s.add_argument("--collection", default=None)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--chroma-dir", default=None)
    s.add_argument("--batch-size", type=int, default=256)
    s.set_defaults(func=cmd_sync_chroma_index)

    s = sub.add_parser("search-vector", help="Search works or chunks with the local vector index")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["works", "chunks"], default="works")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--backend", choices=["local", "chroma"], default="local")
    s.add_argument("--chroma-dir", default=None)
    s.set_defaults(func=cmd_search_vector)

    s = sub.add_parser("search-hybrid", help="Search with BM25 plus vector retrieval; auto mode can route broad queries to works")
    s.add_argument("query")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["auto", "works", "chunks"], default="auto")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--graph-expand", type=int, default=None)
    s.add_argument("--seed-limit", type=int, default=5)
    s.set_defaults(func=cmd_search_hybrid)

    s = sub.add_parser("enrich-inspire-metadata", help="Refresh existing works from INSPIRE and backfill citation graph data")
    s.add_argument("--collection", default="cms_rare_decay")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--force", action="store_true", help="Refresh works even if citations already exist")
    s.add_argument("--timeout", type=int, default=60)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--sleep", type=float, default=0.1)
    s.add_argument("--skip-search", action="store_true")
    s.add_argument("--skip-graph", action="store_true")
    s.add_argument("--min-shared", type=int, default=2)
    s.set_defaults(func=cmd_enrich_inspire_metadata)

    s = sub.add_parser("build-graph", help="Build citation and embedding-based graph edges")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "bibliographic-coupling", "co-citation", "similarity"], default="all")
    s.add_argument("--collection", default=None)
    s.add_argument("--min-shared", type=int, default=2)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--min-score", type=float, default=0.35)
    s.set_defaults(func=cmd_build_graph)

    s = sub.add_parser("sync-graph", help="Incrementally sync graph edges")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--target", choices=["all", "bibliographic-coupling", "co-citation", "similarity"], default="all")
    s.add_argument("--min-shared", type=int, default=2)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--min-score", type=float, default=0.35)
    s.add_argument("--scope", choices=["all", "dirty"], default="dirty")
    s.add_argument("--collection", default=None)
    s.add_argument("--updated-since", default=None)
    s.set_defaults(func=cmd_sync_graph)

    s = sub.add_parser("show-graph", help="Inspect graph neighbors for a work")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--edge-kind", choices=["all", "bibliographic-coupling", "co-citation", "similarity"], default="all")
    s.add_argument("--collection", default=None)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--model", default=DEFAULT_VECTOR_MODEL)
    s.set_defaults(func=cmd_show_graph)

    s = sub.add_parser("import-pdg", help="Register a PDG PDF into the archival ingest flow")
    s.add_argument("--config", default=None, help="Path to hep-rag.yaml")
    s.add_argument("--workspace", default=None, help="Override workspace root")
    s.add_argument("--collection", default="pdg", help="Target collection name")
    s.add_argument("--edition", default=None, help="PDG edition year, e.g. 2024")
    s.add_argument("--source", default=None, help="Local PDG MinerU bundle / parsed source path for direct import")
    s.add_argument("--source-id", default=None, help="Stable PDG source id when importing a local bundle")
    s.add_argument("--title", default=None, help="Display title for a local PDG bundle import")
    s.add_argument("--pdf", default=None, help="Existing local PDG PDF to stage into the workspace")
    s.add_argument("--download", action="store_true", help="Attempt remote PDG PDF download when no local PDF is provided")
    s.set_defaults(func=cmd_import_pdg)

    s = sub.add_parser("import-mineru", help="Import MinerU output and materialize a structured document")
    s.add_argument("--source", required=True, help="ZIP, raw MinerU output dir, manifest.json, or parsed dir")
    s.add_argument("--collection", default=None, help="Override target parsed collection dir")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--replace", action="store_true")
    s.add_argument("--chunk-size", type=int, default=2400)
    s.add_argument("--overlap-blocks", type=int, default=1)
    s.add_argument("--section-parent-char-limit", type=int, default=12000)
    s.set_defaults(func=cmd_import_mineru)

    s = sub.add_parser("bootstrap-legacy-corpus", help="Bootstrap metadata and MinerU parses from the legacy hep_rag corpus")
    s.add_argument("--legacy-db", required=True, help="Path to legacy hep_rag SQLite DB")
    s.add_argument("--parsed-root", required=True, help="Root directory containing legacy parsed MinerU manifests")
    s.add_argument("--collection", default="cms_rare_decay")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--replace", action="store_true")
    s.add_argument("--chunk-size", type=int, default=2400)
    s.add_argument("--overlap-blocks", type=int, default=1)
    s.add_argument("--section-parent-char-limit", type=int, default=12000)
    s.add_argument("--audit-limit", type=int, default=5)
    s.add_argument("--skip-graph", action="store_true")
    s.add_argument("--min-shared", type=int, default=2)
    s.set_defaults(func=cmd_bootstrap_legacy_corpus)

    s = sub.add_parser("show-document", help="Inspect a materialized document and sample clean chunks")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_show_document)

    s = sub.add_parser("audit-document", help="Audit parser noise and retrieval readiness for a materialized document")
    s.add_argument("--work-id", type=int, default=None)
    s.add_argument("--id-type", choices=["inspire", "arxiv", "doi"], default=None)
    s.add_argument("--id-value", default=None)
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_audit_document)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
