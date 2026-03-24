# hep-rag v2 Plan

## 1. Goal

Build a graph-first HEP literature system that can support:

- large-scale paper organization
- survey drafting support
- literature map building
- idea extraction and bridge-paper discovery
- evidence-grounded retrieval over selected full text

This project should not inherit the tutoring, learner-state, or teaching workflow from `icf-tutor-rag`.

## 2. What To Reuse From icf-tutor-rag

Reuse these ideas:

- structured document contracts instead of raw text blobs
- explicit nodes and edges rather than opaque vector-only storage
- section/block/formula/asset/chunk hierarchy for full-text subsets
- graph export as a downstream view, not the only source of truth
- local-first pipeline with inspectable artifacts on disk

Do not reuse these parts:

- learner/session/question/confusion tables
- review queue tied to teaching readiness
- tutoring orchestration
- pedagogy-specific chunk roles unless they also help scientific retrieval

## 3. Core Design Decision

The system should use a two-layer corpus model.

Layer A: global paper graph

- target scale: about 100k works
- inputs: title, abstract, authors, collaboration, venue, year, categories, keywords, references, citation counts
- purpose: clustering, trend mapping, bridge detection, survey scaffolding, candidate set generation

Layer B: full-text evidence graph

- target scale: much smaller at first
- inputs: parsed full text, sections, blocks, formulas, figures, tables, chunks
- purpose: grounded answering, method comparison, formula lookup, figure-aware evidence, detailed survey notes

This means the initial system should not try to fully parse 100k PDFs.

## 4. Immediate Recommendation

Do not choose between:

- abstract-only for everything
- full-text for a tiny corpus

Instead do both, but at different layers.

Recommended start:

- ingest 100k metadata plus abstracts plus references
- build the global graph on top of that
- separately build a full-text pilot set of about 300 papers
- expand the full-text set to 1k to 5k after retrieval and graph contracts are stable

Reason:

- abstract-plus-reference data is enough to build a useful scholarly graph
- full-text parsing is the expensive and noisy part
- survey and paper organization tasks benefit early from citation and similarity structure
- detailed technical evidence needs full text, but only for a much smaller working set

## 5. Why This Is Better Than The Current hep-rag

Current `hep-rag` is still centered on chunk retrieval:

- `papers/chunks/citations/captions` is too shallow for a long-lived literature graph
- retrieval is local chunk similarity first
- graph structure is not yet the main abstraction

The upgrade should invert that priority:

1. build paper graph first
2. use full-text chunks as evidence attached to graph nodes
3. let retrieval route across graph structure before diving into chunk-level evidence

## 6. Constraints

The first version should avoid LLM-driven chunking, embedding generation, or relation extraction as the default pipeline.

That means:

- chunking should be deterministic and parser-driven
- topic and phrase extraction should start with rule-based and statistical methods
- graph edges should come from metadata, parser structure, and numeric similarity
- LLM should be optional at the very end for synthesis, not required for ingestion

This will reduce semantic richness, but it is the right tradeoff for:

- scale
- repeatability
- cost control
- debuggability

## 7. Proposed Node Model

Minimum node families for v2:

- `work`
- `author`
- `collaboration`
- `venue`
- `topic`
- `collection`

Full-text pilot node families:

- `document`
- `section`
- `block`
- `formula`
- `asset`
- `chunk`

Notes:

- `work` is the canonical scholarly paper entity
- `document` is the parsed full-text manifestation of a work
- one work may have zero or one parsed document in early versions

## 8. Proposed Edge Model

Paper-graph edges that do not need LLMs:

- `CITES`
- `CITED_BY`
- `AUTHORED_BY`
- `IN_COLLABORATION`
- `PUBLISHED_IN`
- `HAS_TOPIC`
- `IN_COLLECTION`
- `SIMILAR_TO`
- `BIBLIOGRAPHICALLY_COUPLED`
- `CO_CITED_WITH`

Full-text edges for the pilot layer:

- `HAS_DOCUMENT`
- `HAS_SECTION`
- `HAS_BLOCK`
- `HAS_FORMULA`
- `HAS_ASSET`
- `HAS_CHUNK`
- `NEXT_SECTION`
- `NEXT_CHUNK`
- `MENTIONS_TOPIC`
- `MENTIONS_REFERENCE`
- `HAS_SYMBOL`

Important rule:

- do not create free-form semantic relation types early
- keep relation contracts narrow and measurable

## 9. Data Sources

Primary metadata sources:

- INSPIRE for HEP-native metadata and references
- OpenAlex for cross-checks, broader citation coverage, topics, and external identifiers
- arXiv for direct PDF fallback and category metadata

Primary full-text sources:

- arXiv PDFs
- publisher-open PDFs
- XML full text when available
- parser outputs from MinerU or GROBID-backed workflows

## 10. Parser Strategy

For full text, the parser layer should remain deterministic.

Recommended order:

1. use metadata-only ingestion for all works
2. use MinerU for a curated full-text pilot set
3. add GROBID support for bibliography and structure extraction when MinerU coverage is weak
4. keep raw parser artifacts on disk and materialize a normalized block contract

Do not parse all PDFs up front.

Hard rule:

- `pdftotext` or `pdf2txt` must not remain in the main production pipeline
- dirty plain-text extraction can exist only as a one-off debug utility, or should be removed entirely
- all retrieval-grade full-text evidence must come from structured parser output

## 11. Chunking Strategy

Chunking should be parser-aware and rule-based.

For the full-text pilot:

- respect section boundaries
- preserve equation blocks as atomic units
- keep figure and table captions attached to nearby context
- generate multiple chunk views instead of one universal chunk stream

Recommended chunk types:

- `section_parent`
- `section_child`
- `formula_window`
- `asset_window`
- `abstract_chunk`

The current `hep-rag` block-aware chunker is a useful seed, but it needs a richer contract than only:

- `text`
- `section_hint`
- `page_hint`

## 12. Embedding Strategy

Do not keep the current hashing-vectorizer-only setup as the main path.

Use a hybrid retrieval stack:

- sparse: BM25 over titles, abstracts, section titles, chunk text, captions, formulas
- dense paper-level embeddings: scientific-document encoder such as SPECTER2
- dense evidence-level embeddings: start with a scientific encoder and evaluate cost later

Important detail:

- embeddings are not the graph
- embeddings should generate candidate neighbors, not define truth

Store:

- paper embeddings separately from chunk embeddings
- nearest-neighbor edges as materialized graph edges with scores and build version

## 13. Graph Construction Strategy

The graph should be built in three passes.

Pass 1: deterministic metadata graph

- create work nodes
- resolve references
- materialize citation edges
- attach authors, collaborations, venues, years, categories, keywords

Pass 2: algorithmic similarity graph

- compute paper embeddings
- build mutual kNN
- compute bibliographic coupling
- compute co-citation neighborhoods
- derive cluster labels from graph community detection

Pass 3: full-text evidence graph

- attach document structure for selected works
- materialize sections, blocks, formulas, assets, chunks
- attach mentions of references and topics when detectable without LLMs

This order matters.

If pass 1 is weak, everything above it becomes unstable.

## 14. Scale Plan For 100k Works

The 100k target should be treated as a metadata-scale problem first.

Recommended scale envelope:

- 100k works in relational metadata tables
- 100k abstract embeddings
- kNN graph stored as edge tables
- full text only for a smaller active subset

Do not attempt:

- 100k full parsed PDFs in the first milestone
- all-pairs similarity
- graph edges based on unrestricted phrase extraction from every chunk

Instead:

- use batched ingestion
- materialize embeddings incrementally
- build approximate nearest neighbors
- version graph builds
- separate source-of-truth tables from export tables

## 15. Storage Recommendation

For the first serious version:

- keep relational source-of-truth tables local and explicit
- store parser artifacts and indexes on disk
- allow graph export to Neo4j, but do not require Neo4j for the core pipeline

Pragmatic choice:

- SQLite is acceptable for a local 100k-work MVP if write patterns stay simple
- if full-text and edge volume grow quickly, move the canonical DB to PostgreSQL without changing logical contracts

So the plan should optimize for schema portability, not for one specific database engine.

## 16. Proposed Schema Split

Metadata graph tables:

- `works`
- `work_ids`
- `work_authors`
- `authors`
- `venues`
- `topics`
- `work_topics`
- `citations`
- `similarity_edges`
- `bibliographic_coupling_edges`
- `co_citation_edges`
- `collections`
- `collection_works`
- `graph_builds`

Full-text tables:

- `documents`
- `document_sections`
- `blocks`
- `formulas`
- `assets`
- `chunks`
- `chunk_embeddings`
- `chunk_topic_mentions`

Versioning and observability:

- `ingest_runs`
- `parse_runs`
- `embedding_runs`
- `graph_build_runs`
- `quality_reports`

## 17. Initial Product Surfaces

The first useful surfaces should not be chat-first.

Build these first:

1. paper explorer
2. citation and similarity neighborhood viewer
3. cluster and topic timeline view
4. survey workspace
5. evidence-backed retrieval

Then add:

- literature-summary drafting
- compare papers by topic or method
- bridge-paper finder
- contradiction and tension finder

## 18. Evaluation

Need separate evaluation for each layer.

Metadata graph evaluation:

- reference resolution rate
- coverage of authors and venues
- graph connectivity
- cluster coherence

Retrieval evaluation:

- paper retrieval for topic queries
- bridge-paper retrieval
- citation recommendation quality
- section and chunk retrieval on the full-text pilot

Survey utility evaluation:

- can the system identify key papers for a topic
- can it surface historical progression
- can it provide evidence-backed notes for a survey section

## 19. Phased Build Plan

Phase 0: planning and contracts

- freeze node and edge vocabulary
- define source-of-truth schema
- define on-disk artifact layout
- define evaluation set

Phase 1: metadata graph MVP

- ingest works from INSPIRE
- add reference ingestion
- normalize authors, collaborations, venues
- build citation graph
- add paper-level sparse retrieval

Exit criteria:

- 10k to 30k works stable
- citation graph populated
- paper search and neighborhood inspection available

Phase 2: algorithmic graph enrichment

- add paper embeddings
- build approximate nearest-neighbor graph
- build bibliographic coupling and co-citation edges
- add topic clustering and cluster summaries without LLM dependency

Exit criteria:

- graph neighborhoods useful for survey navigation
- related-paper discovery beats title-only search

Phase 3: full-text pilot

- select 300 papers from a focused domain slice
- parse them with MinerU into normalized structure
- materialize sections, formulas, assets, and chunks
- attach full-text evidence to work nodes
- do not allow fallback to plain `pdftotext` output for this layer

Exit criteria:

- evidence-backed retrieval works for the pilot slice
- formula and figure contexts remain inspectable

Phase 4: synthesis layer

- build survey workspace and note graph
- support comparison tables and topic dossiers
- add optional LLM synthesis on top of graph-routed evidence

Exit criteria:

- draft survey sections can be backed by explicit citations and evidence bundles

## 20. First Concrete Milestone

The next implementation milestone should be:

"Turn `hep-rag` from a 100-paper chunk retriever into a 10k-work paper graph system."

That means the first code iteration should focus on:

- schema redesign
- metadata ingest rebuild
- reference capture
- graph edge materialization
- paper-level retrieval

Not on:

- better answer prompting
- richer chat UX
- end-to-end full-text parsing at scale

## 21. Open Questions To Resolve Before Coding

- should the canonical metadata graph stay in SQLite for v1, or jump directly to PostgreSQL
- what is the exact primary corpus definition for the first 10k and first 100k
- which paper embedding model is acceptable for local batch processing
- whether OpenAlex should be a primary source or only a reconciliation source
- whether GROBID should be introduced in phase 1 or phase 3

## 22. Recommended Immediate Next Steps

1. Define the v2 schema and migration target.
2. Add `references` and related metadata fields to collection ingestion immediately.
3. Remove `pdf2txt` from the planned production path and make MinerU the explicit full-text default.
4. Build a metadata-only prototype over 1k works before touching more parser code.
5. Create a paper-graph inspection UI before reworking answer generation.
6. Select a 300-paper full-text pilot set only after the metadata graph is stable.
