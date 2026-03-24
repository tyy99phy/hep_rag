# HEP Corpus Bootstrap Status

Date: 2026-03-22

## Summary

The local `cms_rare_decay` pilot corpus has now been upgraded from the rough legacy `hep_rag` assets into the `hep_rag_v2` graph-first workspace.

This was done without LLM-driven ingestion.

The current pipeline is:

1. import legacy metadata from the old `hep.db`
2. import legacy MinerU parse bundles
3. materialize structured documents, blocks, formulas, assets, and chunks
4. audit parser noise and retrieval readiness
5. refresh existing works from INSPIRE to recover references
6. rebuild citation-derived graph edges

## Current Corpus State

Current DB snapshot after bootstrap and INSPIRE enrichment:

- works: 101
- documents: 100
- chunks: 3703
- formulas: 227
- assets: 768
- citations: 3862
- resolved citations: 70
- works with any outgoing citations: 98
- works with resolved outgoing citations: 49
- works with resolved incoming citations: 31

Search state:

- work_search: 101
- chunk_search: 3703
- formula_search: 227
- asset_search: 768

## Full-Text Quality

After two rounds of rule-based cleanup and one round of audit-pattern refinement:

- 87 / 100 documents are `ready_for_next_phase`
- 13 / 100 documents still fail the current readiness threshold

The remaining failures are concentrated in a small set of older or noisier parses, not spread across the whole corpus.

Dominant residual issues:

- double punctuation and OCR-style punctuation collisions
- a few remaining orphaned reference phrases
- a small number of literal `frac` remnants
- 2 documents with weak equation-placeholder structure

Interpretation:

- the full-text pilot is good enough to support retrieval and evidence grounding work now
- the remaining 13 documents should be treated as cleanup backlog, not as a blocker

## Citation Graph State

After INSPIRE metadata refresh:

- citation rows written: 3862
- resolved citations inside the current 101-paper corpus: 70

High-precision graph build at `min_shared=2` produced:

- bibliographic coupling edges: 8
- co-citation edges: 5

For the current small pilot corpus, that threshold is too sparse for practical exploration.

The active graph was therefore rebuilt at `min_shared=1`:

- bibliographic coupling edges: 128
- co-citation edges: 21

This is the right operating point for the current 101-paper pilot.

Important note:

- keep `min_shared=2` as the safer default for larger corpora
- use `min_shared=1` for small curated pilot collections

## What This Means

The project is now past the "rough parser demo" stage.

It already has:

- a graph-native scholarly schema
- structured full-text evidence storage
- deterministic chunking
- auditable parser quality gates
- BM25 retrieval over works, chunks, formulas, and assets
- citation-derived graph neighborhoods

It does not yet have:

- a large metadata-first global corpus
- dense paper-level similarity edges
- bibliography extraction beyond what INSPIRE provides
- a workflow layer for survey drafting or bridge-paper discovery

## Recommended Next Step

The next milestone should be:

`metadata-first scale-out`

Concretely:

1. expand the metadata graph well beyond the current 101 papers using INSPIRE queries
2. keep full-text parsing limited to a smaller high-value subset
3. use the 87 ready documents as the initial grounded-evidence pilot set
4. add paper-level algorithmic similarity later, after the larger metadata graph is stable

## Useful Commands

Bootstrap legacy corpus:

```bash
PYTHONPATH=src python3 -m hep_rag_v2.cli bootstrap-legacy-corpus \
  --legacy-db /path/to/legacy/hep-rag/db/hep.db \
  --parsed-root /path/to/legacy/hep-rag/data/parsed/cms_rare_decay \
  --collection cms_rare_decay \
  --replace \
  --audit-limit 5
```

Refresh references from INSPIRE:

```bash
PYTHONPATH=src python3 -m hep_rag_v2.cli enrich-inspire-metadata \
  --collection cms_rare_decay \
  --sleep 0.05
```

Build a strict graph:

```bash
PYTHONPATH=src python3 -m hep_rag_v2.cli build-graph \
  --collection cms_rare_decay \
  --min-shared 2
```

Build a pilot-exploration graph:

```bash
PYTHONPATH=src python3 -m hep_rag_v2.cli build-graph \
  --collection cms_rare_decay \
  --min-shared 1
```
