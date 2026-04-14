from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.modules.pop("hep_rag_v2", None)
sys.path.insert(0, str(SRC))
importlib.invalidate_caches()

from hep_rag_v2.retrieval_adapter import (
    TypedRetrievalMetadata,
    TypedRetrievalResult,
    adapt_chunk_hit,
    build_retrieval_shell,
    adapt_work_hit,
    normalize_retrieval_payload,
)


class TestRetrievalAdapter(unittest.TestCase):
    def test_adapt_work_hit_normalizes_common_fields(self) -> None:
        row = {
            "work_id": 101,
            "raw_title": "Observation of a rare decay",
            "abstract": "A measurement with strong evidence.",
            "year": 2026,
            "canonical_source": "inspire",
            "canonical_id": "101",
            "primary_source_url": "https://inspirehep.net/literature/101",
            "primary_pdf_url": "https://example.test/101.pdf",
            "hybrid_score": 0.91,
            "bm25_rank": 1,
            "vector_rank": 2,
            "rank": 1,
            "search_type": "hybrid",
            "query_group_hits": 2,
            "query_group_coverage": 1.0,
            "family_id": 77,
            "family_label": "CMS rare decay family",
            "family_primary_work_id": 101,
            "related_versions": [{"work_id": 102, "canonical_id": "101v2"}],
        }

        result = adapt_work_hit(row, collection="cms_rare_decay")

        self.assertIsInstance(result, TypedRetrievalResult)
        self.assertEqual(result.object_type, "work")
        self.assertEqual(result.object_id, "work:101")
        self.assertEqual(result.work_id, 101)
        self.assertIsNone(result.chunk_id)
        self.assertEqual(result.title, "Observation of a rare decay")
        self.assertEqual(result.content, "A measurement with strong evidence.")
        self.assertEqual(result.canonical_source, "inspire")
        self.assertEqual(result.canonical_id, "101")
        self.assertEqual(result.collection, "cms_rare_decay")
        self.assertEqual(result.evidence_key, "work:101")
        self.assertIsInstance(result.metadata, TypedRetrievalMetadata)
        self.assertEqual(result.metadata.lane, "works")
        self.assertEqual(result.metadata.rank, 1)
        self.assertEqual(result.metadata.hybrid_score, 0.91)
        self.assertEqual(result.metadata.family_label, "CMS rare decay family")
        self.assertEqual(result.metadata.related_versions[0]["canonical_id"], "101v2")

    def test_adapt_chunk_hit_preserves_chunk_context(self) -> None:
        row = {
            "chunk_id": 301,
            "work_id": 101,
            "chunk_role": "paragraph",
            "section_hint": "4 Results",
            "page_hint": "p.7",
            "clean_text": "Observed significance is 3.2 sigma in the signal region.",
            "raw_title": "Observation of a rare decay",
            "canonical_source": "inspire",
            "canonical_id": "101",
            "hybrid_score": 0.83,
            "bm25_rank": 1,
            "vector_rank": 1,
            "rank": 1,
            "search_type": "hybrid",
            "query_group_hits": 1,
            "query_group_coverage": 0.5,
            "family_id": 77,
            "family_primary_work_id": 101,
            "related_versions": [],
        }

        result = adapt_chunk_hit(row, collection="cms_rare_decay")

        self.assertEqual(result.object_type, "chunk")
        self.assertEqual(result.object_id, "chunk:301")
        self.assertEqual(result.work_id, 101)
        self.assertEqual(result.chunk_id, 301)
        self.assertEqual(result.title, "Observation of a rare decay")
        self.assertEqual(result.content, "Observed significance is 3.2 sigma in the signal region.")
        self.assertEqual(result.section_hint, "4 Results")
        self.assertEqual(result.page_hint, "p.7")
        self.assertEqual(result.evidence_key, "chunk:301")
        self.assertEqual(result.metadata.lane, "chunks")
        self.assertEqual(result.metadata.rank, 1)
        self.assertEqual(result.metadata.family_id, 77)

    def test_normalize_payload_includes_typed_reasoning_objects(self) -> None:
        payload = {
            "query": "vector boson scattering anomalies",
            "collection": "default",
            "requested_target": "result_object",
            "routing": {
                "target": "result_object",
                "graph_expand": 1,
                "reasons": ["manual_target"],
            },
            "result_objects": [
                {
                    "result_id": 41,
                    "title": "Same-sign WW excess in fiducial region",
                    "summary": "Observed fiducial cross section is above the SM central value.",
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                    "score": 0.93,
                    "rank": 1,
                }
            ],
            "method_objects": [
                {
                    "method_id": 17,
                    "title": "Matrix-element unfolding",
                    "summary": "Differential measurements use matrix-element-inspired unfolding.",
                    "score": 0.74,
                    "rank": 2,
                }
            ],
        }

        typed = normalize_retrieval_payload(payload)

        self.assertEqual(typed.metadata.target, "result_object")
        self.assertEqual(len(typed.typed_objects), 2)
        self.assertEqual([item.object_type for item in typed.typed_objects], ["result_object", "method_object"])
        self.assertEqual(typed.primary_items()[0].object_type, "result_object")

        shell = build_retrieval_shell(payload)
        self.assertEqual([item["object_type"] for item in shell["results"]], ["result_object", "method_object"])
        self.assertEqual(shell["evidence_registry"]["items"][0]["evidence_key"], "result_object:41")


if __name__ == "__main__":
    unittest.main()
